import random
from pathlib import Path
import pandas as pd
import os
import matplotlib.pyplot as plt
from glob import glob
from hashlib import sha1

from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.repository.datasets import get_dataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor, EarlyStopping

from estimator import LagGPTEstimator
from pathlib import Path

import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("filename", help = "YAML config file.")
parser.add_argument("--suffix", default="", type=str)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--dataset_path", default="/home/toolkit/datasets", type=str)
parser.add_argument("--precision", default="32", type=str, choices=["32", "16", "bf16-mixed"])
args = parser.parse_args()


with open(args.filename, mode="rt", encoding="utf-8") as file:
    config = yaml.safe_load(file)

pl.seed_everything(args.seed)

class CombinedDatasetIterator:
    def __init__(self, datasets, seed, weights):
        self._datasets = [iter(el) for el in datasets]
        self._weights = weights
        self._rng = random.Random(seed)

    def __next__(self):
        (dataset,) = self._rng.choices(self._datasets, weights=self._weights, k=1)
        return next(dataset)

class CombinedDataset:
    def __init__(self, datasets, seed=None, weights=None):
        self._seed = seed
        self._datasets = datasets
        self._weights = weights
        n_datasets = len(datasets)
        if weights is None:
            self._weights = [1 / n_datasets] * n_datasets

    def __iter__(self):
        return CombinedDatasetIterator(self._datasets, self._seed, self._weights)
    
    def __len__(self):
        return sum([len(ds) for ds in self._datasets])

print("Loading data...")
dataset_path = Path(args.dataset_path)
gluonts_ds = [
        get_dataset("airpassengers", path=dataset_path).train,
        get_dataset("australian_electricity_demand", path=dataset_path).train,
        get_dataset("car_parts_without_missing", path=dataset_path).train,
        get_dataset("cif_2016", path=dataset_path).train,
        get_dataset("covid_deaths", path=dataset_path).train,
        get_dataset("electricity", path=dataset_path).train,
        get_dataset("electricity_weekly", path=dataset_path).train,
        get_dataset("exchange_rate", path=dataset_path).train,
        get_dataset("fred_md", path=dataset_path).train,
        get_dataset("hospital", path=dataset_path).train,
        get_dataset("kaggle_web_traffic_weekly", path=dataset_path).train,
        get_dataset("kdd_cup_2018_without_missing", path=dataset_path).train,
        get_dataset("london_smart_meters_without_missing", path=dataset_path).train,
        get_dataset("nn5_daily_with_missing", path=dataset_path).train,
        get_dataset("nn5_weekly", path=dataset_path).train,
        get_dataset("pedestrian_counts", path=dataset_path).train,
        get_dataset("rideshare_without_missing", path=dataset_path).train,
        get_dataset("saugeenday", path=dataset_path).train,
        get_dataset("solar-energy", path=dataset_path).train,
        get_dataset("solar_10_minutes", path=dataset_path).train,
        get_dataset("solar_weekly", path=dataset_path).train,
        get_dataset("taxi_30min", path=dataset_path).train,
        get_dataset("temperature_rain_without_missing", path=dataset_path).train,
        get_dataset("tourism_monthly", path=dataset_path).train,
        get_dataset("uber_tlc_daily", path=dataset_path).train,
        get_dataset("uber_tlc_hourly", path=dataset_path).train,
        get_dataset("vehicle_trips_without_missing", path=dataset_path).train,
        get_dataset("weather", path=dataset_path).train,
        get_dataset("wiki-rolling_nips", path=dataset_path).train,
        get_dataset("m4_daily", path=dataset_path).train,
        get_dataset("m4_hourly", path=dataset_path).train,
        get_dataset("m4_monthly", path=dataset_path).train,
        get_dataset("m4_quarterly", path=dataset_path).train,
        get_dataset("m4_yearly", path=dataset_path).train,
        get_dataset("wind_farms_without_missing", path=dataset_path).train,
]
dataset = CombinedDataset(gluonts_ds, weights=([sum([len(x["target"]) for x in d]) for d in gluonts_ds] if config["dataset"]["weighted"] else None)   )

val_dataset = get_dataset(config["dataset"]["val"], path=dataset_path).test
meta = get_dataset(config["dataset"]["val"], path=dataset_path).metadata

# Make the experiment_name
experiment_name = ("data-scaling-weighted-"+str(config["gpt"]["aug_prob"])+"_"+args.suffix if config["dataset"]["weighted"] else "data-scaling-uniform-"+str(config["gpt"]["aug_prob"])+"_"+args.suffix)
fulldir = "/home/toolkit/pytorch-transformer-ts/lag-gpt/" + experiment_name
os.makedirs(fulldir, exist_ok=True)

# Code to retrieve the version with the highest #epoch stored and restore it incl directory and its checkpoint
lightning_version_to_use, ckpt_path = None, None
max_epoch = -1
if "lightning_logs" in os.listdir(fulldir):
    for lightning_version in os.listdir(fulldir+"/lightning_logs/"):
        ckpts = glob(fulldir+"/lightning_logs/" + lightning_version + "/checkpoints/*.ckpt")
        if len(ckpts): 
            epoch = int(ckpts[0][ckpts[0].find("=")+1:ckpts[0].find("-step")])
            if epoch > max_epoch:
                lightning_version_to_use = lightning_version
                max_epoch = epoch
                ckpt_path = ckpts[0]
    if lightning_version_to_use: print("Using lightning_version", lightning_version_to_use, "with epoch", max_epoch, "restoring from checkpoint at path", ckpt_path)

# Make a CSV Logger with the specific version
experiment_logger = CSVLogger(save_dir=fulldir)
# experiment_logger = WandbLogger(name=experiment_name + "/" + str(args.seed), save_dir=fulldir, group=experiment_name, \
#                            tags=config["wandb"]["tags"] if "wandb" in config else [], \
#                             project=config["wandb"]["tags"] if "wandb" in config else "scaling_logs", \
#                             config=config, id=sha1(fulldir.encode("utf-8")).hexdigest()[:8])
logger = [experiment_logger]

# checkpoint_callback = ModelCheckpoint(
#     save_last=True,
#     verbose=True,
#     monitor='val_loss',
#     mode='min'
# )
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=20, verbose=True, mode="min")
callbacks=[early_stop_callback]

estimator = LagGPTEstimator(
    prediction_length=meta.prediction_length,
    context_length=config["gpt"]["context_length"], # block_size: int = 2048 
    batch_size=config["gpt"]["batch_size"], # 4
    n_layer=config["gpt"]["n_layer"],
    n_head=config["gpt"]["n_head"],
    n_embd=config["gpt"]["n_embd_per_head"]*config["gpt"]["n_head"], # 4096
    scaling=config["gpt"]["scaling"],
    aug_prob = config["gpt"]["aug_prob"],
    aug_rate = config["gpt"]["aug_rate"],
    num_batches_per_epoch= config["gpt"]["batches_per_epoch"],
    trainer_kwargs=dict(max_epochs=config["gpt"]["max_epochs"], accelerator="gpu", \
                        precision=args.precision, logger=logger, devices=[config["CUDA"]["device_id"]], \
                        callbacks=callbacks, check_val_every_n_epoch=config["check_val_every_n_epoch"] if "check_val_every_n_epoch" in config else 1),
    ckpt_path = ckpt_path
)

predictor = estimator.train(
    training_data=dataset, 
    validation_data=val_dataset,
    shuffle_buffer_length=1000
)


# loss_df = pd.read_csv("data-scaling-logs/"+experiment_name+"/version_"+str(experiment_version)+"/metrics.csv")
# train_loss = loss_df.dropna(subset=["train_loss"])
# val_loss = loss_df.dropna(subset=["val_loss"])

# fig, ax = plt.subplots()
# ax.plot(train_loss["epoch"], train_loss["train_loss"], label= "train")
# ax.plot(val_loss["epoch"], val_loss["val_loss"], label="val")
# ax.legend()
# ax.set_xscale("log")
# fig.savefig("data-scaling-logs/"+experiment_name+"/version_"+str(experiment_version)+"/loss.png") 
# plt.close(fig)  
