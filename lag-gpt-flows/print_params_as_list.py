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
from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor, EarlyStopping, LearningRateFinder, LearningRateMonitor

from estimator import LagGPTFlowsEstimator
from pathlib import Path
import pathlib

import argparse
import yaml
import gc
import torch

parser = argparse.ArgumentParser()
parser.add_argument("filename", help = "YAML config file.")
parser.add_argument("--suffix", default="", type=str, required=True, help="This can be useful information to distinguish runs, like `heads-scaling-5-heads`")
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--dataset_path", default="/home/toolkit/datasets", type=str)
parser.add_argument("--precision", default="32", type=str, choices=["32", "16", "bf16-mixed"])

# For scaling, I am putting all parameters here to save time in creating the yaml files
parser.add_argument("--layers", default=8, type=int)
parser.add_argument("--heads", default=4, type=int)
parser.add_argument("--dims_per_head", default=16, type=int)

# Also the batch size is by default set to a high value and found by the highest possible size at which 1 epoch runs
parser.add_argument("--batch_size", default=128, type=int)

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
dataset = CombinedDataset(gluonts_ds, weights=([sum([len(x["target"]) for x in d]) for d in gluonts_ds] if config["dataset"]["weighted"] else None), seed=args.seed) 

val_dataset = get_dataset(config["dataset"]["val"], path=dataset_path).test
meta = get_dataset(config["dataset"]["val"], path=dataset_path).metadata

# Make the experiment_name
experiment_name = ("data-scaling-weighted-"+str(config["gpt"]["aug_prob"])+"_"+args.suffix if config["dataset"]["weighted"] else "data-scaling-uniform-"+str(config["gpt"]["aug_prob"])+"_"+args.suffix)
fulldir = os.path.join(pathlib.Path(__file__).parent.resolve(), "scaling-logs-parameter-logging", experiment_name, str(args.seed)) # Always creates the experiment directory inside "lag-gpt-flows"
os.makedirs(fulldir, exist_ok=True)

fulldir_experiments = os.path.join(fulldir, "experiments")
os.makedirs(fulldir_experiments, exist_ok=True)

# Code to retrieve the version with the highest #epoch stored and restore it incl directory and its checkpoint
lightning_version_to_use, ckpt_path = None, None
max_epoch = -1
if "scaling_logs" in os.listdir(fulldir_experiments):
    ckpts = glob(fulldir_experiments+"/scaling_logs/" + sha1(fulldir_experiments.encode("utf-8")).hexdigest()[:8] + "/checkpoints/*.ckpt")
    if len(ckpts): ckpt_path = ckpts[0]
elif "lightning_logs" in os.listdir(fulldir_experiments):
    for lightning_version in os.listdir(fulldir_experiments+"/lightning_logs/"):
        ckpts = glob(fulldir_experiments+"/lightning_logs/" + lightning_version + "/checkpoints/*.ckpt")
        if len(ckpts): 
            epoch = int(ckpts[0][ckpts[0].find("=")+1:ckpts[0].find("-step")])
            if epoch > max_epoch:
                lightning_version_to_use = lightning_version
                max_epoch = epoch
                ckpt_path = ckpts[0]
    if lightning_version_to_use: print("Using lightning_version", lightning_version_to_use, "with epoch", max_epoch, "restoring from checkpoint at path", ckpt_path)

# Make a CSV Logger with the specific version
if "metrics" in config:
    if config["metrics"]["logger"] == "csv":
        experiment_logger = CSVLogger(save_dir=fulldir_experiments)
    else:
        tags = config["wandb"]["tags"] if "wandb" in config and "tags" in config["wandb"] else []
        if type(tags) != list: tags = [tags]
        experiment_logger = WandbLogger(name=experiment_name + "/" + str(args.seed), save_dir=fulldir_experiments, group=experiment_name, \
                            tags=tags, entity="arjun-team", \
                                project=config["wandb"]["project"] if "wandb" in config and "project" in config["wandb"] else "scaling_logs", \
                                config=config, id=sha1(fulldir_experiments.encode("utf-8")).hexdigest()[:8])
else:
    experiment_logger = CSVLogger(save_dir=fulldir_experiments)
logger = [experiment_logger]

lr_monitor = LearningRateMonitor(logging_interval="epoch")
lr_callback = LearningRateFinder(min_lr=1e-4, max_lr=1e-1, num_training_steps=100)
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=50, verbose=True, mode="min")
callbacks=[lr_monitor, lr_callback, early_stop_callback]
# callbacks = [] # For data scaling

# Do a batch size search first without any logger
print("DOING BATCH SIZE SEARCH...")
batch_size = args.batch_size

# fulldir_batchsize_search = os.path.join(fulldir, "batch-size-search")
# os.makedirs(fulldir_batchsize_search, exist_ok=True)

# while True:
#     print("Trying batch size:", batch_size)
#     batch_size_search_dir = os.path.join(fulldir_batchsize_search, "batch-size-search-" + str(batch_size))
#     os.makedirs(batch_size_search_dir, exist_ok=True)

#     bsz_logger = None
#     try:
#         estimator = LagGPTFlowsEstimator(
#             prediction_length=config["gpt"]["prediction_length"] if "prediction_length" in config["gpt"] else meta.prediction_length,
#             context_length=config["gpt"]["context_length"], # block_size: int = 2048 
#             batch_size=batch_size, # 4
#             n_layer=args.layers,
#             n_head=args.heads,
#             n_embd=args.dims_per_head*args.heads, # 4096
#             dsf_marginal=config["gpt"]["dsf_marginal"],
#             scaling=config["gpt"]["scaling"],
#             aug_prob = config["gpt"]["aug_prob"],
#             aug_rate = config["gpt"]["aug_rate"],
#             num_batches_per_epoch= 10,
#             trainer_kwargs=dict(max_epochs=1, accelerator="gpu", \
#                                 precision=args.precision, logger=False, devices=[config["CUDA"]["device_id"]], \
#                                 callbacks=[], default_root_dir=batch_size_search_dir),
#             ckpt_path=None
#         )

#         predictor = estimator.train(
#             training_data=dataset, 
#             validation_data=val_dataset,
#             shuffle_buffer_length=1000,
#             ckpt_path=None
#         )

#         print("Succesfully found batch size:", batch_size)
#         break
#     except RuntimeError as e:
#         if "out of memory" in str(e):
#             gc.collect()
#             torch.cuda.empty_cache()
#             if batch_size == 1: 
#                 print("Batch is already at the minimum. Cannot reduce further. Exiting...")
#                 exit(0)
#             else:
#                 print("Caught OutOfMemoryError. Reducing batch size...")
#                 batch_size //= 2
#                 continue

# if batch_size != 1:
#     batch_size //= 2
#     print("Using batch size:", batch_size)

# gc.collect()
# torch.cuda.empty_cache()
# print("Training...")

parameters = []

for args.layers in [2,4]:
    args.dims_per_head = args.layers * 2

    estimator = LagGPTFlowsEstimator(
        prediction_length=config["gpt"]["prediction_length"] if "prediction_length" in config["gpt"] else meta.prediction_length,
        context_length=config["gpt"]["context_length"], # block_size: int = 2048 
        batch_size=batch_size, # 4
        n_layer=args.layers,
        n_head=args.heads,
        n_embd=args.dims_per_head*args.heads, # 4096
        dsf_marginal=config["gpt"]["dsf_marginal"],
        scaling=config["gpt"]["scaling"],
        aug_prob = config["gpt"]["aug_prob"],
        aug_rate = config["gpt"]["aug_rate"],
        num_batches_per_epoch= config["gpt"]["batches_per_epoch"],
        trainer_kwargs=dict(max_epochs=config["gpt"]["max_epochs"], accelerator="gpu", \
                            precision=args.precision, logger=logger, devices=[config["CUDA"]["device_id"]], \
                            callbacks=callbacks, default_root_dir=fulldir_experiments),
        ckpt_path=ckpt_path
    )

    num_parameters = sum(p.numel() for p in estimator.create_lightning_module().parameters())
    # print(f"Number of parameters: {num_parameters}")

    parameters.append(num_parameters)

print(list(parameters))