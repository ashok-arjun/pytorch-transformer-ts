import random
import time
from pathlib import Path
import pandas as pd
import os
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
import wandb


def select_datasets(datasets, target_percentage=1.0):
    if target_percentage == 1.0: return datasets
    datasets_lengths = []

    for dataset in datasets:
        datasets_lengths.append(sum([len(x["target"]) for x in dataset]))

    datasets_lengths_normalized = []
    for i in range(len(datasets_lengths)):
        datasets_lengths_normalized.append(datasets_lengths[i] / sum(datasets_lengths))

    datasets_lengths_normalized.sort()
    datasets = [x for _, x in sorted(zip(datasets_lengths_normalized, datasets))]

    selected_datasets = []
    total_percentage = 0
    
    for i, dataset_length in enumerate(datasets_lengths_normalized):
        if total_percentage + dataset_length <= target_percentage:
            selected_datasets.append(datasets[i])
            total_percentage += dataset_length
        else:
            break
    
    print("Selected", len(selected_datasets), "with total percentage", total_percentage)

    return selected_datasets

parser = argparse.ArgumentParser()
parser.add_argument("filename", help = "YAML config file.")
parser.add_argument("--suffix", default="", type=str, required=True, help="This can be useful information to distinguish runs, like `heads-scaling-5-heads`")
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--dataset_path", default="/gpfs/alpine/csc499/scratch/arjunashok/datasets", type=str)
parser.add_argument("--precision", default="32", type=str, choices=["32", "16", "bf16-mixed"])

# For scaling, I am putting all parameters here to save time in creating the yaml files
parser.add_argument("--layers", default=8, type=int)
parser.add_argument("--heads", default=4, type=int)
parser.add_argument("--dims_per_head", default=16, type=int)
parser.add_argument("--context_length", default=1024, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--weight_decay", default=0, type=float)

# Also the batch size is by default set to a high value and found by the highest possible size at which 1 epoch runs
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
parser.add_argument("--early_stopping_patience", default=50, type=int)

# Only evaluation on traffic
parser.add_argument("--test_only", action="store_true")

# Dataset percentage selection
parser.add_argument("--filter_datasets_percentage", type=float, default=1.)


args = parser.parse_args()

device = torch.cuda.current_device()
memory_stats = torch.cuda.memory_stats(device=device)
t = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
allocated_memory_gb = memory_stats["allocated_bytes.all.current"] / (1024 ** 3)

print(f"Total Memory: {t:.2f} GB")
print(f"Allocated Memory: {allocated_memory_gb:.2f} GB")

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


if args.filter_datasets_percentage < 1.:
    gluonts_ds = select_datasets(gluonts_ds, args.filter_datasets_percentage)

dataset = CombinedDataset(gluonts_ds, weights=([sum([len(x["target"]) for x in d]) for d in gluonts_ds] if config["dataset"]["weighted"] else None), seed=args.seed) 

val_dataset = get_dataset(config["dataset"]["val"], path=dataset_path).test
meta = get_dataset(config["dataset"]["val"], path=dataset_path).metadata

# Make the experiment_name
experiment_name = ("data-scaling-weighted-"+str(config["gpt"]["aug_prob"])+"_"+args.suffix if config["dataset"]["weighted"] else "data-scaling-uniform-"+str(config["gpt"]["aug_prob"])+"_"+args.suffix)
fulldir = os.path.join(pathlib.Path(__file__).parent.resolve(), "scaling-logs", experiment_name, str(args.seed)) # Always creates the experiment directory inside "lag-gpt-flows"
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
                                config=config, id=sha1(fulldir_experiments.encode("utf-8")).hexdigest()[:8], allow_val_change=True, \
                                dir="/gpfs/alpine/csc499/scratch/arjunashok/pytorch-transformer-ts/wandb-gradnorms", mode="offline")
else:
    experiment_logger = CSVLogger(save_dir=fulldir_experiments)
logger = [experiment_logger]

lr_monitor = LearningRateMonitor(logging_interval="epoch")
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=int(args.early_stopping_patience), verbose=True, mode="min")
lr_finder = LearningRateFinder(min_lr=1e-6, max_lr=1e-1, num_training_steps=100, early_stop_threshold=None)

if not ckpt_path:
    callbacks=[lr_finder, lr_monitor, early_stop_callback]
else:
    callbacks=[lr_monitor, early_stop_callback]

# callbacks = [] # For data scaling

# Do a batch size search first without any logger
batch_size = args.batch_size

print("DOING BATCH SIZE SEARCH...")
fulldir_batchsize_search = os.path.join(fulldir, "batch-size-search")
os.makedirs(fulldir_batchsize_search, exist_ok=True)
while True:
    print("Trying batch size:", batch_size)
    batch_size_search_dir = os.path.join(fulldir_batchsize_search, "batch-size-search-" + str(batch_size))
    os.makedirs(batch_size_search_dir, exist_ok=True)

    bsz_logger = None
    try:
        estimator = LagGPTFlowsEstimator(
            prediction_length=config["gpt"]["prediction_length"] if "prediction_length" in config["gpt"] else meta.prediction_length,
            context_length=args.context_length, # block_size: int = 2048 
            batch_size=batch_size, # 4
            n_layer=args.layers,
            n_head=args.heads,
            n_embd=args.dims_per_head*args.heads, # 4096
            dsf_marginal=config["gpt"]["dsf_marginal"],
            scaling=config["gpt"]["scaling"],
            lr=args.lr,
            lrs=config["gpt"]["lrs"],
            lrs_patience=int(config["gpt"]["lrs_patience"]),
            weight_decay=args.weight_decay,
            aug_prob = config["gpt"]["aug_prob"],
            aug_rate = config["gpt"]["aug_rate"] if "aug_rate" in config["gpt"] else 0.,
            aug_range = config["gpt"]["aug_range"] if "aug_range" in config["gpt"] else None,
            num_batches_per_epoch= 10,
            trainer_kwargs=dict(max_epochs=1, accelerator="gpu", \
                                precision=args.precision, logger=False, devices=[0], \
                                callbacks=[], default_root_dir=batch_size_search_dir, accumulate_grad_batches=args.gradient_accumulation_steps),
            ckpt_path=None
        )

        predictor = estimator.train(
            training_data=dataset, 
            validation_data=val_dataset,
            shuffle_buffer_length=1000,
            ckpt_path=None
        )

        print("Succesfully found batch size:", batch_size)
        break
    except RuntimeError as e:
        if "out of memory" in str(e):
            gc.collect()
            torch.cuda.empty_cache()
            if batch_size == 1: 
                print("Batch is already at the minimum. Cannot reduce further. Exiting...")
                exit(0)
            else:
                print("Caught OutOfMemoryError. Reducing batch size...")
                batch_size //= 2
                continue
        else:
            print(e)
            exit(1)        
if batch_size != 1:
    batch_size //= 2
    print("Using batch size:", batch_size)


if type(logger[0]) == WandbLogger: 
    wandb.config.update({"batch_size": batch_size}, allow_val_change=True)
    wandb.config.update({"gradient_accumulation_steps": args.gradient_accumulation_steps}, allow_val_change=True)

gc.collect()
torch.cuda.empty_cache()
print("Training...")

estimator = LagGPTFlowsEstimator(
    prediction_length=config["gpt"]["prediction_length"] if "prediction_length" in config["gpt"] else meta.prediction_length,
    context_length=args.context_length, # block_size: int = 2048 
    batch_size=batch_size, # 4
    n_layer=args.layers,
    n_head=args.heads,
    n_embd=args.dims_per_head*args.heads, # 4096
    dsf_marginal=config["gpt"]["dsf_marginal"],
    scaling=config["gpt"]["scaling"],
    lr=args.lr,
    lrs=config["gpt"]["lrs"],
    lrs_patience=int(config["gpt"]["lrs_patience"]),
    weight_decay=args.weight_decay,
    aug_prob = config["gpt"]["aug_prob"],
    aug_rate = config["gpt"]["aug_rate"] if "aug_rate" in config["gpt"] else 0.,
    aug_range = config["gpt"]["aug_range"] if "aug_range" in config["gpt"] else None,
    num_batches_per_epoch= config["gpt"]["batches_per_epoch"],
    trainer_kwargs=dict(max_epochs=config["gpt"]["max_epochs"], accelerator="gpu", \
                        precision=args.precision, logger=logger, devices=[0], \
                        callbacks=callbacks, default_root_dir=fulldir_experiments, accumulate_grad_batches=args.gradient_accumulation_steps),
    ckpt_path=ckpt_path,
    num_parallel_samples=10
)

num_parameters = sum(p.numel() for p in estimator.create_lightning_module().parameters())
if type(logger[0]) == WandbLogger: wandb.config.update({"num_parameters": num_parameters})

start_time = time.time()
predictor = estimator.train(
    training_data=dataset, 
    validation_data=val_dataset,
    shuffle_buffer_length=1000,
    ckpt_path=ckpt_path
)
end_time = time.time()

# # Perform evaluation on the val dataset
# forecast_it, ts_it = make_evaluation_predictions(dataset=val_dataset,
#                                              predictor=predictor,
#                                              num_samples=100)
# forecasts = list(forecast_it)
# targets = list(ts_it)
# evaluator = MultivariateEvaluator(quantiles=(np.arange(20)/20.0)[1:],
#                                   target_agg_funcs={'sum': np.sum})
# agg_metric, _ = evaluator(targets, forecasts, num_series=len(dataset_test))
# agg_metric_modified = {}
# for key ,value in agg_metric.items():
#     agg_metric_modified["val/"+key] = value
# if type(logger[0] == CSVLogger):
#     logger[0].log_metrics(agg_metric_modified, step=0)

# # Perform evaluation on the test dataset with the same length
# test_dataset = get_dataset(config["dataset"]["test"], path=dataset_path).test
# test_meta = get_dataset(config["dataset"]["test"], path=dataset_path).metadata
# forecast_it, ts_it = make_evaluation_predictions(dataset=test_dataset,
#                                              predictor=predictor,
#                                              num_samples=100)
# forecasts = list(forecast_it)
# targets = list(ts_it)
# evaluator = MultivariateEvaluator(quantiles=(np.arange(20)/20.0)[1:],
#                                   target_agg_funcs={'sum': np.sum})
# agg_metric, _ = evaluator(targets, forecasts, num_series=len(dataset_test))
# agg_metric_modified = {}
# for key ,value in agg_metric.items():
#     agg_metric_modified["test-same-len/"+key] = value
# if type(logger[0] == CSVLogger):
#     logger[0].log_metrics(agg_metric_modified, step=0)


# # Perform evaluation on the test dataset with its length
# test_dataset = get_dataset(config["dataset"]["test"], path=dataset_path).test
# test_meta = get_dataset(config["dataset"]["test"], path=dataset_path).metadata
# estimator = LagGPTFlowsEstimator(
#     prediction_length=test_meta.prediction_length,
#     context_length=args.context_length, # block_size: int = 2048 
#     batch_size=batch_size, # 4
#     n_layer=args.layers,
#     n_head=args.heads,
#     n_embd=args.dims_per_head*args.heads, # 4096
#     dsf_marginal=config["gpt"]["dsf_marginal"],
#     scaling=config["gpt"]["scaling"],
#     lr=args.lr,
#     lrs=config["gpt"]["lrs"],
#     lrs_patience=int(config["gpt"]["lrs_patience"]),
#     weight_decay=args.weight_decay,
#     aug_prob = config["gpt"]["aug_prob"],
#     aug_rate = config["gpt"]["aug_rate"] if "aug_rate" in config["gpt"] else 0.,
#     aug_range = config["gpt"]["aug_range"] if "aug_range" in config["gpt"] else None,
#     num_batches_per_epoch= config["gpt"]["batches_per_epoch"],
#     trainer_kwargs=dict(max_epochs=config["gpt"]["max_epochs"], accelerator="gpu", \
#                         precision=args.precision, logger=None, devices=[0], \
#                         callbacks=callbacks, default_root_dir=fulldir_experiments, accumulate_grad_batches=args.gradient_accumulation_steps)
# )
# forecast_it, ts_it = make_evaluation_predictions(dataset=test_dataset,
#                                              predictor=predictor,
#                                              num_samples=100)
# forecasts = list(forecast_it)
# targets = list(ts_it)
# evaluator = MultivariateEvaluator(quantiles=(np.arange(20)/20.0)[1:],
#                                   target_agg_funcs={'sum': np.sum})
# agg_metric, _ = evaluator(targets, forecasts, num_series=len(dataset_test))
# agg_metric_modified = {}
# for key ,value in agg_metric.items():
#     agg_metric_modified["test/"+key] = value
# if type(logger[0] == CSVLogger):
#     logger[0].log_metrics(agg_metric_modified, step=0)