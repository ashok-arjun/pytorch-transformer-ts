import os
import wandb
import pandas as pd
from glob import glob
from hashlib import sha1

### Define these arguments manually
# FOLDERS = [
#     "lag-transformer/arian_exps/data-scaling-uniform-0.5_encoder8",
#     "lag-transformer/arian_exps/data-scaling-uniform-0.5_encoder9",
#     "lag-transformer/arian_exps/data-scaling-uniform-0.5_encoder10",
#     "lag-transformer/arian_exps/data-scaling-uniform-0.5_encoder11",
#     "lag-transformer/arian_exps/data-scaling-uniform-0.5_encoder12",
#     "lag-transformer/arian_exps/data-scaling-uniform-0.5_encoder13"
# ]
# FOLDERS = [
#     "lag-transformer/arian_exps/data-scaling-uniform-0.5_dims_per_head_128",
#     "lag-transformer/arian_exps/data-scaling-uniform-0.5_dims_per_head_256"
# ]
# FOLDERS = [
#     "lag-transformer/arian_exps/data-scaling-uniform-0.5_heads10",
#     "lag-transformer/arian_exps/data-scaling-uniform-0.5_heads11",
#     "lag-transformer/arian_exps/data-scaling-uniform-0.5_heads12",
#     "lag-transformer/arian_exps/data-scaling-uniform-0.5_heads13",
#     "lag-transformer/arian_exps/data-scaling-uniform-0.5_heads14",
#     "lag-transformer/arian_exps/data-scaling-uniform-0.5_heads17",
#     "lag-transformer/arian_exps/data-scaling-uniform-0.5_heads18",
#     "lag-transformer/arian_exps/data-scaling-uniform-0.5_heads19"
# ]
# FOLDERS = [
#     "lag-transformer/arian_exps/data-scaling-uniform-0.5_decoder7",
#     "lag-transformer/arian_exps/data-scaling-uniform-0.5_decoder8",
#     "lag-transformer/arian_exps/data-scaling-uniform-0.5_decoder9",
#     "lag-transformer/arian_exps/data-scaling-uniform-0.5_decoder15",
#     "lag-transformer/arian_exps/data-scaling-uniform-0.5_decoder16",
#     "lag-transformer/arian_exps/data-scaling-uniform-0.5_decoder17",
#     "lag-transformer/arian_exps/data-scaling-uniform-0.5_decoder18",
#     "lag-transformer/arian_exps/data-scaling-uniform-0.5_decoder19"
# ]
FOLDERS = [
    # "lag-gpt-flows/scaling-logs/data-scaling-uniform-1.0_augprob_1_layers_dims_per_head_ratio_scaling_layers_2_dims_per_head_4",
    # "lag-gpt-flows/scaling-logs/data-scaling-uniform-1.0_augprob_1_layers_dims_per_head_ratio_scaling_layers_4_dims_per_head_8",
    # "lag-gpt-flows/scaling-logs/data-scaling-uniform-1.0_augprob_1_layers_dims_per_head_ratio_scaling_layers_8_dims_per_head_16",
    # "lag-gpt-flows/scaling-logs/data-scaling-uniform-1.0_augprob_1_layers_dims_per_head_ratio_scaling_layers_10_dims_per_head_20",
    # "lag-gpt-flows/scaling-logs/data-scaling-uniform-1.0_augprob_1_layers_dims_per_head_ratio_scaling_layers_12_dims_per_head_24",
    # "lag-gpt-flows/scaling-logs/data-scaling-uniform-1.0_augprob_1_layers_dims_per_head_ratio_scaling_layers_14_dims_per_head_28",
    # "lag-gpt-flows/scaling-logs/data-scaling-uniform-1.0_augprob_1_layers_dims_per_head_ratio_scaling_layers_16_dims_per_head_32",
    # "lag-gpt-flows/scaling-logs/data-scaling-uniform-1.0_augprob_1_layers_dims_per_head_ratio_scaling_layers_18_dims_per_head_36",
    # "lag-gpt-flows/scaling-logs/data-scaling-uniform-1.0_augprob_1_layers_dims_per_head_ratio_scaling_layers_20_dims_per_head_40",
    # "lag-gpt-flows/scaling-logs/data-scaling-uniform-1.0_augprob_1_layers_dims_per_head_ratio_scaling_layers_22_dims_per_head_44",
    # "lag-gpt-flows/scaling-logs/data-scaling-uniform-1.0_augprob_1_layers_dims_per_head_ratio_scaling_layers_24_dims_per_head_48",
    # "lag-gpt-flows/scaling-logs/data-scaling-uniform-1.0_augprob_1_layers_dims_per_head_ratio_scaling_layers_26_dims_per_head_52",
    # "lag-gpt-flows/scaling-logs/data-scaling-uniform-1.0_augprob_1_layers_dims_per_head_ratio_scaling_layers_28_dims_per_head_56",
    # "lag-gpt-flows/scaling-logs/data-scaling-uniform-1.0_augprob_1_layers_dims_per_head_ratio_scaling_layers_30_dims_per_head_60",
    # "lag-gpt-flows/scaling-logs/data-scaling-uniform-1.0_augprob_1_layers_dims_per_head_ratio_scaling_layers_32_dims_per_head_64",
    # "lag-gpt-flows/scaling-logs/data-scaling-uniform-1.0_augprob_1_layers_dims_per_head_ratio_scaling_layers_34_dims_per_head_68",
    # "lag-gpt-flows/scaling-logs/data-scaling-uniform-1.0_augprob_1_layers_dims_per_head_ratio_scaling_layers_36_dims_per_head_72",
    # "lag-gpt-flows/scaling-logs/data-scaling-uniform-1.0_augprob_1_layers_dims_per_head_ratio_scaling_layers_38_dims_per_head_76",
    # "lag-gpt-flows/scaling-logs/data-scaling-uniform-1.0_augprob_1_layers_dims_per_head_ratio_scaling_layers_40_dims_per_head_80"
]
TAGS = ["lag_gpt_with_flows_param_scaling"]

for FOLDER_NAME in FOLDERS:
    # Get the seeds present in the directory
    SEEDS = sorted(os.listdir(FOLDER_NAME))
    NAME = FOLDER_NAME.split("/")[-1]
    for seed in SEEDS:
        print(NAME, "Seed:", seed)
        fulldir = FOLDER_NAME+"/"+str(seed)+"/experiments"
        if not os.path.isdir(fulldir): continue

        # Start W&B run - change these arguments if you'd like to
        wandb.init(project="scaling_logs", name=NAME+"/"+str(seed), group=NAME, tags=TAGS, id=sha1(fulldir.encode("utf-8")).hexdigest()[:8])

        # Use epoch# from checkpoint to select the lightning version
        lightning_version_to_use = None
        max_epoch = -1
        if "lightning_logs" in os.listdir(fulldir):
            for lightning_version in os.listdir(fulldir+"/lightning_logs/"):
                ckpts = glob(fulldir+"/lightning_logs/" + lightning_version + "/checkpoints/*.ckpt")
                if len(ckpts): 
                    epoch = int(ckpts[0][ckpts[0].find("=")+1:ckpts[0].find("-step")])
                    if epoch > max_epoch:
                        lightning_version_to_use = lightning_version
                        max_epoch = epoch
            if lightning_version_to_use: print("Using lightning_version", lightning_version_to_use, "with epoch", max_epoch)
            else: continue

        loss_df = pd.read_csv(fulldir+"/lightning_logs/" + lightning_version_to_use + "/metrics.csv")
        train_loss = loss_df.dropna(subset=["train_loss"])
        val_loss = loss_df.dropna(subset=["val_loss"])
        # Log the losses
        for index, row in train_loss.iterrows():
            wandb.log({"train_loss":row['train_loss'], "epoch": row['epoch'], "step": row["step"]})
        for index, row in val_loss.iterrows():
            wandb.log({"val_loss":row['val_loss'], "epoch": row['epoch'], "step": row["step"]})
        # # Log minimum loss across epochs
        # min_val_loss = val_loss['val_loss'].min()
        # wandb.log({"best_val_loss":float(min_val_loss)})
        # Check if there was early stopping
        if "wait_count" in loss_df.columns:
            wait_count = loss_df.dropna(subset=["wait_count"])
            for index, row in wait_count.iterrows():
                wandb.log({"wait_count":wait_count, "epoch": row['epoch'], "step": row["step"]})
        if "best_val_loss" in loss_df.columns:
            best_val_loss = loss_df.dropna(subset=["best_val_loss"])
            for index, row in best_val_loss.iterrows():
                wandb.log({"best_val_loss":row['best_val_loss'], "epoch": row['epoch'], "step": row["step"]})
        else:
            # Log minimum loss across epochs
            min_val_loss = val_loss['val_loss'].min()
            wandb.log({"best_val_loss_final":float(min_val_loss)})
        # Finish
        wandb.finish()