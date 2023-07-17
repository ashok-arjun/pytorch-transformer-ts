import os
import wandb
import pandas as pd
from glob import glob
from hashlib import sha1

### Define these arguments manually
FOLDERS = [
    "lag-transformer/data-scaling-uniform-0.5_split_exps_4_heads_32dims_per_head",
    "lag-transformer/data-scaling-uniform-0.5_split_exps_5_heads_32dims_per_head",
    "lag-transformer/data-scaling-uniform-0.5_split_exps_6_heads_32dims_per_head",
    "lag-transformer/data-scaling-uniform-0.5_split_exps_7_heads_32dims_per_head",
    "lag-transformer/data-scaling-uniform-0.5_split_exps_8_heads_32dims_per_head",
    "lag-transformer/data-scaling-uniform-0.5_split_exps_9_heads_32dims_per_head"
]
TAGS = ["lag_transformer_param_scaling_heads_scaling_logall_again"]

for FOLDER_NAME in FOLDERS:
    # Get the seeds present in the directory
    SEEDS = os.listdir(FOLDER_NAME)
    NAME = FOLDER_NAME.split("/")[-1]
    for seed in SEEDS:
        print(NAME, "Seed:", seed)
        fulldir = FOLDER_NAME+"/"+str(seed)
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
        # Finish
        wandb.finish()