import os
import wandb
import pandas as pd

### Define these arguments manually
FOLDER_NAME="lag-transformer/data-scaling-uniform-0.5_4096_dims_bsz1"
TAGS = ["dims_scaling"]

# Get the seeds present in the directory
SEEDS = os.listdir(FOLDER_NAME)
# Get the name
NAME = FOLDER_NAME.split("/")[-1]
for version in SEEDS:
    # Start W&B run - change these arguments if you'd like to
    wandb.init(project="scaling_logs", name=NAME+"_seed_"+str(version), group=NAME, tags=TAGS)
    # Read DF
    loss_df = pd.read_csv(FOLDER_NAME+"/"+str(version)+"/lightning_logs/version_0/metrics.csv")
    train_loss = loss_df.dropna(subset=["train_loss"])
    val_loss = loss_df.dropna(subset=["val_loss"])
    # Log the losses
    for index, row in train_loss.iterrows():
        wandb.log({"train_loss":row['train_loss'], "epoch": row['epoch'], "step": row["step"]})
    for index, row in val_loss.iterrows():
        wandb.log({"val_loss":row['val_loss'], "epoch": row['epoch'], "step": row["step"]})
    # Finish
    wandb.finish()