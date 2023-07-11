for SEED in 1 2 3 4 5
do
    NAME="128_dims"
    python lag-transformer/lag-transformer-scaling-data.py \
    repo_scripts/lag-transformer/dims_scaling/128/config.yaml --seed $SEED --suffix $NAME \
    --dataset_path "/home/toolkit/datasets" --precision "32"
done
