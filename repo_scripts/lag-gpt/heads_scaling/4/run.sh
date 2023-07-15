for SEED in 1 2 3 4 5
do
    NAME="test_4_heads"
    python lag-gpt/lag-gpt-scaling-data.py \
    repo_scripts/lag-gpt/heads_scaling/4/config.yaml --seed $SEED --suffix $NAME \
    --dataset_path "/home/toolkit/datasets" --precision "32"
done
