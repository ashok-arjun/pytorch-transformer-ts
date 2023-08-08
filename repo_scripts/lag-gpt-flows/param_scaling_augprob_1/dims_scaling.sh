for DIMS_PER_HEAD in 16 32 64 128 256 512 1024 2048
do
    for SEED in {1..10}
    do
        NAME="augprob_1_dims_per_head_${DIMS_PER_HEAD}"
        python lag-gpt-flows/lag-gpt-with-flows-scaling-data.py \
        repo_scripts/lag-gpt-flows/param_scaling_augprob_1/base_config.yaml --suffix "${NAME}" --seed $SEED --dims_per_head $DIMS_PER_HEAD
    done
done