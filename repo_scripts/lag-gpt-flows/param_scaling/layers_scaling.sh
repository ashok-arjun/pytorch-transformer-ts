for LAYERS in 1 2 3 5 6 7 9 10 11 13 14 15 17 18 19
do
    for SEED in {1..10}
    do
        python lag-gpt-flows/lag-gpt-with-flows-scaling-data.py \
        repo_scripts/lag-gpt-flows/param_scaling/base_config.yaml --suffix "${NAME}" --seed $SEED --layers $LAYERS
    done
done