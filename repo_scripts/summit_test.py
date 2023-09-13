# This is for the given experiment configuration - say parameter scaling with layers:dims
# A name + config file is passed to this file
# It scales the parmeters as we want (code we can set)
# For each config, it lauches a script with 6 random seeds by appropriately modifying name like the .sh files

import argparse
import os
from os.path import dirname, abspath

def jsrun_6seeds(name, config):
    # Modify the script content with the arguments
    script_content = \
'''
#!/bin/bash

#BSUB -P CSC499
#BSUB -W 24:00
#BSUB -J {0}
#BSUB -q killable
#BSUB -o /gpfs/alpine/csc499/scratch/arjunashok/pytorch-transformer-ts/summit_job_outputs/job%J.out
#BSUB -e /gpfs/alpine/csc499/scratch/arjunashok/pytorch-transformer-ts/summit_job_outputs/job%J.err
#BSUB -nnodes 1
#BSUB -alloc_flags gpudefault
# END LSF DIRECTIVES

number_of_nodes=$(cat $LSB_DJOB_HOSTFILE | uniq | head -n -1 | wc -l)
# echo $number_of_nodes

# JSRUN OPTIONS (CONFIGURE ME!)
number_of_resource_sets=1
physical_cores_per_resource_set=2
gpus_per_resource_set=1
mpi_ranks_per_resource_set=1

# maybe cd to the directory of the code
cd /gpfs/alpine/csc499/scratch/arjunashok/pytorch-transformer-ts/

module load python
module load cuda
source activate /gpfs/alpine/csc499/scratch/arjunashok/conda/scaling_opence_wandb

for i in 11 23 34 45 56 67
do
    SEED=$i
    RANDOM_STRING=$(openssl rand -hex 3 | tr -d '\\n' | cut -c1-5)
    NAME="{0}"
    jsrun -n 1 \
    -c 2 \
    -g 1 \
    -a 1 \
    /gpfs/alpine/csc499/scratch/arjunashok/conda/scaling_opence_wandb/bin/python lag-gpt-flows/lag-gpt-with-flows-scaling-data.py "{3}" \
    --suffix "{0}" &
done

wait
'''

    return script_content.format(name, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="repo_scripts/summit_test_config.yaml")

    args = parser.parse_args()

    directory = dirname(abspath(__file__))
    scripts_dir = os.path.join(directory, "scripts")
    os.makedirs(scripts_dir, exist_ok=True)

    name = "summit_test_"+str(layers)+"_dims_per_head_"+str(dims_per_head)
    config = args.config

    assert os.path.isfile(config), "Config Path is not a file"
    config = os.path.abspath(config)

    jsrun_string = jsrun_6seeds(name, config, layers=layers, dims_per_head=dims_per_head)
    script_path = os.path.abspath(scripts_dir) + "/" + name + ".lsf"

    with open(script_path, "w") as f: 
        f.write(jsrun_string)
    print("Submitting", script_path)
    os.system("bsub " + script_path)


