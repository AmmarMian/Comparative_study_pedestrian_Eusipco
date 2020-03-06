#!/bin/bash
#SBATCH --time=48:00:00      # 10 hours
#SBATCH --mem=32000   # 32G of memory
#SBATCH --cpus-per-task=12


python3 execute_simulation_local.py Simulation_setups/INRIA_4_fold_sample_each.yaml
#python3 execute_simulation_local.py Simulation_setups/INRIA_4_fold_sample_one.yaml
#python3 execute_simulation_local.py Simulation_setups/INRIA_4_fold_sample_one_pos_one_neg.yaml