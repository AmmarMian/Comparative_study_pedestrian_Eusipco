#!/bin/bash
#SBATCH --time=10:00:00      # 10 hours
#SBATCH --mem=32000   # 32G of memory
#SBATCH --cpus-per-task=8

python3 execute_simulation_local.py Simulation_setups/INRIA_basic_test.yaml