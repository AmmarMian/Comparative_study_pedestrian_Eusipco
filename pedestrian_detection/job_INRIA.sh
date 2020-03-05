#!/bin/bash
#SBATCH --time=20:00:00      # 10 hours
#SBATCH --mem=32000   # 32G of memory
#SBATCH --cpus-per-task=12

python3 execute_simulation_local.py Simulation_setups/INRIA_4_fold.yaml
python3 Scripts/compute_train_test_knn_mdm.py "pedestrian_detection/Simulation_data/Simulations_setups_data/INRIA_4_fold/machine_learning_features_method_SCM normalized by image" 100 4 -p -j 24 &> log_knn_mdm_INRIA.txt
