#!/bin/bash
#SBATCH --time=24:00:00      # 24 hours
#SBATCH --mem=48000   # 32G of memory
#SBATCH --cpus-per-task=12

python3 execute_simulation_local.py Simulation_setups/DaimerChrysler_3_fold.yaml
python3 Scripts/compute_train_test_knn_mdm.py "pedestrian_detection/Simulation_data/Simulations_setups_data/DaimerChrysler_3_fold/machine_learning_features_method_SCM normalized by image" 100 3 -p -j 24 &> log_knn_mdm_DaimerChrysler.txt
