#!/bin/bash
#SBATCH --time=24:00:00      # 48 hours
#SBATCH --mem=24000   # 32G of memory
#SBATCH --cpus-per-task=12

#python3 Scripts/robustness_study.py Scripts/Tuning/Data/INRIA -p -j 36 &> ./log_robustness_INRIA.txt
python3 Scripts/robustness_study.py Scripts/Tuning/Data/DaimerChrysler -p -j 24 &> ./log_robustness_DaimerChrysler.txt
