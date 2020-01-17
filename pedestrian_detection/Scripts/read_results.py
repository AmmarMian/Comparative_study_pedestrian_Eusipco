# -*- coding: utf-8 -*-
# @Author: miana1
# @Description: Read results file
# @Date:   2020-01-14 11:19:58
# @E-mail: ammar.mian@aalto.fi
# @Last Modified by:   miana1
# @Last Modified time: 2020-01-17 12:40:45
# ----------------------------------------------------------------------------
# Copyright 2019 Aalto University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

import os
import sys
import logging
import argparse
import pickle
import numpy as np
from tqdm import tqdm, trange
from sklearn.model_selection import KFold
from copy import deepcopy
from joblib import Parallel, delayed
from sklearn.utils import shuffle


if __name__ == '__main__':

    # Managing inputs of this script
    parser = argparse.ArgumentParser(description='Compute training phase for all methods')
    parser.add_argument("data_file", help="Path (From base) to the file containing results")
    args = parser.parse_args()

    # We always need to know where this script is with regards to base of
    # project, so we define these variables to make everything run smoothly
    path_to_base = "../../"
    folder_of_present_script = os.path.dirname(os.path.realpath(__file__))
    absolute_base_path = os.path.join(folder_of_present_script, path_to_base)
    path_to_results_file = os.path.join(absolute_base_path, args.data_file)

    # Init paths, and import needed packages
    sys.path.insert(0, absolute_base_path)
    from global_utils import *


    # Read logging configuration
    configure_logging(os.path.join(absolute_base_path, "logging.yaml"))

    # Read data from pickle dump
    logging.info("Reading results data from file %s", path_to_results_file)
    with open(path_to_results_file, 'rb') as f:
        results = pickle.load(f)
