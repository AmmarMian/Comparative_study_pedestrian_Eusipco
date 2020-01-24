# -*- coding: utf-8 -*-
# @Author: miana1
# @Description: A file that reads the simulation setup and launch the training of
#               all the methods chosen on the training data.
#               Version where everything is done on a local machine and there
#               is no job system.
# @Date:   2020-01-14 11:19:58
# @E-mail: ammar.mian@aalto.fi
# @Last Modified by:   miana1
# @Last Modified time: 2020-01-24 12:03:30
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
from sklearn.utils import shuffle

def _do_train_test_one_method(X_train, y_train, X_test, y_test, method):
    method_this_fold = deepcopy(method)
    method_this_fold.fit(X_train, y_train)
    y_predicted = method_this_fold.predict(X_test)
    accuracy = (y_test == y_predicted).sum() / len(y_test)
    return [accuracy, method_this_fold]


if __name__ == '__main__':

    # Managing inputs of this script
    parser = argparse.ArgumentParser(description='Compute training phase for all methods')
    parser.add_argument("data_file", help="Path (From base) to the file containing machine learning features")
    parser.add_argument("simulation_setup",  help="Path (From base) to the simulation setup file")
    parser.add_argument("-p", "--parallel", action="store_true",
                         help="Enable parallel computation")
    parser.add_argument("-j", "--n_jobs", default=8, type=int,
                         help="Number of jobs for parallel computation")
    args = parser.parse_args()

    # We always need to know where this script is with regards to base of
    # project, so we define these variables to make everything run smoothly
    path_to_base = "../../"
    folder_of_present_script = os.path.dirname(os.path.realpath(__file__))
    absolute_base_path = os.path.join(folder_of_present_script, path_to_base)
    path_to_machine_learning_features_data_file = os.path.join(absolute_base_path,
                                                                    args.data_file)

    # Init paths, and import needed packages
    sys.path.insert(0, absolute_base_path)
    from global_utils import *
    from psdlearning.utils import algebra
    from psdlearning import parsing_methods

    # Read logging configuration
    configure_logging(os.path.join(absolute_base_path, "logging.yaml"))

    # Read data from pickle dump
    logging.info("Reading machine learning features data from file %s",
                                path_to_machine_learning_features_data_file)
    with open(path_to_machine_learning_features_data_file, 'rb') as f:
        dataset = pickle.load(f)

    # Parsing YAML file
    with open(args.simulation_setup) as f:
        simulation_setup = yaml.safe_load(f)

    # Getting train data and formatting it
    logging.info('Getting and formatting training data')
    train_samples = [dataset['features'][i] for i in dataset['indexes train']]
    train_labels = [dataset['labels'][i] for i in dataset['indexes train']]
    dataset = None
    X_train = []
    y_train = []
    number_non_spd = 0
    for index in trange(len(train_samples)):
        regions_samples = train_samples[index]
        for sample in regions_samples:
            covariance = algebra.unvech(sample)

            # Discarding the non SPD matrices
            if algebra.is_pos_def(covariance):
                X_train.append(covariance)
                y_train.append(train_labels[index])
            else:
                number_non_spd += 1
    if number_non_spd > 0:
        logging.warning(f'There was {number_non_spd} non SPD matrices discarded among the samples')
    X = np.array(X_train)
    y = np.array(y_train)
    train_samples = None
    train_labels = None


    # Parsing methods
    logging.info('Parsing classification methods')
    methods_list = []
    for method_input in simulation_setup['classification_methods'].values():
        method = parsing_methods.parse_machine_learning_method(method_input['parsing_string'],
                            method_input['method_name'], method_input['method_args'])
        method.set_parallel(args.parallel, args.n_jobs)
        methods_list.append(method)

    # Doing K-fold splitting
    if simulation_setup['train']['pre-shuffle']:
        X, y,  = shuffle(X, y, random_state=simulation_setup['train']['seed'])

    kf = KFold(n_splits=simulation_setup['train']['n_splits'],
                random_state= simulation_setup['train']['seed'])
    kf.get_n_splits(X)

    # Training for each fold and each method
    logging.info('Doing training')
    methods_k_fold = [] # Container of classifers trained on each fold
    accuracy_list = []
    i = 1
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        logging.info(f'Doing fold {i}')
        methods_list_this_fold = []
        accuracy_list_this_fold = []

        for method in tqdm(methods_list):
            accuracy, method_this_fold =_do_train_test_one_method(X_train,
                                            y_train, X_test, y_test, method)
            accuracy_list_this_fold.append(accuracy)
            methods_list_this_fold.append(method_this_fold)

        methods_k_fold.append(methods_list_this_fold)
        accuracy_list.append(accuracy_list_this_fold)
        i += 1

    accuracy_list = np.array(accuracy_list)
    print(accuracy_list)

    # Saving results
    logging.info('Saving training results')
    with open(os.path.join(os.path.dirname(path_to_machine_learning_features_data_file), 'Results_training'), 'wb') as f:
        pickle.dump([methods_list, accuracy_list], f)

