# -*- coding: utf-8 -*-
# @Author: miana1
# @Description: A file that compute a prediction on the test set usigng the classifiers
#               obtained from k fold testing.
# @Date:   2020-01-28 10:35:30
# @E-mail: ammar.mian@aalto.fi
# @Last Modified by:   miana1
# @Last Modified time: 2020-01-31 11:18:03
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

if __name__ == '__main__':

    # Managing inputs of this script
    parser = argparse.ArgumentParser(description='Compute testing phase for all methods')
    parser.add_argument("data_file", help="Path (From base) to the file containing machine learning features")
    parser.add_argument("result_testing", help="Path (From base) to the file containing results of testing")
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
    path_to_training_results = os.path.join(absolute_base_path, args.result_testing)

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

    # Getting test data and formatting it
    logging.info('Getting and formatting testing data')
    test_samples = [dataset['features'][i] for i in dataset['indexes test']]
    test_labels = [dataset['labels'][i] for i in dataset['indexes test']]
    dataset = None
    X_test = []
    y_test = []
    number_non_spd = 0
    for index in trange(len(test_samples)):
        regions_samples = test_samples[index]
        for sample in regions_samples:
            covariance = algebra.unvech(sample)

            # Discarding the non SPD matrices
            if algebra.is_pos_def(covariance):
                X_test.append(covariance)
                y_test.append(test_labels[index])
            else:
                number_non_spd += 1
    if number_non_spd > 0:
        logging.warning(f'There was {number_non_spd} non SPD matrices discarded among the samples')
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    test_samples = None
    test_labels = None


    # Read methods from result of training
    with open(path_to_training_results, 'rb') as f:
        methods_k_fold, training_accuracy_list = pickle.load(f)


    # Doing the prediciton and computing accuracy
    logging.info('Doing testing')
    testing_accuracy_list = []
    for methods_list_one_fold in tqdm(methods_k_fold):
        accuracy_this_fold = []
        for method in methods_list_one_fold:
            # Setting parallel if applicable
            set_parallel = getattr(method, "set_parallel", None)
            if callable(set_parallel):
                method.set_parallel(args.parallel, args.n_jobs)
            y_predicted = method.predict(X_test)
            accuracy = (y_test == y_predicted).sum() / len(y_test)
            accuracy_this_fold.append(accuracy)
        testing_accuracy_list.append(accuracy_this_fold)
    testing_accuracy_list = np.array(testing_accuracy_list)

    # Printing results
    print('\nResults are the following:')
    for i, method in enumerate(methods_list_one_fold):
        print(f'Method {i}: {method.method_name}')
        print(f'Training: {training_accuracy_list[:,i]}')
        print(f'Testing: {testing_accuracy_list[:,i]}\n')

    # Saving results
    logging.info('Saving testing results')
    with open(os.path.join(os.path.dirname(path_to_machine_learning_features_data_file), 'Results_training_testing'), 'wb') as f:
        pickle.dump([methods_k_fold, training_accuracy_list, testing_accuracy_list], f)

    logging.info('Removing previous result file')
    os.remove(path_to_training_results)
