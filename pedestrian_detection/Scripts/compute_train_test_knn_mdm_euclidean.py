# -*- coding: utf-8 -*-
# @Author: miana1
# @Description: File for doing the training and testing for the Riemannian knn
#               and mdm which cannot be saved using pickle.
# @Date:   2020-02-14 13:15:18
# @E-mail: ammar.mian@aalto.fi
# @Last Modified by:   miana1
# @Last Modified time: 2020-02-14 15:12:08
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
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid


if __name__ == '__main__':

    # Managing inputs of this script
    parser = argparse.ArgumentParser(description='Compute training and testing for Euclidean knn and mdm')
    parser.add_argument("data_file", help="Path (From base) to the file containing machine learning features")
    parser.add_argument("seed", type=int, help="random state seed for shuffling data")
    parser.add_argument("n_folds", type=int, help="Number of folds")
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


    # Read logging configuration
    configure_logging(os.path.join(absolute_base_path, "logging.yaml"))

    # KNN parameters
    n_neighbors = 5
    if args.parallel:
        n_jobs=args.n_jobs
    else:
        n_jobs = 1


    # Read data from pickle dump
    logging.info("Reading machine learning features data from file %s",
                                path_to_machine_learning_features_data_file)
    with open(path_to_machine_learning_features_data_file, 'rb') as f:
        dataset = pickle.load(f)

    # Getting train data and formatting it
    logging.info('Getting and formatting training data')
    train_samples = [dataset['features'][i] for i in dataset['indexes train']]
    train_labels = [dataset['labels'][i] for i in dataset['indexes train']]
    X_train = []
    y_train = []
    number_non_spd = 0
    for index in trange(len(train_samples)):
        regions_samples = train_samples[index]
        for sample in regions_samples:
            covariance = algebra.unvech(sample)

            # Discarding the non SPD matrices
            if algebra.is_pos_def(covariance):
                X_train.append(sample)
                y_train.append(train_labels[index])
            else:
                number_non_spd += 1
    if number_non_spd > 0:
        logging.warning(f'There was {number_non_spd} non SPD matrices discarded among the samples')
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    train_samples = None
    train_labels = None

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
                X_test.append(sample)
                y_test.append(test_labels[index])
            else:
                number_non_spd += 1
    if number_non_spd > 0:
        logging.warning(f'There was {number_non_spd} non SPD matrices discarded among the samples')
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    test_samples = None
    test_labels = None


    # Doing K-fold splitting
    X_train, y_train  = shuffle(X_train, y_train, random_state=args.seed)
    kf = KFold(n_splits=args.n_folds, random_state=args.seed)
    kf.get_n_splits(X_train)


    # Training for each fold and each method
    logging.info('Doing training')
    clf_knn_k_fold = [] # Container of classifers trained on each fold
    clf_mdm_k_fold = [] # Container of classifers trained on each fold
    accuracy_list_training_knn = []
    accuracy_list_training_mdm = []
    i = 1
    for train_index, test_index in kf.split(X_train):

        logging.info(f'Doing fold {i}')
        clf_knn = KNeighborsClassifier(n_neighbors, n_jobs=n_jobs)
        clf_mdm = NearestCentroid()
        X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

        clf_knn.fit(X_train_fold, y_train_fold)
        y_predicted = clf_knn.predict(X_test_fold)
        accuracy = (y_test_fold == y_predicted).sum() / len(y_test_fold)
        clf_knn_k_fold. append(clf_knn)
        accuracy_list_training_knn.append(accuracy)

        clf_mdm.fit(X_train_fold, y_train_fold)
        y_predicted = clf_mdm.predict(X_test_fold)
        accuracy = (y_test_fold == y_predicted).sum() / len(y_test_fold)
        clf_mdm_k_fold. append(clf_mdm)
        accuracy_list_training_mdm.append(accuracy)

        i += 1


    # Testing on test dataset
    logging.info('Doing testing')
    accuracy_list_testing_knn = []
    accuracy_list_testing_mdm = []
    X_test, y_test  = shuffle(X_test, y_test, random_state=args.seed)

    for clf_knn in clf_knn_k_fold:
        y_predicted =  clf_knn.predict(X_test)
        accuracy = (y_test == y_predicted).sum() / len(y_test)
        accuracy_list_testing_knn.append(accuracy)

    for clf_mdm in clf_mdm_k_fold:
        y_predicted =  clf_mdm.predict(X_test)
        accuracy = (y_test == y_predicted).sum() / len(y_test)
        accuracy_list_testing_mdm.append(accuracy)

    print('Results Euclidean KNN:')
    print('Training: ' + str(accuracy_list_training_knn))
    print('Testing: ' + str(accuracy_list_testing_knn))

    print('\nResults Euclidean MDM:')
    print('Training: ' + str(accuracy_list_training_mdm))
    print('Testing: ' + str(accuracy_list_testing_mdm))
