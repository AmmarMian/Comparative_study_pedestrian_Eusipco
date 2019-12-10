# -*- coding: utf-8 -*-
# @Author: miana1
# @Description: A script to reduce the number of positive samples accroding to some criterion
#               such as riemannian variance.
#               Will replace original data file which is aimed to be deleted after processing.
# @Date:   2019-11-04 13:27:55
# @E-mail: ammar.mian@aalto.fi
# @Last Modified by:   miana1
# @Last Modified time: 2019-12-10 17:07:15
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
from sklearn.utils import shuffle

if __name__ == '__main__':

    # Managing inputs of this script
    parser = argparse.ArgumentParser(description='Reduce positive samples according to some criterion and format data.')
    parser.add_argument("data_file", help="Path (From base to the file containing machine learning features")
    parser.add_argument("method_name",  help="Name of method to reduce samples.")
    parser.add_argument("method_args",  help="Arguments to the method")
    parser.add_argument("number_to_keep",  type=int, help="Number of positive samples to keep")
    parser.add_argument("-s", "--shuffle",  action="store_true",
                                help="Shuffle the features or not.")
    parser.add_argument("--shuffle_seed",  type=int, default=0,
                                help="Seed for shuffling the features")
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
    from utils.reducing_training_samples import *

    # Read logging configuration
    configure_logging(os.path.join(absolute_base_path, "logging.yaml"))

    logging.info("Reducing positive samples with method %s and arguments %s",
                                        args.method_name, args.method_args)

    # Parse method for computing machine learning features
    method = parse_reducing_samples_method(args.method_name, args.method_args)
    if args.parallel:
        method.set_parallel_variables(True, args.n_jobs)

    # Read data from pickle dump
    logging.info("Reading machine learning features data from file %s",
                                path_to_machine_learning_features_data_file)
    with open(path_to_machine_learning_features_data_file, 'rb') as f:
        dataset = pickle.load(f)

    # First obtaining training and testing so we don't lose the information
    train_samples = [dataset['features'][i] for i in dataset['indexes train']]
    test_samples = [dataset['features'][i] for i in dataset['indexes test']]
    train_labels = [dataset['labels'][i] for i in dataset['indexes train']]
    test_labels = [dataset['labels'][i] for i in dataset['indexes test']]
    train_images_paths = [dataset['images paths'][i] for i in dataset['indexes train']]
    test_images_paths = [dataset['images paths'][i] for i in dataset['indexes test']]

    # Obtaining positive and negative samples
    positive_train_indices = [i for i, x in enumerate(train_labels) if x == 1]
    negative_train_indices = [i for i, x in enumerate(train_labels) if x == -1]
    positive_test_indices = [i for i, x in enumerate(test_labels) if x == 1]
    negative_test_indices = [i for i, x in enumerate(test_labels) if x == -1]

    positive_train_samples = [train_samples[i] for i in positive_train_indices]
    positive_test_samples = [test_samples[i] for i in positive_test_indices]
    negative_train_samples = [train_samples[i] for i in negative_train_indices]
    negative_test_samples = [test_samples[i] for i in negative_test_indices]

    # Reducing positive samples
    positive_samples = np.array(positive_train_samples + positive_test_samples)
    indexes = method.reduce_samples(positive_samples, args.number_to_keep)
    positive_samples = positive_samples[:, indexes, :]

    positive_sub_regions = [dataset['positive sub-regions'][i] for i in indexes]

    # Merging positive and negative samples
    positive_samples = [list(i) for i in list(positive_samples)]
    positive_train_samples = positive_samples[:len(positive_train_indices)]
    positive_test_samples = positive_samples[len(positive_train_indices):]

    machine_learning_features_list = positive_train_samples + negative_train_samples +\
                                     positive_test_samples + negative_test_samples

    machine_learning_labels_list = [1]*len(positive_train_samples) + \
                                   [-1]*len(negative_train_samples) + \
                                   [1]*len(positive_test_samples) + \
                                   [-1]*len(negative_test_samples)


    image_paths = [train_images_paths[i] for i in positive_train_indices] + \
                  [train_images_paths[i] for i in negative_train_indices] + \
                  [test_images_paths[i] for i in positive_test_indices] + \
                  [test_images_paths[i] for i in negative_test_indices]

    indexes_train_temp = np.arange( len(positive_train_indices)+len(negative_train_indices))
    indexes_test_temp = np.arange( len(positive_train_indices)+len(negative_train_indices),
                                   len(machine_learning_labels_list) )

    if args.shuffle:
        temp = list(np.arange(len(machine_learning_labels_list)))
        machine_learning_features_list, machine_learning_labels_list, image_paths, temp = \
                                    shuffle(machine_learning_features_list,
                                            machine_learning_labels_list,
                                            image_paths, temp,
                                            random_state=args.shuffle_seed)
        indexes_train = []
        for i in indexes_train_temp:
            indexes_train.append(temp.index(i))
        indexes_test = []
        for i in indexes_test_temp:
            indexes_test.append(temp.index(i))

    else:
        indexes_train = indexes_train_temp
        indexes_test = indexes_test_temp

    dataset['features'] = machine_learning_features_list
    dataset['labels'] = machine_learning_labels_list
    dataset['positive sub-regions'] = positive_sub_regions
    dataset['images paths'] = image_paths
    dataset['indexes train']= indexes_train
    dataset['indexes test']= indexes_test

    with open(path_to_machine_learning_features_data_file, 'wb') as f:
        logging.info("Rewriting temporary machine learning features data into %s",
                                        path_to_machine_learning_features_data_file)
        logging.info("Don't forget to delete the file after use!")
        pickle.dump(dataset, f)
