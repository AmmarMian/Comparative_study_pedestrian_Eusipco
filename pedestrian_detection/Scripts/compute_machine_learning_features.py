# -*- coding: utf-8 -*-
# @Author: miana1
# @Description: A script to compute features that will be used in a machine learning algorithm
#               Can be covariance from subregions on feature tensors or
#               only the sub-region feature tensor.
#               Will store in a tmp file wchich is aimed to be deleted after processing.
# @Date:   2019-10-24 15:58:49
# @E-mail: ammar.mian@aalto.fi
# @Last Modified by:   miana1
# @Last Modified time: 2019-12-10 17:40:24
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
    parser = argparse.ArgumentParser(description='Compute features that will be used in machine learning algorithms.')
    parser.add_argument("dataset", help="Name of dataset to read")
    parser.add_argument("sub_regions_file_name", help="Name of the pickled file storing list of sub-regions for postive and negative examples.")
    parser.add_argument("path_to_machine_learning_features_storage",  help="Path (From base) to the folder where to store the machine learning features.")
    parser.add_argument("method_name",  help="Name of method to compute machine learning features.")
    parser.add_argument("method_args",  help="Arguments to the method")
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
    path_to_data_storage_file = os.path.join(folder_of_present_script,
            "../Simulation_data/Eight_dimensional_features", args.dataset)
    path_to_sub_regions_storage_file = os.path.join(folder_of_present_script,
            "../Simulation_data/Sub_regions/", args.sub_regions_file_name)
    path_to_machine_learning_features_storage_file = os.path.join(absolute_base_path,
        args.path_to_machine_learning_features_storage, "machine_learning_features_method_%s"%args.method_name)

    # Init paths, and import needed packages
    sys.path.insert(0, absolute_base_path)
    from global_utils import *
    from utils.machine_learning_feature import *

    # Read logging configuration
    configure_logging(os.path.join(absolute_base_path,"logging.yaml"))

    if not os.path.isfile(path_to_machine_learning_features_storage_file):
        try:
            # Read data from pickle dump
            logging.info("Computing machine learning features with method %s and arguments %s",
                                                                args.method_name, args.method_args)

            # Parse method for computing machine learning features
            method = parse_machine_learning_feature_method(args.method_name, args.method_args)

            logging.info("Reading data from file %s", path_to_data_storage_file)
            with open(path_to_data_storage_file, 'rb') as f:
                dataset = pickle.load(f)

            # Read sub_regions_lists
            logging.info("Reading sub-regions from file %s", path_to_sub_regions_storage_file)
            with open(path_to_sub_regions_storage_file, 'rb') as f:
                temp = pickle.load(f)
                pos_sub_regions_list = temp[0]
                neg_sub_regions_list = temp[1]


            # First let's obtain train and test samples to keep this information
            train_feature_tensors, train_labels, train_images_paths = dataset.get_train_feature_tensors()
            test_feature_tensors, test_labels, test_images_paths = dataset.get_test_feature_tensors()

            positive_train_indices = [i for i, x in enumerate(train_labels) if x == 1]
            negative_train_indices = [i for i, x in enumerate(train_labels) if x == -1]
            positive_test_indices = [i for i, x in enumerate(test_labels) if x == 1]
            negative_test_indices = [i for i, x in enumerate(test_labels) if x == -1]

            positive_feature_tensors_list = [train_feature_tensors[i] for i in positive_train_indices] + \
                                            [test_feature_tensors[i] for i in positive_test_indices]
            negative_feature_tensors_list = [train_feature_tensors[i] for i in negative_train_indices] + \
                                            [test_feature_tensors[i] for i in negative_test_indices]


            # Compute features for positive and negative examples separately
            logging.info("Computing features for positive examples")
            positive_machine_learning_features = \
                compute_machine_learning_features_batch(positive_feature_tensors_list,
                                                        pos_sub_regions_list, method,
                                                        parallel=args.parallel, n_jobs=args.n_jobs)
            logging.info("Computing features for negative examples")
            negative_machine_learning_features = \
                compute_machine_learning_features_batch(negative_feature_tensors_list,
                                                        neg_sub_regions_list, method,
                                                        parallel=args.parallel, n_jobs=args.n_jobs)

            # Merging features and shuffling
            train_positive_machine_learning_features = positive_machine_learning_features[:len(positive_train_indices)]
            test_positive_machine_learning_features = positive_machine_learning_features[len(positive_train_indices):]
            train_negative_machine_learning_features = negative_machine_learning_features[:len(negative_train_indices)]
            test_negative_machine_learning_features = negative_machine_learning_features[len(negative_train_indices):]

            machine_learning_features_list = train_positive_machine_learning_features + \
                                             train_negative_machine_learning_features + \
                                             test_positive_machine_learning_features + \
                                             test_negative_machine_learning_features

            machine_learning_labels_list = [1]*len(train_positive_machine_learning_features) + \
                                           [-1]*len(train_negative_machine_learning_features) + \
                                           [1]*len(test_positive_machine_learning_features) + \
                                           [-1]*len(test_negative_machine_learning_features)

            image_paths = [train_images_paths[i] for i in positive_train_indices] + \
                          [train_images_paths[i] for i in negative_train_indices] + \
                          [test_images_paths[i] for i in positive_test_indices] + \
                          [test_images_paths[i] for i in negative_test_indices]

            indexes_train_temp = np.arange( len(positive_train_indices)+len(negative_train_indices))
            indexes_test_temp = np.arange( len(positive_train_indices)+len(negative_train_indices),
                                      len(machine_learning_labels_list))

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

            # Storing results
            with open(path_to_machine_learning_features_storage_file, 'wb') as f:
                logging.info("Writing temporary machine learning features data into %s",
                                        path_to_machine_learning_features_storage_file)
                logging.info("Don't forget to delete the file after use!")
                dataset = {'features': machine_learning_features_list,
                           'labels': np.array(machine_learning_labels_list),
                           'images paths': image_paths,
                           'indexes train': indexes_train,
                           'indexes test':indexes_test,
                           'positive sub-regions': pos_sub_regions_list,
                           'negative sub-regions': neg_sub_regions_list}
                pickle.dump(dataset, f)

        except (FileNotFoundError, MethodNotRecognized) as e:
            logging.error(e)
    else:
        logging.info("File %s already exists, ending here", path_to_machine_learning_features_storage_file)
