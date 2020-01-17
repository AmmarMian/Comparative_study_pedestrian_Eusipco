# -*- coding: utf-8 -*-
# @Author: Mian Ammar
# @Description: A script allowing to read pedestrian detection datasets
#               and computing features
# @Date:   2019-10-17 15:21:21
# @E-mail: ammar.mian@aalto.fi
# @Last Modified by:   miana1
# @Last Modified time: 2020-01-16 10:57:57
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

def read_dataset_from_choice(choice, absolute_base_path, paths, shuffling, shuffle_seed):
    """ Function that read the data chosen from its name.
        Inputs:
            * choice = an str that is either 'INRIA' or
                        'DaimerChrysler'.
            * absolute_base_path = an str containing the absolute path
                                    to the base of this project.
            * paths = a dictionary obtained by reading the file
                    Simulations/Global/logging.yaml.
            * shuffling = a boolean in order to shuffle the data or not.
            * shuffle_seed = an int which is the seed for rng when shuffling the dataset
        Outputs:
            * dataset = an pedestrian_dataset object containing the
                        dataset chosen.
    """

    if choice == 'INRIA':
        path_to_dataset_base = os.path.join(absolute_base_path,
                        paths['data_path']['INRIA']['base'])
        dataset = INRIA_dataset(path_to_dataset_base)
        dataset.read_data(paths['data_path']['INRIA']['train'],
                        paths['data_path']['INRIA']['test'], shuffling=shuffling, seed=shuffle_seed)
        return dataset

    elif choice == 'DaimerChrysler':
        path_to_dataset_base = os.path.join(absolute_base_path,
                        paths['data_path']['DaimlerChrysler']['base'])
        dataset = DaimerChrysler_base_dataset(path_to_dataset_base)
        dataset.read_data(shuffling=shuffling, seed=shuffle_seed)
        return dataset

    else:
        logging.error('Sorry %s dataset is not recognized' % choice)
        raise NameError


if __name__ == '__main__':

    # Managing inputs of this script
    parser = argparse.ArgumentParser(description='Read some pedestrian detection dataset ' + \
                                                'and compute 8-dimensional features')
    parser.add_argument("dataset", help="Name of dataset to read")
    parser.add_argument("-s", "--shuffle", action="store_true",
                         help="Shuffle the dataset")
    parser.add_argument("--shuffle_seed", type=int, default=None,
                         help="Seed for rng when shuffling the dataset")
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

    # Init paths, and import needed packages
    sys.path.insert(0, absolute_base_path)
    from global_utils import *
    from utils.data_management import *

    # Read logging configuration
    configure_logging(os.path.join(absolute_base_path,"logging.yaml"))

    # Importing path file to have info about all needed data
    paths = read_paths(os.path.join(absolute_base_path, "paths.yaml"))

    if not os.path.isfile(path_to_data_storage_file):

        try:
            # Reading dataset
            dataset = read_dataset_from_choice(args.dataset, absolute_base_path, paths,
                                                        args.shuffle, args.shuffle_seed)

            # Compute 8-dimensionnal features
            if dataset is not None:
                dataset.compute_eight_dimensional_feature_tensors(args.parallel, args.n_jobs)

            # Storing data using pickle
            with open(path_to_data_storage_file, 'wb') as f:
                logging.info("Writing dataset into %s", path_to_data_storage_file)
                pickle.dump(dataset, f)

        except NameError as e:
            logging.info("NameError, Ending execution here")
            logging.error(str(e))

    else:
        logging.info("File %s already exists, ending here", path_to_data_storage_file)

