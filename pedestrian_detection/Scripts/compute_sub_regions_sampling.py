# -*- coding: utf-8 -*-
# @Author: miana1
# @Description: A file that allows to compute a sampling of sub_regions for both 
#               positive and negative training images
# @Date:   2019-10-23 17:17:49
# @E-mail: ammar.mian@aalto.fi
# @Last Modified by:   miana1
# @Last Modified time: 2019-12-10 17:31:40
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
from tqdm import tqdm
import time
import numpy as np


def generate_sub_regions_random(N_R, h, w, n_h, n_w, seed=None, progress=False):
    """ Function to obtain a uniformly random sampling for the sub_regions.

        Usage: sub_regions_list = generate_sub_regions_random(N_R, h, w, n_h, n_w, seed, progress)
        Inputs:
            * N_R = an int corresponding to number of regions to generate.
            * h = an int corresponding to height of the image.
            * w = an int corresponding to width of the image.
            * n_w = an int so that the sub_regions are of width that is minimum \ceil{w/n_w}.
            * n_h = an int so that the sub_regions are of height that is minimum \ceil{h/n_h}.
            * seed = an int which is the seed for rng so that it is reproducible.
            * progress = a boolean to show or not a progress bar for the generation.
        Outputs:
            * sub_regions_list = a list of ints [x_j, y_j, w_j, h_j] where:
                - (x_j, y_j) are the coordinates of the left corner of the region
                - (w_j, h_j) are the width and heigth of the region
    """


    if type(seed) is int:
        rng = np.random.RandomState(seed)
    else:
        rng = seed

    sub_regions_list = []
    if progress:
        pbar = tqdm(total=N_R)
    for j in range(N_R):
        x_j = rng.randint(0, w - int(np.ceil(w/n_w)))
        y_j = rng.randint(0, h - int(np.ceil(h/n_h)))
        w_j = rng.randint(int(np.ceil(w/n_w)), w - x_j)
        h_j = rng.randint(int(np.ceil(h/n_h)), h - y_j)
        sub_regions_list.append( (x_j, y_j, w_j, h_j) )
        if progress:
                pbar.update(1)
    return sub_regions_list


def compute_overlap_percent(region_1, region_2):
    """ Function to compute the overlap in percent between two regions.

        Usage: overlap_percent = compute_overlap_percent(region_1, region_2)
        Inputs:
            * region_1 = an list of ints corresponding to x_1, y_1, w_1, h_1
            * region_2 = an list of ints corresponding to x_2, y_2, w_2, h_2
        Outputs:
            * overlap_percent = a float corresponding to the overlap in percent.
    """
    x_1, y_1, w_1, h_1 = region_1
    x_2, y_2, w_2, h_2 = region_2

    if x_1 < x_2:
        w_j_tilde = w_1
    else:
        w_j_tilde = w_2

    if y_1 < y_2:
        h_j_hat = h_1
    else:
        h_j_hat = h_2

    if (np.abs(x_1-x_2) < w_j_tilde) and ( np.abs(y_1-y_2)<h_j_hat ):
        overlap_percent = 100 * (h_j_hat - np.abs(y_1-y_2)) * \
                 (w_j_tilde - np.abs(x_1-x_2)) / np.min( (h_1*w_1, h_2*w_2) )
    else:
        overlap_percent = 0

    return overlap_percent


def generate_sub_regions_random_with_overlap_constraint(N_R, h, w, n_h, n_w, overlap_threshold=75, timeout=30,
                                                    seed=None, progress=False):
    """ Function to obtain a uniformly random sampling for the sub_regions.

        Usage: sub_regions_list = generate_sub_regions_random_with_overlap_constraint(N_R, h, w, n_h,
                                                    n_w, overlap_threshold, timeout, seed, progress)
        Inputs:
            * N_R = an int corresponding to number of regions to generate.
            * h = an int corresponding to height of the image.
            * w = an int corresponding to width of the image.
            * n_w = an int so that the sub_regions are of width that is minimum \ceil{w/n_w}.
            * n_h = an int so that the sub_regions are of height that is minimum \ceil{h/n_h}.
            * overlap_threshold = a float corresponding to the maximum overlap in percent.
            * timeout = a float corresponding to the timeout limit in minutes of this function.
            * seed = an int or a rng which is the seed for rng so that it is reproducible.
            * progress = a boolean to show or not a progress bar for the generation.
        Outputs:
            * sub_regions_list = a list of ints [x_j, y_j, w_j, h_j] where:
                - (x_j, y_j) are the coordinates of the left corner of the region,
                - (w_j, h_j) are the width and heigth of the region.
    """

    if type(seed) is int:
        rng = np.random.RandomState(seed)
    else:
        rng = seed

    sub_regions_list = []
    j = 0
    if progress:
        pbar = tqdm(total=N_R)
    t_beginning = time.time()
    timeout_seconds = timeout*60
    while j<N_R and (time.time()-t_beginning)<timeout_seconds:

        # randomly generate using uniform distribution a set of values
        x_j = rng.randint(0, w - int(np.ceil(w/n_w)))
        y_j = rng.randint(0, h - int(np.ceil(h/n_h)))
        w_j = rng.randint(int(np.ceil(w/n_w)), w - x_j)
        h_j = rng.randint(int(np.ceil(h/n_h)), h - y_j)

        # Checking overlap
        is_not_overlapping = True
        for region in sub_regions_list:
            if compute_overlap_percent( region, (x_j, y_j, w_j, h_j) ) > overlap_threshold:
                is_not_overlapping = False
                break
        if is_not_overlapping:
            sub_regions_list.append( (x_j, y_j, w_j, h_j) )
            j = j + 1
            if progress:
                pbar.update(1)
    if j < N_R:
        logging.warning("Timed out after %.2f minutes and %d sub_regions", (time.time()-t_beginning)/60, j)
        logging.warning("The remainder of generated sub-regions won't have the overlap constraint")
        for index in range(j, N_R):
            x_j = rng.randint(0, w - int(np.ceil(w/n_w)))
            y_j = rng.randint(0, h - int(np.ceil(h/n_h)))
            w_j = rng.randint(int(np.ceil(w/n_w)), w - x_j)
            h_j = rng.randint(int(np.ceil(h/n_h)), h - y_j)
            sub_regions_list.append( (x_j, y_j, w_j, h_j) )
            if progress:
                pbar.update(1)

    return sub_regions_list


if __name__ == '__main__':

    # Managing inputs of this script
    parser = argparse.ArgumentParser(description='Sample sub-regions.')
    parser.add_argument("dataset", help="Name of dataset to read")
    parser.add_argument("seed", type=int, default=None,
                         help="Seed for rng to have reproducible results")
    parser.add_argument("N_R_pos",  type=int, help="Number of sub_regions to create for positive examples"+\
                        " (Not guaranteed in the case of overlap constraint).")
    parser.add_argument("N_R_neg",  type=int, help="Number of sub_regions to create for positive examples"+\
                        " (Not guaranteed in the case of overlap constraint).")
    parser.add_argument("n_w",  type=int, help="An int so that the sub_regions are of width that is minimum \ceil{w/n_w}")
    parser.add_argument("n_h",  type=int, help="An int so that the sub_regions are of height that is minimum \ceil{h/n_h}")
    parser.add_argument("-m", "--method",  choices=['random', 'random_overlap'],
                        default='random', help="Method for sampling. " +\
                            "Either 'random' (default) or 'random_overlap'")
    parser.add_argument("-o", "--overlap_threshold", type=float, default=75,
                         help="Maximum overlap in percent. Default is 75 percent.")
    parser.add_argument("-t", "--timeout", type=float, default=30,
                         help="Timeout limit for overlap method in minutes. Default is 30 minutes.")
    parser.add_argument("-p", "--progress", action="store_true",
                         help="Show a tqdm progress bar or not")
    args = parser.parse_args()

    # We always need to know where this script is with regards to base of
    # project, so we define these variables to make everything run smoothly
    path_to_base = "../../"
    folder_of_present_script = os.path.dirname(os.path.realpath(__file__))
    absolute_base_path = os.path.join(folder_of_present_script, path_to_base)
    path_to_data_storage_file = os.path.join(folder_of_present_script,
            "../Simulation_data/Eight_dimensional_features", args.dataset)
    path_to_sub_regions_storage_file = os.path.join(folder_of_present_script,
            "../Simulation_data/Sub_regions/", args.dataset+"_method_"+args.method+\
            "_pos_"+str(args.N_R_pos)+"_neg_"+str(args.N_R_neg)+\
            "_nh_"+str(args.n_h)+"_nw_"+str(args.n_w)+"_seed_"+str(args.seed))

    # Init paths, and import needed packages
    sys.path.insert(0, absolute_base_path)
    from global_utils import *

    # Read logging configuration
    configure_logging(os.path.join(absolute_base_path,"logging.yaml"))

    if not os.path.isfile(path_to_sub_regions_storage_file):

        # Read data from pickle dump
        logging.info("Computing windows sampling with method %s and seed %d", args.method, args.seed)
        logging.info("Reading data from file %s", path_to_data_storage_file)
        try:
            with open(path_to_data_storage_file, 'rb') as f:
                dataset = pickle.load(f)

            # Taking into account the seed for the generation
            rng = np.random.RandomState(args.seed)

            # Computing sampling for positive images
            logging.info("Doing %d positive sub-regions", args.N_R_pos)
            feature_tensors, image_paths = dataset.get_positive_examples_feature_tensors()
            h = np.inf
            w = np.inf
            # Just in case, the shape is not consistent, we take the lowest possible size
            for feature_tensor in feature_tensors:
                shape = feature_tensor.shape
                h = int(np.min( [h, shape[0]] ))
                w = int(np.min( [w, shape[1]] ))

            if args.method == "random":
                pos_sub_regions_list = generate_sub_regions_random(args.N_R_pos, h, w, args.n_h, args.n_w,
                                                                seed=rng, progress=args.progress)
            else:
                pos_sub_regions_list = generate_sub_regions_random_with_overlap_constraint(args.N_R_pos, h, w, args.n_h, args.n_w,
                                overlap_threshold=args.overlap_threshold, timeout=args.timeout, seed=rng, progress=args.progress)

            # Computing sampling for negative images
            logging.info("Doing %d negative sub-regions", args.N_R_neg)
            feature_tensors, image_paths = dataset.get_negative_examples_feature_tensors()
            h = np.inf
            w = np.inf
            # Just in case, the shape is not consistent, we take the lowest possible size
            for feature_tensor in feature_tensors:
                shape = feature_tensor.shape
                h = int(np.min( [h, shape[0]] ))
                w = int(np.min( [w, shape[1]] ))

            if args.method == "random":
                neg_sub_regions_list = generate_sub_regions_random(args.N_R_neg, h, w, args.n_h, args.n_w,
                                                                seed=rng, progress=args.progress)
            else:
                neg_sub_regions_list = generate_sub_regions_random_with_overlap_constraint(args.N_R_neg, h, w, args.n_h, args.n_w,
                                overlap_threshold=args.overlap_threshold, timeout=args.timeout, seed=rng, progress=args.progress)

            # Storing data using pickle
            with open(path_to_sub_regions_storage_file, 'wb') as f:
                logging.info("Writing sub-regions into %s", path_to_sub_regions_storage_file)
                pickle.dump([pos_sub_regions_list, neg_sub_regions_list], f)

        except FileNotFoundError:
            logging.error("Data file %s was not found, ending here.", path_to_data_storage_file)
    else:
        logging.info("File %s already exists, ending here", path_to_sub_regions_storage_file)

