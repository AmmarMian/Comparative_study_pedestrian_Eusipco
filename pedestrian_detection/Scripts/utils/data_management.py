# -*- coding: utf-8 -*-
# @Author: Mian Ammar
# @Description: File containing classes and functions to make data_reading smoother
# @Date:   2019-10-17 13:39:08
# @E-mail: ammar.mian@aalto.fi
# @Last Modified by:   miana1
# @Last Modified time: 2020-01-28 14:50:01
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

import logging
import os
import sys
from PIL import Image
from sklearn.utils import shuffle
import numpy as np
from tqdm import tqdm
folder_of_present_script = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, folder_of_present_script)
from feature import *

class pedestrian_dataset():
    """Class storing datasets and additional information."""

    def __init__(self, name, dataset_path):
        """ Initialise pedestrian dataset.
            Inputs:
                * dataset_path = an str leading to the base of the dataset.
            Outputs: None.
        """
        self.name = name
        self.dataset_path = dataset_path
        self.images_list = None
        self.images_labels = None
        self.image_paths = None
        self.features = None
        self.features_labels = None
        self.number_of_images = 0
        self.indexes_training = None
        self.indexes_testing = None
        logging.debug("Initialized pedestrian_class of name %s and path %s",
                       self.name, self.dataset_path)


    def read_data(self):
        """ Read data into the class, must be specified for each different dataset.
            Inputs: None.
            Outputs: None.
        """
        logging.warning("The read_data method has not been implemented for the class %s yet",
                        self.name)


    def set_dataset_manual(self, images_list, images_labels):
        """ Manually set the data if wanted.
            Inputs:
                * images_list = a list of 2D numpy arrays corresponding to images.
                * images_labels = a list ont int corresponding to the labels in
                           {-1,1} of the images.
            Outputs: None.
        """
        self.images_list = images_list
        self.images_labels = images_labels
        logging.debug("Images manually set for class %s ", self.name)


    def set_features_manual(self, features, features_labels):
        """ Manually set the features if wanted.
            Inputs:
                * features = a numpy array of size (n_samples, n_features)
                * features_labels = a nump array of size (n_samples,)
                                    of corresponding labels
            Outputs: None.
        """
        self.features = features
        self.features_labels = features_labels
        logging.debug("Features manually set for class %s ", self.name)


    def compute_eight_dimensional_feature_tensors(self, parallel=False, n_jobs=8):
        """ Compute 8 dimensional feature for all the images of the dataset.
            To save memory, delete the original images.
            Inputs:
                * parallel = a boolean to activate parallel computation or not.
                * n_jobs = number of jobs to create for parallel computation.
            Outputs: None.
        """

        self.feature_tensors = compute_features_batch(self.images_list,
                                        parallel=parallel, n_jobs=n_jobs)
        self.images_list = None


    def get_positive_examples_feature_tensors(self):
        """ Return only positive examples tensor features.
            Inputs: None.
            Outputs:
                * feature_tensors = a list of numpy 3-D arrays corresponding to
                                 positive examples.
        """

        indices = [i for i, x in enumerate(self.images_labels) if x == 1]
        return [self.feature_tensors[i] for i in indices], [self.images_paths[i] for i in indices]


    def get_negative_examples_feature_tensors(self):
        """ Return only negative examples tensor features.
            Inputs: None.
            Outputs:
                * feature_tensors = a list of numpy 3-D arrays corresponding to
                                 negative examples.
                * image_paths = a list of paths to the image used to compute
                                the feature tensor
        """

        indices = [i for i, x in enumerate(self.images_labels) if x == -1]
        return [self.feature_tensors[i] for i in indices], [self.images_paths[i] for i in indices]


    def get_train_images(self):
        """ Return training data as originally splitted in INRIA dataset.
            Inputs: None.
            Outputs:
                * train_images = a list of numpy 2-D arrays corresponding to
                                 original training set.
                * train_labels = a list of int of corresponding labels in {-1,1}.
        """

        if (self.images_list is not None) and (self.images_labels is not None):
            train_images = [self.images_list[i] for i in self.indexes_training]
            train_labels = [self.images_labels[i] for i in self.indexes_training]
            images_paths = [self.images_paths[i] for i in self.indexes_training]

            return train_images, train_labels, images_paths

        else:
            logging.error("The data was not available, returning None")
            return None


    def get_test_images(self):
        """ Return testing data as originally splitted in INRIA dataset.
            Inputs: None.
            Outputs:
                * test_images = a list of numpy 2-D arrays corresponding to
                                 original training set.
                * test_labels = a list of int of corresponding labels in {-1,1}.
        """

        if (self.images_list is not None) and (self.images_labels is not None):
            test_images = [self.images_list[i] for i in self.indexes_testing]
            test_labels = [self.images_labels[i] for i in self.indexes_testing]
            images_paths = [self.images_paths[i] for i in self.indexes_testing]

            return test_images, test_labels, images_paths

        else:
            logging.error("The data was not available, returning None")
            return None


    def get_train_feature_tensors(self):
        """ Return training data as originally splitted in INRIA dataset.
            Inputs: None.
            Outputs:
                * train_feature_tensors = a list of numpy 3-D arrays corresponding to
                                 original training set with 8-dimensional computed.
                * train_labels = a list of int of corresponding labels in {-1,1}.
        """

        if (self.feature_tensors is not None) and (self.images_labels is not None):
            train_feature_tensors = [self.feature_tensors[i] for i in self.indexes_training]
            train_labels = [self.images_labels[i] for i in self.indexes_training]
            images_paths = [self.images_paths[i] for i in self.indexes_training]

            return train_feature_tensors, train_labels, images_paths

        else:
            logging.error("The data was not available, returning None")
            return None


    def get_test_feature_tensors(self):
        """ Return testing data as originally splitted in INRIA dataset.
            Inputs: None.
            Outputs:
                * test_feature_tensors = a list of numpy 3-D arrays corresponding to
                                 original training set with 8-dimensional computed.
                * test_labels = a list of int of corresponding labels in {-1,1}.
        """

        if (self.feature_tensors is not None) and (self.images_labels is not None):
            test_feature_tensors = [self.feature_tensors[i] for i in self.indexes_testing]
            test_labels = [self.images_labels[i] for i in self.indexes_testing]
            images_paths = [self.images_paths[i] for i in self.indexes_testing]

            return test_feature_tensors, test_labels, images_paths

        else:
            logging.error("The data was not available, returning None")
            return None

    def shuffle_images(self, seed):
        logging.info("Shuffling images as requested for class %s",
                            self.name)
        # We have to permute and find again where the training and test set have
        # been put on if it is needed to know
        temp = list(np.arange(self.number_of_images))
        self.images_list, self.images_labels, self.images_paths, temp = \
                                                        shuffle(self.images_list,
                                                        self.images_labels, self.images_paths,
                                                        temp, random_state=seed)
        self.indexes_training = []
        for i in indexes_training:
            self.indexes_training.append(temp.index(i))
        self.indexes_testing = []
        for i in indexes_testing:
            self.indexes_testing.append(temp.index(i))


class INRIA_dataset(pedestrian_dataset):
    """Class to store data of INRIA pedestrian dataset available at:
       http://pascal.inrialpes.fr/data/human/
    """

    def __init__(self, dataset_path):
        """ Initialise INRIA pedestrian dataset.
            Inputs:
                * dataset_path = an str which is the path to the base of the
                                 dataset as extracted through the zip on the website.
            Outputs: None.
        """
        super(INRIA_dataset, self).__init__("INRIA pedestrian dataset", dataset_path)
        self.positive_image_width = 64
        self.positive_image_height = 128


    def read_data(self, train_directory, test_directory, shuffling=False, seed=None):
        """ Read INRIA pedestrian detection dataset into memory.
            Inputs:
                * train_directory = an str which tells which directory in the base
                                    may be taken as a train if no K-fold is done.
                * test_directory = an str which tells which directory in the base
                                    may be taken as a test if no K-fold is done.
                * shuffling = an optional boolean in order to shuffle the data or not.
                * seed = an optional int to give random seed generator for the shuffling

            Outputs: None.
        """

        # --------------------------------------------------------------------
        # 1 - Reading training images
        # --------------------------------------------------------------------
        path_to_train = os.path.join(self.dataset_path, train_directory)
        logging.info("Reading training images at %s set for class %s",
                      path_to_train, self.name)

        # Reading positive training images into a list
        positive_training_images_names_list = [os.path.join(self.dataset_path, train_directory, 'pos', file)
            for file in os.listdir(os.path.join(self.dataset_path, train_directory, 'pos'))
            if file.endswith('.png')]

        positive_training_images = []
        for image_path in tqdm(positive_training_images_names_list):
            image = np.array(Image.open(image_path).convert('L'))
            x_1, y_1, x_2, y_2 = _calculate_centered_window(image.shape, self.positive_image_width, self.positive_image_height)
            positive_training_images.append(image[y_1:y_2, x_1:x_2])

        # Reading negative training images into a list
        negative_training_images_names_list = [os.path.join(self.dataset_path, train_directory, 'neg', file)
            for file in os.listdir(os.path.join(self.dataset_path, train_directory, 'neg'))
            if file.endswith('.png')]

        negative_training_images = []
        for image_path in tqdm(negative_training_images_names_list):
            image = np.array(Image.open(image_path).convert('L'))
            negative_training_images.append(image)

        # Getting the indexes of traning samples for when needed
        indexes_training = np.arange(len(positive_training_images) + len(negative_training_images))
        logging.info('%d training images read', len(indexes_training))

        # --------------------------------------------------------------------
        # 2 - Reading testing images
        # --------------------------------------------------------------------
        path_to_test = os.path.join(self.dataset_path, test_directory)
        logging.info("Reading testing images at %s set for class %s",
                      path_to_train, self.name)

        # Reading positive testing images into a list
        positive_testing_images_names_list = [os.path.join(self.dataset_path, test_directory, 'pos', file)
            for file in os.listdir(os.path.join(self.dataset_path, test_directory, 'pos'))
            if file.endswith('.png')]

        positive_testing_images = []
        for image_path in tqdm(positive_testing_images_names_list):
            image = np.array(Image.open(image_path).convert('L'))
            x_1, y_1, x_2, y_2 = _calculate_centered_window(image.shape, self.positive_image_width, self.positive_image_height)
            positive_testing_images.append(image[y_1:y_2, x_1:x_2])

        # Reading negative testing images into a list
        negative_testing_images_names_list = [os.path.join(self.dataset_path, test_directory, 'neg', file)
            for file in os.listdir(os.path.join(self.dataset_path, test_directory, 'neg'))
            if file.endswith('.png')]

        negative_testing_images = []
        for image_path in tqdm(negative_testing_images_names_list):
            image = np.array(Image.open(image_path).convert('L'))
            negative_testing_images.append(image)

        # Getting the indexes of traning samples for when needed
        indexes_testing = np.arange(indexes_training[-1]+1,
                        len(positive_training_images) + len(negative_training_images) + \
                        len(positive_testing_images) + len(negative_testing_images))
        logging.info('%d testing images read', len(indexes_testing))

        # --------------------------------------------------------------------
        # 3 - Merging all images
        # --------------------------------------------------------------------
        self.images_list = positive_training_images + negative_training_images + \
                           positive_testing_images + negative_testing_images
        self.images_labels = [1]*len(positive_training_images) + [-1]*len(negative_training_images) + \
                 [1]*len(positive_testing_images) +  [-1]*len(negative_testing_images)
        self.images_paths = positive_training_images_names_list + negative_training_images_names_list + \
                            positive_testing_images_names_list + negative_testing_images_names_list
        self.number_of_images = len(self.images_labels)

        # --------------------------------------------------------------------
        # 4 - Shuffling images if needed
        # --------------------------------------------------------------------
        if shuffling:
           self.shuffle_images(seed)
        else:
            self.indexes_training = indexes_training
            self.indexes_testing = indexes_testing


class DaimerChrysler_base_dataset(pedestrian_dataset):
    """Class to store data of DaimerChrysler benchmark pedestrian dataset available at:
       http://www.gavrila.net/Datasets/Daimler_Pedestrian_Benchmark_D/Daimler_Mono_Ped__Class__Bench/daimler_mono_ped__class__bench.html
    """

    def __init__(self, dataset_path):
        """ Initialise DaimerChrysler pedestrian dataset.
            Inputs:
                * dataset_path = an str which is the path to the base of the
                                 dataset as extracted through the zip on the website.
            Outputs: None.
        """
        super(DaimerChrysler_base_dataset, self).__init__("DaimerChrysler pedestrian dataset", dataset_path)
        self.positive_image_width = 18
        self.positive_image_height = 36


    def read_data(self, shuffling=False, seed=None):
        """ Read DaimerChrysler pedestrian detection dataset into memory.
            Inputs:
                * shuffling = an optional boolean in order to shuffle the data or not.
                * seed = an optional int to give random seed generator for the shuffling

            Outputs: None.
        """

        # --------------------------------------------------------------------
        # 1 - Reading training images
        # --------------------------------------------------------------------
        logging.info("Reading training images for class %s",  self.name)

        images_list = []
        image_paths = []
        images_labels = []
        for data_index in range(1,4):
            path_this_data_batch = os.path.join(self.dataset_path, str(data_index))

            # Positive images
            positive_training_images_names_list_temp = [os.path.join(path_this_data_batch, 'ped_examples', file)
            for file in os.listdir(os.path.join(path_this_data_batch, 'ped_examples'))
            if file.endswith('.pgm')]

            image_paths += positive_training_images_names_list_temp
            images_labels += [1]*len(positive_training_images_names_list_temp)
            for image_path in tqdm(positive_training_images_names_list_temp):
                image = np.array(_read_pgm(image_path))
                images_list.append(image)

            # Negative images
            negative_training_images_names_list_temp = [os.path.join(path_this_data_batch, 'non-ped_examples', file)
            for file in os.listdir(os.path.join(path_this_data_batch, 'non-ped_examples'))
            if file.endswith('.pgm')]

            image_paths += negative_training_images_names_list_temp
            images_labels += [-1]*len(negative_training_images_names_list_temp)
            for image_path in tqdm(negative_training_images_names_list_temp):
                image = np.array(_read_pgm(image_path))
                images_list.append(image)

        indexes_training = np.arange(len(images_list))
        logging.info('%d training images read', len(indexes_training))

        # --------------------------------------------------------------------
        # 2 - Reading testing images
        # --------------------------------------------------------------------
        logging.info("Reading testing images for class %s", self.name)

        for data_index in ['T1', 'T2']:
            path_this_data_batch = os.path.join(self.dataset_path, data_index)

            # Positive images
            positive_testing_images_names_list_temp = [os.path.join(path_this_data_batch, 'ped_examples', file)
            for file in os.listdir(os.path.join(path_this_data_batch, 'ped_examples'))
            if file.endswith('.pgm')]

            image_paths += positive_testing_images_names_list_temp
            images_labels += [1]*len(positive_testing_images_names_list_temp)
            for image_path in tqdm(positive_testing_images_names_list_temp):
                image = np.array(_read_pgm(image_path))
                images_list.append(image)

            # Negative images
            negative_testing_images_names_list_temp = [os.path.join(path_this_data_batch, 'non-ped_examples', file)
            for file in os.listdir(os.path.join(path_this_data_batch, 'non-ped_examples'))
            if file.endswith('.pgm')]

            image_paths += negative_testing_images_names_list_temp
            images_labels += [-1]*len(negative_testing_images_names_list_temp)
            for image_path in tqdm(negative_testing_images_names_list_temp):
                image = np.array(_read_pgm(image_path))
                images_list.append(image)

        indexes_testing = np.arange(len(indexes_training),len(images_labels))
        logging.info('%d testing images read', len(indexes_testing))

        self.images_labels = images_labels
        self.images_list = images_list
        self.images_paths = image_paths
        self.number_of_images = len(self.images_labels)

        # --------------------------------------------------------------------
        # 3 - Shuffling images if needed
        # --------------------------------------------------------------------
        if shuffling:
            self.shuffle_images(seed)
        else:
            self.indexes_training = indexes_training
            self.indexes_testing = indexes_testing


def _calculate_centered_window(image_dim, width,height):
        x1 = int(round((image_dim[1]-width)/2))
        y1 = int(round((image_dim[0]-height)/2))
        x2 = x1 + width
        y2 = y1 + height
        return (x1,y1,x2,y2)


def _read_pgm(_image_path):
    """Return a raster of integers from a PGM as a list of lists."""

    with open(_image_path, 'rb') as pgmf:
        assert pgmf.readline() == b'P5\n'
        (width, height) = [int(i) for i in pgmf.readline().split()]
        depth = int(pgmf.readline())
        assert depth <= 255

        raster = []
        for y in range(height):
            row = []
            for y in range(width):
                row.append(ord(pgmf.read(1)))
            raster.append(row)
        return raster
