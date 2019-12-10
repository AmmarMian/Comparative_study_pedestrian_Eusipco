# -*- coding: utf-8 -*-
# @Author: Mian Ammar
# @Description: A file to gather functions relative to features computation on images.
# @Date:   2019-10-16 17:30:50
# @E-mail: ammar.mian@aalto.fi
# @Last Modified by:   miana1
# @Last Modified time: 2019-10-24 16:53:24
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

import numpy as np
from joblib import Parallel, delayed
from scipy import ndimage
import logging
from tqdm import tqdm


def compute_eight_dimensional_feature(image):
    """ A function to compute the 8-dimensional features for an image as used in:
        O. Tuzel, F. Porikli and P. Meer,
        "Pedestrian Detection via Classification on Riemannian Manifolds",
        in IEEE Transactions on Pattern Analysis and Machine Intelligence,
        vol. 30, no. 10, pp. 1713-1727, Oct. 2008.
        doi: 10.1109/TPAMI.2008.75,

        at eq. (11) p. 1716.

        Usage: image_features = compute_eight_dimensional_feature(image)
        Inputs:
            * image = a numpy array of shape (h, w) corresponding to the image.
        Outputs:
            * image_features = a numpy array of shape (h, w, 8) corresponding
                                to the tensor of image features."""

    if len(image.shape) != 2:
        logging.warning("The input given is not an image, outputting None")
        return None

    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    X, Y = np.meshgrid(x,y)
    Ix = ndimage.sobel(image,axis=1,mode='constant')
    Ixx = ndimage.sobel(Ix,axis=1,mode='constant')
    Iy = ndimage.sobel(image,axis=0,mode='constant')
    Iyy = ndimage.sobel(Iy,axis=0,mode='constant')
    I_abs = np.hypot(np.abs(Ix), np.abs(Iy))
    A = np.arctan2(np.abs(Iy), np.abs(Ix))

    return np.dstack([X, Y, np.abs(Ix), np.abs(Iy),
                        I_abs, np.abs(Ixx), np.abs(Iyy), A])


def compute_features_batch(images_list, parallel=False, n_jobs=8):
    """ A function to compute the 8-dimensional features for a batch of images.

        Usage: image_features_list = compute_features_batch(images_list, parallel, n_jobs)
        Inputs:
            * image = a list of 2-D numpy arrays corresponding to the images.
            * parallel = a boolean to activate parallel computation or not.
            * n_jobs = number of jobs to create for parallel computation.
        Outputs:
            * image_features_list = a list of 3-D numpy arrays corresponding
                                to the tensors of image features."""


    logging.info("Computing 8-dimensional features for %d images", len(images_list))
    if not parallel:
        image_features_list = []
        for image in tqdm(images_list):
            image_features_list.append(compute_eight_dimensional_feature(image))
    else:
        image_features_list = Parallel(n_jobs=n_jobs)(delayed(compute_eight_dimensional_feature)(image)
            for image in images_list)

    return image_features_list
