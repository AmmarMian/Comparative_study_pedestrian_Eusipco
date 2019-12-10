# -*- coding: utf-8 -*-
# @Author: miana1
# @Description: A file to gather functions used to compute the final features
#               for machine learning algorithms.
# @Date:   2019-10-24 16:19:39
# @E-mail: ammar.mian@aalto.fi
# @Last Modified by:   miana1
# @Last Modified time: 2019-12-10 16:53:29
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
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from pyriemann.utils.base import invsqrtm
from psdlearning.utils.algebra import vech
from global_utils import MethodNotRecognized

# ----------------------------------------------------------------------------
# 1 - Global parser for all methods
# ----------------------------------------------------------------------------
def parse_machine_learning_feature_method(method_name, method_args_string):
    """ A function to parse choices for the methodology automatically.

        Usage: method, method_args, method_need_whole_data =
                parse_machine_learning_feature_method(method_name, method_args_string)
        Inputs:
            * method_name = an str corresponding to the name of the method
            * method_args_string = a string corresponding to the methods arguments to parse
        Outputs:
            * method = a machine_learning_feature_computation_method object corresponding to
                       the method.
        """

    if method_name == 'SCM normalized by image':
        return sample_covariance_feature_normalized_method(method_args_string)

    elif method_name == 'SCM':
        return sample_covariance_feature_method(method_args_string)

    elif method_name == 'SSCM':
        return spatial_sign_covariance_feature_method(method_args_string)

    elif method_name == 'Raw tensor':
        return raw_tensor_method(method_args_string)

    else:
        logging.error("The method %s is not recognized, ending here", method_name)
        raise MethodNotRecognized


# ----------------------------------------------------------------------------
# 2 - Definition of methods
# ----------------------------------------------------------------------------
class machine_learning_feature_computation_method():
    """A class to formally define a feature computation method."""
    def __init__(self, name, need_preprocessing_on_whole_image):
        self.name = name
        self.need_preprocessing_on_whole_image = need_preprocessing_on_whole_image

    def parse_arguments(self, args):
        logging.error("The method parse_arguments is not defined"+\
            " for machine_learning_feature_computation_method class named %s" %self.name)

    def compute_feature(self, X):
        logging.error("The method compute_feature is not defined"+\
            " for machine_learning_feature_computation_method class named %s" %self.name)

    def preprocessing(self, feature_tensor):
        logging.error("The method preprocessing is not defined or not needed"+\
            " for machine_learning_feature_computation_method class named %s" %self.name)


class sample_covariance_feature_method(machine_learning_feature_computation_method):
    """ A wrapper class for numpy.cov function as a feature. See:
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.cov.html

        Inputs:
            * X = a 3-D numpy array corresponding to the 8-dimensional feature
                  tensor restricted to some sub-region
        Parameters:
            * bias = correspond to the bias parameter:
                Normalization is by (N-1) where N is the number of observations
                (unbiased estimate). If bias is True then normalization is by N.
        Outputs:
            * Sigma = the vech of estimate using sample covariance. """

    def __init__(self, args):
        super(sample_covariance_feature_method, self).__init__("Sample covariance feature", False)
        self.args = args
        self.parse_arguments()

    def parse_arguments(self):
        self.bias = (self.args == 'True')

    def compute_feature(self, X):
        w_j, h_j, d = X.shape
        return vech(np.cov(X.reshape(w_j*h_j, d).T, bias=self.bias))


class spatial_sign_covariance_feature_method(machine_learning_feature_computation_method):
    """ A wrapper class for spatial sign covariance estimator.

        Inputs:
            * X = a 3-D numpy array corresponding to the 8-dimensional feature
                  tensor restricted to some sub-region
        Parameters: None
        Outputs:
            * Sigma = the vech of estimate using spatial sign covariance. """

    def __init__(self, args):
        super(spatial_sign_covariance_feature_method, self).__init__("Spatial sign covariance feature", False)
        self.args = args

    def parse_arguments(self):
        pass

    def compute_feature(self, X):
        w_j, h_j, d = X.shape
        return vech(spatial_sign_cov(X.reshape(w_j*h_j, d)))


class sample_covariance_feature_normalized_method(machine_learning_feature_computation_method):
    """ A class to wrap the computation the normalized covariance estimator proposed in:
        O. Tuzel, F. Porikli and P. Meer,
        "Pedestrian Detection via Classification on Riemannian Manifolds",
        in IEEE Transactions on Pattern Analysis and Machine Intelligence,
        vol. 30, no. 10, pp. 1713-1727, Oct. 2008.
        doi: 10.1109/TPAMI.2008.75,

        at eq. (12) p. 1716.

        Preprocessing:
            * feature_tensor = a 3-D numpy array corresponding to the whole image
                                to compute normalization
        Inputs:
            * X = a 3-D numpy array corresponding to the 8-dimensional feature
                  tensor restricted to some sub-region.
        Parameters:
            * bias = correspond to the bias parameter:
                Normalization is by (N-1) where N is the number of observations
                (unbiased estimate). If bias is True then normalization is by N.
            * isqrtm_C_R = inverse square matrix of the covariance of the whole
                           image used for normalization.
        Outputs:
            * Sigma = the vech of estimate using sample covariance. """

    def __init__(self, args):
        super(sample_covariance_feature_normalized_method, self).__init__("Sample covariance feature normalized by image",
                                                                                True)
        self.args = args
        self.parse_arguments()

    def parse_arguments(self):
        self.bias = (self.args == 'True')

    def compute_feature(self, X):
        w_j, h_j, d = X.shape
        return vech(self.isqrtm_C_R @ np.cov(X.reshape(w_j*h_j, d).T, bias=self.bias) @ self.isqrtm_C_R)

    def preprocessing(self, feature_tensor):
        h, w, d = feature_tensor.shape
        C_R = np.cov(feature_tensor.reshape(h*w,d).T, bias=self.bias)
        self.isqrtm_C_R = invsqrtm(C_R)


class raw_tensor_method(machine_learning_feature_computation_method):
    """ A class to just output the tensor as is for tensor regression methods.

        Inputs:
            *  X = a 3-D numpy array corresponding to the 8-dimensional feature
                  tensor restricted to some sub-region.
        Parameters: None
        Outputs: X """

    def __init__(self, args):
        super(raw_tensor_method, self).__init__("Raw tensor", False)

    def parse_arguments(self):
        pass

    def compute_feature(self, X):
        return X


# ----------------------------------------------------------------------------
# 2 - Functions to manage feature computation on all images and sub-regions
# ----------------------------------------------------------------------------
def compute_machine_learning_features_for_one_feature_tensor(feature_tensor,
                                                    sub_regions_list, method, pbar=None):
    """ A function to compute machine learning features for all regions on a
        single feature tensor with a given method.

        Usage: machine_learning_features_list_temp =
                compute_machine_learning_features_for_one_feature_tensor(feature_tensor,
                                                    sub_regions_list, method, pbar)
        Inputs:
            * feature_tensor = 3-D numpy array of shape (h, w, p), where:
                - h is the height of the image,
                - w is the width of the image,
                - p is the dimension of the features, usually 8.
            * sub_regions_list = a list of tuples (x_j, y_j, w_j, h_j), where:
                - (x_j, y_j) are the coordinates of the upper-left corner,
                - (w_j, h_j) are the with and height of the region.
            * method = a machine_learning_feature_computation_method class.
            * pbar = A tqdm progessbar object. If none, no progressbar is displayed.
                     If set to a boolean, create a new pbar.
        Outputs:
            * machine_learning_features_list_temp = the corresponding features in a list
        """

    # Preprocessing on the whole feature tensor if needed
    if method.need_preprocessing_on_whole_image:
        method.preprocessing(feature_tensor)

    # Managing progress bar
    if isinstance(pbar, bool):
        pbar_tmp = tqdm(total=len(sub_regions_list))
    else:
        pbar_tmp = pbar

    # Iterating through regions to compute the features
    machine_learning_features_list_temp = []
    if pbar_tmp is not None:
        for region in sub_regions_list:
            x_j, y_j, w_j, h_j = region
            region_data = feature_tensor[y_j:y_j+h_j, x_j:x_j+w_j, :]
            machine_learning_features_list_temp.append( method.compute_feature(region_data) )
            pbar_tmp.update(1)
    else:
        for region in sub_regions_list:
            x_j, y_j, w_j, h_j = region
            region_data = feature_tensor[y_j:y_j+h_j, x_j:x_j+w_j, :]
            machine_learning_features_list_temp.append( method.compute_feature(region_data) )

    return machine_learning_features_list_temp


def compute_machine_learning_features_batch(feature_tensors_list,
                                                sub_regions_list, method,
                                                        parallel=False, n_jobs=8):
    """ A function to compute machine learning features for all images and sub-regions.

        Usage: machine_learning_features_images_list = \
                compute_machine_learning_features_batch(feature_tensors_list,
                                                sub_regions_list, method,
                                                        parallel, n_jobs)
        Inputs:
            * feature_tensors_list = a list of 3-D numpy arrays corresponding to the
                                     8-dimensional feature tensors.
            * sub_regions_list = a list of tuples (x_j, y_j, w_j, h_j), where:
                - (x_j, y_j) are the coordinates of the upper-left corner,
                - (w_j, h_j) are the with and height of the region.
            * method = a machine_learning_feature_computation_method class.
            * parallel = a boolean to activate parallel computation or not.
            * n_jobs = number of jobs to create for parallel computation.
        Outputs:
            * machine_learning_features_images_list = a nested list of size:
                                    (len(feature_tensors_list), len(sub_regions_list))
                                    of features to input in a machine
                                          learning method.
    """


    logging.info("Computing machine learning features for %d feature tensors", len(feature_tensors_list))

    if not parallel:
        pbar = tqdm(total=len(feature_tensors_list)*len(sub_regions_list))
        machine_learning_features_images_list = []
        for feature_tensor in feature_tensors_list:
            machine_learning_features_sub_region_list = \
                compute_machine_learning_features_for_one_feature_tensor(feature_tensor,
                                                    sub_regions_list, method, pbar=pbar)
            machine_learning_features_images_list.append(machine_learning_features_sub_region_list)
        pbar.close()
    else:
        # Computing things ang parallel and obtaining the result in a list
        machine_learning_features_images_list = Parallel(n_jobs=n_jobs)(delayed(
            compute_machine_learning_features_for_one_feature_tensor)(feature_tensor,
                                                    sub_regions_list, method)
                                        for feature_tensor in feature_tensors_list )

    return machine_learning_features_images_list


# ----------------------------------------------------------------------------
# 3 - TODO: Rewrite functions into method classes
# ----------------------------------------------------------------------------
def wrapper_tyler_estimator_covariance(X, args):
    """ A wrapper function that computes the Tyler Fixed Point Estimator as machine learning feature.

        Usage: Sigma = tyler_estimator_covariance(X, args)
        Inputs:
            * X = a 2-D numpy array of shape (n_samples, n_features)
            * args = a tuple corresponding to (tol, iter_max) where:
                - tol = tolerance for convergence of estimator,
                - iter_max = number of maximum iterations,
        Outputs:
            * Sigma = the vech of estimate using Tyler estimator.
     """

    # Fetching arguments
    tol, iter_max = args
    mean_vector = np.mean(X, axis=0)
    Sigma, delta, iteration = tyler_estimator_covariance(
                            X - np.tile(mean_vector, (X.shape[0], 1)),
                                                        tol, iter_max)
    return vech(Sigma)


def wrapper_student_t_estimator_covariance_mle(X, args):
    """ A wrapper function that computes the MLE for covariance matrix estimation for
        a student t distribution when the degree of freedom is known

        TODO: Substract the mean.

        Usage: (Sigma, delta, iteration) = student_t_estimator_covariance_mle(X, args)
        Inputs:
            * X = a 2-D numpy array of shape (n_samples, n_features)
            * args = a tuple corresponding to (tol, iter_max, init) where:
                - d = number of degrees of freedom for Student-t distribution
                - tol = tolerance for convergence of estimator,
                - iter_max = number of maximum iterations,
                - init = Initial value of estimate, leave to None to initiaize to identity.
        Outputs:
            * Sigma = the estimate
            * delta = the final distance between two iterations.
            * iteration = number of iterations til convergence. """


    # Fetching arguments
    d, tol, iter_max, init = args
    mean_vector = np.mean(X, axis=0)
    Sigma, delta, iteration = student_t_estimator_covariance_mle(
                                X - np.tile(mean_vector, (X.shape[0], 1)),
                                                        d, tol, iter_max)
    return vech(Sigma)

