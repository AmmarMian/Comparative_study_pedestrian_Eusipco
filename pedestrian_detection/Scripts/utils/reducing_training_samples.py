# -*- coding: utf-8 -*-
# @Author: miana1
# @Description: A father to gather functions related to filter features samples
#               according to some criterion
# @Date:   2019-11-01 14:31:43
# @E-mail: ammar.mian@aalto.fi
# @Last Modified by:   miana1
# @Last Modified time: 2019-12-10 16:55:03
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
from tqdm import tqdm, trange
from pyriemann.utils.base import expm, logm, sqrtm, invsqrtm
from pyriemann.utils.mean import _get_sample_weight
from pyriemann.utils.distance import distance_logeuclid, distance_riemann
import warnings
from psdlearning.utils.algebra import unvech
from psdlearning.utils.covariance_estimation import *

# ----------------------------------------------------------------------------
# 1 - Global parser for all methods
# ----------------------------------------------------------------------------
def parse_reducing_samples_method(method_name, method_args_string):
    """ A function to parse choices for the methodology to reduce samples automatically.

        Usage: method =
                parse_reducing_samples_method(method_name, method_args_string)
        Inputs:
            * method_name = an str corresponding to the name of the method
            * method_args_string = a string corresponding to the methods arguments to parse
        Outputs:
            * method = a reducing_samples_method object corresponding to
                       the method.
        """

    if method_name == 'Variance to Riemannian mean':
        return variance_to_Riemannian_mean_method(method_args_string)

    elif method_name == 'Variance to Log-Euclidean mean':
        return variance_to_Log_Euclidean_mean_method(method_args_string)

    elif method_name == 'None':
        return no_reduction_method()

    else:
        logging.error("The method %s is not recognized, ending here", method_name)
        raise MethodNotRecognized

# ----------------------------------------------------------------------------
# 2 - Definition of methods
# ----------------------------------------------------------------------------
class reducing_samples_method():
    """A class to formally define a feature samples reduction method."""
    def __init__(self, name, parallel=False, n_jobs = 8):
        self.name = name
        self.parallel = parallel
        self.n_jobs = n_jobs

    def parse_arguments(self, args):
        logging.error("The method parse_arguments is not defined"+\
            " for machine_learning_feature_computation_method class named %s" %self.name)

    def reduce_samples(self, X, number_to_keep):
        logging.error("The method reduce_samples is not defined"+\
            " for machine_learning_feature_computation_method class named %s" %self.name)

    def set_parallel_variables(self, parallel, n_jobs):
        self.parallel = parallel
        self.n_jobs = n_jobs


class no_reduction_method(reducing_samples_method):
    """When we don't need to reduce samples."""
    def __init__(self):
        super(no_reduction_method, self).__init__('No reduction of samples.')

    def parse_arguments(self, args):
        pass

    def reduce_samples(self, X, number_to_keep):
        return np.arange(X.shape[1])


class variance_to_some_mean_method(reducing_samples_method):
    """Class for methods relying on sorting the variance to the mean of the region."""

    def __init__(self, name, distance_function, distance_args,
                                        mean_function, mean_args):
        super(variance_to_some_mean_method, self).__init__(name)
        self.distance_function = distance_function
        self.distance_args = distance_args
        self.mean_function = mean_function
        self.mean_args = mean_args


    def reduce_samples(self, X, number_to_keep):
        """ A method to compute the variance to the mean of samples for every sub-region
            and select the sub-regions with the lowest variance.

            The method doesn't consider the possibility that the number of sub-regions
            vary from image to image since it doesn't make sense to reduce the number of
            sub-regions in this case.

            Inputs:
                * X = a 3-D numpy array corresponding to machine learning
                      features. The first dimension is linked to the image where the
                      feature was computed, the second dimension is linked to the
                      sub-regions to compute the features.
                * number_to_keep = number of samples to keep.

            Parameters:
                * parallel = a boolean to activate parallel computation or not.
                * n_jobs = number of jobs to create for parallel computation.

            Outputs:
                * indexes = indexes corresponding to the data to keep"""

        # Basic check
        if number_to_keep >= X.shape[1]:
            logging.warning('The number of sub-region to keep is greater or equal ' + \
                             'to the number of sub-regions.')
            return X

        # Compute variances
        if self.parallel:
            variances_list = \
            Parallel(n_jobs=self.n_jobs)(delayed(compute_variance_one_region)(X[:,region_index,:],
                                                        self.distance_function, self.distance_args,
                                                        self.mean_function, self.mean_args)
                                                        for region_index in range(X.shape[1]))
        else:
            variances_list = []
            for region_index in trange(X.shape[1]):
                variances_list.append( compute_variance_one_region(X[:,region_index,:],
                self.distance_function, self.distance_args, self.mean_function, self.mean_args) )

        # Making sure to discard nan values if it happens by assigning them to np.inf
        variances_list = np.array(variances_list)
        variances_list[np.isnan(variances_list)] = np.inf

        # Sorting variances and keeping only the number_to_keep lowest features
        indexes = variances_list.argsort()

        return indexes[:number_to_keep]


class variance_to_Riemannian_mean_method(variance_to_some_mean_method):
    """Class for variance_to_some_mean_method with a Riemannian distance
       and mean on the SPD manifold."""

    def __init__(self,  method_args):
        mean_args, distance_args = self.parse_arguments(method_args)
        super(variance_to_Riemannian_mean_method, self).__init__('Variance to Riemannian mean',
            wrapper_distance_riemann, distance_args, mean_riemann_custom, mean_args)


    def parse_arguments(self, method_args):
        mean_args = []
        for x in method_args.replace(" ", "").split('],')[0].strip('][').split(','):
            try:
                temp = float(x)
            except ValueError:
                temp = None
            mean_args.append(temp)

        distance_args = None
        return mean_args, distance_args


class variance_to_Log_Euclidean_mean_method(variance_to_some_mean_method):
    """Class for variance_to_some_mean_method with a Log-Euclidean distance
       and mean on the SPD manifold."""

    def __init__(self,  method_args):
        distance_args, mean_args = self.parse_arguments(method_args)
        super(variance_to_Log_Euclidean_mean_method, self).__init__('Variance to Log-Euclidean mean',
            wrapper_distance_logeuclid, distance_args, mean_logeuclid_custom, mean_args)


    def parse_arguments(self, method_args):
        mean_args = None
        distance_args = None
        return distance_args, mean_args


# ----------------------------------------------------------------------------
# 3 - Useful functions
# ----------------------------------------------------------------------------
def compute_variance_one_region(vech_covariances_region, distance_function,
                                distance_args, mean_function, mean_args):
    """ A function variance to the mean of samples for on sub-region.

        Inputs:
            * vech_covariances_region = a 2-D numpy array where the first dimension is
                                        the images index and the second correspond to the
                                        vech of the covariance.
            * distance_function = a function to compute the distance.
            * distance_args = arguments to pass to distance_function.
            * mean_function = a function to compute the mean of the covariance of the region.
            * mean_args = arguments to pass to mean_function.

        Outputs:
            * variance_region = The variance of this region."""

    # Unpack the covariances
    covariances_array = []
    for image_index in range(vech_covariances_region.shape[0]):
        covariances_array.append( unvech(vech_covariances_region[image_index,:]) )
    covariances_array = np.swapaxes(np.dstack(covariances_array), 2,0)

    # Compute the mean of the region
    covariance_mean = mean_function(covariances_array, mean_args)

    # Compute the variance of the region
    var_region = 0
    for image_index in range(vech_covariances_region.shape[0]):
        var_region += distance_function(covariances_array[image_index, :, :],
                                                covariance_mean, distance_args)
    return var_region / vech_covariances_region.shape[0]


def wrapper_distance_riemann(A, B, args):
    """
        A wrapper for pyriemann.utils.distance.distance_riemann to handle  I/O with
        regards to reducing samples classes.

        For function doc refer to the doc of pyriemann.utils.mean.distance_riemann.
    """
    return distance_riemann(A, B)


def wrapper_distance_logeuclid(A, B, args):
    """
        A wrapper for pyriemann.utils.distance.distance_logeuclid to handle  I/O with
        regards to reducing samples classes.

        For function doc refer to the doc of pyriemann.utils.mean.distance_logeuclid.
    """
    return distance_logeuclid(A, B)


def mean_riemann_custom(covmats, mean_args):
    """
        A custom version of pyriemann.utils.mean.mean_riemann to handle singular matrices
        and I/O with regards to reducing samples classes.

        For function doc refer to the doc of pyriemann.utils.mean.mean_riemann.
    """

    # Taking arguments
    tol, maxiter, init, sample_weight = mean_args

    # init
    sample_weight = _get_sample_weight(sample_weight, covmats)
    Nt, Ne, Ne = covmats.shape
    if init is None:
        C = np.mean(covmats, axis=0)
    else:
        C = init
    k = 0
    nu = 1.0
    tau = np.finfo(np.float64).max
    crit = np.finfo(np.float64).max

    # stop when J<10^-9 or max iteration = 50
    while (crit > tol) and (k < maxiter) and (nu > tol):
        k = k + 1
        C12 = sqrtm(C)
        Cm12 = invsqrtm(C)
        J = np.zeros((Ne, Ne))

        for index in range(Nt):
            tmp = np.dot(np.dot(Cm12, covmats[index, :, :]), Cm12)
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    J += sample_weight[index] * logm(tmp)
                except RuntimeWarning:
                    pass

        crit = np.linalg.norm(J, ord='fro')
        h = nu * crit
        C = np.dot(np.dot(C12, expm(nu * J)), C12)
        if h < tau:
            nu = 0.95 * nu
            tau = h
        else:
            nu = 0.5 * nu

    return C


def mean_logeuclid_custom(covmats, sample_weight=None):
    """
        A custom version of pyriemann.utils.mean.mean_logeuclid to handle singular matrices
        and I/O with regards to reducing samples classes.

        For function doc refer to the doc of pyriemann.utils.mean.mean_logeuclid.
    """

    sample_weight = _get_sample_weight(sample_weight, covmats)
    Nt, Ne, Ne = covmats.shape
    T = np.zeros((Ne, Ne))
    for index in range(Nt):
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                T += sample_weight[index] * logm(covmats[index, :, :])
            except RuntimeWarning:
                pass
    C = expm(T)

    return C