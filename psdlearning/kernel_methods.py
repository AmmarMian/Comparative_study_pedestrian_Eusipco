# -*- coding: utf-8 -*-
# @Author: miana1
# @Description: File regrouping Riemannian kernel methodologies
# @Date:   2020-01-21 14:25:05
# @E-mail: ammar.mian@aalto.fi
# @Last Modified by:   miana1
# @Last Modified time: 2020-02-11 15:52:49
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
from tqdm import trange, tqdm
from itertools import combinations
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from .utils.base import *
from pyriemann.utils.base import logm
from pyriemann.utils.distance import distance_logeuclid
from pyriemann.utils.mean import mean_riemann, mean_logeuclid
from psdlearning.utils.algebra import vec
from sklearn.svm import SVC


def vec_logm(X):
    return vec(logm(X))


class spd_rbf_kernel_svc(machine_learning_method):
    """
       SVM with a Riemannian RBF kernel as described in:
       S. Jayasumana, R. Hartley, M. Salzmann, H. Li and M. Harandi,
       "Kernel Methods on the Riemannian Manifold of Symmetric Positive Definite
        Matrices," 2013 IEEE Conference on Computer Vision and Pattern Recognition*, Portland, OR, 2013, pp. 73-80.
       doi: 10.1109/CVPR.2013.17
       URL: https://ieeexplore.ieee.org/document/6618861.

       The implementation is based on the SVC module of sklearn:
       https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

       Parameters:
            * gamma: either 'auto' or a float.
            Width of the kernel.
            In the first case, it is defined as:
                gamma = 1 / var(X), where var is the variance on the training samples
                as defined in eq (2) of the paper.
            In the second case, the value is used.
            * p: power on the distance for computing the gamma automatically
                 default is 2.

       Attributs:
            * classifier: a sklearn SVC instance which is used for the classification
            * logm_X_train = contain the logm of all training samples used to fit
                             for faster computation.
    """

    def __init__(self, method_name, method_args):
        super(spd_rbf_kernel_svc, self).__init__(method_name, method_args)
        self.init_method()


    def init_method(self):
        svc_args = self.method_args
        svc_args['kernel'] = 'precomputed'

        # Setting parameter gamma
        if isinstance(svc_args['gamma'], float):
            if svc_args['gamma'] > 0:
                self.gamma = svc_args['gamma']
            else:
                logging.error("gamma should be a positive float")
                raise ValueError("gamma should be a positive float")
        else:
            self.gamma = 'auto'

        # Setting parameter 2
        if 'p' in svc_args:
            self.p = svc_args['p']
        else:
            self.p = 2
        # Setting the internal SVC classifier
        del svc_args['p']
        self.classifier = SVC(**self.method_args)

        # Will serve to keep in memory the training samples needed
        # To compute the Gram matrix in prediction
        self.X_train = None


    def __str__(self):
        string_to_print = f"spd_rbf_kernel_svc(gamma={self.gamma}, p={self.p})"
        string_to_print += f"\n Intern SVC classifier:\n{str(self.classifier)}"
        return string_to_print


    def _compute_gamma_auto(self, X_train):
        # mu = mean_riemann(X_train)
        mu = mean_logeuclid(X_train)
        sigma = 0
        for matrix in X_train:
            sigma += np.power(distance_logeuclid(matrix, mu), self.p)
        self.gamma = len(X_train) / sigma


    def fit(self, X_train, y_train):
        """
        Inputs:
            * X_train = ndarray, shape (n_samples, n_channels, n_channels)
            ndarray of SPD matrices.
            * y_train = ndarray shape (n_samples, 1)
            labels corresponding to each sample.
        """

        logging.debug(f'Training Riemannian RBF kernel SVC on {len(y_train)} samples')
        # Computing gamma if needed:
        if self.gamma == 'auto':
            logging.debug('Computing gamma automatically')
            self._compute_gamma_auto(X_train)
            logging.debug('Computing done')

        # Computing Gram matrix
        logging.debug('Computing Gram matrix for training')

        # Compute logm then using scipy functions to gain time
        logging.debug(f'Computing logm of {X_train.shape[0]} matrices')
        if self.parallel:
            logm_X_train = Parallel(n_jobs=self.n_jobs)(delayed(vec_logm)(Sigma) for Sigma in X_train)
        else:
            logm_X_train = np.empty((X_train.shape[0], X_train.shape[1]**2))
            for i in range(X_train.shape[0]):
                logm_X_train[i] = vec_logm(X_train[i])

        logging.debug(f'Computing pairwise distances')
        pairwise_distances = cdist(logm_X_train, logm_X_train, 'sqeuclidean')

        gram_matrix = np.exp(-self.gamma*pairwise_distances)
        logging.debug('Done')

        logging.debug('Fitting SVC')
        self.classifier.fit(gram_matrix, y_train)
        logging.debug('Done')
        gram_matrix = None

        self.logm_X_train = np.array(logm_X_train)
        logging.debug('Finished training')

        return self


    def predict(self, X_test):
        """
        Inputs:
            * X_test = ndarray, shape (n_samples, n_channels, n_channels)
            ndarray of SPD matrices.
        Outputs:
            * y_test = ndarray shape (n_samples, 1) of predicted
            labels corresponding to each sample.
        """

        # Computing Gram matrix
        logging.debug(f'Predicting {X_test.shape[0]} samples based on Riemannian RBF kernel SVC')
        n_samples_train = self.logm_X_train.shape[0]
        n_samples_test = X_test.shape[0]

        # Computing Gram matrix
        logging.debug('Computing Gram matrix for training')

        # Compute logm then using scipy functions to gain time
        logging.debug(f'Computing logm of {X_test.shape[0]} matrices')
        if self.parallel:
            logm_X_test = Parallel(n_jobs=self.n_jobs)(delayed(vec_logm)(Sigma) for Sigma in X_test)
        else:
            logm_X_test = np.empty((X_test.shape[0], X_test.shape[1]**2))
            for i in range(X_test.shape[0]):
                logm_X_test[i] = vec_logm(X_test[i])

        logging.debug(f'Computing pairwise distances')
        pairwise_distances = cdist(logm_X_test, self.logm_X_train, 'sqeuclidean')

        gram_matrix = np.exp(-self.gamma*pairwise_distances)
        logging.debug('Done')

        logging.debug('Doing prediction')
        return self.classifier.predict(gram_matrix)
        logging.debug('Finished predicting')
