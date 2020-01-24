# -*- coding: utf-8 -*-
# @Author: miana1
# @Description: File regrouping Riemannian kernel methodologies
# @Date:   2020-01-21 14:25:05
# @E-mail: ammar.mian@aalto.fi
# @Last Modified by:   miana1
# @Last Modified time: 2020-01-24 14:49:32
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
from .utils.base import *
from sklearn.svm import SVC
from pyriemann.utils.distance import distance_logeuclid
from pyriemann.utils.mean import mean_riemann, mean_logeuclid
import numpy as np
from tqdm import trange, tqdm
from itertools import combinations
from joblib import Parallel, delayed


def riemannian_rbf_kernel(gamma):

    def _riemannian_rbf_kernel_fixed_gamma(X, Y):
        return np.exp(-gamma*distance_logeuclid(X,Y))
    return _riemannian_rbf_kernel_fixed_gamma


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
            * kernel: a function to compute the rbf kernel given the gamma used
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
                self.kernel = riemannian_rbf_kernel(self.gamma)
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
        self.kernel = riemannian_rbf_kernel(np.power(sigma, -1/self.p))


    def fit(self, X_train, y_train):
        """
        Inputs:
            * X_train = ndarray, shape (n_samples, n_channels, n_channels)
            ndarray of SPD matrices.
            * y_train = ndarray shape (n_samples, 1)
            labels corresponding to each sample.
        """

        logging.info(f'Training Riemannian RBF kernel SVC on {len(y_train)} samples')
        # Computing gamma if needed:
        if self.gamma == 'auto':
            logging.info('Computing gamma automatically')
            self._compute_gamma_auto(X_train)
            logging.info('Computing done')

        # Computing Gram matrix
        logging.info('Computing Gram matrix')
        n_samples = X_train.shape[0]
        G = np.eye(n_samples)
        indices = combinations(range(n_samples), r=2)
        if self.parallel:
            logging.info('Doing it in parallel')
            G_temp = Parallel(n_jobs=self.n_jobs)(delayed(
                        self.kernel)(X_train[i,:,:], X_train[j,:,:])
                                            for i,j in indices )
            for in_product, index in zip(G_temp, indices):
                G[index[0],index[1]] = in_product
                G[index[1],index[0]] = in_product
        else:
            logging.info('Doing it sequentially')
            for i,j in tqdm(indices):
                G[i,j] = self.kernel(X_train[i,:,:], X_train[j,:,:])
                G[j,i] = G[i,j]
        logging.info('Done')
        self.classifier.fit(G, y_train)
        G = None
        self.X_train = X_train
        logging.info('Finished training')


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
        logging.info(f'Predicting {X_test.shape[0]} samples based on Riemannian RBF kernel SVC')
        n_samples_train = self.X_train.shape[0]
        n_samples_test = X_test.shape[0]
        logging.info('Computing Gram matrix')
        G = np.zeros((n_samples_test, n_samples_train))

        # Separating the case n_samples_test < n_samples_train from the other
        if n_samples_test <= n_samples_train:
            indices = np.triu_indices(n_samples_test,m=n_samples_train)
        else:
            indices = np.tril_indices(n_samples_test,m=n_samples_train)

        if self.parallel:
            G_temp = Parallel(n_jobs=self.n_jobs)(delayed(
                       self.kernel)(X_test[i, :, :], self.X_train[j,:,:])
                                            for i,j in zip(*(indices)) )
            for in_product, index in zip(G_temp, indices):
                G[index[0],index[1]] = in_product
        else:
            for i,j in tqdm(zip(*(indices))):
                G[i,j] = self.kernel(X_test[i,:,:], self.X_train[j,:,:])

        # Filling the other triangular part of the matrix
        if n_samples_test <= n_samples_train:
            indices_other = np.tril_indices(n_samples_test,m=n_samples_train, k=-1)
        else:
            indices_other = np.triu_indices(n_samples_test,m=n_samples_train, k=1)

        for i,j in zip(*(indices_other)):
                G[i,j] = G[j,i]

        logging.info('Done')
        return self.classifier.predict(G)
        logging.info('Finished predicting')
