# -*- coding: utf-8 -*-
# @Author: miana1
# @Description: File gathering covariance estimation functions
# @Date:   2019-10-25 13:47:41
# @E-mail: ammar.mian@aalto.fi
# @Last Modified by:   miana1
# @Last Modified time: 2019-11-13 16:32:36
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

def spatial_sign_cov(X):
    """ Spatial sign covariance matrix estimator.

        Usage: Sigma = spatial_sign_cov(X)
        Inputs:
            * X = a 2-D numpy array of shape (n_samples, n_features)
        Outputs:
            * Sigma = the estimate.
     """

    N, p = X.shape
    mean_vector = np.mean(X, axis=0)
    Y = (X - np.tile(mean_vector, (N, 1))).T
    tau = np.diagonal(Y.conj().T @ Y)
    Y = Y / np.sqrt(tau)
    Y[:, tau==0] = 0
    return p/N * (Y @ Y.conj().T)


def tyler_estimator_covariance(X, tol=1e-3, iter_max=30, init=None):
    """ A function that computes the Tyler Fixed Point Estimator for covariance matrix
    	estimation.

    	Usage: (Sigma, delta, iteration) = tyler_estimator_covariance(X, tol,
    																iter_max, init)
        Inputs:
            * X = a 2-D numpy array of shape (n_samples, n_features)
            * tol = tolerance for convergence of estimator,
            * iter_max = number of maximum iterations,
            * init = Initial value of estimate, leave to None to initiaize to identity.
        Outputs:
            * Sigma = the estimate.
            * delta = the final distance between two iterations.
            * iteration = number of iterations til convergence. """


    Y = X.T

    # Initialisation
    (p, N) = Y.shape
    delta = np.inf # Distance between two iterations
    if init is None:
        Sigma = np.eye(p).astype(Y.dtype) # Initialise estimate to identity
    else:
        Sigma = init

    # Recursive algorithm
    iteration = 0
    while (delta>tol) and (iteration<iter_max):

        # Computing expression of Tyler estimator (with matrix multiplication)
        tau = np.diagonal(Y.conj().T@np.linalg.inv(Sigma)@Y)
        X_bis = Y / np.sqrt(tau)
        Sigma_new = (p/N) * X_bis@X_bis.conj().T

        # Imposing trace constraint: Tr(Sigma) = p
        Sigma_new = p*Sigma_new/np.trace(Sigma_new)

        # Condition for stopping
        delta = np.linalg.norm(Sigma_new - Sigma, 'fro') / np.linalg.norm(Sigma, 'fro')
        iteration = iteration + 1

        # Updating Sigma
        Sigma = Sigma_new

    if iteration == iter_max:
        logging.warning(
        'Tyler estimator: recursive algorithm did not converge after %d iterations', iter_max)

    return (Sigma, delta, iteration)


def student_t_estimator_covariance_mle(X, d, tol=1e-3, iter_max=0, init=None):
    """ A function that computes the MLE for covariance matrix estimation for
    	a student t distribution when the degree of freedom is known

    	Usage: (Sigma, delta, iteration) = student_t_estimator_covariance_mle(X, d,
    															tol, iter_max, init)
        Inputs:
            * X = a 2-D numpy array of shape (n_samples, n_features)
        	* d = number of degrees of freedom for Student-t distribution
            * tol = tolerance for convergence of estimator,
            * iter_max = number of maximum iterations,
            * init = Initial value of estimate, leave to None to initiaize to identity.
        Outputs:
            * Sigma = the estimate
            * delta = the final distance between two iterations.
            * iteration = number of iterations til convergence. """


    # Initialisation
    (p,N) = X.shape
    delta = np.inf # Distance between two iterations
    if init is None:
        Sigma = np.eye(p).astype(Y.dtype) # Initialise estimate to identity
    else:
        Sigma = init

    # Recursive algorithm
    iteration = 0
    while (delta>tol) and (iteration<iter_max):

        # Computing expression of Tyler estimator (with matrix multiplication)
        tau = d + np.diagonal(X.conj().T@np.linalg.inv(Sigma)@X)
        X_bis = X / np.sqrt(tau)
        Sigma_new = ((d+p)/N) * X_bis@X_bis.conj().T

        # Condition for stopping
        delta = np.linalg.norm(Sigma_new - Sigma, 'fro') / np.linalg.norm(Sigma, 'fro')
        iteration = iteration + 1

        # Updating Sigma
        Sigma = Sigma_new

    if iteration == iter_max:
        logging.warning(
        'Student-mle estimator: recursive algorithm did not converge after %d iterations', iter_max)

    return (Sigma, delta, iteration)

