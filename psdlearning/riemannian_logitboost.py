# -*- coding: utf-8 -*-
# @Author: miana1
# @Description: An implementaiton of Riemannian logitboost algorithm on SPD
#               matrices features.
# @Date:   2020-01-29 14:21:22
# @E-mail: ammar.mian@aalto.fi
# @Last Modified by:   miana1
# @Last Modified time: 2020-02-11 15:43:11
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

import warnings

import numpy as np
from scipy.special import expit, softmax
from sklearn.base import ClassifierMixin
from sklearn.base import clone, is_regressor
from sklearn.ensemble import BaseEnsemble
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (check_X_y, check_is_fitted,
                                      check_random_state)
import sys
import os
from .utils.logitboost import *
from .utils.base import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.tangentspace import tangent_space
from tqdm import trange

class wrapper_riemannian_logitboost(machine_learning_method):
    def __init__(self, method_name, method_args):
        super(wrapper_riemannian_logitboost, self).__init__(method_name, method_args)
        self.init_method()

    def init_method(self):
        # Parsing arguments
        if self.method_args['base_estimator'] == 'decision stump':
            self.method_args['base_estimator'] = DecisionTreeRegressor(max_depth=1)
        # TODO: Add other options
        else:
            logging.error(f'Sorry base estimator option %s is not recognised', self.method_args['base_estimator'])
            return None
        self.classifier = riemannian_logitboost(**self.method_args)

    def __str__(self):
        string_to_print += f"\n Riemannian LogitBoost classifier with arguments:\n{str(self.method_args)}"
        return string_to_print

    def fit(self, X, y):
        return self.classifier.fit(X, y)

    def predict(self, X):
        return self.classifier.predict(X)



class riemannian_logitboost(LogitBoost):
    """docstring for riemannian_logitboost"""
    def __init__(self, base_estimator=None, n_estimators=50,
                 weight_trim_quantile=0.05, max_response=4.0, learning_rate=1.0,
                 bootstrap=False, random_state=None):
        super(riemannian_logitboost, self).__init__(base_estimator, n_estimators,
                            weight_trim_quantile, max_response, learning_rate,
                            bootstrap, random_state)

        # Usually To control eventually some parallel execution,
        # But here is not used. We put it here so that when set_parallel
        # method is used in a script, we don't have an error
        self.parallel = False
        self.n_jobs = 8

        # Contain the weighted mean for each boost iteration that is used in the
        # prediction
        self.mean_spd_matrices = []


    def set_parallel(self, is_parallel=False, n_jobs=8):
        self.parallel = is_parallel
        self.n_jobs = n_jobs


    def fit(self, X, y, **fit_params):
        """Build a LogitBoost classifier from the training data (`X`, `y`).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features, n_features)
            The training covariance features data.

        y : array-like of shape (n_samples,)
            The target values (class labels).

        **fit_params : keyword arguments
            Additional keyword arguments to pass to the base estimator's `fit()`
            method.

        Returns
        -------
        self : :class:`LogitBoost`
            Returns this LogitBoost estimator.
        """
        # Validate __init__() parameters
        self._validate_estimator()
        random_state = check_random_state(self.random_state)

        # Validate training data
        X, y = check_X_y(X, y, allow_nd=True)
        check_classification_targets(y)

        # Convert y to class label indices
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = self.classes_.shape[0]

        # Extract number of features in X
        self.n_features_ = X.shape[1]

        # Clear any previous estimators and create a new list of estimators
        self.estimators_ = []

        # Check extra keyword arguments for sample_weight: if the user specifies
        # the sample weight manually, then the boosting iterations will never
        # get to update them themselves
        if fit_params.pop('sample_weight', None) is not None:
            warnings.warn('Ignoring sample_weight.', RuntimeWarning)

        # Delegate actual fitting to helper methods
        if self.n_classes_ == 2:
            self._fit_binary(X, y, random_state, fit_params)
        else:
            self._fit_multiclass(X, y, random_state, fit_params)

        return self


    def _fit_binary(self, X, y, random_state, fit_params):
        """Fit a binary LogitBoost model.

        This is Algorithm 3 in Friedman, Hastie, & Tibshirani (2000).
        """
        # Initialize with uniform class probabilities
        p = np.full(shape=X.shape[0], fill_value=0.5, dtype=np.float64)

        # Initialize zero scores for each observation
        scores = np.zeros(X.shape[0], dtype=np.float64)

        # Do the boosting iterations to build the ensemble of estimators
        for i in range(self.n_estimators):
            # Update the working response and weights for this iteration
            sample_weight, z = self._weights_and_response(y, p)

            # Mapping the data to tangent space of the Riemannian mean
            mu = mean_riemann(X, sample_weight=sample_weight)
            self.mean_spd_matrices.append(mu)
            X_tspace = tangent_space(X, mu)

            # Create and fit a new base estimator
            X_train, z_train, kwargs = self._boost_fit_args(X_tspace, z, sample_weight,
                                                            random_state)
            estimator = self._make_estimator(append=True,
                                             random_state=random_state)
            kwargs.update(fit_params)
            estimator.fit(X_train, z_train, **kwargs)

            # Update the scores and the probability estimates, unless we're
            # doing the last iteration
            if i < self.n_estimators - 1:
                new_scores = estimator.predict(X_tspace)
                scores += self.learning_rate * new_scores
                p = expit(scores)


    def _fit_multiclass(self, X, y, random_state, fit_params):
        """Fit a multiclass LogitBoost model.

        This is Algorithm 6 in Friedman, Hastie, & Tibshirani (2000).
        """
        # Initialize with uniform class probabilities
        p = np.full(shape=(X.shape[0], self.n_classes_),
                    fill_value=(1. / self.n_classes_), dtype=np.float64)

        # Initialize zero scores for each observation
        scores = np.zeros((X.shape[0], self.n_classes_), dtype=np.float64)

        # Convert y to a one-hot-encoded vector
        y = np.eye(self.n_classes_)[y]

        # Do the boosting iterations to build the ensemble of estimators
        for iboost in range(self.n_estimators):
            # List of estimators for this boosting iteration
            new_estimators = []
            new_mean_spd_matrices = []

            # Create a new estimator for each class
            new_scores = []
            for j in range(self.n_classes_):
                # Compute the working response and weights
                sample_weight, z = self._weights_and_response(y[:, j], p[:, j])

                # Mapping the data to tangent space of the Riemannian mean
                mu = mean_riemann(X, sample_weight=sample_weight)
                new_mean_spd_matrices.append(mu)
                X_tspace = tangent_space(X, mu)

                # Fit a new base estimator
                X_train, z_train, kwargs = self._boost_fit_args(X_tspace, z,
                                                                sample_weight,
                                                                random_state)
                estimator = self._make_estimator(append=False,
                                                 random_state=random_state)
                kwargs.update(fit_params)
                estimator.fit(X_train, z_train, **kwargs)
                new_estimators.append(estimator)

                # Update the scores and the probability estimates
                if iboost < self.n_estimators - 1:
                    new_scores.append(estimator.predict(X_tspace))

            if iboost < self.n_estimators - 1:
                new_scores = np.asarray(new_scores).T
                new_scores -= new_scores.mean(axis=1, keepdims=True)
                new_scores *= (self.n_classes_ - 1) / self.n_classes_

                scores += self.learning_rate * new_scores
                p = softmax(scores, axis=1)

            self.estimators_.append(new_estimators)
            self.mean_spd_matrices.append(new_mean_spd_matrices)


    def decision_function(self, X):
        """Compute the decision function of `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features, n_features)
            The input covariance features data.

        Returns
        -------
        scores : :class:`numpy.ndarray` of shape (n_samples, k)
            The decision function of the input samples. The order of outputs is
            the same of that of the `classes_` attribute. Binary classification
            is a special cases with `k` = 1, otherwise `k` = `n_classes`. For
            binary classification, positive values indicate class 1 and negative
            values indicate class 0.
        """
        check_is_fitted(self, "estimators_")
        if self.n_classes_ == 2:
            scores = [estimator.predict(tangent_space(X, mu)) for estimator, mu in zip(self.estimators_, self.mean_spd_matrices)]
            scores = np.sum(scores, axis=0)
        else:
            scores = [[estimator.predict(tangent_space(X, mu)) for estimator, mu in zip(estimators, means)]
                      for estimators, means in zip(self.estimators_, self.mean_spd_matrices)]
            scores = np.sum(scores, axis=0).T

        return scores


    def contributions(self, X, sample_weight=None):
        """Average absolute contribution of each estimator in the ensemble.

        This can be used to compare how much influence different estimators in
        the ensemble have on the final predictions made by the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features, n_features)
            The input samples to average over.

        sample_weight : array-like of shape (n_samples,) (default=None)
            Weights for the samples, for averaging.

        Returns
        -------
        contrib : :class:`numpy.ndarray` of shape (n_estimators,)
            Average absolute contribution of each estimator in the ensemble.
        """
        check_is_fitted(self, 'estimators_')

        if self.n_classes_ == 2:
            predictions = [estimator.predict(tangent_space(X, mu)) for estimator, mu in zip(self.estimators_, self.mean_spd_matrices)]
            predictions = np.abs(predictions)
        else:
            predictions = [[estimator.predict(tangent_space(X, mu)) for estimator, mu in zip(estimators, means)]
                      for estimators, means in zip(self.estimators_, self.mean_spd_matrices)]
            predictions = np.abs(predictions)
            predictions = np.mean(predictions, axis=1)

        return np.average(predictions, axis=-1, weights=sample_weight)


    def staged_decision_function(self, X):
        """Compute decision function of `X` for each boosting iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each boosting iteration.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features, n_features)
            The input data.

        Yields
        ------
        scores : :class:`numpy.ndarray` of shape (n_samples, k)
            The decision function of the input samples. The order of outputs is
            the same of that of the `classes_` attribute. Binary classification
            is a special cases with `k` = 1, otherwise `k` = `n_classes`. For
            binary classification, positive values indicate class 1 and negative
            values indicate class 0.
        """
        check_is_fitted(self, 'estimators_')

        if self.n_classes_ == 2:
            scores = 0
            for estimator, mu in zip(self.estimators_, self.mean_spd_matrices):
                scores += estimator.predict(tangent_space(X, mu))
                yield scores
        else:
            scores = 0
            for estimators, means in zip(self.estimators_, self.mean_spd_matrices):
                new_scores = [estimator.predict(tangent_space(X, mu)) for estimator, mu in zip(estimators, means)]
                scores += np.asarray(new_scores).T
                yield scores

