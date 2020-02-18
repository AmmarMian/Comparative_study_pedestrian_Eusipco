# -*- coding: utf-8 -*-
# @Author: miana1
# @Description: File regrouping euclidean based methodologies
# @Date:   2020-01-21 14:23:19
# @E-mail: ammar.mian@aalto.fi
# @Last Modified by:   miana1
# @Last Modified time: 2020-01-29 17:00:46
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

from .utils.base import *
from .utils.logitboost import LogitBoost
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor


class sklearn_svc_method(machine_learning_method):
    """
       A wrapper for the svc algorithm of sklearn: More details at:
       https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    """
    def __init__(self, method_name, method_args):
        super(sklearn_svc_method, self).__init__(method_name, method_args)
        self.init_method()

    def init_method(self):
        self.classifier = SVC(**self.method_args)

    def __str__(self):
        string_to_print += f"\n Intern SVC classifier:\n{str(self.classifier)}"
        return string_to_print

    def fit(self, X_train, y_train):
        """
        Inputs:
            * X_train = ndarray, shape (n_samples, n_channels, n_channels)
            ndarray of SPD matrices.
            * y_train = ndarray shape (n_samples, 1)
            labels corresponding to each sample.
        """
        n_samples, n_channels = X_train.shape[:2]
        self.classifier.fit(X_train.reshape((n_samples, n_channels**2)), y_train)
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
        n_samples, n_channels = X_test.shape[:2]
        return self.classifier.predict(X_test.reshape((n_samples, n_channels**2)))


class logitboost_method(machine_learning_method):
    """
       A wrapper for the Euclidean logitboost. More details at:
       https://github.com/artemmavrin/logitboost/blob/master/src/logitboost/logitboost.py
    """
    def __init__(self, method_name, method_args):
        super(logitboost_method, self).__init__(method_name, method_args)
        self.init_method()


    def init_method(self):

        # Parsing arguments
        if self.method_args['base_estimator'] == 'decision stump':
            base_estimator = DecisionTreeRegressor(max_depth=1)
        # TODO: Add other options
        else:
            logging.error(f'Sorry base estimator option %s is not recognised', self.method_args['base_estimator'])
            return None

        self.classifier = LogitBoost(base_estimator,
                        self.method_args['n_estimators'],
                        self.method_args['weight_trim_quantile'],
                        self.method_args['max_response'],
                        self.method_args['learning_rate'],
                        self.method_args['bootstrap'],
                        self.method_args['random_state'])

    def __str__(self):
        string_to_print += f"\n Riemannian LogitBoost on SPD matrices with args:\n" + self.method_args
        return string_to_print

    def fit(self, X_train, y_train):
        """
        Inputs:
            * X_train = ndarray, shape (n_samples, n_channels, n_channels)
            ndarray of SPD matrices.
            * y_train = ndarray shape (n_samples, 1)
            labels corresponding to each sample.
        """
        n_samples, n_channels = X_train.shape[:2]
        self.classifier.fit(X_train.reshape((n_samples, n_channels**2)), y_train)
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
        n_samples, n_channels = X_test.shape[:2]
        return self.classifier.predict(X_test.reshape((n_samples, n_channels**2)))
