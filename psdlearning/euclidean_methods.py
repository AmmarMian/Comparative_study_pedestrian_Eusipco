# -*- coding: utf-8 -*-
# @Author: miana1
# @Description: File regrouping euclidean based methodologies
# @Date:   2020-01-21 14:23:19
# @E-mail: ammar.mian@aalto.fi
# @Last Modified by:   miana1
# @Last Modified time: 2020-01-24 12:56:40
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
from sklearn.svm import SVC

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

