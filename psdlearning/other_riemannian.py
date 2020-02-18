# -*- coding: utf-8 -*-
# @Author: miana1
# @Description: wrapper for pyriemann methods to work in our environment.
# @Date:   2020-01-31 11:18:18
# @E-mail: ammar.mian@aalto.fi
# @Last Modified by:   miana1
# @Last Modified time: 2020-01-31 11:39:59
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

import sys
import os
from .utils.base import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pyriemann.classification import KNearestNeighbor, TSclassifier, MDM
import logging


class wrapper_KNN(machine_learning_method):
    """wrapper for pyriemann KNN"""
    def __init__(self, method_name, method_args):
        super(wrapper_KNN, self).__init__(method_name, method_args)
        self.init_method()

    def init_method(self, n_jobs=1):
        self.classifier = KNearestNeighbor(n_neighbors = self.method_args['n_neighbors'],
                                           metric = self.method_args['metric'],
                                           n_jobs=n_jobs)


    def set_parallel(self, is_parallel=False, n_jobs=8):
        logging.warning('The call to this set_parallel method is reseting the class, and must be fitted again')
        self.parallel = is_parallel
        self.n_jobs = n_jobs

        if self.parallel:
            self.init_method(n_jobs)


    def fit(self, X, y):
        return self.classifier.fit(X, y)


    def predict(self, X):
        return self.classifier.predict(X)


class wrapper_TSclassifier(machine_learning_method):
    """wrapper for pyriemann TSclassifier"""
    def __init__(self, method_name, method_args):
        super(wrapper_TSclassifier, self).__init__(method_name, method_args)
        self.init_method()

    def init_method(self):
        self.classifier = TSclassifier(metric = self.method_args['metric'],
                                       tsupdate = self.method_args['tsupdate'])


    def fit(self, X, y):
        return self.classifier.fit(X, y)


    def predict(self, X):
        return self.classifier.predict(X)


class wrapper_MDM(machine_learning_method):
    """wrapper for pyriemann MDM"""
    def __init__(self, method_name, method_args):
        super(wrapper_MDM, self).__init__(method_name, method_args)
        self.init_method()

    def init_method(self, n_jobs=1):
        self.classifier = MDM(metric = self.method_args['metric'],
                              n_jobs=n_jobs)

    def set_parallel(self, is_parallel=False, n_jobs=8):
        logging.warning('The call to this set_parallel method is reseting the class, and must be fitted again')
        self.parallel = is_parallel
        self.n_jobs = n_jobs

        if self.parallel:
            self.init_method(n_jobs)


    def fit(self, X, y):
        return self.classifier.fit(X, y)


    def predict(self, X):
        return self.classifier.predict(X)
