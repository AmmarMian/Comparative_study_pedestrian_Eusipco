# -*- coding: utf-8 -*-
# @Author: miana1
# @Description: Some useful definitions
# @Date:   2020-01-21 15:28:53
# @E-mail: ammar.mian@aalto.fi
# @Last Modified by:   miana1
# @Last Modified time: 2020-01-29 13:37:48
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

from sklearn.base import ClassifierMixin
from sklearn.ensemble import BaseEnsemble


class machine_learning_method(BaseEnsemble, ClassifierMixin):
    """
        A formal class to organize machine learning methods in order to have
        a common framework.
    """
    def __init__(self, method_name, method_args):
        self.method_name = method_name
        self.method_args = method_args

        # To control eventually some parallel execution
        self.parallel = False
        self.n_jobs = 8


    def set_parallel(self, is_parallel=False, n_jobs=8):
        self.parallel = is_parallel
        self.n_jobs = n_jobs


    def init_method(self):
        logging.warning('Sorry the method %s has not been defined yet', self.method_name)


    def fit(self, X ,y):
        logging.error('Sorry the method %s has not been defined yet', self.method_name)
        logging.error('Returning None')
        return None

    def predict(self, X):
        logging.error('Sorry the method %s has not been defined yet', self.method_name)
        logging.error('Returning None')
        return None