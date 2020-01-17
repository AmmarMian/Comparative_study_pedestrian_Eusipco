# -*- coding: utf-8 -*-
# @Author: miana1
# @Description: File containing definitions of methods used in pedestrian
#               detection task
# @Date:   2019-11-11 16:36:04
# @E-mail: ammar.mian@aalto.fi
# @Last Modified by:   miana1
# @Last Modified time: 2020-01-14 13:58:58
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
from sklearn.svm import SVC

# ----------------------------------------------------------------------------
# 1 - Global parser for all methods
# ----------------------------------------------------------------------------
class MethodNotRecognized(Exception):
    pass

def parse_machine_learning_method(parsing_string, method_name, method_args):
    """ A function to parse choices for the machine learning algorithm

        Usage: method = parse_machine_learning_method_pedestrian_detection(method_name,
                                                                     method_args_string)
        Inputs:
            * parsing_string = an str to parse the methodology
            * method_name = an str corresponding to the name of the method
            * method_args = a dictionary of paraemters to give to the method
        Outputs:
            * method = a machine_learning_method object corresponding to
                       the method.
        """

    if parsing_string == 'sklearn svc':
        return sklearn_svc_method(method_name, method_args)

    else:
        logging.error("The method %s is not recognized, ending here", method_name)
        raise MethodNotRecognized

# ----------------------------------------------------------------------------
# 2 - Definition of methods
# ----------------------------------------------------------------------------
class machine_learning_method():
    """
        A formal class to organize machine learning methods in order to have
        a common framework.
    """
    def __init__(self, method_name, method_args):
        self.method_name = method_name
        self.method_args = method_args


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

