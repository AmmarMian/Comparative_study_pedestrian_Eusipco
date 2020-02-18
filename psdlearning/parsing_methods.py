# -*- coding: utf-8 -*-
# @Author: miana1
# @Description: File containing definitions of methods used in pedestrian
#               detection task
# @Date:   2019-11-11 16:36:04
# @E-mail: ammar.mian@aalto.fi
# @Last Modified by:   miana1
# @Last Modified time: 2020-01-31 11:35:50
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
from .euclidean_methods import *
from .kernel_methods import *
from .riemannian_logitboost import *
from .other_riemannian import *


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

    elif parsing_string == 'spd kernel rbf svc':
        return spd_rbf_kernel_svc(method_name, method_args)

    elif parsing_string == 'logitboost':
        return logitboost_method(method_name, method_args)

    elif parsing_string == 'riemannian logitboost':
        return wrapper_riemannian_logitboost(method_name, method_args)

    elif parsing_string == 'riemannian knn':
        return wrapper_KNN(method_name, method_args)

    elif parsing_string == 'riemannian mdm':
        return wrapper_MDM(method_name, method_args)

    elif parsing_string == 'ts logistic regression':
        return wrapper_TSclassifier(method_name, method_args)

    else:
        logging.error("The method %s is not recognized, ending here", method_name)
        raise MethodNotRecognized
