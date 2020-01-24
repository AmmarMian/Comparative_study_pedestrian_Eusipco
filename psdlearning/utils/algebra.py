# -*- coding: utf-8 -*-
# @Author: miana1
# @Description: A file to gather functions relative to algebra manipulation.
# @Date:   2019-10-24 16:01:13
# @E-mail: ammar.mian@aalto.fi
# @Last Modified by:   miana1
# @Last Modified time: 2020-01-24 14:39:33
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
import scipy as sp
from joblib import Parallel, delayed
import logging
from tqdm import tqdm


def is_pos_def(x):
    eigvals, eigvects = sp.linalg.eigh(x, check_finite=False)
    return np.all(eigvals > 0)


def matprint(mat, fmt="g"):
    """ A function to pretty print a matrix.

        Usage: matprint(mat, fmt)
        Inputs:
            * mat = a 2-D numpy array to print.
            * fmt = an str corresponding to the formatiing of the numbers.
        Outputs: None.
    """

    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")


def toeplitz_matrix(rho, p):
    """ A function that computes a Hermitian semi-positive matrix.

    	Usage: matrix = toeplitz_matrix(rho, p)
        Inputs:
            * rho = a float corresponding to the toeplitz coefficient.
            * p = size of matrix.
        Outputs:
            * matrix = a 2D numpy array of shape (p,p) coresponding
            		   to the matrix.
    """

    return sp.linalg.toeplitz(np.power(rho, np.arange(0, p)))


def vec(mat):
    return mat.ravel('F')


def vech(mat):
    # Gets Fortran-order
    return mat.T.take(_triu_indices(len(mat)))


def _tril_indices(n):
    rows, cols = np.tril_indices(n)
    return rows * n + cols


def _triu_indices(n):
    rows, cols = np.triu_indices(n)
    return rows * n + cols


def _diag_indices(n):
    rows, cols = np.diag_indices(n)
    return rows * n + cols


def unvec(v):
    k = int(np.sqrt(len(v)))
    assert(k * k == len(v))
    return v.reshape((k, k), order='F')


def unvech(v):
    # quadratic formula, correct fp error
    rows = .5 * (-1 + np.sqrt(1 + 8 * len(v)))
    rows = int(np.round(rows))

    result = np.zeros((rows, rows), dtype=v.dtype)
    result[np.triu_indices(rows)] = v
    result = result + result.T.conj()

    # divide diagonal elements by 2
    result[np.diag_indices(rows)] /= 2

    return result