# -*- coding: utf-8 -*-
# @Author: miana1
# @Description: Gridsearch cv for Euclidean RBF kernel to generate a plot
# @Date:   2020-01-31 13:10:02
# @E-mail: ammar.mian@aalto.fi
# @Last Modified by:   miana1
# @Last Modified time: 2020-02-03 14:32:01
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

import os
import sys
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import argparse
import pickle
import numpy as np
from tqdm import tqdm, trange
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.utils import shuffle

def miss_rate_score(y_true, y_pred):
    """
        Function to compute the miss rate score for pedestrian detection classification.
    """

    false_neg = np.logical_and(y_true==1,  y_pred==-1).sum()
    true_pos = np.logical_and(y_true==1,  y_pred==1).sum()
    return false_neg / (false_neg + true_pos)


def ffpw_score(y_true, y_pred):
    """
        Function to compute the false positives per window score for pedestrian detection classification.
    """

    false_pos = np.logical_and(y_true==-1,  y_pred==1).sum()
    true_neg = np.logical_and(y_true==-1,  y_pred==-1).sum()
    return false_pos / (true_neg + false_pos)

if __name__ == '__main__':

    # Managing inputs of this script
    parser = argparse.ArgumentParser(description='Compute training phase for all methods')
    parser.add_argument("data_file", help="Path (From base) to the file containing machine learning features")
    parser.add_argument("-s", "--seed", type=int, default=None,
                         help="Seed for rng to have reproducible results")
    parser.add_argument("-p", "--parallel", action="store_true",
                         help="Enable parallel computation")
    parser.add_argument("-j", "--n_jobs", default=8, type=int,
                         help="Number of jobs for parallel computation")
    args = parser.parse_args()


    # We always need to know where this script is with regards to base of
    # project, so we define these variables to make everything run smoothly
    path_to_base = "../../../"
    folder_of_present_script = os.path.dirname(os.path.realpath(__file__))
    absolute_base_path = os.path.join(folder_of_present_script, path_to_base)


    # Init paths, and import needed packages
    sys.path.insert(0, absolute_base_path)
    from global_utils import *
    from psdlearning.utils import algebra

    # Read data from pickle dump
    logger.info("Reading machine learning features data from file %s", args.data_file)
    with open(args.data_file, 'rb') as f:
        dataset = pickle.load(f)
    X_train, y_train = dataset['Train']
    X_test, y_test = dataset['Test']

    # Shuffling data
    X_train, y_train = shuffle(X_train, y_train, random_state=args.seed)
    X_test, y_test = shuffle(X_test, y_test, random_state=args.seed)

    # Setting parameters of the gridsearch instance for classifier
    # param_grid={'kernel': ['rbf'],
    #     'C':[1, 10, 100, 1000],
    #     'gamma':np.logspace(-4,2,10),
    #     }
    param_grid={'kernel': ['rbf'],
        'C': [1],
        'gamma':np.linspace(0.1,20,20),
        }

    # Setting scoring metrics
    scoring = {'Miss rate': make_scorer(miss_rate_score, greater_is_better=False),
               'False positives per window': make_scorer(ffpw_score, greater_is_better=False)}

    # Setting grid-search
    if dataset['name'] == 'INRIA':
        cv = 4
    else:
        cv = 3
    if args.parallel:
        gs = GridSearchCV(SVC(random_state=args.seed),
                      cv=cv,
                      param_grid=param_grid,
                      scoring=scoring, return_train_score=True,
                      refit=False,
                      verbose=2,
                      n_jobs=args.n_jobs)
    else:
        gs = GridSearchCV(SVC(random_state=args.seed),
                      cv=cv,
                      param_grid=param_grid,
                      refit=False,
                      verbose=2,
                      scoring=scoring, return_train_score=True)


    # Doing grid search
    n_samples, n_features = X_train.shape[:2]
    gs.fit(X_train.reshape((n_samples, n_features**2)), y_train)

    print('Miss rates:')
    means = gs.cv_results_['mean_test_Miss rate']
    stds = gs.cv_results_['std_test_Miss rate']
    for mean, std, params in zip(means, stds, gs.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()


    print('False positives per window:')
    means = gs.cv_results_['mean_test_False positives per window']
    stds = gs.cv_results_['std_test_False positives per window']
    for mean, std, params in zip(means, stds, gs.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()


    fig, ax = create_matplotlib_figure()
    plt.scatter(-gs.cv_results_['mean_test_False positives per window'], -gs.cv_results_['mean_test_Miss rate'], marker='x')
    ax.set_xscale('log')
    plt.ylim([np.min(-gs.cv_results_['mean_test_Miss rate']), np.max(-gs.cv_results_['mean_test_Miss rate'])])
    plt.savefig('results_euclidean_rbf_svc_%s.png' % dataset['name'])