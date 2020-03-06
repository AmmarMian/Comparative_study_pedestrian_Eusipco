# -*- coding: utf-8 -*-
# @Author: miana1
# @Description: File to do a robustness to outliers study
# @Date:   2020-02-11 13:07:13
# @E-mail: ammar.mian@aalto.fi
# @Last Modified by:   miana1
# @Last Modified time: 2020-02-14 14:24:33
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
import argparse
from sklearn.svm import SVC
import pickle
import numpy as np
import scipy as sp
from tqdm import tqdm, trange
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.neighbors import  KNeighborsClassifier, NearestCentroid
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
from tqdm import tqdm, trange
from scipy.stats import t, wishart
from copy import deepcopy


def gen_outlier(p):
    #return np.cov(t.rvs(0.5, size=(p,3*p)))
    return wishart.rvs(300, np.random.gamma(10)*np.random.rand()*sp.linalg.toeplitz(np.power(abs(np.random.rand()), np.arange(0, 8))) )


def do_one_trial(X_train, y_train, X_test, y_test, ML_methods, alpha_vec, seed):

    accuracy_trial = np.zeros((len(alpha_vec), len(ML_methods)))
    for i, alpha in enumerate(alpha_vec):

        # Generate outlier and reshuffle training data
        X_train_other = np.copy(X_train)
        y_train_other = np.copy(y_train)
        X_train_other,  y_train_other = shuffle(X_train_other, y_train_other, random_state=seed)
        n_samples_outlier = int(alpha*X_train.shape[0])

        for j in range(n_samples_outlier):
            outlier = gen_outlier(X_train.shape[1])
            while not algebra.is_pos_def(outlier):
                outlier = gen_outlier(X_train.shape[1])
            X_train_other[j,:,:] = outlier
        X_train_other,  y_train_other = shuffle(X_train_other, y_train_other, random_state=seed+1)

        for j, method in enumerate(ML_methods):
            method.fit(X_train_other, y_train_other)
            y_predicted = method.predict(X_test)
            accuracy = (y_test == y_predicted).sum() / len(y_test)
            accuracy_trial[i, j] = accuracy

    return accuracy_trial


if __name__ == '__main__':

    # ------------------------------------------------------------------------------------------------------------
    # Script Execution setup
    # ------------------------------------------------------------------------------------------------------------

    # Managing inputs of this script
    parser = argparse.ArgumentParser(description='Compute testing phase for all methods')
    parser.add_argument("data_file", help="Path (From base) to the file containing machine learning features")
    parser.add_argument("--seed", type=int, default=1,
                         help="Seed for rng when shuffling the dataset")
    parser.add_argument("-p", "--parallel", action="store_true",
                         help="Enable parallel computation")
    parser.add_argument("-j", "--n_jobs", default=8, type=int,
                         help="Number of jobs for parallel computation")
    args = parser.parse_args()

    # We always need to know where this script is with regards to base of
    # project, so we define these variables to make everything run smoothly
    path_to_base = "../../"
    folder_of_present_script = os.path.dirname(os.path.realpath(__file__))
    absolute_base_path = os.path.join(folder_of_present_script, path_to_base)
    path_to_machine_learning_features_data_file = os.path.join(absolute_base_path,
                                                                    args.data_file)

    # Init paths, and import needed packages
    sys.path.insert(0, absolute_base_path)
    from global_utils import *
    from psdlearning.utils import algebra
    from psdlearning import parsing_methods
    from psdlearning.euclidean_methods import *
    from psdlearning.kernel_methods import spd_rbf_kernel_svc
    from psdlearning.riemannian_logitboost import wrapper_riemannian_logitboost
    from psdlearning.other_riemannian import wrapper_TSclassifier, wrapper_KNN, wrapper_MDM

    # Read logging configuration
    configure_logging(os.path.join(absolute_base_path, "logging.yaml"))


    # ------------------------------------------------------------------------------------------------------------
    # Data reading
    # ------------------------------------------------------------------------------------------------------------

    # Read data from pickle dump
    logging.info("Reading machine learning features data from file %s", args.data_file)
    with open(args.data_file, 'rb') as f:
        dataset = pickle.load(f)
    X_train, y_train = dataset['Train']
    X_test, y_test = dataset['Test']

    # Shuffling data
    X_train, y_train = shuffle(X_train, y_train, random_state=args.seed)
    X_test, y_test = shuffle(X_test, y_test, random_state=args.seed)


    # Less samples to run faster
    X_train = X_train[:800,:,:]
    y_train = y_train[:800]
    X_test = X_test[:200,:,:]
    y_test = y_test[:200]


    # ------------------------------------------------------------------------------------------------------------
    # Setting the ML methods
    # ------------------------------------------------------------------------------------------------------------

    # 1) Linear SVM
    method_args = {'kernel':'linear', 'random_state':args.seed, 'C':1.0}
    linear_svm = sklearn_svc_method('Linear SVM', method_args)

    # 2) Euclidean RBF kernel SVM
    method_args = {'kernel':'rbf', 'random_state':args.seed, 'C':1.0, 'gamma':'scale'}
    euclidean_rbf_svm = sklearn_svc_method('Euclidean RBF SVM', method_args)

    # 3) Riemannian RBF kernel SVM
    method_args = {
       'gamma': 'auto', # Width of the kernel function
       'p': 2, # Power to computer the variance for gamma = 'auto'
       'C': 1.0, # Penalty parameter C of the error term.
       'random_state': args.seed
        }
    riemannian_rbf_svm = spd_rbf_kernel_svc('Riemannian RBF SVM', method_args)


    # 4) Euclidean logitboost
    method_args = {
       'base_estimator': 'decision stump', # The base estimator from which the LogitBoost classifier is built.
       'n_estimators': 50, # The number of estimators per class in the ensemble.
       'weight_trim_quantile': 0.05, # Threshold for weight trimming
       'max_response': 4.0, # Maximum response value to allow when fitting the base estimators.
       'learning_rate': 1.0, # The learning rate shrinks the contribution of each classifier by `learning_rate` during fitting.
       'bootstrap': False, # If True, each boosting iteration trains the base estimator using a weighted bootstrap sample of the training data.
       'random_state': args.seed# Seed used by the random number generator.
    }
    euclidean_logitboost = logitboost_method('Euclidean logitboost', method_args)


    # 5) Riemannian logitboost
    method_args={
       'base_estimator': 'decision stump', # The base estimator from which the LogitBoost classifier is built.
       'n_estimators': 50, # The number of estimators per class in the ensemble.
       'weight_trim_quantile': 0.05, # Threshold for weight trimming
       'max_response': 4.0, # Maximum response value to allow when fitting the base estimators.
       'learning_rate': 1.0, # The learning rate shrinks the contribution of each classifier by `learning_rate` during fitting.
       'bootstrap': False, # If True, each boosting iteration trains the base estimator using a weighted bootstrap sample of the training data.
       'random_state': args.seed # Seed used by the random number generator.
    }
    riemannian_logitboost = wrapper_riemannian_logitboost('Riemannian logitboost', method_args)

    # 6) Riemannian KNN
    method_args = {
       'n_neighbors': 5,
       'metric': 'riemann'
    }
    riemannian_knn = wrapper_KNN('Riemannian KNN', method_args)

    # 7) TS logistic regression
    method_args = {
       'tsupdate': True,
       'metric': 'riemann'
    }
    ts_logistic_regression = wrapper_TSclassifier('Tangent space logistic regression', method_args)


    # 8) Riemannian MDM
    method_args = {
       'metric': 'riemann'
    }
    riemannian_mdm = wrapper_MDM('Riemannian MDM', method_args)

    # 9) Euclidean knn
    method_args = {
       'n_neighbors': 5
    }
    euclidean_knn = sklearn_knn_method('Euclidean KNN', method_args)

    # 10) Euclidean mdm
    euclidean_mdm = sklearn_mdm_method('Euclidean MDM', None)

    # 11) Euclidan LogisticRegression
    method_args = {
       'C': 1.0,
       'max_iter': 100
    }
    euclidean_logistic_regression = sklearn_LogisticRegression_method('Euclidean LogisticRegression', method_args)


    ML_methods = [euclidean_rbf_svm, riemannian_rbf_svm,
            euclidean_logitboost, riemannian_logitboost, riemannian_knn,
            ts_logistic_regression, riemannian_mdm, euclidean_knn,
            euclidean_mdm, euclidean_logistic_regression]


    # ------------------------------------------------------------------------------------------------------------
    # Doing simulation by generating outliers
    # ------------------------------------------------------------------------------------------------------------
    logging.info('Computing simulation')


    alpha_vec = np.linspace(0.01, 0.1, 5)
    number_trials = 144

    if args.parallel:
        logging.info('Parallel processing chosen')
        accuracy_list = Parallel(n_jobs=args.n_jobs)(delayed(do_one_trial)(X_train, y_train, X_test, y_test,
                                ML_methods, alpha_vec, args.seed+trial) for trial in trange(number_trials))
        accuracy_list = np.array(accuracy_list)
    else:
        accuracy_list = np.zeros((number_trials, len(alpha_vec), len(ML_methods)))
        for trial in trange(number_trials):
            accuracy_trial = do_one_trial(X_train, y_train, X_test, y_test,
                                            ML_methods, alpha_vec, args.seed+trial)
            accuracy_list[trial,:,:] = accuracy_trial

    logging.info('Done')

    print(accuracy_list)
    logging.info('Results:\n'+str(accuracy_list))
    with open('results_robustness_study', 'wb') as f:
        pickle.dump(accuracy_list, f)

    # Saving results for plotting in a dictionary for each method
    results = {'alpha':alpha_vec}
    for i, method in enumerate(ML_methods):
        res_mean = np.mean(accuracy_list[:,:,i], axis=0)
        res_max = np.max(accuracy_list[:,:,i], axis=0)
        res_min = np.min(accuracy_list[:,:,i], axis=0)
        results[method.method_name] = {'mean': res_mean, 'min': res_min, 'max':res_max,
                                        'values':accuracy_list[:,:,i]}
    with open('results_robustness_study', 'wb') as f:
        pickle.dump(results, f)

    fig, ax = create_matplotlib_figure()
    markers = ['x', 'o', 's', 'd', '+', '>', '^', '8', 'p', 'h', '<', 'v']
    for method, marker in zip(results, markers):
        if method != 'alpha':
            plt.plot(results['alpha'], results[method]['mean'], marker=marker, label=method)
            print(f'{method}: ' +str(results[method]['mean']))
    plt.legend()
    plt.savefig('results_robustness_study.png')
    plt.show()
