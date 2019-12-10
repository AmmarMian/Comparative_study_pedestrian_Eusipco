# -*- coding: utf-8 -*-
# @Author: Mian Ammar
# @Description: Some general use functions for managing scripts execution.
# @Date:   2019-10-10 18:01:54
# @E-mail: ammar.mian@aalto.fi
# @Last Modified by:   miana1
# @Last Modified time: 2019-12-10 17:16:49
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

import logging
import logging.config
import os
import yaml
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc

class MethodNotRecognized(Exception):
    pass


def configure_logging(path_to_config_file):
    """ A short function to put at the begginning of a python script
        to configure logging package."""

    try:
        # Verifying that the config file is present
        if not os.path.isfile(path_to_config_file):  # Checking whether file exist
            raise FileNotFoundError

        # Reading the config file
        with open(path_to_config_file) as f:
            conf = yaml.safe_load(f)

        # Setting the configuration for the logs
        if "logging" in conf:
            logging.config.dictConfig(conf["logging"])
            logging.info("Logging configuration loaded from: %s", path_to_config_file)


    except FileNotFoundError:
        logging.critical("Unable to find the logging config file: %s",
                            path_to_config_file)
        logging.critical("Logging config will be set to default")


def read_paths(path_to_yaml):
    """ A short function to put read the paths to the datasets which
        are stored in a YAML file.

        Usage: paths = read_paths(path_to_yaml)

        Inputs:
            * path_to_yaml = a string giving the path to the path YAML file.
        Outputs:
            * paths = a dictionary containing all the paths to datasets."""

    try:
        # Verifying that the config file is present
        if not os.path.isfile(path_to_yaml):  # Checking whether file exist
            raise FileNotFoundError

        # Reading the config file
        with open(path_to_yaml) as f:
            paths = yaml.safe_load(f)
            logging.info("Paths loaded from : %s", path_to_yaml)
            return paths

    except FileNotFoundError:
        logging.warning("Unable to find the path file: %s",
                            path_to_yaml)
        return None


def create_matplotlib_figure(figsize=(5,3.5), dpi=120, grid=True, xlabel='',
                             ylabel='', title='', usetex=False):
    """ A function to instanciate a formatted matplotlib figure which
        use latex font and has a nice color scheme.

        Usage: fig, ax = create_matplotlib_figure(figsize, dpi, grid, xlabel,
                                                    ylabel, title)
        Inputs:
            * figsize = tuple of numbers corresponding size of the figure,
                        default is (5,3.5).
            * dpi = int corresponding to the dpi of the figure, default is 120.
            * grid = boolean to show or not the grid inthe plot.
            * xlabel = an str to put into xlabel. If not precised, it isn't used.
            * ylabel = an str to put into ylabel. If not precised, it isn't used.
            * title = an str to put into title. If not precised, it isn't used.
            * usetex = a boolean to activate or not latex rendering.
        Outputs:
            * fig = a matplotlib figure instance.
            * ax = a matplotlib axis instance."""

    if usetex:
        rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
        rc('text', usetex=True)

    # Creating figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.tight_layout()

    # Creating axis of one plot
    ax = fig.add_subplot(1, 1, 1) # nrows, ncols, index
    ax.set_facecolor((0.917647058823529,0.917647058823529,0.949019607843137))

    # Borderless axis
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Setting labels and title
    if not (xlabel==''):
        plt.xlabel(xlabel)
    if not (ylabel==''):
        plt.ylabel(ylabel)
    if not (title==''):
        plt.title(title)

    # Grid
    plt.grid(grid)

    return fig, ax

