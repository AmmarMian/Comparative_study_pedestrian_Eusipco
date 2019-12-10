# -*- coding: utf-8 -*-
# @Author: miana1
# @Description: A file that read a YAML simulation setup file and produce a
#               bash file that will run python scripts according to the
#               simulation and finally run it.
# @Date:   2019-10-28 15:17:49
# @E-mail: ammar.mian@aalto.fi
# @Last Modified by:   miana1
# @Last Modified time: 2019-12-10 17:43:24
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
import yaml
import argparse
from shutil import copyfile
from datetime import datetime
import stat
import subprocess

if __name__ == '__main__':

    # Managing inputs of this script
    parser = argparse.ArgumentParser(description='Execute a simulation according to a simulation setup file on a local computer.')
    parser.add_argument("simulation_setup", help="Path to the simulation setup file.")
    args = parser.parse_args()

    # We always need to know where this script is with regards to base of
    # project, so we define these variables to make everything run smoothly
    path_to_base = "../"
    folder_of_present_script = os.path.dirname(os.path.realpath(__file__))
    absolute_base_path = os.path.join(folder_of_present_script, path_to_base)
    path_to_simulation_folder = os.path.join(folder_of_present_script,
            "Simulation_data/Simulations_setups_data/", os.path.splitext(os.path.basename(args.simulation_setup))[0])

    if os.path.isfile(args.simulation_setup):
        # Creating simulation folder if it doesn't exist
        os.makedirs(path_to_simulation_folder, exist_ok=True)

        # Init paths, and import needed packages
        sys.path.insert(0, absolute_base_path)
        from global_utils import *

        # Read logging configuration
        configure_logging(os.path.join(absolute_base_path,"logging.yaml"))

        # First copy the yaml file into the simulation data folder in case it is lost...
        logging.info(f'Copying simulation setup file into simulaiton directory {path_to_simulation_folder}')
        copyfile(args.simulation_setup, os.path.join(path_to_simulation_folder, 'simulation_setup.yaml'))

        # Parsing YAML file
        with open(args.simulation_setup) as f:
            simulation_setup = yaml.safe_load(f)

        # Creating the bash file to run
        execution_script_path = os.path.join(path_to_simulation_folder, 'running_script.sh')
        logging.info(f'Writing execution script into {execution_script_path}')
        with open(execution_script_path, 'w') as f:

            # ---------------------------------------------------------------------------------------------------------------------------
            # Header
            # ---------------------------------------------------------------------------------------------------------------------------
            dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            header = f"#!/bin/bash\n# Script file generated on {dt_string}.\n" \
                     f"# Purpose is to run simulation setup:\n# {args.simulation_setup}\n# on a local machine.\n" \
                     f"# -------------------------------------------------------------------------------------\n\n"
            f.write(header)

            # ---------------------------------------------------------------------------------------------------------------------------
            # Reading of data
            # ---------------------------------------------------------------------------------------------------------------------------
            temp = os.path.join(folder_of_present_script, "Scripts/read_data_and_compute_eight_dimensional_features.py").replace(" ", "\\ ")
            read_data_script = f"python3 {temp} " \
                               f"{simulation_setup['dataset']['name']}"
            if simulation_setup['global_setups']['Triton_jobs']['parallel']:
                read_data_script += f" -p -j {simulation_setup['global_setups']['Triton_jobs']['number_of_jobs']}"
            log_file_temp = os.path.join(path_to_simulation_folder, 'log_reading_data.txt').replace(" ", "\\ ")
            read_data_script += f" &> {log_file_temp}\n"
            f.write(read_data_script)

            # ---------------------------------------------------------------------------------------------------------------------------
            # Sub-regions sampling
            # ---------------------------------------------------------------------------------------------------------------------------
            temp = os.path.join(folder_of_present_script, "Scripts/compute_sub_regions_sampling.py").replace(" ", "\\ ")
            sub_regions_sampling_script = f"python3 {temp} {simulation_setup['dataset']['name']} {simulation_setup['sub-regions']['seed']}" \
                               f" {simulation_setup['sub-regions']['number_positive_windows']} {simulation_setup['sub-regions']['number_negative_windows']}" \
                               f" {simulation_setup['sub-regions']['n_w']} {simulation_setup['sub-regions']['n_h']}" \
                               f" -m {simulation_setup['sub-regions']['method']} -o {simulation_setup['sub-regions']['overlap_percent']}" \
                               f" -t {simulation_setup['sub-regions']['timeout_method']}"
            if simulation_setup['sub-regions']['progressbar']:
                sub_regions_sampling_script += " -p"

            log_file_temp = os.path.join(path_to_simulation_folder, f'log_sub_regions_sampling.txt').replace(" ", "\\ ")
            sub_regions_sampling_script += f" &> {log_file_temp}\n"
            f.write(sub_regions_sampling_script)

            # ---------------------------------------------------------------------------------------------------------------------------
            # Machine learning feature computation
            # ---------------------------------------------------------------------------------------------------------------------------
            temp = os.path.join(folder_of_present_script, "Scripts/compute_machine_learning_features.py").replace(" ", "\\ ")

            sub_region_filename = os.path.join(folder_of_present_script,
            "Simulation_data/Sub_regions/", f"{simulation_setup['dataset']['name']}_method_{simulation_setup['sub-regions']['method']}_" \
            f"pos_{simulation_setup['sub-regions']['number_positive_windows']}_neg_{simulation_setup['sub-regions']['number_negative_windows']}_" \
            f"nh_{simulation_setup['sub-regions']['n_h']}_nw_{simulation_setup['sub-regions']['n_w']}_seed_{simulation_setup['sub-regions']['seed']}").replace(" ", "\\ ")

            path_to_simulation_folder_tmp = path_to_simulation_folder.replace(" ", "\\ ")
            machine_learning_features_script = f"python3 {temp} {simulation_setup['dataset']['name']} {sub_region_filename}" \
                               f" {path_to_simulation_folder_tmp} \"{simulation_setup['machine_learning_features']['method']}\" " \
                               f"{simulation_setup['machine_learning_features']['method_args']}"

            if simulation_setup['machine_learning_features']['shuffle']:
                machine_learning_features_script += f" -s --shuffle_seed {simulation_setup['machine_learning_features']['seed']}"
            if simulation_setup['global_setups']['Triton_jobs']['parallel']:
                machine_learning_features_script += f" -p -j {simulation_setup['global_setups']['Triton_jobs']['number_of_jobs']}"

            log_file_temp = os.path.join(path_to_simulation_folder, f'log_machine_learning_features.txt').replace(" ", "\\ ")
            machine_learning_features_script += f" &> {log_file_temp}\n"
            f.write(machine_learning_features_script)

            # ---------------------------------------------------------------------------------------------------------------------------
            # Reducing positive samples and formatting data
            # ---------------------------------------------------------------------------------------------------------------------------
            temp = os.path.join(folder_of_present_script, "Scripts/reduce_positive_samples.py").replace(" ", "\\ ")

            path_to_data_storage_file = "\"" + os.path.join(path_to_simulation_folder,
                f"machine_learning_features_method_{simulation_setup['machine_learning_features']['method']}") + "\""

            reducing_samples_script = f"python3 {temp} " \
                               f" {path_to_data_storage_file} \"{simulation_setup['reducing_samples']['method']}\" " \
                               f"\"{simulation_setup['reducing_samples']['method_args']}\" {simulation_setup['reducing_samples']['number_to_keep']}"

            if simulation_setup['reducing_samples']['shuffle']:
                reducing_samples_script += f" -s --shuffle_seed {simulation_setup['reducing_samples']['seed']}"
            if simulation_setup['global_setups']['Triton_jobs']['parallel']:
                reducing_samples_script += f" -p -j {simulation_setup['global_setups']['Triton_jobs']['number_of_jobs']}"

            log_file_temp = os.path.join(path_to_simulation_folder, f'log_reducing_samples.txt').replace(" ", "\\ ")
            reducing_samples_script += f" &> {log_file_temp}\n"
            f.write(reducing_samples_script)

        # Running the bash file
        logging.info(f'Executing script {execution_script_path}')
        st = os.stat(execution_script_path)
        os.chmod(execution_script_path, st.st_mode | stat.S_IEXEC) # Making the file an executable
        subprocess.call(execution_script_path)

    else:
        logging.error(f'The simulation setup file {args.simulation_setup} was not found. Exiting here...')