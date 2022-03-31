from joblib import Parallel, delayed
import numpy as np
import os
import pathlib

from global_variables_config import *

import run_EP

def create_init(anatomy_values, param_values, i):
    """Function to create the init file of the ekbatch simulation.

        @param anatomy_values: Array where each line corresponds to one mesh. Each component is a single
        str with the values of the modes (1 to 18)
        @param param_values: Array where each line corresponds to one mesh. Each component is a single
        str with the values of EP parameters.
        @param i: Index of the mesh to create from anatomy values.

        @returns The .init file in the EP_simulations folder.
    """
    if not os.path.isfile(os.path.join(PROJECT_PATH,"meshes","heart_" + anatomy_values[i+1].replace(",","")[:-36],"biv",
                                "EP_simulations", '{0:.2f}'.format(round(float(param_values[i].split(' ')[0]),2)) +\
                                    '{0:.2f}'.format(round(float(param_values[i].split(' ')[1]),2)) +\
                                    '{0:.2f}'.format(round(float(param_values[i].split(' ')[2]),2)) +\
                                    '{0:.2f}'.format(round(float(param_values[i].split(' ')[3]),2)) +\
                                    '{0:.2f}'.format(round(float(param_values[i].split(' ')[4]),2)) +\
                                    ".init")):
        pathlib.Path(os.path.join(PROJECT_PATH,"meshes","heart_" + anatomy_values[i+1].replace(",","")[:-36],"biv","EP_simulations")).mkdir(parents=True, exist_ok=True)


        run_EP.carp2init(fourch_name = "heart_" + anatomy_values[i + 1].replace(",", "")[:-36],
                         lastfectag=100*round(float(param_values[i].split(' ')[1]), 2),
                         CV_l=round(float(param_values[i].split(' ')[2]),2),
                         k_fibre=round(float(param_values[i].split(' ')[3]),2),
                         k_fec=round(float(param_values[i].split(' ')[4]),2),
                         simulation_file_name=os.path.join(PROJECT_PATH,"meshes","heart_" + anatomy_values[i+1].replace(",","")[:-36],"biv",
                                "EP_simulations",
                                    '{0:.2f}'.format(round(float(param_values[i].split(' ')[0]),2)) +\
                                    '{0:.2f}'.format(round(float(param_values[i].split(' ')[1]),2)) +\
                                    '{0:.2f}'.format(round(float(param_values[i].split(' ')[2]),2)) +\
                                    '{0:.2f}'.format(round(float(param_values[i].split(' ')[3]),2)) +\
                                    '{0:.2f}'.format(round(float(param_values[i].split(' ')[4]),2)) +\
                                    ".dat"
                                    ),
                         path_ep=os.path.join(PROJECT_PATH,"meshes","heart_" + anatomy_values[i+1].replace(",","")[:-36],"biv",
                                "EP_simulations"),
                         subfolder="meshes"
                         )

def launch_simulation(anatomy_values, param_values, i):
    """Function to run the ekbatch simulation.

        @param anatomy_values: Array where each line corresponds to one mesh. Each component is a single
        str with the values of the modes (1 to 18)
        @param param_values: Array where each line corresponds to one mesh. Each component is a single
        str with the values of EP parameters.
        @param i: Index of the mesh to create from anatomy values.

        @returns The .dat file in the EP_simulations folder.
    """
    if not os.path.isfile(
            os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv",
                         "EP_simulations", '{0:.2f}'.format(round(float(param_values[i].split(' ')[0]), 2)) + \
                                           '{0:.2f}'.format(round(float(param_values[i].split(' ')[1]), 2)) + \
                                           '{0:.2f}'.format(round(float(param_values[i].split(' ')[2]), 2)) + \
                                           '{0:.2f}'.format(round(float(param_values[i].split(' ')[3]), 2)) + \
                                           '{0:.2f}'.format(round(float(param_values[i].split(' ')[4]), 2)) + \
                                           ".dat")):
        print("Couldnt find " + os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv",
                         "EP_simulations", '{0:.2f}'.format(round(float(param_values[i].split(' ')[0]), 2)) + \
                                           '{0:.2f}'.format(round(float(param_values[i].split(' ')[1]), 2)) + \
                                           '{0:.2f}'.format(round(float(param_values[i].split(' ')[2]), 2)) + \
                                           '{0:.2f}'.format(round(float(param_values[i].split(' ')[3]), 2)) + \
                                           '{0:.2f}'.format(round(float(param_values[i].split(' ')[4]), 2)) + \
                                           ".dat\n"))
        print("Running " + "heart_" + anatomy_values[i+1].replace(",","")[:-36] + "...\n")
        run_EP.launch_init(fourch_name="heart_" + anatomy_values[i+1].replace(",","")[:-36],
                       alpha_endo=round(float(param_values[i].split(' ')[0]),2),
                       alpha_epi=-round(float(param_values[i].split(' ')[0]),2),
                       simulation_file_name=os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv",
                         "EP_simulations", '{0:.2f}'.format(round(float(param_values[i].split(' ')[0]), 2)) + \
                                           '{0:.2f}'.format(round(float(param_values[i].split(' ')[1]), 2)) + \
                                           '{0:.2f}'.format(round(float(param_values[i].split(' ')[2]), 2)) + \
                                           '{0:.2f}'.format(round(float(param_values[i].split(' ')[3]), 2)) + \
                                           '{0:.2f}'.format(round(float(param_values[i].split(' ')[4]), 2)) + \
                                           ".dat"),
                       path_ep= os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv",
                         "EP_simulations"),
                       subfolder="meshes"
                       )

def run(subfolder="initial_sweep", anatomy_csv_file="input_anatomy_training.csv", ep_dat_file="input_ep_training.dat"):
    """Function to create the simulation files and run them.

        @param subfolder: Folder within PROJECT_PATH where the input data will be read.
        @param anatomy_csv_file: Name of the csv file containing _all_ the meshes modes values.
        @param ep_dat_file: Name of the dat file containing the values of the EP parameters.

        @return The .dat file in the EP_simulations file.
    """

    with open(os.path.join(PROJECT_PATH, subfolder, anatomy_csv_file)) as f:
        anatomy_values = f.read().splitlines()

    with open(os.path.join(PROJECT_PATH, subfolder, ep_dat_file)) as f:
        param_values = f.read().splitlines()

    Parallel(n_jobs=20)(delayed(create_init)(anatomy_values=anatomy_values, param_values=param_values, i=i) for i in range(len(anatomy_values)-1))
    # Parallel(n_jobs=20)(delayed(launch_simulation)(anatomy_values=anatomy_values, param_values=param_values, i=i) for i in range(len(anatomy_values)-1))
    for i in range(len(anatomy_values) - 1):
        launch_simulation(anatomy_values=anatomy_values, param_values=param_values, i=i)