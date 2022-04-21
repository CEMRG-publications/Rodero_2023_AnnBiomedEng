import numpy as np
import os
import time

from global_variables_config import *

import gpytGPE

def load_training(folders, load_x, load_y, biomarker):
    """Function to load the training data set (suffix _training).

    @param folders: array of strings with the name of the folders to load the biomarkers from.
    @param load_x: If true, loads the input points.
    @param load_y: If true, load the biomarker values.
    @param biomarker: Name of the biomarker to load if load_y is True.

    @return Either x_train or y_train, or both as numpy arrays.
    """

    if load_x:
        for folder_i in range(len(folders)):
            current_x = np.loadtxt(os.path.join(PROJECT_PATH, folders[folder_i], "input_space_training.dat"), dtype=float)
            if folder_i > 0:
                x_train = np.vstack((current_x, x_train))
            else:
                x_train = current_x
    if load_y:
        for folder_i in range(len(folders)):
            current_y = np.loadtxt(os.path.join(PROJECT_PATH, folders[folder_i], biomarker + "_training.dat"), dtype=float)
            if folder_i > 0:
                y_train = np.append(current_y, y_train)
            else:
                y_train = current_y

    if load_x:
        return x_train
    if load_y:
        return y_train
    if load_x and load_y:
        return x_train, y_train

def load_validation(folders, load_x, load_y, biomarker):
    """Function to load the validation data set (suffix _validation).

    @param folders: array of strings with the name of the folders to load the biomarkers from.
    @param load_x: If true, loads the input points.
    @param load_y: If true, load the biomarker values.
    @param biomarker: Name of the biomarker to load if load_y is True.

    @return Either x_train or y_train, or both as numpy arrays.
    """
    if load_x:
        for folder_i in range(len(folders)):
            current_x = np.loadtxt(os.path.join(PROJECT_PATH, folders[folder_i], "input_space_validation.dat"),
                                   dtype=float)
            if folder_i > 0:
                x_train = np.vstack((current_x, x_train))
            else:
                x_train = current_x
    if load_y:
        for folder_i in range(len(folders)):
            current_y = np.loadtxt(
                os.path.join(PROJECT_PATH, folders[folder_i], biomarker + "_validation.dat"), dtype=float)
            if folder_i > 0:
                y_train = np.append(current_y, y_train)
            else:
                y_train = current_y

    if load_x:
        return x_train
    if load_y:
        return y_train
    if load_x and load_y:
        return x_train, y_train

def load_test(folders, load_x, load_y, biomarker):
    """Function to load the test data set (suffix _test).

    @param folders: array of strings with the name of the folders to load the biomarkers from.
    @param load_x: If true, loads the input points.
    @param load_y: If true, load the biomarker values.
    @param biomarker: Name of the biomarker to load if load_y is True.

    @return Either x_train or y_train, or both as numpy arrays.
    """
    if load_x:
        for folder_i in range(len(folders)):
            current_x = np.loadtxt(os.path.join(PROJECT_PATH, folders[folder_i], "input_space_test.dat"),
                                   dtype=float)
            if folder_i > 0:
                x_train = np.vstack((current_x, x_train))
            else:
                x_train = current_x
    if load_y:
        for folder_i in range(len(folders)):
            current_y = np.loadtxt(
                os.path.join(PROJECT_PATH, subfolder, folders[folder_i], biomarker + "_test.dat"), dtype=float)
            if folder_i > 0:
                y_train = np.append(current_y, y_train)
            else:
                y_train = current_y

    if load_x:
        return x_train
    if load_y:
        return y_train
    if load_x and load_y:
        return x_train, y_train

def train(folders, verbose=True):
    """Function to train emulators using the data present in the folders specified.

    @param folders Array of strings with the name of the folders where the training set is loaded from.

    @return An array of emulators, one per biomarker.
    """

    x_train = load_training(folders=folders, load_x=True, load_y=False, biomarker=None)

    if folders == ['initial_sweep']:
        x_val = load_validation(folders=folders, load_x=True, load_y=False, biomarker=None)
    else:
        x_val = None

    biomarkers = np.loadtxt(os.path.join(PROJECT_PATH, "biomarkers_labels.txt"), dtype=str)

    emulators = []

    for biomarkers_i in range(len(biomarkers)):
        y_train = load_training(folders=folders, load_x=False, load_y=True, biomarker=biomarkers[biomarkers_i])

        if folders == ['initial_sweep']:
            y_val = load_validation(folders=folders, load_x=False, load_y=True, biomarker=biomarkers[biomarkers_i])
        else:
            y_val = None

        if os.path.isfile(os.path.join(PROJECT_PATH, folders[-1], biomarkers[biomarkers_i] + '_' + '_'.join(folders).replace("/","_") + '.gpe')):
            emul = gpytGPE.gpe.GPEmul.load(X_train=x_train, y_train=y_train,
                                           loadpath=os.path.join(PROJECT_PATH, folders[-1] + "/"),
                                           filename=biomarkers[biomarkers_i] + '_' + '_'.join(folders).replace("/","_") + '.gpe',
                                           verbose=verbose)
        else:
            print("Looking for " + os.path.join(PROJECT_PATH, folders[-1],
                                                biomarkers[biomarkers_i] + '_' + '_'.join(folders).replace("/",
                                                                                                           "_") + '.gpe'))
            time.sleep(4)
            emul = gpytGPE.gpe.GPEmul(X_train=x_train, y_train=y_train)
            emul.train(X_val=x_val, y_val=y_val, max_epochs=100, n_restarts=5,
                       savepath=os.path.join(PROJECT_PATH, folders[-1] + "/"))
            emul.save(biomarkers[biomarkers_i] + '_' + '_'.join(folders).replace("/","_") + '.gpe')

        emulators.append(emul)

    if len((np.array(emulators)).shape) > 1:
        emulators = [item for sublist in emulators for item in sublist]

    return emulators