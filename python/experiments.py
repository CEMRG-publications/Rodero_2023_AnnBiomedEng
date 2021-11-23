import numpy as np
import os
import sys

import postprocessing
import fitting_hm
import global_variables_config
from Historia.shared.design_utils import read_labels, get_minmax
from Historia.history import hm

def summary_plots(wave_to_plot, experiment_name,
                    output_labels_dir = os.path.join("/data","fitting", "EP_output_labels.txt"),
                    exp_mean_name = "exp_mean.txt", exp_std_name = "exp_std.txt",
                    only_feasible = True, units_dir = ""):
    """Function to plot a series of different plots after a wave to analyse
    the output.

    Args:
        wave_to_plot (int): Number of wave to plot.
        experiment_name (str): Name of the folder for the output.
        output_labels_dir (str, optional): Directory where the output labels 
        are. Defaults to os.path.join("/data","fitting", "EP_output_labels.txt").
        exp_mean_name (str, optional): Name of the file where the experimental
        mean is. Defaults to "exp_mean.txt".
        exp_std_name (str, optional): Name of the file where the experimental
        standard deviation is. Defaults to "exp_std.txt".
        only_feasible (bool, optional): If True, it uses only the feasible
        points (under 5SD). Defaults to True.
        units_dir (str, optional): Directory where the file with the units of
        the output is. Defaults to "".
    """

    postprocessing.plot_var_quotient(first_wave = 1, last_wave = wave_to_plot,
                                    subfolder = experiment_name,
                                    plot_title = "Evolution of variance quotient for " + experiment_name)
    postprocessing.plot_output_evolution_seaborn(first_wave = 0, last_wave = wave_to_plot,
                                            subfolder = experiment_name,
                                            only_feasible = only_feasible,
                                            output_labels_dir = output_labels_dir,
                                            exp_mean_name = exp_mean_name,
                                            exp_std_name = exp_std_name,
                                            units_dir = units_dir)
    postprocessing.plot_percentages_NROY(subfolder = experiment_name, last_wave = wave_to_plot)

def summary_statistics(last_wave,experiment_name):
    """Function to print some statistics about the history matching wave 
    including the minimum variance, the wave where that minimal variance is 
    reached and the NROY size in that wave.

    Args:
        last_wave (str): Last wave to include in the analysis. The first waves
        are always included.
        experiment_name (str): Name of the folder where the emulators are.
    """

    mean_var_quotient_vec = []
    NROY_rel = []
    NROY_abs = []
    for i in range(last_wave + 1):
        W_PV = np.loadtxt(os.path.join("/data","fitting",experiment_name,"wave" + str(i),"variance_quotient.dat"),
                        dtype=float)

        mean_var_quot = np.mean(W_PV)

        mean_var_quotient_vec.append(mean_var_quot)
    
        NROY_perc = np.loadtxt(os.path.join("/data","fitting",experiment_name,"wave" + str(i), "NROY_rel.dat"),dtype=float)
        NROY_rel.append(float(NROY_perc))
        if i == 0:
            NROY_abs.append(float(NROY_perc))
        else:
            NROY_abs.append(1e-2*float(NROY_perc)*NROY_abs[i-1])
    
    min_var_wave = mean_var_quotient_vec.index(min(mean_var_quotient_vec))
    print("Min. variance: " + str(round(mean_var_quotient_vec[min_var_wave],2)))
    print("Wave with min. var.: " + str(min_var_wave))
    print("NROY size: " + str(round(NROY_abs[min_var_wave],2)))

def print_NROY_boundaries_anatomy_EP(num_wave=2, n_simul_wave0=280, cutoff=2.5):
    wave = hm.Wave()
    wave.load(os.path.join(PROJECT_PATH, "anatomy", "wave" + str(num_wave - 1),
                           "wave_" + str(num_wave - 1)))

    active_features = ["LVV", "RVV", "LAV", "RAV", "LVOTdiam", "RVOTdiam", "LVmass",
                       "LVWT", "LVEDD", "SeptumWT", "RVlongdiam", "RVbasaldiam",
                       "TAT", "TATLVendo"]
    emulator = []

    for output_name in active_features:
        _, _, emul = fitting_hm.run_GPE(waveno=num_wave, train=False,
                                        active_feature=[output_name],
                                        n_samples=-1,
                                        training_set_memory=2,
                                        subfolder="anatomy",
                                        only_feasible=False)
        emulator.append(emul)
    # emulator now might be a list of list, we need to flatten it
    em_shape = (np.array(emulator)).shape
    if len(em_shape) > 1:
        emulator_flat = [item for sublist in emulator for item in sublist]
        emulator = emulator_flat

    wave.emulator = emulator

    idx_train = round(0.8 * 0.8 * n_simul_wave0)
    x = np.loadtxt(os.path.join(PROJECT_PATH, "anatomy", "wave0", "X.dat"),
                   dtype=float)
    x_train = x[:idx_train]
    emul_i_train = get_minmax(x_train)

    wave.Itrain = emul_i_train
    wave.cutoff = cutoff
    wave.maxno = 1

    path_match = os.path.join(PROJECT_PATH, "match")

    exp_mean = np.loadtxt(os.path.join(path_match, "exp_mean_anatomy_EP.txt"),
                          dtype=float)
    exp_std = np.loadtxt(os.path.join(path_match, "exp_std_anatomy_EP.txt"),
                         dtype=float)
    exp_var = np.power(exp_std, 2)

    wave.mean = exp_mean
    wave.var = exp_var

    xlabels_ep = read_labels(os.path.join(PROJECT_PATH, "anatomy",
                                          "EP_funct_labels.txt"))
    xlabels_anatomy = read_labels(os.path.join(PROJECT_PATH, "anatomy",
                                               "modes_labels.txt"))
    xlabels = [lab for sublist in [xlabels_anatomy, xlabels_ep] for lab in sublist]

    test_points = np.loadtxt(os.path.join(PROJECT_PATH, "anatomy",
                                          "wave" + str(num_wave), "X_test.dat"),
                             dtype=float)
    wave.find_regions(test_points)

    nimp_rangion = wave.NIMP
    ranges = get_minmax(nimp_rangion)

    for input_number in range(len(xlabels)):
        msg = "Range for " + xlabels[input_number] + ": [" +\
              str(round(ranges[input_number][0], 2)) + ", " +\
              str(round(ranges[input_number][1], 2)) + "]"
        print(msg)

def experiment_coveney(only_plot = True):
    """History matching pipeline similar to the one presented in the paper
    by Sam coveney. Implausibility thresholds, training sets size and number
    of waves are detailed in the script.

    Args:
        only_plot (bool, optional): If True, it only prints the plots. 
        Defaults to True.
    """
    
    experiment_name = "coveney"
    original_training_set_size = 50

    if only_plot:
        summary_plots(wave_to_plot = 9, experiment_name = experiment_name)
    else:

        wave_to_plot = -1

        fitting_hm.run_new_wave(num_wave = 0, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 4, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 4)
        wave_to_plot += 1

        fitting_hm.run_new_wave(num_wave = 1, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 4, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 4)
        wave_to_plot += 1

        fitting_hm.run_new_wave(num_wave = 2, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.8, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 4)
        wave_to_plot += 1

        fitting_hm.run_new_wave(num_wave = 3, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.6, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 4)
        wave_to_plot += 1

        fitting_hm.run_new_wave(num_wave = 4, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.4, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 4)
        wave_to_plot += 1

        fitting_hm.run_new_wave(num_wave = 5, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.2, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 4)
        wave_to_plot += 1

        fitting_hm.run_new_wave(num_wave = 6, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 4)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)
        
        fitting_hm.run_new_wave(num_wave = 7, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 4)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 8, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 4)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 9, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 4)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

def experiment_longobardi(only_plot = True):
    """History matching pipeline similar to the one presented in the paper
    by Stefano Longobardi. Implausibility thresholds, training sets size and number
    of waves are detailed in the script.

    Args:
        only_plot (bool, optional): If True, it only prints the plots. 
        Defaults to True.
    """
    
    experiment_name = "longobardi"
    original_training_set_size = 150

    if only_plot:
        summary_plots(wave_to_plot = 5, experiment_name = experiment_name)
    else:

        wave_to_plot = -1

        fitting_hm.run_new_wave(num_wave = 0, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 5.5, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 40)
        wave_to_plot += 1

        fitting_hm.run_new_wave(num_wave = 1, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 5, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 40)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 2, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 4.5, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 40)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 3, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 4, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 40)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 4, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.5, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 40)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 5, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3., n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 440)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

def experiment_3(only_plot = True):
    """History matching pipeline to test different scenarios. Implausibility 
    thresholds, training sets size and number of waves are detailed in the 
    script.

    Args:
        only_plot (bool, optional): If True, it only prints the plots. 
        Defaults to True.
    """
    
    
    original_training_set_size = 100
    experiment_name = "experiment_3"
    wave_to_plot = -1


    if only_plot:
        summary_plots(wave_to_plot = 6, experiment_name = experiment_name)
    else:

        fitting_hm.run_new_wave(num_wave = 0, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 4, n_samples = original_training_set_size,
                    generate_simul_pts = 10, subfolder = experiment_name, training_set_memory = 40)
        wave_to_plot += 1

        fitting_hm.run_new_wave(num_wave = 1, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.8, n_samples = original_training_set_size,
                    generate_simul_pts = 10, subfolder = experiment_name, training_set_memory = 40)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 2, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.6, n_samples = original_training_set_size,
                    generate_simul_pts = 10, subfolder = experiment_name, training_set_memory = 40)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 3, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.4, n_samples = original_training_set_size,
                    generate_simul_pts = 10, subfolder = experiment_name, training_set_memory = 40)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 4, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.2, n_samples = original_training_set_size,
                    generate_simul_pts = 10, subfolder = experiment_name, training_set_memory = 40)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 5, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3., n_samples = original_training_set_size,
                    generate_simul_pts = 10, subfolder = experiment_name, training_set_memory = 440)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 6, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3., n_samples = original_training_set_size,
                    generate_simul_pts = 10, subfolder = experiment_name, training_set_memory = 440)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

def experiment_4(only_plot = True):
    """History matching pipeline to test different scenarios. Implausibility 
    thresholds, training sets size and number of waves are detailed in the 
    script.

    Args:
        only_plot (bool, optional): If True, it only prints the plots. 
        Defaults to True.
    """
    # Accurate initial emulator, slow descent (memory 2)

    original_training_set_size = 100
    experiment_name = "experiment_4"
    wave_to_plot = -1

    if only_plot:
        summary_plots(wave_to_plot = 6, experiment_name = experiment_name)
    else:

        fitting_hm.run_new_wave(num_wave = 0, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 4, n_samples = original_training_set_size,
                    generate_simul_pts = 20, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1

        fitting_hm.run_new_wave(num_wave = 1, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.8, n_samples = original_training_set_size,
                    generate_simul_pts = 20, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 2, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.6, n_samples = original_training_set_size,
                    generate_simul_pts = 20, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 3, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.4, n_samples = original_training_set_size,
                    generate_simul_pts = 20, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 4, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.2, n_samples = original_training_set_size,
                    generate_simul_pts = 20, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 5, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3., n_samples = original_training_set_size,
                    generate_simul_pts = 20, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 6, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3., n_samples = original_training_set_size,
                    generate_simul_pts = 20, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

def experiment_5(only_plot = True):
    """History matching pipeline to test different scenarios. Implausibility 
    thresholds, training sets size and number of waves are detailed in the 
    script.

    Args:
        only_plot (bool, optional): If True, it only prints the plots. 
        Defaults to True.
    """
    # Balanced emulator, initial and pre-final convergence (memory 2)

    original_training_set_size = 50
    experiment_name = "experiment_5"
    wave_to_plot = -1

    if only_plot:
        summary_plots(wave_to_plot = 9, experiment_name = experiment_name)
    else:

        fitting_hm.run_new_wave(num_wave = 0, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 4, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1

        fitting_hm.run_new_wave(num_wave = 1, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 4, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 2, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 4, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 3, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.8, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 4, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.6, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 5, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.4, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 6, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.2, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 7, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.2, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 8, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.2, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 9, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3., n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

def experiment_6(only_plot = True):
    """History matching pipeline to test different scenarios. Implausibility 
    thresholds, training sets size and number of waves are detailed in the 
    script.

    Args:
        only_plot (bool, optional): If True, it only prints the plots. 
        Defaults to True.
    """
    # Cut to the chase (total memory)

    original_training_set_size = 100
    experiment_name = "experiment_6"
    wave_to_plot = -1

    if only_plot:
        summary_plots(wave_to_plot = 5, experiment_name = experiment_name)
    else:

        fitting_hm.run_new_wave(num_wave = 0, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.2, n_samples = original_training_set_size,
                    generate_simul_pts = 20, subfolder = experiment_name, training_set_memory = 20)
        wave_to_plot += 1

        fitting_hm.run_new_wave(num_wave = 1, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.2, n_samples = original_training_set_size,
                    generate_simul_pts = 20, subfolder = experiment_name, training_set_memory = 20)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 2, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.2, n_samples = original_training_set_size,
                    generate_simul_pts = 20, subfolder = experiment_name, training_set_memory = 20)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 3, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3., n_samples = original_training_set_size,
                    generate_simul_pts = 20, subfolder = experiment_name, training_set_memory = 20)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 4, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3., n_samples = original_training_set_size,
                    generate_simul_pts = 20, subfolder = experiment_name, training_set_memory = 20)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 5, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3., n_samples = original_training_set_size,
                    generate_simul_pts = 20, subfolder = experiment_name, training_set_memory = 20)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

def experiment_7(only_plot = False):
    """History matching pipeline to test different scenarios. Implausibility 
    thresholds, training sets size and number of waves are detailed in the 
    script.

    Args:
        only_plot (bool, optional): If True, it only prints the plots. 
        Defaults to True.
    """
     # Cut to the chase (memory 2)

    original_training_set_size = 100
    experiment_name = "experiment_7"
    wave_to_plot = -1

    if only_plot:
        summary_plots(wave_to_plot = 4, experiment_name = experiment_name)
    else:

        fitting_hm.run_new_wave(num_wave = 0, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.2, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1

        fitting_hm.run_new_wave(num_wave = 1, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.2, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 2, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.2, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 3, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3., n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 4, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3., n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 5, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3., n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

def experiment_anatomy():
    """History matching pipeline for the EP + anatomy scenario. Implausibility 
    thresholds, training sets size and number of waves are detailed in the 
    script.

    Args:
        only_plot (bool, optional): If True, it only prints the plots. 
        Defaults to True.
    """
    num_input_param = 14
    experiment_name = "anatomy"
    wave_to_plot = -1
    output_labels_dir = os.path.join("/data","fitting",experiment_name,"output_labels.txt")
    units_dir = os.path.join("/data","fitting",experiment_name,"output_units.txt")
    exp_mean_name = "exp_mean_anatomy_EP.txt"
    exp_std_name = "exp_std_anatomy_EP.txt"

    xlabels_EP = read_labels(os.path.join("/data","fitting", experiment_name, "EP_funct_labels_latex.txt"))
    xlabels_anatomy = read_labels(os.path.join("/data","fitting", experiment_name, "modes_labels.txt"))
    xlabels = [lab for sublist in [xlabels_anatomy,xlabels_EP] for lab in sublist]

    # fitting_hm.anatomy_new_wave(num_wave = 0, run_simulations = True, train_GPE = True,
    #                     fill_wave_space = True, cutoff = 3.2, n_samples = int(num_input_param*20),
    #                     generate_simul_pts = int(num_input_param*10), subfolder = experiment_name,
    #                     training_set_memory = 2)

    wave_to_plot = wave_to_plot + 1

    # fitting_hm.anatomy_new_wave(num_wave = 1, run_simulations = True, train_GPE = True,
    #                     fill_wave_space = True, cutoff = 3.2, n_samples = int(num_input_param*20),
    #                     generate_simul_pts = int(num_input_param*10), subfolder = experiment_name,
    #                     training_set_memory = 2)
    
    wave_to_plot = wave_to_plot + 1


    # fitting_hm.anatomy_new_wave(num_wave = 2, run_simulations = True, train_GPE = True,
    #                     fill_wave_space = True, cutoff = 3.0, n_samples = int(num_input_param*20),
    #                     generate_simul_pts = int(num_input_param*10), subfolder = experiment_name,
    #                     training_set_memory = 2)

    wave_to_plot = wave_to_plot + 1
    
    summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name,
                        only_feasible = False, output_labels_dir = output_labels_dir,
                        exp_mean_name = exp_mean_name, exp_std_name = exp_std_name,
                        units_dir = units_dir)
    print("Summary plots finished")
    # custom_plots.full_GSA(emul_num = wave_to_plot, subfolder = experiment_name,
    #                     output_labels_dir = output_labels_dir,
    #                     input_labels = xlabels)

def experiment_anatomy_bigwave0():
    """History matching pipeline for the EP + anatomy scenario. Implausibility
    thresholds, training sets size and number of waves are detailed in the
    script.

    Args:
        only_plot (bool, optional): If True, it only prints the plots.
        Defaults to True.
    """
    num_input_param = 14
    experiment_name = "anatomy/meshes"
    wave_to_plot = -1
    output_labels_dir = os.path.join("/data", "fitting", experiment_name, "output_labels.txt")
    units_dir = os.path.join("/data", "fitting", experiment_name, "output_units.txt")
    exp_mean_name = "exp_mean_anatomy_EP.txt"
    exp_std_name = "exp_std_anatomy_EP.txt"

    xlabels_EP = read_labels(os.path.join("/data", "fitting", experiment_name, "EP_funct_labels_latex.txt"))
    xlabels_anatomy = read_labels(os.path.join("/data", "fitting", experiment_name, "modes_labels.txt"))
    xlabels = [lab for sublist in [xlabels_anatomy, xlabels_EP] for lab in sublist]

    fitting_hm.anatomy_new_wave(num_wave = 0, run_simulations = True, train_gpe = True,
                        fill_wave_space = True, cutoff = 3.2, n_samples = int(num_input_param*30),
                        generate_simul_pts = int(num_input_param*10), subfolder = experiment_name,
                        training_set_memory = 2)

    wave_to_plot = wave_to_plot + 1
    #
    # fitting_hm.anatomy_new_wave(num_wave = 1, run_simulations = True, train_gpe = True,
    #                     fill_wave_space = True, cutoff = 3.2, n_samples = int(num_input_param*30),
    #                     generate_simul_pts = int(num_input_param*10), subfolder = experiment_name,
    #                     training_set_memory = 2)
    #
    # wave_to_plot = wave_to_plot + 1
    #
    # fitting_hm.anatomy_new_wave(num_wave = 2, run_simulations = True, train_gpe = True,
    #                     fill_wave_space = True, cutoff = 3.0, n_samples = int(num_input_param*30),
    #                     generate_simul_pts = int(num_input_param*10), subfolder = experiment_name,
    #                     training_set_memory = 2)
    #
    # wave_to_plot = wave_to_plot + 1
    #
    # summary_plots(wave_to_plot=wave_to_plot, experiment_name=experiment_name,
    #               only_feasible=False, output_labels_dir=output_labels_dir,
    #               exp_mean_name=exp_mean_name, exp_std_name=exp_std_name,
    #               units_dir=units_dir)
    # print("Summary plots finished")
    # custom_plots.full_GSA(emul_num = wave_to_plot, subfolder = experiment_name,
    #                     output_labels_dir = output_labels_dir,
    #                     input_labels = xlabels)

def experiment_anatomy_max_range():
    """
    Same as anatomy but taking the input parameter range as the maximum of the CT cases instead of 2SD.
    """
    num_input_param = 14
    experiment_name = "anatomy_max_range_limit_CT25percent"
    wave_to_plot = -1
    output_labels_dir = os.path.join("/data","fitting",experiment_name,"output_labels.txt")
    units_dir = os.path.join("/data","fitting",experiment_name,"output_units.txt")
    exp_mean_name = "exp_mean_anatomy_EP.txt"
    exp_std_name = "exp_std_anatomy_EP.txt"

    xlabels_EP = read_labels(os.path.join("/data","fitting", experiment_name, "EP_funct_labels_latex.txt"))
    xlabels_anatomy = read_labels(os.path.join("/data","fitting", experiment_name, "modes_labels.txt"))
    xlabels = [lab for sublist in [xlabels_anatomy,xlabels_EP] for lab in sublist]

    fitting_hm.anatomy_new_wave(num_wave = 0, run_simulations = True, train_gpe = True,
                        fill_wave_space = True, cutoff = 3.2, n_training_pts = int(num_input_param*20),
                        generate_simul_pts = int(num_input_param*10), subfolder = experiment_name,
                        training_set_memory = 2, max_range = True)

    wave_to_plot = wave_to_plot + 1

    fitting_hm.anatomy_new_wave(num_wave = 1, run_simulations = True, train_gpe = True,
                        fill_wave_space = True, cutoff = 3.2, n_training_pts = int(num_input_param*20),
                        generate_simul_pts = int(num_input_param*10), subfolder = experiment_name,
                        training_set_memory = 2, max_range = True)

    wave_to_plot = wave_to_plot + 1

    fitting_hm.anatomy_new_wave(num_wave = 2, run_simulations = True, train_gpe = True,
                        fill_wave_space = True, cutoff = 3.0, n_training_pts = int(num_input_param*20),
                        generate_simul_pts = int(num_input_param*10), subfolder = experiment_name,
                        training_set_memory = 2, max_range = True)

    wave_to_plot = wave_to_plot + 1

    summary_plots(wave_to_plot=wave_to_plot, experiment_name=experiment_name,
                  only_feasible=False, output_labels_dir=output_labels_dir,
                  exp_mean_name=exp_mean_name, exp_std_name=exp_std_name,
                  units_dir=units_dir)
    print("Summary plots finished")
    postprocessing.full_GSA(emul_num = wave_to_plot, subfolder = experiment_name,
                        output_labels_dir = output_labels_dir,
                        input_labels = xlabels)

def EP_template(only_plot = False):
    """History matching pipeline to test different scenarios. Implausibility
    thresholds, training sets size and number of waves are detailed in the
    script.

    Args:
        only_plot (bool, optional): If True, it only prints the plots.
        Defaults to True.
    """
     # Cut to the chase (memory 2)

    original_training_set_size = 100
    experiment_name = "EP_template"
    wave_to_plot = -1

    if only_plot:
        summary_plots(wave_to_plot = 4, experiment_name = experiment_name)
    else:

        fitting_hm.run_new_wave(num_wave = 0, run_simulations = True, train_gpe = True,
                    fill_wave_space = True, cutoff = 3.2, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1

        fitting_hm.run_new_wave(num_wave = 1, run_simulations = True, train_gpe = True,
                    fill_wave_space = True, cutoff = 3.2, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1

        fitting_hm.run_new_wave(num_wave = 2, run_simulations = True, train_gpe = True,
                    fill_wave_space = True, cutoff = 3.2, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1

        fitting_hm.run_new_wave(num_wave = 3, run_simulations = True, train_gpe = True,
                    fill_wave_space = True, cutoff = 3., n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1

        fitting_hm.run_new_wave(num_wave = 4, run_simulations = True, train_gpe = True,
                    fill_wave_space = True, cutoff = 3., n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

def summary_R2_ISE_anatomy():
    active_features = ["LVV", "RVV", "LAV", "RAV", "LVOTdiam", "RVOTdiam", "LVmass",
                       "LVWT", "LVEDD", "SeptumWT", "RVlongdiam",
                       "TAT", "TATLVendo"]

    R2_vec, ISE_vec = fitting_hm.first_GPE(active_features=active_features,
                                           train=False, saveplot=False,
                                           start_size=280,
                                           subfolder="anatomy_max_range",
                                           only_feasible=False, return_scores=True)

    R2_mean = round(np.mean([float(i) for i in R2_vec]),2)
    ISE_mean = round(np.mean([float(i) for i in ISE_vec]),2)
    R2_std = round(np.std([float(i) for i in R2_vec]),2)
    ISE_std = round(np.std([float(i) for i in ISE_vec]),2)

    print("R2: " + str(R2_mean) + "+-" + str(R2_std))
    print("ISE: " + str(ISE_mean) + "+-" + str(ISE_std))

def run_experiment(experiment_name):
    """Function to run one of the previously defined history matching pipelines.

    Args:
        experiment_name (str): Name of the scenario to run.
    """
    experiment_name = str(experiment_name)

    if experiment_name == "1" or experiment_name == "coveney":
        experiment_coveney()
    elif experiment_name == "2" or experiment_name == "longobardi":
        experiment_longobardi()
    elif experiment_name == "3":
        experiment_3()
    elif experiment_name == "4":
        experiment_4()
    elif experiment_name == "5":
        experiment_5()
    elif experiment_name == "6":
        experiment_6()
    elif experiment_name == "7":
        experiment_7(only_plot=True)
    elif experiment_name == "anatomy":
        experiment_anatomy()
    elif experiment_name == "anatomy_bigwave0":
        experiment_anatomy_bigwave0()
    elif experiment_name == "all":
        for i in range(1,8):
            run_experiment(i)
    
if __name__ == "__main__":
    run_experiment(sys.argv[1])
    # print_NROY_boundaries_anatomy_EP()
