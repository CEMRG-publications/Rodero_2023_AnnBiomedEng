import diversipy as dp
from pickle import FALSE, TRUE
import random
import numpy as np
import torch
import torchmetrics
import os
import pathlib
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import iqr
import seaborn
from itertools import combinations
from SALib.sample import saltelli
from scipy.special import binom
from SALib.analyze import sobol
import multiprocessing
import tqdm
import template_EP
from scipy.stats import gaussian_kde
import pandas as pd
import seaborn as sns
import skopt
import time

from Historia.shared.design_utils import get_minmax, lhd, read_labels
from Historia.shared.plot_utils import plot_pairwise_waves, interp_col, get_col
from gpytGPE.gpe import GPEmul
from gpytGPE.utils.metrics import IndependentStandardError as ISE
from Historia.history import hm
from gpytGPE.utils.plotting import gsa_box, gsa_donut

import anatomy
import custom_plots
import mechanics

SEED = 2
# ----------------------------------------------------------------
# Make the code reproducible
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def first_GPE(active_features = ["TAT","TATLV"], train = False, saveplot = True,
              start_size = np.inf, figure_name = "inference_on_testset",
              return_scores = False, subfolder = ".",only_feasible = True):
    """Function to run the GPE with the original training set and testing it.

    Args:
        active_features (list, optional): List of output features to train the
        GPEs. Defaults to ["TAT","TATLV"].
        train (bool, optional): If True, trains the GPE, otherwise it loads it.
        Defaults to False.
        saveplot (bool, optional): Boolean to save the plots or just show them.
        Defaults to True.
        start_size (int, optional): Number of points to use in train/validation.
        Validation will be 20% of that value, and the 80% remaining will be
        split in 80%/20% for training/test. Defaults to np.inf to read the whole
        input file (just in case you want to use a subset of those points).
        figure_name (str, optional): Filename of the plot. Defaults to 
        "inference_on_testset".
        return_scores (bool, optional): If True, shows the MSE and ISE. Defaults
        to False.
        subfolder (str, optional): Name of the subfolder in /data/fitting to
        work on. Defaults to ".".
        only_feasible (bool, optional): If True, works only with "feasible"
        point (within 5SD of the mean). Defaults to True.

    Returns:
        X_train, y_train, emul (list): List with the vector (or matrix) of
        training points, simulations of those points and the resulting emulator.
    """
    print(start_size)

    SEED = 2
    # ----------------------------------------------------------------
    # Make the code reproducible
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    path_gpes = os.path.join("/data","fitting",subfolder, "wave0")
    path_figures = os.path.join("/data","fitting",subfolder, "figures")


    if only_feasible:
        X = np.loadtxt(os.path.join(path_gpes, "X_feasible.dat"), dtype=float)
    else:
        X = np.loadtxt(os.path.join(path_gpes, "X.dat"), dtype=float)

    n_samples = min(X.shape[0], start_size)

    idx_train = round(0.8*0.8*n_samples) # 100 approx (5 input x 20)
    idx_val = idx_train + round(0.2*0.8*n_samples) # Therefore 25
    idx_test = idx_val + round(0.2*n_samples) # Therefore 31

    X_train = X[:idx_train]
    X_val = X[idx_train:idx_val]
    X_test = X[idx_val:idx_test]

    mean_list = []
    std_list = []
    emulator = []
    R2score_vec = []
    ISE_vec = []

    # (7) Plotting mean predictions + uncertainty vs observations
    if saveplot:
        height = 9.36111
        width = 5.91667
        fig, axes = plt.subplots(1, 2, figsize=(2 * width, 2 * height / 4))

    for i, output_name in enumerate(active_features):
        print(output_name)
        if only_feasible:
            y = np.loadtxt(os.path.join(path_gpes, output_name + "_feasible.dat"),dtype=float)
        else:
            y = np.loadtxt(os.path.join(path_gpes, output_name + ".dat"),dtype=float)
        y_train = y[:idx_train]
        y_val = y[idx_train:idx_val]
        y_test =  y[idx_val:idx_test]

        if train:
            emul = GPEmul(X_train, y_train)
            emul.train(X_val, y_val, max_epochs = 100, n_restarts = 5,savepath = path_gpes + "/")
            emul.save("wave0_" + output_name + ".gpe")
    # ----------------------------------------------------------------
        else:
            emul = GPEmul.load(X_train, y_train, loadpath=path_gpes + "/",filename = "wave0_" + output_name + ".gpe")

        emulator.append(emul)

        y_pred_mean, y_pred_std = emul.predict(X_test)
        mean_list.append(y_pred_mean)
        std_list.append(y_pred_std)

        R2Score = torchmetrics.R2Score()(emul.tensorize(y_pred_mean), emul.tensorize(y_test))

        iseScore = ISE(
            emul.tensorize(y_test),
            emul.tensorize(y_pred_mean),
            emul.tensorize(y_pred_std),
        )

        R2string = f"{R2Score:.2f}"

        print(f"\nStatistics on test set for GPE trained for the output " + output_name + ":")
        print(f"  R2 = {R2Score:.2f}")
        # print(f"  R2 = " + R2string)
        print(f"  %ISE = {iseScore:.2f} %\n")

        R2score_vec.append(f"{R2Score:.2f}")
        ISE_vec.append(f"{iseScore:.2f}")

        ci = 2 #~95% confidance interval

        inf_bound = []
        sup_bound = []

        l = np.argsort(y_pred_mean)  # for the sake of a better visualisation
        # l = np.argsort(-y_test)
        # l = np.array(range(len(y_pred_mean)))
        inf_bound.append((y_pred_mean - ci * y_pred_std).min())
        sup_bound.append((y_pred_mean + ci * y_pred_std).max())



        if saveplot:
            axes[i].scatter(
                np.arange(1, len(l) + 1),
                y_test[l],
                facecolors="none",
                edgecolors="C0",
                label="simulated",
            )
            axes[i].scatter(
                np.arange(1, len(l) + 1),
                y_pred_mean[l],
                facecolors="C0",
                s=16,
                label="emulated",
            )
            axes[i].errorbar(
                np.arange(1, len(l) + 1),
                y_pred_mean[l],
                yerr=ci * y_pred_std[l],
                c="C0",
                ls="none",
                lw=0.5,
                label=f"uncertainty ({ci} SD)",
            )

            axes[i].set_xticks([])
            axes[i].set_xticklabels([])
            axes[i].set_ylabel("ms", fontsize=12)
            axes[i].set_title(
                f"GPE {output_name} | R2 = {R2Score:.2f} | %ISE = {iseScore:.2f}",
                fontsize=12,
            )
            axes[i].legend(loc="upper left")

    if saveplot:
        axes[0].set_ylim([np.min(inf_bound), np.max(sup_bound)])
        axes[1].set_ylim([np.min(inf_bound), np.max(sup_bound)])


        fig.tight_layout()
        plt.savefig(
            os.path.join(path_figures, figure_name + ".png"), bbox_inches="tight", dpi=300
        )

    if return_scores:
        return R2score_vec, ISE_vec
    else:
        return X_train, y_train, emulator

def run_GPE(waveno = 2, train = False, active_feature = ["TAT"],
            n_samples = 125, training_set_memory = 100, subfolder = ".",
            only_feasible = True):
    """Function to train or evaluate a GPE of a given wave.

    Args:
        waveno (int, optional): Wave number (establishes the folder name).
        Defaults to 2.
        train (bool, optional): If True, trains the GPE, otherwise it loads it.
        Defaults to False.
        active_feature (list, optional): List of output features to train the
        GPEs. Defaults to ["TAT"].
        n_samples (int, optional): Number of points to use in train/validation.
        Validation will be 20% of that value, and the 80% remaining will be
        split in 80%/20% for training/test. Defaults to 125.
        training_set_memory (int, optional): The training set of the current 
        emulator will be expanded with the previous training_set_memory training
        sets. Defaults to 100.
        subfolder (str, optional): Name of the subfolder in /data/fitting to
        work on. Defaults to ".".
        only_feasible (bool, optional): If True, works only with "feasible"
        point (within 5SD of the mean). Defaults to True.

    Returns:
        X_train, y_train, emul (list): List with the vector (or matrix) of
        training points, simulations of those points and the resulting emulator.
    """

    SEED = 2
    # ----------------------------------------------------------------
    # Make the code reproducible
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if waveno == 0:
        X_train, y_train, emul = first_GPE(active_features = active_feature,
                                            train = train, saveplot=False,
                                            start_size = n_samples,
                                            subfolder = subfolder,
                                            only_feasible = only_feasible)
    else:
        if only_feasible:
            current_X = np.loadtxt(os.path.join("/data","fitting",subfolder,"wave" + str(waveno),"X_feasible.dat"), dtype=float)
            current_y = np.loadtxt(os.path.join("/data","fitting",subfolder,"wave" + str(waveno), active_feature[0] + "_feasible.dat"),dtype=float)
        else:
            current_X = np.loadtxt(os.path.join("/data","fitting",subfolder,"wave" + str(waveno),"X.dat"), dtype=float)
            current_y = np.loadtxt(os.path.join("/data","fitting",subfolder,"wave" + str(waveno), active_feature[0] + ".dat"),dtype=float)
        
        if training_set_memory > 0:
            X_prev, y_prev, _ = run_GPE(waveno = waveno - 1, train = False,
                                        active_feature = active_feature,
                                        n_samples = n_samples, subfolder = subfolder,
                                        training_set_memory = training_set_memory - 1,
                                        only_feasible = only_feasible)

            X_train = np.vstack((current_X, X_prev))
            y_train = np.append(current_y,y_prev)
        else:
            X_train = current_X
            y_train = current_y

        if train:
            emul = GPEmul(X_train, y_train)
            emul.train(X_val = None, y_val = None, max_epochs = 100, n_restarts = 5,savepath = os.path.join("/data","fitting",subfolder,"wave" + str(waveno) + "/"))
            emul.save("wave" + str(waveno) + "_" + active_feature[0] + ".gpe")
        else:
            emul = GPEmul.load(X_train, y_train, loadpath=os.path.join("/data","fitting",subfolder,"wave" + str(waveno) + "/"),filename = "wave" + str(waveno) + "_" + active_feature[0] + ".gpe")
        # ----------------------------------------------
    return X_train, y_train, emul

def run_new_wave(num_wave = 3, run_simulations = False, train_GPE = False,
                fill_wave_space = False, cutoff = 2.0, n_samples = 150,
                generate_simul_pts = 10, subfolder = ".", training_set_memory = 100,
                skip_filtering = True):

    #==========================================================================
    # For each wave i, with a implausibility threshold cutoff_i and a maximum
    # variance var_i (i > 0), we have:
    # 1. If var_i < var_{i-1} -> We can either decrease or keep constant
    # cutoff_{i+1}
    # 2. If var_i approx = var_{i-1} (and low) --> We decrease cutoff_{i+1}
    # 3. If var_i approx = var_{i-1} (and high) --> We keep cutoff_{i+1}
    # 4. If var_i > var_{i-1} --> The emulator got worse so we need to restart
    # probably with a different emulator or a better training set
    #==========================================================================
    SEED = 2
    # ----------------------------------------------------------------
    # Make the code reproducible
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    #========== Constant variables =============#

    xlabels = read_labels(os.path.join("/data","fitting", "EP_funct_labels_latex.txt"))
    idx_train = round(0.8*0.8*n_samples)

    active_features = ["TAT","TATLV"]
    emulator = []

    n_tests = int(1e5)
    path_match = os.path.join("/data","fitting", "match")
    exp_mean = np.loadtxt(os.path.join(path_match, "exp_mean.txt"), dtype=float)
    exp_std = np.loadtxt(os.path.join(path_match, "exp_std.txt"), dtype=float)
    exp_var = np.power(exp_std, 2)

    #================ Load training sets or run simulations =================#
    if run_simulations:
        if num_wave == 0:
            template_EP.EP_funct_param(n_samples = round(n_samples/(0.8*0.8)),
            waveno = 0, subfolder = subfolder)
        had_to_run_new = template_EP.template_EP_parallel(line_from = 0, line_to = np.inf,
                                            waveno = num_wave, subfolder = subfolder)
    
        if not had_to_run_new:
            if os.path.isfile(os.path.join("/data","fitting",subfolder,"wave" + str(num_wave),"X_feasible.dat")):
                for i in range(len(active_features)):
                    if os.path.isfile(os.path.join("/data","fitting",subfolder,"wave" + str(num_wave),active_features[i] + ".dat")):
                        Y_QC = np.loadtxt(os.path.join("/data","fitting",subfolder,"wave" + str(num_wave),active_features[i] + ".dat"), dtype=float)
                        if max(Y_QC) > 1e5:
                            had_to_run_new = True
                            break
                    else:
                        had_to_run_new = True
                        break
            else:
                had_to_run_new = True


        if had_to_run_new:
            template_EP.EP_output(waveno = num_wave, subfolder = subfolder)
            template_EP.filter_output(waveno = num_wave, subfolder = subfolder, skip = skip_filtering)

    X = np.loadtxt(os.path.join("/data","fitting",subfolder,"wave0","X_feasible.dat"), dtype=float)
    X_train = X[:idx_train]
    I = get_minmax(X_train)


    #=================== Train or load GPE =========================#
    for output_name in active_features:
        _, _, emul = run_GPE(waveno = num_wave, train = train_GPE,
                            active_feature = [output_name],
                            n_samples = n_samples,
                            training_set_memory = training_set_memory,
                            subfolder = subfolder)
        emulator.append(emul)
    # emulator now might be a list of list, we need to flatten it
    em_shape = (np.array(emulator)).shape
    if len(em_shape) > 1:
        emulator_flat = [item for sublist in emulator for item in sublist]
        emulator = emulator_flat
    #=================== Load or create the wave object ===================#
    if num_wave > 0:
        W = hm.Wave()
        W.load(os.path.join("/data","fitting",subfolder,"wave" + str(num_wave-1),
                    "wave_" + str(num_wave-1)))
        W.emulator = emulator
        W.Itrain = I
        W.cutoff = cutoff
        W.maxno = 1
        W.mean = exp_mean
        W.var = exp_var
    else:
        W = hm.Wave(emulator = emulator,
                    Itrain = I,
                    cutoff = cutoff,
                    maxno = 1,
                    mean = exp_mean,
                    var = exp_var)



    #============= Load or run the cloud technique =========================#
    param_ranges_lower = np.loadtxt(os.path.join(path_match, "input_range_lower.dat"), dtype=float)
    param_ranges_upper = np.loadtxt(os.path.join(path_match, "input_range_upper.dat"), dtype=float)

    param_ranges = np.column_stack((param_ranges_lower, param_ranges_upper))
    if fill_wave_space:
        print("Experiment " + subfolder + ", wave " + str(num_wave))
        print("Adding points at " + time.strftime("%H:%M",time.localtime()))
        if num_wave == 0:
            TESTS = lhd(param_ranges, n_tests, SEED)
            
        else:
            TESTS = W.add_points(n_tests, scale = 0.05, param_ranges = param_ranges)  # use the "cloud technique" to populate
        print("Points added")
        # what is left from W.NIMP\SIMULS (set difference) if points left are <
        # the chosen n_tests

        np.savetxt(os.path.join("/data","fitting",subfolder,"wave" + str(num_wave),"X_test.dat"), TESTS, fmt="%.2f")
    else:
        TESTS = np.loadtxt(os.path.join("/data","fitting",subfolder,"wave" + str(num_wave),"X_test.dat"),
                        dtype=float)


    #============= We finally print and show the wave we wanted =============#
    W.find_regions(TESTS)  # enforce the implausibility criterion to detect regions of
                            # non-implausible and of implausible points
    W.print_stats()  # show statistics about the two obtained spaces
    
    nimp = len(W.nimp_idx)
    imp = len(W.imp_idx)
    tests = nimp + imp
    perc = 100 * nimp / tests

    np.savetxt(os.path.join("/data","fitting",subfolder,"wave" + str(num_wave),"NROY_rel.dat"), [perc], fmt='%.2f')

    pathlib.Path(os.path.join("/data","fitting",subfolder,"figures")).mkdir(parents=True, exist_ok=True)


    custom_plots.plot_wave(W = W, xlabels = xlabels,
                            filename = os.path.join("/data","fitting",
                            subfolder,"figures","wave" + str(num_wave) + "_impl_min"),
                            waveno = num_wave, reduction_function = "min",
                            plot_title = subfolder + ", wave " + str(num_wave) + ": taking the min of each slice",
                            param_ranges = param_ranges)
    custom_plots.plot_wave(W = W, xlabels = xlabels,
                            filename = os.path.join("/data","fitting",
                            subfolder,"figures","wave" + str(num_wave) + "_impl_max"),
                            waveno = num_wave, reduction_function = "max",
                            plot_title = subfolder + ", wave " + str(num_wave) + ": taking the max of each slice",
                            param_ranges = param_ranges)
    custom_plots.plot_wave(W = W, xlabels = xlabels,
                            filename = os.path.join("/data","fitting",
                            subfolder,"figures","wave" + str(num_wave) + "_prob_imp"),
                            waveno = num_wave, reduction_function = "prob_IMP",
                            plot_title = subfolder + ", wave " + str(num_wave) + ": percentage of implausible points",
                            param_ranges = param_ranges)
    W.plot_wave(xlabels=xlabels, display="var", filename=os.path.join("/data","fitting",subfolder,"figures","wave" + str(num_wave) + "_var.png"))

    #=================== Generate data for next wave =====================#
    SIMULS = W.get_points(generate_simul_pts)  # actual matrix of selected points

    pathlib.Path(os.path.join("/data","fitting",subfolder,"wave" + str(num_wave + 1))).mkdir(parents=True, exist_ok=True)
    np.savetxt(os.path.join("/data","fitting",subfolder,"wave" + str(num_wave + 1),"X.dat"), SIMULS, fmt="%.2f")

    W.save(os.path.join("/data","fitting",subfolder,"wave" + str(num_wave),
                    "wave_" + str(num_wave)))
    np.savetxt(os.path.join("/data","fitting",subfolder,"wave" + str(num_wave),"variance_quotient.dat"), W.PV, fmt="%.2f")

def anatomy_new_wave(num_wave = 0, run_simulations = False, train_GPE = False,
                fill_wave_space = False, cutoff = 0, n_samples = 150,
                generate_simul_pts = 10, subfolder = "anatomy", training_set_memory = 100,
                only_feasible = False):

    SEED = 2
    # ----------------------------------------------------------------
    # Make the code reproducible
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    #========== Constant variables =============#

    xlabels_EP = read_labels(os.path.join("/data","fitting", "EP_funct_labels_latex.txt"))
    xlabels_anatomy = read_labels(os.path.join("/data","fitting", "modes_labels.txt"))
    xlabels = [lab for sublist in [xlabels_anatomy,xlabels_EP] for lab in sublist]
    
    idx_train = round(0.8*0.8*n_samples)

    active_features = ["LVV","RVV","LAV","RAV","LVOTdiam","RVOTdiam","LVmass",
                        "LVWT","LVEDD","SeptumWT","RVlongdiam","RVbasaldiam",
                        "TAT","TATLVendo"]
    emulator = []

    n_tests = int(1e5)
    path_match = os.path.join("/data","fitting", "match")
    exp_mean = np.loadtxt(os.path.join(path_match, "exp_mean_anatomy_EP.txt"), dtype=float)
    exp_std = np.loadtxt(os.path.join(path_match, "exp_std_anatomy_EP.txt"), dtype=float)
    exp_var = np.power(exp_std, 2)

    #================ Load training sets or run simulations =================#
    if run_simulations:
        if num_wave == 0:
            anatomy.input(n_samples = n_samples/(0.8*0.8), waveno = num_wave, subfolder = subfolder)
        else:
            anatomy.preprocess_input(waveno = num_wave, subfolder = subfolder)

        anatomy.build_meshes(waveno = num_wave, subfolder = subfolder)
        anatomy.EP_setup(waveno = num_wave, subfolder = subfolder)
        anatomy.EP_simulations(waveno = num_wave, subfolder = subfolder)

        anatomy.write_output_casewise(waveno = num_wave, subfolder = subfolder)
        anatomy.collect_output(waveno = num_wave, subfolder = subfolder)

    X = np.loadtxt(os.path.join("/data","fitting",subfolder,"wave0","X.dat"), dtype=float)
    X_train = X[:idx_train]
    I = get_minmax(X_train)


    #=================== Train or load GPE =========================#
    for output_name in active_features:
        _, _, emul = run_GPE(waveno = num_wave, train = train_GPE,
                            active_feature = [output_name],
                            n_samples = n_samples,
                            training_set_memory = training_set_memory,
                            subfolder = subfolder, only_feasible=only_feasible)
        emulator.append(emul)
    # emulator now might be a list of list, we need to flatten it
    em_shape = (np.array(emulator)).shape
    if len(em_shape) > 1:
        emulator_flat = [item for sublist in emulator for item in sublist]
        emulator = emulator_flat
    #=================== Load or create the wave object ===================#
    if num_wave > 0:
        W = hm.Wave()
        W.load(os.path.join("/data","fitting",subfolder,"wave" + str(num_wave-1),
                    "wave_" + str(num_wave-1)))
        W.emulator = emulator
        W.Itrain = I
        W.cutoff = cutoff
        W.maxno = 1
        W.mean = exp_mean
        W.var = exp_var
    else:
        W = hm.Wave(emulator = emulator,
                    Itrain = I,
                    cutoff = cutoff,
                    maxno = 1,
                    mean = exp_mean,
                    var = exp_var)



    #============= Load or run the cloud technique =========================#

    param_ranges_lower_anatomy = np.loadtxt(os.path.join(path_match, "anatomy_input_range_lower.dat"), dtype=float)
    param_ranges_upper_anatomy = np.loadtxt(os.path.join(path_match, "anatomy_input_range_upper.dat"), dtype=float)

    param_ranges_lower_EP = np.loadtxt(os.path.join(path_match, "EP_input_range_lower.dat"), dtype=float)
    param_ranges_upper_EP = np.loadtxt(os.path.join(path_match, "EP_input_range_upper.dat"), dtype=float)

    param_ranges_lower = np.append(param_ranges_lower_anatomy, param_ranges_lower_EP)
    param_ranges_upper = np.append(param_ranges_upper_anatomy, param_ranges_upper_EP)

    param_ranges = np.column_stack((param_ranges_lower, param_ranges_upper))
    if fill_wave_space:
        print("Experiment " + subfolder + ", wave " + str(num_wave))
        print("Adding points at " + time.strftime("%H:%M",time.localtime()))
        if num_wave == 0:
            TESTS = lhd(param_ranges, n_tests, SEED)
            
        else:
            TESTS = modified_add_points(W = W, n_tests = n_tests, scale = 0.05, param_ranges = param_ranges)
            # TESTS = W.add_points(n_tests, scale = 0.05, param_ranges = param_ranges)  # use the "cloud technique" to populate
        print("Points added")
        # what is left from W.NIMP\SIMULS (set difference) if points left are <
        # the chosen n_tests

        np.savetxt(os.path.join("/data","fitting",subfolder,"wave" + str(num_wave),"X_test.dat"), TESTS, fmt="%.2f")
    else:
        TESTS = np.loadtxt(os.path.join("/data","fitting",subfolder,"wave" + str(num_wave),"X_test.dat"),
                        dtype=float)


    #============= We finally print and show the wave we wanted =============#
    W.find_regions(TESTS)  # enforce the implausibility criterion to detect regions of
                            # non-implausible and of implausible points
    W.print_stats()  # show statistics about the two obtained spaces
    
    nimp = len(W.nimp_idx)
    imp = len(W.imp_idx)
    tests = nimp + imp
    perc = 100 * nimp / tests

    np.savetxt(os.path.join("/data","fitting",subfolder,"wave" + str(num_wave),"NROY_rel.dat"), [perc], fmt='%.2f')

    pathlib.Path(os.path.join("/data","fitting",subfolder,"figures")).mkdir(parents=True, exist_ok=True)

    print("Printing impl. min...")
    custom_plots.plot_wave(W = W, xlabels = xlabels,
                            filename = os.path.join("/data","fitting",
                            subfolder,"figures","wave" + str(num_wave) + "_impl_min"),
                            waveno = num_wave, reduction_function = "min",
                            plot_title = subfolder + ", wave " + str(num_wave) + ": taking the min of each slice",
                            param_ranges = param_ranges)
    print("Printing impl. max...")
    custom_plots.plot_wave(W = W, xlabels = xlabels,
                            filename = os.path.join("/data","fitting",
                            subfolder,"figures","wave" + str(num_wave) + "_impl_max"),
                            waveno = num_wave, reduction_function = "max",
                            plot_title = subfolder + ", wave " + str(num_wave) + ": taking the max of each slice",
                            param_ranges = param_ranges)
    print("Printing variance...")
    W.plot_wave(xlabels=xlabels, display="var", filename=os.path.join("/data","fitting",subfolder,"figures","wave" + str(num_wave) + "_var.png"))

    #=================== Generate data for next wave =====================#
    SIMULS = W.get_points(generate_simul_pts)  # actual matrix of selected points

    pathlib.Path(os.path.join("/data","fitting",subfolder,"wave" + str(num_wave + 1))).mkdir(parents=True, exist_ok=True)
    np.savetxt(os.path.join("/data","fitting",subfolder,"wave" + str(num_wave + 1),"X.dat"), SIMULS, fmt="%.2f")

    W.save(os.path.join("/data","fitting",subfolder,"wave" + str(num_wave),
                    "wave_" + str(num_wave)))
    np.savetxt(os.path.join("/data","fitting",subfolder,"wave" + str(num_wave),"variance_quotient.dat"), W.PV, fmt="%.2f")

def modified_add_points(W, n_tests, scale=0.1, param_ranges = None):
# NOTE: the Wave object instance internal structure will be compromised after calling this method: we recommend calling self.save() beforehand!
    nsidx = W.nsimul_idx
    NROY = np.copy(W.NIMP[nsidx])

    if param_ranges is None:
        lbounds = W.Itrain[:, 0]
        ubounds = W.Itrain[:, 1]
    else:
        lbounds = param_ranges[:, 0]
        ubounds = param_ranges[:, 1]

    print(
        f"\nRequested points: {n_tests}\nAvailable points: {NROY.shape[0]}\nStart searching..."
    )

    count = 0
    a, b = (
        NROY.shape[0] if NROY.shape[0] < n_tests else n_tests,
        n_tests - NROY.shape[0] if n_tests - NROY.shape[0] > 0 else 0,
    )
    print(
        f"\n[Iteration: {count:<2}] Found: {a:<{len(str(n_tests))}} ({'{:.2f}'.format(100*a/n_tests):>6}%) | Missing: {b:<{len(str(n_tests))}}"
    )

    while NROY.shape[0] < n_tests:
        count += 1

        I = get_minmax(NROY)
        SCALE = scale * np.array(
            [I[i, 1] - I[i, 0] for i in range(NROY.shape[1])]
        )

        temp = np.random.normal(loc=NROY, scale=SCALE)
        while True:
            l = []
            for i in range(temp.shape[0]):
                d1 = temp[i, :] - lbounds
                d2 = ubounds - temp[i, :]
                if (
                    np.sum(np.sign(d1)) != temp.shape[1]
                    or np.sum(np.sign(d2)) != temp.shape[1]
                ):
                    l.append(i)
            if l:
                temp[l, :] = np.random.normal(loc=NROY[l, :], scale=SCALE)
            else:
                break

        W.find_regions(temp)
        NROY = np.vstack((NROY, W.NIMP))

        a, b = (
            NROY.shape[0] if NROY.shape[0] < n_tests else n_tests,
            n_tests - NROY.shape[0] if n_tests - NROY.shape[0] > 0 else 0,
        )
        print(
            f"[Iteration: {count:<2}] Found: {a:<{len(str(n_tests))}} ({'{:.2f}'.format(100*a/n_tests):>6}%) | Missing: {b:<{len(str(n_tests))}}"
        )

    print("\nDone.")
    TESTS = np.vstack(
        (
            NROY[: len(nsidx)],
            dp.subset.psa_select(NROY[len(nsidx) :], n_tests - len(nsidx)),
        )
    )
    return TESTS

def mechanics_new_wave(num_wave = 0, run_simulations = False, train_GPE = False,
                fill_wave_space = False, cutoff = 0, n_samples = -1,
                generate_simul_pts = -1, subfolder = "mechanics", training_set_memory = 2,
                only_feasible = False):

    SEED = 2
    # ----------------------------------------------------------------
    # Make the code reproducible
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    #========== Constant variables =============#
    
    xlabels_anatomy = read_labels(os.path.join("/data","fitting", "mechanics_anatomy_labels.txt"))
    xlabels_EP = read_labels(os.path.join("/data","fitting", "mechanics_EP_funct_labels_latex.txt"))
    xlabels_mechanics = read_labels(os.path.join("/data","fitting", "mechanics_funct_labels.txt"))

    xlabels = [lab for sublist in [xlabels_anatomy,xlabels_EP,xlabels_mechanics] for lab in sublist]
    
    idx_train = round(0.8*0.8*n_samples)

    active_features = ["LVV","RVV","LAV","RAV","LVOTdiam","RVOTdiam","LVmass",
                        "LVWT","LVEDD","SeptumWT","RVlongdiam","RVbasaldiam",
                        "TAT","TATLVendo","xxxx","xxxxx","xxxx"]
    emulator = []

    n_tests = int(1e5)
    path_match = os.path.join("/data","fitting", "match")
    exp_mean = np.loadtxt(os.path.join(path_match, "exp_mean_anatomy_EP_mechanics.txt"), dtype=float)
    exp_std = np.loadtxt(os.path.join(path_match, "exp_std_anatomy_EP_mechanics.txt"), dtype=float)
    exp_var = np.power(exp_std, 2)

    #================ Load training sets or run simulations =================#
    if run_simulations:
        if num_wave == 0:
            mechanics.input(n_samples = n_samples/(0.8*0.8), waveno = num_wave, subfolder = subfolder)
        else:
            mechanics.preprocess_input(waveno = num_wave, subfolder = subfolder)

        mechanics.build_meshes(waveno = num_wave, subfolder = subfolder)
        anatomy.EP_setup(waveno = num_wave, subfolder = subfolder)
        anatomy.EP_simulations(waveno = num_wave, subfolder = subfolder)

        anatomy.write_output_casewise(waveno = num_wave, subfolder = subfolder)
        anatomy.collect_output(waveno = num_wave, subfolder = subfolder)

    X = np.loadtxt(os.path.join("/data","fitting",subfolder,"wave0","X.dat"), dtype=float)
    X_train = X[:idx_train]
    I = get_minmax(X_train)


    #=================== Train or load GPE =========================#
    for output_name in active_features:
        _, _, emul = run_GPE(waveno = num_wave, train = train_GPE,
                            active_feature = [output_name],
                            n_samples = n_samples,
                            training_set_memory = training_set_memory,
                            subfolder = subfolder, only_feasible=only_feasible)
        emulator.append(emul)
    # emulator now might be a list of list, we need to flatten it
    em_shape = (np.array(emulator)).shape
    if len(em_shape) > 1:
        emulator_flat = [item for sublist in emulator for item in sublist]
        emulator = emulator_flat
    #=================== Load or create the wave object ===================#
    if num_wave > 0:
        W = hm.Wave()
        W.load(os.path.join("/data","fitting",subfolder,"wave" + str(num_wave-1),
                    "wave_" + str(num_wave-1)))
        W.emulator = emulator
        W.Itrain = I
        W.cutoff = cutoff
        W.maxno = 1
        W.mean = exp_mean
        W.var = exp_var
    else:
        W = hm.Wave(emulator = emulator,
                    Itrain = I,
                    cutoff = cutoff,
                    maxno = 1,
                    mean = exp_mean,
                    var = exp_var)



    #============= Load or run the cloud technique =========================#

    param_ranges_lower_anatomy = np.loadtxt(os.path.join(path_match, "anatomy_input_range_lower.dat"), dtype=float)
    param_ranges_upper_anatomy = np.loadtxt(os.path.join(path_match, "anatomy_input_range_upper.dat"), dtype=float)

    param_ranges_lower_EP = np.loadtxt(os.path.join(path_match, "EP_input_range_lower.dat"), dtype=float)
    param_ranges_upper_EP = np.loadtxt(os.path.join(path_match, "EP_input_range_upper.dat"), dtype=float)

    param_ranges_lower = np.append(param_ranges_lower_anatomy, param_ranges_lower_EP)
    param_ranges_upper = np.append(param_ranges_upper_anatomy, param_ranges_upper_EP)

    param_ranges = np.column_stack((param_ranges_lower, param_ranges_upper))
    if fill_wave_space:
        print("Experiment " + subfolder + ", wave " + str(num_wave))
        print("Adding points at " + time.strftime("%H:%M",time.localtime()))
        if num_wave == 0:
            TESTS = lhd(param_ranges, n_tests, SEED)
            
        else:
            TESTS = modified_add_points(W = W, n_tests = n_tests, scale = 0.05, param_ranges = param_ranges)
            # TESTS = W.add_points(n_tests, scale = 0.05, param_ranges = param_ranges)  # use the "cloud technique" to populate
        print("Points added")
        # what is left from W.NIMP\SIMULS (set difference) if points left are <
        # the chosen n_tests

        np.savetxt(os.path.join("/data","fitting",subfolder,"wave" + str(num_wave),"X_test.dat"), TESTS, fmt="%.2f")
    else:
        TESTS = np.loadtxt(os.path.join("/data","fitting",subfolder,"wave" + str(num_wave),"X_test.dat"),
                        dtype=float)


    #============= We finally print and show the wave we wanted =============#
    W.find_regions(TESTS)  # enforce the implausibility criterion to detect regions of
                            # non-implausible and of implausible points
    W.print_stats()  # show statistics about the two obtained spaces
    
    nimp = len(W.nimp_idx)
    imp = len(W.imp_idx)
    tests = nimp + imp
    perc = 100 * nimp / tests

    np.savetxt(os.path.join("/data","fitting",subfolder,"wave" + str(num_wave),"NROY_rel.dat"), [perc], fmt='%.2f')

    pathlib.Path(os.path.join("/data","fitting",subfolder,"figures")).mkdir(parents=True, exist_ok=True)

    print("Printing impl. min...")
    custom_plots.plot_wave(W = W, xlabels = xlabels,
                            filename = os.path.join("/data","fitting",
                            subfolder,"figures","wave" + str(num_wave) + "_impl_min"),
                            waveno = num_wave, reduction_function = "min",
                            plot_title = subfolder + ", wave " + str(num_wave) + ": taking the min of each slice",
                            param_ranges = param_ranges)
    print("Printing impl. max...")
    custom_plots.plot_wave(W = W, xlabels = xlabels,
                            filename = os.path.join("/data","fitting",
                            subfolder,"figures","wave" + str(num_wave) + "_impl_max"),
                            waveno = num_wave, reduction_function = "max",
                            plot_title = subfolder + ", wave " + str(num_wave) + ": taking the max of each slice",
                            param_ranges = param_ranges)
    print("Printing variance...")
    W.plot_wave(xlabels=xlabels, display="var", filename=os.path.join("/data","fitting",subfolder,"figures","wave" + str(num_wave) + "_var.png"))

    #=================== Generate data for next wave =====================#
    SIMULS = W.get_points(generate_simul_pts)  # actual matrix of selected points

    pathlib.Path(os.path.join("/data","fitting",subfolder,"wave" + str(num_wave + 1))).mkdir(parents=True, exist_ok=True)
    np.savetxt(os.path.join("/data","fitting",subfolder,"wave" + str(num_wave + 1),"X.dat"), SIMULS, fmt="%.2f")

    W.save(os.path.join("/data","fitting",subfolder,"wave" + str(num_wave),
                    "wave_" + str(num_wave)))
    np.savetxt(os.path.join("/data","fitting",subfolder,"wave" + str(num_wave),"variance_quotient.dat"), W.PV, fmt="%.2f")

if __name__ == "__main__":

    num_input_param = 14

    anatomy_new_wave(num_wave = 0, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.2, n_samples = int(14*20),
                generate_simul_pts = int(14*10), subfolder = "anatomy",
                training_set_memory = 2)

    # original_training_set_size = 400
    # template_EP.EP_funct_param(original_training_set_size)
    # template_EP.template_EP_parallel(line_from = 312, line_to = original_training_set_size - 1, waveno = 0)
    # template_EP.EP_output(at_least, waveno = 0)
    # template_EP.filter_output(waveno = 0)

    # for num_wave in range(1,11):
    #     run_new_wave(num_wave = num_wave, run_simulations = False, train_GPE = False, fill_wave_space = False, cutoff = 2.0)
    
    # subfolders_array = ["coveney","longobardi","experiment_3","experiment_4","experiment_5","experiment_6","experiment_7"]
    # number_waves = np.array([9,5,6,6,9,5,5])

    # for i in range(len(number_waves)):
    #     plot_percentages_NROY(subfolder = subfolders_array[i], last_wave = number_waves[i])
        # plot_waves(wavesno = number_waves[i], subfolder = subfolders_array[i])
        # plot_var_quotient(first_wave = 0, last_wave = number_waves[i], subfolder =subfolders_array[i])
        # plot_output_evolution_complete(first_wave = 0, last_wave = number_waves[i], subfolder =subfolders_array[i])
        # plot_output_evolution(first_wave = 0, last_wave = number_waves[i], subfolder = subfolders_array[i])

    # GSA(emul_num = 6, feature = "TAT", generate_Sobol = True, subfolder ="coveney")
    # GSA(emul_num = 6, feature = "TATLV", generate_Sobol = True, subfolder ="coveney")

    # GSA(emul_num = 0, feature = "TAT", generate_Sobol = True, subfolder ="longobardi")
    # GSA(emul_num = 0, feature = "TATLV", generate_Sobol = True, subfolder ="longobardi")

    # GSA(emul_num = 4, feature = "TAT", generate_Sobol = True, subfolder ="experiment_3")
    # GSA(emul_num = 4, feature = "TATLV", generate_Sobol = True, subfolder ="experiment_3")

    # GSA(emul_num = 7, feature = "TAT", generate_Sobol = True, subfolder ="experiment_5")
    # GSA(emul_num = 7, feature = "TATLV", generate_Sobol = True, subfolder ="experiment_5")

    # GSA(emul_num = 5, feature = "TAT", generate_Sobol = True, subfolder ="experiment_6")
    # GSA(emul_num = 5, feature = "TATLV", generate_Sobol = True, subfolder ="experiment_6")

    # custom_plots.GSA(emul_num = 4, feature = "TAT", generate_Sobol = False, subfolder ="experiment_7")
    # custom_plots.GSA(emul_num = 4, feature = "TATLV", generate_Sobol = False, subfolder ="experiment_7")

    # Xdata =  np.loadtxt(os.path.join("/data","fitting","coveney","wave0", "X_test.dat"),dtype=float)
    # xlabels = read_labels(os.path.join("/data","fitting", "EP_funct_labels_latex.txt"))
    # # plot_output_evolution(first_wave = 0, last_wave = 9, subfolder = "coveney")
    # W = hm.Wave()
    # W.load(os.path.join("/data","fitting","coveney","wave0","wave_0"))

    # # X_test = W.reconstruct_tests()
    # # print(W.__dict__.keys())
    # Impl = W.I
    # cutoff = W.cutoff
    
    # custom_plots.plot_dataset_modified(Xdata, xlabels, Impl, cutoff)
    # custom_plots.plot_dataset_modified(W.IMP, xlabels, Impl[W.imp_idx], cutoff)
    # custom_plots.plot_dataset_modified(W.NIMP, xlabels, Impl[W.nimp_idx], cutoff)
    # custom_plots.plot_dataset_modified(np.vstack([W.NIMP,W.IMP]), xlabels, np.append(Impl[W.nimp_idx],Impl[W.imp_idx]), cutoff)
    # plot_impl_binary(W,xlabels)
    # W.print_stats()

    # plot_accumulated_waves_points(xlabels)

    # run_new_wave(num_wave = 4, run_simulations = False, train_GPE = False,
    #             fill_wave_space = False, cutoff = 3., n_samples = 100,
    #             generate_simul_pts = 50, subfolder = "experiment_7", training_set_memory = 2)