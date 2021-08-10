import csv
from datetime import date
from functools import partial
from itertools import combinations
import numpy as np
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.gridspec as grsp
import multiprocessing
import scipy
from scipy.stats import iqr
import random
import torch
import os
from SALib.analyze import sobol
from SALib.sample import saltelli
from scipy.special import binom
import seaborn as sns
import tqdm
import pandas as pd

from Historia.shared.design_utils import read_labels, get_minmax
from gpytGPE.gpe import GPEmul
from gpytGPE.utils.plotting import gsa_box, gsa_donut, gsa_network, gsa_heat, correct_index
from Historia.history import hm
from Historia.shared.plot_utils import interp_col, get_col

import fitting_hm

from global_variables_config import *
# ----------------------------------------------------------------
# Make the code reproducible
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def plot_wave(W, xlabels=None, filename="./wave_impl",
             reduction_function = "min", plot_title = "Implausibility space",
             param_ranges = None):
    """Function to save a plot of a specific wave with the colorscale of the
    implausibility. NROY regions are grey.

    Args:
        W (wave object): wave to plot
        xlabels (array of string, optional): Array with the labels for the 
        input parameters. Defaults to None.
        filename (str, optional): Name of the file where the plot is saved. 
        Defaults to "./wave_impl".
        reduction_function (str, optional): For every pair of points there are
        infinite points considering the other parameter values. This parameter
        specifies how to color it. Defaults to "min", so plot the point with 
        the minimum implausibility. prob_IMP is another option, in which case
        it plot the percentage of points "behind" it that are implausible.
        plot_title (str, optional): Title of the plot. Defaults to 
        "Implausibility space".
        param_ranges (matrix of lists, optional): Param ranges to set the 
        axis correctly. If None, it sets the training points range. Defaults to 
        None.
    """

    X = W.reconstruct_tests()

    if xlabels is None:
        xlabels = [f"p{i+1}" for i in range(X.shape[1])]

    C = W.I
    cmap = "jet"
    vmin = 1.0
    vmax = W.cutoff

    height = 9.36111
    width = 5.91667
    fig = plt.figure(figsize=(3 * width, 3 * height / 3))
    gs = grsp.GridSpec(
        W.input_dim - 1,
        W.input_dim,
        width_ratios=(W.input_dim - 1) * [1] + [0.1],
    )

    for k in range(W.input_dim * W.input_dim):
        i = k % W.input_dim
        j = k // W.input_dim

        if i > j:
            axis = fig.add_subplot(gs[i - 1, j])
            axis.set_facecolor("xkcd:light grey")

            if (xlabels[i] == "$\mathregular{k_{fibre}}$") or (xlabels[j] == "$\mathregular{k_{fibre}}$"):
                hexagon_size = 9
            else:
                hexagon_size = 25
            
            if reduction_function == "min":
                cbar_label = "Implausibility measure"
                im = axis.hexbin(
                    X[:, j],
                    X[:, i],
                    C=C,
                    reduce_C_function=np.min,
                    gridsize=hexagon_size,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
            if reduction_function == "max":
                cbar_label = "Implausibility measure"
                im = axis.hexbin(
                    X[:, j],
                    X[:, i],
                    C=C,
                    reduce_C_function=np.max,
                    gridsize=hexagon_size,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
            if reduction_function == "prob_IMP":
                cbar_label = "Implasible points (%)"
                def prob_IMP(x, threshold):
                    return 100*sum(i >= threshold for i in x)/len(x)

                reduce_function = partial(prob_IMP, threshold=vmax)

                im = axis.hexbin(
                    X[:, j],
                    X[:, i],
                    C=C,
                    reduce_C_function=reduce_function,
                    gridsize=hexagon_size,
                    cmap=cmap,
                    vmin=0,
                    vmax=100,
                )

            if param_ranges is None:
                param_ranges = W.Itrain

            axis.set_xlim([param_ranges[j, 0], param_ranges[j, 1]])
            axis.set_ylim([param_ranges[i, 0], param_ranges[i, 1]])

            if i == W.input_dim - 1:
                axis.set_xlabel(xlabels[j], fontsize=12)
            else:
                axis.set_xticklabels([])
            if j == 0:
                axis.set_ylabel(xlabels[i], fontsize=12)
            else:
                axis.set_yticklabels([])

    cbar_axis = fig.add_subplot(gs[:, W.input_dim - 1])
    cbar = fig.colorbar(im, cax=cbar_axis, format = '%.2f')
    cbar.set_label(cbar_label, size=12)
    fig.tight_layout()
    plt.suptitle(plot_title, fontsize = 18)
    plt.savefig(filename + ".png", bbox_inches="tight", dpi=300)

def plot_var_quotient(first_wave = 0, last_wave = 9, subfolder = ".",
                        plot_title = "Evolution of variance quotient"):
    """Function to plot the uncertainty quantification of an emulator. The 
    quotient plotted corresponds to the highest quotient of the variance of the
    emulator over the variance of the output.

    Args:
        first_wave (int, optional): First wave to plot. Defaults to 0.
        last_wave (int, optional): Last wave to plot. Defaults to 9.
        subfolder (str, optional): Subfolder to work on. Defaults to ".".
        plot_title (str, optional): Title of the plot. Defaults to "Evolution of
        variance quotient".
    """

    matplotlib.rcParams.update({'font.size': 22})

    max_var_quotient_vec = []
    median_var_quotient_vec = []
    mean_var_quotient_vec = []

    for i in range(first_wave,last_wave + 1):
        emulator = []
        W_PV = np.loadtxt(os.path.join("/data","fitting",subfolder,"wave" + str(i),"variance_quotient.dat"),
                        dtype=float)

        max_var_quot = np.min([np.percentile(W_PV, 75) + 1.5 * iqr(W_PV),
                                W_PV.max()])
        median_var_quot = np.median(W_PV)
        mean_var_quot = np.mean(W_PV)

        max_var_quotient_vec.append(max_var_quot)
        median_var_quotient_vec.append(median_var_quot)
        mean_var_quotient_vec.append(mean_var_quot)

    height = 9.36111
    width = 5.91667
    fig = plt.figure(figsize=(3 * width, 3 * height / 3))
    plt.plot(range(first_wave,last_wave+1),max_var_quotient_vec, '.r-',
            label = "Upper whisker", markersize = 16)
    plt.plot(range(first_wave,last_wave+1),median_var_quotient_vec,'.k-',
            label = "Median", markersize = 16)
    plt.plot(range(first_wave,last_wave+1),mean_var_quotient_vec,'.b-',
            label = "Mean", markersize = 16)
    plt.title(plot_title)
    plt.xlabel("wave")
    plt.ylabel("GPE variance / EXP. variance")
    plt.legend(loc='upper right')
    fig.tight_layout()
    plt.savefig(os.path.join("/data","fitting",subfolder,"figures","variance_quotient.png"), bbox_inches="tight", dpi=300)

def plot_output_evolution_seaborn(first_wave = 0, last_wave = 9,
                                   subfolder = ".", only_feasible = True,
                                   output_labels_dir = "",
                                   exp_mean_name = "",
                                   exp_std_name = "",
                                   units_dir = ""):
    """Function to plot the evolution of the output (both simulations and 
    emulations).

    Args:
        first_wave (int, optional): First wave to plot. Defaults to 0.
        last_wave (int, optional): Last wave to plot. Defaults to 9.
        subfolder (str, optional): Subfolder to work on. Defaults to ".".
        only_feasible (bool, optional): If True, works only with feasible points
        (within 5SD of the mean). Defaults to True.
        output_labels_dir (str, optional): Directory where the file containing
        the names of the output labels is. Defaults to "".
        exp_mean_name (str, optional): Name of the file containing the 
        experimental mean value. Defaults to "".
        exp_std_name (str, optional): Name of the file containing the
        experimental standard deviation value. Defaults to "".
        units_dir (str, optional): Directory where the file containing the 
        labels for the units of the output is. Defaults to "".
    """
    matplotlib.rcParams.update({'font.size': 9})
    SEED = 2
    # ----------------------------------------------------------------
    # Make the code reproducible
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    exp_means = np.loadtxt(os.path.join("/data","fitting","match", exp_mean_name), dtype=float)
    exp_stds = np.loadtxt(os.path.join("/data","fitting","match", exp_std_name), dtype=float)
    output_labels = read_labels(output_labels_dir)

    if units_dir != "":
        units = read_labels(units_dir)

    for i in range(len(output_labels)):
        if only_feasible:
            X_name = "X_feasible.dat"
            y_name = output_labels[i] + "_feasible.dat"
        else:
            X_name = "X.dat"
            y_name = output_labels[i] + ".dat"

        fig, ax = plt.subplots()
        
        data_for_df = []
        min_value_axis = 1e5
        max_value_axis = -1e5

        for w in range(first_wave, last_wave + 1):

            X_test = np.loadtxt(os.path.join("/data","fitting",subfolder,"wave" + str(w), "X_test.dat"),dtype=float)
            X_train = np.loadtxt(os.path.join("/data","fitting",subfolder,"wave" + str(w), X_name),dtype=float)
            y_train = np.loadtxt(os.path.join("/data","fitting",subfolder,"wave" + str(w), y_name),dtype=float)
            
            emul = GPEmul.load(X_train, y_train, loadpath=os.path.join("/data","fitting",subfolder,"wave" + str(w) + "/"),filename = "wave" + str(w) + "_" + output_labels[i] + ".gpe")
            emulated_means, _ = emul.predict(X_test)

            min_value_axis = min(min_value_axis,min(y_train))
            min_value_axis = min(min_value_axis,min(emulated_means))

            max_value_axis = max(max_value_axis,max(y_train))
            max_value_axis = max(max_value_axis,max(emulated_means))

            for mean_value in emulated_means:
                data_for_df.append([mean_value, 'emulated', w, None])
            for training_point in y_train:
                data_for_df.append([training_point, 'simulated', w, w-0.02])
            

        df = pd.DataFrame(data_for_df, columns=["value", "emul_simul", "wave", "moved_wave"])

        sns.violinplot(data=df, x="wave", y="value", hue="emul_simul", hue_order= ["simulated", "emulated"],
               split=True, inner=None, linewidth = 0, bw = 1, cut = 0, scale = "width", 
               palette={"emulated": "dodgerblue", "simulated": "blue"}, saturation = 0.75)
        sns.scatterplot(data = df, x = "moved_wave", y = "value",size=3,
                        hue = "emul_simul", linewidth = 0.1,
                        palette={"emulated": "black", "simulated": "black"})
        
        ax.fill_between(np.array([first_wave-0.5, len(range(first_wave,last_wave)) + 0.5]),
                        max(0,exp_means[i] - 3*exp_stds[i]),
                        exp_means[i] + 3*exp_stds[i],
                        facecolor='gray', alpha=0.2)

        min_value_axis = min(min_value_axis, max(0,exp_means[i] - 3*exp_stds[i]))
        max_value_axis = max(max_value_axis, exp_means[i] + 3*exp_stds[i])
        

        legend_3SD, = ax.fill(np.NaN, np.NaN, 'gray', alpha=0.2, linewidth=0)
        emul_box, = ax.fill(np.NaN, np.NaN, 'dodgerblue', alpha=0.75, linewidth=0)
        simul_box, = ax.fill(np.NaN, np.NaN, 'blue', alpha=0.75, linewidth=0)

        hor_line = plt.axhline(y=exp_means[i], c='red', linestyle='dashed', label="Exp. mean")


        ax.legend([hor_line, legend_3SD, emul_box, simul_box],
                    ["Exp. mean", r'Exp. mean $\pm 3$SD',"Emulation mean of the previous NROY", "Added training points"])

        plt.title("Distribution of the outputs for " + output_labels[i])
        plt.xlabel("Wave")
        if units_dir == "":
            plt.ylabel("ms")
        else:
            plt.ylabel(units[i])
        plt.xlim([first_wave-0.5, len(range(first_wave,last_wave)) + 0.5])
        plt.ylim([1.1*min_value_axis,1.1*max_value_axis])
        plt.xticks(np.arange(first_wave, len(range(first_wave,last_wave)) + 1, 1.0))
        ax.set_xticklabels(np.arange(first_wave,len(range(first_wave,last_wave)) + 1, 1.0))
        fig.tight_layout()
        plt.savefig(os.path.join("/data","fitting",subfolder,"figures", output_labels[i] + "_complete_evolution.png"), bbox_inches="tight", dpi=300)

def plot_dataset_modified(Xdata, xlabels, Impl, cutoff):
    """Plot X high-dimensional dataset by pairwise plotting its features against each other.
    Args:
        Xdata (matrix): n*m matrix
        xlabels (array): list of m strings representing the name of X dataset's 
        features.
        Impl (array): Array with the implausibility value for each point.
        cutoff (float): Threshold to set what's implausible and what's not.
    """

    sample_dim = Xdata.shape[0]
    in_dim = Xdata.shape[1]
    # Impl[np.isinf(Impl)] = cutoff + 1


    in_dim = 2

    _, axes = plt.subplots(
        nrows=in_dim,
        ncols=in_dim,
        sharex="col",
        sharey="row",
        figsize=(20, 11.3),
    )
    for i, axis in enumerate(axes.flatten()):
        axis.scatter(
            Xdata[:, i % in_dim], Xdata[:, i // in_dim], c=Impl, vmin=1, vmax=cutoff, cmap="jet",
            s = 1, marker = 's'
        )
        if i // in_dim == in_dim - 1:
            axis.set_xlabel(xlabels[i % in_dim])
        if i % in_dim == 0:
            axis.set_ylabel(xlabels[i // in_dim])

    for i in range(in_dim):
        for j in range(in_dim):
            if i < j:
                axes[i, j].set_visible(False)
    plt.suptitle(
        "Sample dimension = {} points".format(sample_dim),
        x=0.1,
        y=0.95,
        ha="left",
        va="top",
    )
    plt.show()
    return

def plot_accumulated_waves_points(subfolder, last_wave):
    """Function to plot the evolution of all the waves in a single plot.

    Args:
        subfolder (str): Subfolder to work on.
        last_wave (int): Last wave to add to the plot. It will always add the 
        first ones.
    """
    
    XL = []
    xlabels = read_labels(os.path.join("/data","fitting", "EP_funct_labels_latex.txt"))
    for counter, idx in enumerate(range(last_wave+1)):
        print(f"\nLoading wave {idx}...")
        W = hm.Wave()
        path_to_waves = os.path.join("/data","fitting",subfolder,"wave" + str(idx),"wave_" + str(idx))
        W.load(path_to_waves)
        W.print_stats()
        if counter == 0:
            X_test = W.reconstruct_tests()
            XL.append(X_test)
        else:
            XL.append(W.IMP)

    colors = interp_col(get_col("blue"), last_wave + 1)

    wnames = ["Initial space"] + [f"wave_{idx}" for idx in range(last_wave+1)]

    handles, labels = (0, 0)
    L = len(XL)
    in_dim = XL[0].shape[1]
    height = 9.36111
    width = 5.91667
    fig, axes = plt.subplots(
        nrows=in_dim,
        ncols=in_dim,
        sharex="col",
        sharey="row",
        figsize=(2 * width, 2 * height/2),
    )
    for t, ax in enumerate(axes.flatten()):
        i = t % in_dim
        j = t // in_dim
        if j >= i:
            # if xlabels[i] == "$\mathregular{k_{fibre}}$" or xlabels[j] == "$\mathregular{k_{fibre}}$":
            #     pointsize = 5
            # else:
            #     pointsize = 1

            sns.scatterplot(
                ax=ax,
                x=XL[0][:, i],
                y=XL[0][:, j],
                color=colors[0],
                edgecolor=colors[0],
                label=wnames[0],
                size = 3
            )
            for k in range(1, L):
                sns.scatterplot(
                    ax=ax,
                    x=XL[k][:, i],
                    y=XL[k][:, j],
                    color=colors[k],
                    edgecolor=colors[k],
                    label=wnames[k],
                    size = 3
                )
                handles, labels = ax.get_legend_handles_labels()
                ax.get_legend().remove()
        else:
            ax.set_axis_off()
        if i == 0:
            ax.set_ylabel(xlabels[j])
        if j == in_dim - 1:
            ax.set_xlabel(xlabels[i])
        if i == in_dim - 1 and j == 0:
            ax.legend(handles, labels, loc="center")

    plt.savefig(os.path.join(os.path.join("/data","fitting",subfolder,"figures"),"NROY_reduction_"  + subfolder + ".png"), bbox_inches="tight", dpi=300)
    print("Printed in " + os.path.join(os.path.join("/data","fitting",subfolder,"figures"),"NROY_reduction_"  + subfolder + ".png"))

def plot_percentages_NROY_break(subfolder = ".", last_wave = 9):
    """Function to plot the evolution of the NROY region in absolute and 
    relative terms. It breaks the y-axis to improve the aesthetics.

    Args:
        subfolder (str, optional): Subfolder to work on. Defaults to ".".
        last_wave (int, optional): Last wave to add to the plot. It will always add the 
        first ones. Defaults to 9.
    """
    matplotlib.rcParams.update({'font.size': 22})
    NROY_rel = []
    NROY_abs = []
    NROY_reduction = []
    for w in range(last_wave + 1):
        NROY_perc = np.loadtxt(os.path.join("/data","fitting",subfolder,"wave" + str(w), "NROY_rel.dat"),dtype=float)
        NROY_rel.append(float(NROY_perc))
        if w == 0:
            NROY_abs.append(float(NROY_perc))
        else:
            NROY_abs.append(1e-2*float(NROY_perc)*NROY_abs[w-1])
            NROY_reduction.append(NROY_abs[w-1]-NROY_abs[w])
    
    height = 9.36111
    width = 7.5
    fig = plt.figure(figsize=(3 * width, 3 * height / 3))


    f, (ax, ax2) = plt.subplots(2, 1, sharex=True)

    ax.plot(range(last_wave+1),NROY_abs, '.r-',
            label = "NROY region w.r.t. original space", markersize = 16)
    ax.plot(range(last_wave+1),NROY_rel,'.k-',
            label = "NROY region w.r.t. previous wave", markersize = 16)
    ax.plot(range(1,last_wave+1),NROY_reduction,'.m-',
            label = "NROY region reduction", markersize = 16)
    

    ax2.plot(range(last_wave+1),NROY_abs, '.r-',
            label = "NROY region w.r.t. original space", markersize = 16)
    ax2.plot(range(last_wave+1),NROY_rel,'.k-',
            label = "NROY region w.r.t. previous wave", markersize = 16)
    ax2.plot(range(1,last_wave+1),NROY_reduction,'.m-',
            label = "NROY region reduction", markersize = 16)

    ax.set_ylim(np.floor(min(NROY_abs)/10)*10, 100)  
    ax2.set_ylim(0, np.floor((max(NROY_reduction)+5)/5)*5)

    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs) 

    ax2.set_ylabel('%')
    ax2.yaxis.set_label_coords(0.04, 1.4, transform=fig.transFigure)
    f.subplots_adjust(top=0.8)
    f.suptitle("Evolution of NROY region size\n for " + subfolder)
    plt.xlabel("Wave")
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -1.6))
    fig.tight_layout()
    plt.savefig(os.path.join("/data","fitting",subfolder,"figures","NROY_size.png"), bbox_inches="tight", dpi=300)

def plot_percentages_NROY(subfolder = ".", last_wave = 9):
    """Function to plot the evolution of the NROY region in absolute and 
    relative terms. 

    Args:
        subfolder (str, optional): Subfolder to work on. Defaults to ".".
        last_wave (int, optional): Last wave to add to the plot. It will always add the 
        first ones. Defaults to 9.
    """
    matplotlib.rcParams.update({'font.size': 22})
    NROY_rel = []
    NROY_abs = []
    NROY_reduction = []
    for w in range(last_wave + 1):
        NROY_perc = np.loadtxt(os.path.join("/data","fitting",subfolder,"wave" + str(w), "NROY_rel.dat"),dtype=float)
        NROY_rel.append(float(NROY_perc))
        if w == 0:
            NROY_abs.append(float(NROY_perc))
        else:
            NROY_abs.append(1e-2*float(NROY_perc)*NROY_abs[w-1])
            NROY_reduction.append(NROY_abs[w-1]-NROY_abs[w])
    
    height = 9.36111
    width = 7.5
    fig = plt.figure(figsize=(3 * width, 3 * height / 3))

    plt.plot(range(last_wave+1),NROY_abs, '.r-',
            label = "NROY region w.r.t. original space", markersize = 16)
    plt.plot(range(last_wave+1),NROY_rel,'.k-',
            label = "NROY region w.r.t. previous wave", markersize = 16)
    plt.plot(range(1,last_wave+1),NROY_reduction,'.m-',
            label = "NROY region reduction", markersize = 16)
    

    plt.ylim(0, 100)  
    plt.ylabel('%')
    plt.title("Evolution of NROY region size\n for " + subfolder)
    plt.xlabel("Wave")
    plt.xticks(np.arange(0, last_wave, step=1))
    plt.legend(loc='lower left')
    plt.savefig(os.path.join("/data","fitting",subfolder,"figures","NROY_size.png"), bbox_inches="tight", dpi=300)

def GSA(emul_num = 5, feature = "TAT", generate_Sobol = False, subfolder =".",
        input_labels = []):
    """Function to perform the pie charts of the Global Sensitivity Analysis.

    Args:
        emul_num (int, optional): Number of the wave where the emulator is. 
        Defaults to 5.
        feature (str, optional): Output label of the emulator to use. Defaults 
        to "TAT".
        generate_Sobol (bool, optional): If True, generates the (computationally
        expensive) Sobol' semi-random sequence to evaluate the emulator. 
        Otherwise, it loads the first-order and total effects from file.
        Defaults to False.
        subfolder (str, optional): Subfolder to work on. Defaults to ".".
        input_labels (list, optional): List with the labels for the input
        parameters. Defaults to [].
    """

    in_out_path = os.path.join("/data","fitting",subfolder,"wave" + str(emul_num))

    # ================================================================
    # GPE loading
    # ================================================================

    X_train = np.loadtxt(os.path.join(in_out_path, "X.dat"), dtype=float)
    y_train = np.loadtxt(os.path.join(in_out_path, feature + ".dat"), dtype=float)

    emul = GPEmul.load(X_train, y_train,
                        loadpath=in_out_path + "/",
                        filename = "wave" + str(emul_num) + "_" + feature + ".gpe")

    # ================================================================
    # Estimating Sobol' sensitivity indices
    # ================================================================
    n = 1000 # Increase this to reduce the integral uncertainty. The output grows as n x (2*input + 2), so careful!
    n_draws = 1000

    D = X_train.shape[1]
    I = get_minmax(X_train)
    
    if input_labels == []:
        index_i = read_labels(os.path.join("/data","fitting","EP_funct_labels_latex.txt"))
    else:
        index_i = input_labels
    # index_ij = [f"({c[0]}, {c[1]})" for c in combinations(index_i, 2)]
    index_ij = [list(c) for c in combinations(index_i, 2)]

    if generate_Sobol:
        problem = {"num_vars": D, "names": index_i, "bounds": I}

        X_sobol = saltelli.sample(
            problem, n, calc_second_order=True
        )  # n x (2D + 2) | if calc_second_order == False --> n x (D + 2)
        Y = emul.sample(X_sobol, n_draws=n_draws)

        ST = np.zeros((0, D), dtype=float)
        S1 = np.zeros((0, D), dtype=float)
        S2 = np.zeros((0, int(binom(D, 2))), dtype=float)

        for i in tqdm.tqdm(range(n_draws)):
            S = sobol.analyze(
                problem,
                Y[i],
                calc_second_order=True,
                parallel=True,
                n_processors=multiprocessing.cpu_count(),
                seed=SEED,
            )
            total_order, first_order, (_, second_order) = sobol.Si_to_pandas_dict(S)

            ST = np.vstack((ST, total_order["ST"].reshape(1, -1)))
            S1 = np.vstack((S1, first_order["S1"].reshape(1, -1)))
            S2 = np.vstack((S2, np.array(second_order["S2"]).reshape(1, -1)))

        np.savetxt(os.path.join(in_out_path,"STi_" + feature + ".txt"), ST, fmt="%.6f")
        np.savetxt(os.path.join(in_out_path,"Si_" + feature + ".txt"), S1, fmt="%.6f")
        np.savetxt(os.path.join(in_out_path,"Sij_" + feature + ".txt"), S2, fmt="%.6f")
    else:

        ST = np.loadtxt(os.path.join(in_out_path,"STi_" + feature + ".txt"), dtype=float)
        S1 = np.loadtxt(os.path.join(in_out_path,"Si_" + feature + ".txt"), dtype=float)
        S2 = np.loadtxt(os.path.join(in_out_path,"Sij_" + feature + ".txt"), dtype=float)

    # ================================================================
    # Plotting
    # ================================================================
    thre = 1e-6

    # gsa_box(ST, S1, S2, index_i, index_ij, ylabel = feature, savepath = in_out_path + "/", correction=thre)
    # gsa_donut_anotated(ST, S1, index_i, preffix = feature + "_" + str(emul_num), savepath = in_out_path + "/../figures/", correction=thre)
    # gsa_network(ST, S1, S2, index_i, index_ij, ylabel = feature + "_" + str(emul_num), savepath = in_out_path + "/../figures/", correction=thre)
    gsa_donut_single(ST, S1, index_i, feature = feature, savepath = in_out_path + "/../figures/", correction=thre)

    # gsa_donut(ST = ST, S1 = S1, index_i = index_i, ylabel = feature, savepath = in_out_path + "/", correction=None)
    # gsa_box(ST = ST, S1 = S1, S2 = S2, index_i = index_i, index_ij = index_ij, ylabel = feature, savepath= in_out_path + "/", correction=None)
    # gsa_network(ST = ST, S1 = S1, S2 = S2, index_i = index_i, index_ij = index_ij, ylabel = feature, savepath = in_out_path + "/", correction=None)

def full_GSA(emul_num, subfolder, output_labels_dir, input_labels):
    """Function to perform the global sensitivity analysis of all the output
    values.

    Args:
        emul_num (int, optional): Number of the wave where the emulator is. 
        subfolder (str, optional): Subfolder to work on.
        output_labels_dir (str): Directory where the output labels file is.
        input_labels (list): List with the labels of the input parameters.
    """

    SEED = 2
    # ----------------------------------------------------------------
    # Make the code reproducible
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    output_labels = read_labels(output_labels_dir)
    
    for i in range(len(output_labels)):
        if os.path.isfile(os.path.join("/data","fitting",subfolder,"wave" + str(emul_num),"Sij_" + output_labels[i] + ".txt")):
            flag_Sobol = False
        else:
            flag_Sobol = True

        GSA(emul_num = emul_num, feature = output_labels[i], generate_Sobol = flag_Sobol,
        subfolder = subfolder, input_labels = input_labels)

def gsa_donut_anotated(ST, S1, index_i, preffix, savepath, correction=None):
    """Function to do a pie chart for the GSA, where the chunks are annotated
    and color-blind safe.

    Args:
        ST (numpy array): Total effects.
        S1 (numpy array): First order effects
        index_i (list): Labels of the input parameters.
        preffix (str): Preffix to add to the plotted file.
        savepath (str): Path where the plot is saved.
        correction (float, optional): Threshold that sets to 0 every value under
        it to avoid numerical issues. Defaults to None.
    """
    if correction is not None:
        ST = correct_index(ST, correction)
        S1 = correct_index(S1, correction)

    ST_mean = np.mean(ST, axis=0)
    S1_mean = np.mean(S1, axis=0)

    sum_s1 = S1_mean.sum()
    sum_st = ST_mean.sum()
    ho = sum_st - sum_s1
    x_si = np.array(list(S1_mean) + [ho])
    x_sti = ST_mean

    height = 9.36111
    width = 5.91667
    fig, axes = plt.subplots(1, 2, figsize=(2 * width, 2 * height / 4))

    # c = "blue"
    # colors = interp_col(get_col(c), len(index_i))
    # colors += [interp_col(get_col("gray"), 6)[2]]

    colors_modes_list = sns.color_palette("colorblind", 9)
    colors_EP_list = sns.color_palette("colorblind", 5)
    colors = []
    colors += [np.array(colors_modes_list[i]) for i in range(len(colors_modes_list))]
    colors += [np.array(colors_EP_list[i]) for i in range(len(colors_EP_list))]
    colors += [np.array([0,0,0])]

    all_xlabels = index_i + ["higher-order int."]

    x_si_indices_numbers = np.where(x_si>0.01*sum(x_si))
    x_si_indices = np.in1d(range(x_si.shape[0]),x_si_indices_numbers)

    x_si_combined = x_si[x_si_indices]
    x_si_combined = np.append(x_si_combined,sum(x_si[~x_si_indices]))

    colors_si_combined = np.array([], dtype=np.int64).reshape(0,3)
    for i in x_si_indices_numbers[0]:
        colors_si_combined = np.vstack([colors_si_combined,colors[i]])
    colors_si_combined = np.vstack([colors_si_combined,np.array([220/256.,220/256.,220/256.])])

    x_si_labels_combined = [np.array(all_xlabels[i]) for i in x_si_indices_numbers[0]]
    x_si_labels_combined = np.append(x_si_labels_combined,"Other factors")


    x_sti_indices_numbers = np.where(x_sti>0.01*sum(x_sti))
    x_sti_indices = np.in1d(range(x_sti.shape[0]),x_sti_indices_numbers)

    x_sti_combined = x_sti[x_sti_indices]
    x_sti_combined = np.append(x_sti_combined,sum(x_sti[~x_sti_indices]))

    colors_sti_combined = np.array([], dtype=np.int64).reshape(0,3)
    for i in x_sti_indices_numbers[0]:
        colors_sti_combined = np.vstack([colors_sti_combined,colors[i]])
    colors_sti_combined = np.vstack([colors_sti_combined,np.array([220/256.,220/256.,220/256.])])

    x_sti_labels_combined = [np.array(all_xlabels[i]) for i in x_sti_indices_numbers[0]]
    x_sti_labels_combined = np.append(x_sti_labels_combined,"Other factors")

    wedges_S1, _ = axes[0].pie(
        x_si_combined,
        radius=1,
        colors=colors_si_combined,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.3, edgecolor="w", linewidth=1),
        normalize=True,
    )
    axes[0].set_title("First order effects", fontsize=12, fontweight="bold", pad = 20)

    wedges_ST, _ =axes[1].pie(
        x_sti_combined,
        radius=1,
        colors=colors_sti_combined,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.3, edgecolor="w", linewidth=1),
        normalize=True,
    )
    axes[1].set_title("Total effects", fontsize=12, fontweight="bold", pad = 20)

    # hatches = ['x','x','x','x','x','x','x','x','x','/', '\\', '|', '-', '+','x']
    # print(wedges[0])
    # for patch, hatch in zip(wedges[0],hatches):
    #     patch.set_hatch(hatch)

    bbox_props = dict(boxstyle="square,pad=0", fc="w", ec="k", lw=0, alpha = 0)

    kw = dict(arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges_S1):
        # if (p.theta2 - p.theta1) > 5:
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        if x_si_labels_combined[i] == "Other factors":
            axes[0].annotate(x_si_labels_combined[i], xy=(x, y), xytext=(1.2*x, 1.3*y),
                    horizontalalignment=horizontalalignment, **kw, size=10)
        else:
            axes[0].annotate(x_si_labels_combined[i], xy=(x, y), xytext=(1.2*x, 1.2*y),
                    horizontalalignment=horizontalalignment, **kw, size=10)

    for i, p in enumerate(wedges_ST):
        # if (p.theta2 - p.theta1) > 5:
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        if x_sti_labels_combined[i] == "Other factors":
            axes[1].annotate(x_sti_labels_combined[i], xy=(x, y), xytext=(1.2*x, 1.3*y),
                    horizontalalignment=horizontalalignment, **kw, size=10)
        else:
            axes[1].annotate(x_sti_labels_combined[i], xy=(x, y), xytext=(1.2*x, 1.2*y),
                    horizontalalignment=horizontalalignment, **kw, size=10)


    # plt.figlegend(
    #     wedges_S1, index_i + ["higher-order int."], ncol=5, loc="lower center"
    # )
    plt.savefig(
        savepath + preffix + "_donut.png", bbox_inches="tight", dpi=300
    )

def print_ranking(emul_num, subfolder, output_labels, input_labels):
    """Function to print the ranking of the importance of the input parameters
    based on median positions across all the outputs.

    Args:
        emul_num (int): Wave number where the emulator is.
        subfolder (str): Subfolder to work on.
        output_labels (array): Labels of the output values.
        input_labels (array): Labels of the input values.
    """

    SEED = 2
    # ----------------------------------------------------------------
    # Make the code reproducible
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # output_labels = read_labels(output_labels_dir)
    ranking_matrix = np.zeros((len(output_labels), len(input_labels)))
    
    for i in range(len(output_labels)):
        if os.path.isfile(os.path.join("/data","fitting",subfolder,"wave" + str(emul_num),"Sij_" + output_labels[i] + ".txt")):
            generate_Sobol = False
        else:
            generate_Sobol = True

        in_out_path = os.path.join("/data","fitting",subfolder,"wave" + str(emul_num))
        feature = output_labels[i]

        # ================================================================
        # GPE loading
        # ================================================================

        X_train = np.loadtxt(os.path.join(in_out_path, "X.dat"), dtype=float)
        y_train = np.loadtxt(os.path.join(in_out_path, feature + ".dat"), dtype=float)

        emul = GPEmul.load(X_train, y_train,
                            loadpath=in_out_path + "/",
                            filename = "wave" + str(emul_num) + "_" + feature + ".gpe")

        # ================================================================
        # Estimating Sobol' sensitivity indices
        # ================================================================
        n = 1000 # Increase this to reduce the integral uncertainty. The output grows as n x (2*input + 2), so careful!
        n_draws = 1000

        D = X_train.shape[1]
        I = get_minmax(X_train)
        
        if input_labels == []:
            index_i = read_labels(os.path.join("/data","fitting","EP_funct_labels_latex.txt"))
        else:
            index_i = input_labels
        # index_ij = [f"({c[0]}, {c[1]})" for c in combinations(index_i, 2)]
        index_ij = [list(c) for c in combinations(index_i, 2)]

        if generate_Sobol:
            problem = {"num_vars": D, "names": index_i, "bounds": I}

            X_sobol = saltelli.sample(
                problem, n, calc_second_order=True
            )  # n x (2D + 2) | if calc_second_order == False --> n x (D + 2)
            Y = emul.sample(X_sobol, n_draws=n_draws)

            ST = np.zeros((0, D), dtype=float)
            S1 = np.zeros((0, D), dtype=float)
            S2 = np.zeros((0, int(binom(D, 2))), dtype=float)

            for i in tqdm.tqdm(range(n_draws)):
                S = sobol.analyze(
                    problem,
                    Y[i],
                    calc_second_order=True,
                    parallel=True,
                    n_processors=multiprocessing.cpu_count(),
                    seed=SEED,
                )
                total_order, first_order, (_, second_order) = sobol.Si_to_pandas_dict(S)

                ST = np.vstack((ST, total_order["ST"].reshape(1, -1)))
                S1 = np.vstack((S1, first_order["S1"].reshape(1, -1)))
                S2 = np.vstack((S2, np.array(second_order["S2"]).reshape(1, -1)))

            np.savetxt(os.path.join(in_out_path,"STi_" + feature + ".txt"), ST, fmt="%.6f")
            np.savetxt(os.path.join(in_out_path,"Si_" + feature + ".txt"), S1, fmt="%.6f")
            np.savetxt(os.path.join(in_out_path,"Sij_" + feature + ".txt"), S2, fmt="%.6f")
        else:

            ST = np.loadtxt(os.path.join(in_out_path,"STi_" + feature + ".txt"), dtype=float)
            S1 = np.loadtxt(os.path.join(in_out_path,"Si_" + feature + ".txt"), dtype=float)
            S2 = np.loadtxt(os.path.join(in_out_path,"Sij_" + feature + ".txt"), dtype=float)

        ST = correct_index(ST, 1e-6)
        S1 = correct_index(S1, 1e-6)

        ST_mean = np.mean(ST, axis=0)

        # The lower the ranking position, the more important it is

        ranking_position = np.argsort(-ST_mean)
        ranking_matrix[i,:] = ranking_position

        score_matrix = np.argsort(ranking_matrix, axis = 1)
    
    final_score = np.median(score_matrix, axis = 0)
    labels_sorted = [input_labels[int(idx)] for idx in np.argsort(final_score)]

    print("According to the median score the ranking is ")
    print(labels_sorted)
    print("...and the corresponding scores")
    print(np.sort(final_score))

def gsa_donut_single(ST, S1, index_i, feature, savepath, correction=None):
    """Function to do a pie chart for the GSA, where the chunks are annotated
    and color-blind safe. It plots only the first order effects

    Args:
        ST (numpy array): Total effects.
        S1 (numpy array): First order effects
        index_i (list): Labels of the input parameters.
        feature (str): Preffix to add to the plotted file.
        savepath (str): Path where the plot is saved.
        correction (float, optional): Threshold that sets to 0 every value under
        it to avoid numerical issues. Defaults to None.
    """

    if correction is not None:
        ST = correct_index(ST, correction)
        S1 = correct_index(S1, correction)

    ST_mean = np.mean(ST, axis=0)
    S1_mean = np.mean(S1, axis=0)

    sum_s1 = S1_mean.sum()
    sum_st = ST_mean.sum()
    ho = sum_st - sum_s1
    x_si = np.array(list(S1_mean) + [ho])

    height = 9.36111
    width = 5.91667
    fig, axes = plt.subplots()

    colors_modes_list = sns.color_palette("colorblind", 9)
    colors_EP_list = sns.color_palette("colorblind", 5)
    colors = []
    colors += [np.array(colors_modes_list[i]) for i in range(len(colors_modes_list))]
    colors += [np.array(colors_EP_list[i]) for i in range(len(colors_EP_list))]
    colors += [np.array([0,0,0])]

    all_xlabels = index_i + ["higher-order int."]

    x_si_indices_numbers = np.where(x_si>0.01*sum(x_si))
    x_si_indices = np.in1d(range(x_si.shape[0]),x_si_indices_numbers)

    x_si_combined = x_si[x_si_indices]
    x_si_combined = np.append(x_si_combined,sum(x_si[~x_si_indices]))

    colors_si_combined = np.array([], dtype=np.int64).reshape(0,3)
    for i in x_si_indices_numbers[0]:
        colors_si_combined = np.vstack([colors_si_combined,colors[i]])
    colors_si_combined = np.vstack([colors_si_combined,np.array([220/256.,220/256.,220/256.])])

    x_si_labels_combined = [np.array(all_xlabels[i]) for i in x_si_indices_numbers[0]]
    x_si_labels_combined = np.append(x_si_labels_combined,"Other factors")

    wedges_S1, _ = axes.pie(
        x_si_combined,
        radius=1,
        colors=colors_si_combined,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.3, edgecolor="w", linewidth=1),
        normalize=True,
    )
    # axes.set_title(feature, fontsize=12, fontweight="bold", pad = 20)
    axes.text(0., 0., feature, horizontalalignment='center', verticalalignment='center', fontsize=20, fontweight="bold")

    bbox_props = dict(boxstyle="square,pad=0", fc="w", ec="k", lw=0, alpha = 0)

    kw = dict(arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges_S1):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        if x_si_labels_combined[i] == "Other factors":
            axes.annotate(x_si_labels_combined[i], xy=(x, y), xytext=(1.2*x, 1.3*y),
                    horizontalalignment=horizontalalignment, **kw, size=10)
        else:
            axes.annotate(x_si_labels_combined[i], xy=(x, y), xytext=(1.2*x, 1.2*y),
                    horizontalalignment=horizontalalignment, **kw, size=10)

    plt.savefig(
        savepath + feature + "_first_order_effects.png", bbox_inches="tight", dpi=300
    )


def emulate_output_anatomy(feature, ci, normalise = False):
    # To predict: feature
    # As much input as possible

    ct_modes_csv = os.path.join("/data","fitting","match","CT_cohort_modes_weights.csv")
    path_figures = os.path.join("/data","fitting","anatomy","figures")

    all_ct_modes = np.genfromtxt(ct_modes_csv, delimiter=',', skip_header=True)
    wanted_ct_modes = all_ct_modes[:,1:10]

    ep_param_paper = np.array([80, 70, 0.8, 0.29, 7])

    output_labels = read_labels(os.path.join("/data","fitting","CT_anatomy","output_labels.txt"))
    output_units = read_labels(os.path.join("/data","fitting","CT_anatomy","output_units.txt"))

    if feature == "all":
        for feature_i in output_labels:
            emulate_output_anatomy(feature=feature_i, ci=ci, normalise=normalise)
    else:

        y_simul = np.genfromtxt(os.path.join("/data","fitting","CT_anatomy",feature + ".dat"))


        _, _, emul = fitting_hm.run_GPE(waveno=2, train=False, active_feature=[feature], n_samples=280, training_set_memory=2,
                                        subfolder="anatomy", only_feasible=False)

        prediction_input = np.hstack((wanted_ct_modes,np.tile(ep_param_paper,[wanted_ct_modes.shape[0],1])))

        y_pred_mean, y_pred_std = emul.predict(prediction_input)

        if normalise:
            y_pred_std = np.copy(y_pred_std/y_pred_mean)
            y_simul = np.copy(y_simul/y_pred_mean)
            y_pred_mean = np.copy(y_pred_mean / y_pred_mean)

        inf_bound = []
        sup_bound = []

        height = 9.36111
        width = 5.91667
        fig, axes = plt.subplots(1, 1, figsize=(2 * width, 2 * height / 4))

        inf_bound.append([(y_pred_mean - ci * y_pred_std).min(), np.nanmin(y_simul)])
        sup_bound.append([(y_pred_mean + ci * y_pred_std).max(), np.nanmax(y_simul)])

        l = np.array(range(len(y_pred_mean)))

        axes.scatter(
            np.arange(len(l)),
            y_simul[l],
            facecolors="none",
            edgecolors="C0",
            label="Reported simulations",
        )

        axes.scatter(
            np.arange(len(l)),
            y_pred_mean[l],
            facecolors="C0",
            s=16,
            label="Emulated results",
        )
        axes.errorbar(
            np.arange(len(l)),
            y_pred_mean[l],
            yerr=ci * y_pred_std[l],
            c="C0",
            ls="none",
            lw=0.5,
            label=f"uncertainty ({ci} SD)",
        )

        axes.set_xticks(np.arange(19))
        axes.set_xticklabels(np.arange(1,20))
        axes.set_xlabel("CT subject", fontsize=12)
        if not normalise:
            axes.set_ylabel(output_units[output_labels.index(feature)], fontsize=12)
            axes.set_title(feature + " emulation vs simulation results in the CT cohort | Mean std: "
                           +str(round(np.mean(y_pred_std),4)), fontsize=12,)
            figure_name = "emul_vs_simul_CT_" + feature
            axes.set_ylim([0.95 * np.nanmin(inf_bound), 1.05 * np.nanmax(sup_bound)])
        else:
            axes.set_title("Normalised " + feature + " emulation vs simulation results in the CT "
                                                     "cohort | Mean std: " +
                           str(round(np.mean(y_pred_std),4)), fontsize=12, )
            figure_name = "emul_vs_simul_CT_" + feature + "_normalised"
            axes.set_ylim([0,2])

        axes.legend(loc="upper left")


        fig.tight_layout()


        plt.savefig(os.path.join(path_figures, figure_name + ".png"), bbox_inches="tight", dpi=300)
        plt.close()


def infer_parameters_anatomy():

    # active_features = ["LVV", "RVV", "LAV", "RAV", "LVOTdiam", "RVOTdiam", "LVmass",
    #                    "LVWT", "LVEDD", "SeptumWT", "RVlongdiam", "RVbasaldiam",
    #                    "TAT", "TATLVendo"]

    active_features = ["LVV", "RVV", "LVmass", "TAT"]

    ct_simulation_output_csv = os.path.join("/data", "fitting", "match", "CT_simulation_output.csv")

    all_ct_simulation_output = np.genfromtxt(ct_simulation_output_csv, delimiter=',',
                                             skip_header=True)
    wanted_ct_simulation_output = np.insert(all_ct_simulation_output[:, 1:], 8, np.nan, axis=0)
    wanted_ct_simulation_output = np.insert(wanted_ct_simulation_output, 8, np.nan, axis=0)

    index_output_dict = {"TAT": 12,
                         "LVV": 0,
                         "LVmass": 1,
                         "RVV": 15}

    idx_to_use = [index_output_dict[feature] for feature in active_features]

    wanted_ct_simulation_output = wanted_ct_simulation_output[:,idx_to_use]

    exp_mean = np.nanmean(wanted_ct_simulation_output,axis=0)
    # exp_std = np.nanmean(wanted_ct_simulation_output,axis=0)

    wave = hm.Wave()
    wave.load(os.path.join("/data/fitting", "anatomy", "wave2", "wave_2"))

    difference_matrix = np.empty((0,len(wave.NIMP)))

    for i, output_name in enumerate(active_features):
        _, _, emul = fitting_hm.run_GPE(waveno=2, train=False, active_feature=[output_name], n_samples=280,
                             training_set_memory=2, subfolder="anatomy", only_feasible=False)
        mean_prediction, std_prediction = emul.predict(wave.NIMP)
        difference_matrix = np.vstack((difference_matrix, (exp_mean[i] - mean_prediction)/exp_mean[i]))

    matrix_to_minimise = np.linalg.norm(difference_matrix, axis=0)

    winner_idx = np.argmin(matrix_to_minimise)

    xlabels_EP = read_labels(os.path.join("/data/fitting/anatomy", "EP_funct_labels.txt"))
    xlabels_anatomy = read_labels(os.path.join("/data/fitting/anatomy", "modes_labels.txt"))
    xlabels = [lab for sublist in [xlabels_anatomy, xlabels_EP] for lab in sublist]

    for i, lab in enumerate(xlabels):
        print("Emulated " + lab + "=" + str(round(wave.NIMP[winner_idx][i],6)))


def plot_inference_emulation_discrepancy(plot_title, filename, reduction_function, max_threshold):

    features_names = ["TAT", "LVV"]
    features_values = [90., 120]

    wave = hm.Wave()
    wave.load(os.path.join("/data/fitting", "anatomy", "wave2", "wave_2"))
    X = wave.reconstruct_tests()

    difference_matrix = np.empty((0, len(X)))

    for i, output_name in enumerate(features_names):
        _, _, emul = fitting_hm.run_GPE(waveno=2, train=False, active_feature=[output_name], n_samples=280,
                                        training_set_memory=2, subfolder="anatomy", only_feasible=False)
        mean_prediction, std_prediction = emul.predict(X)
        difference_matrix = np.vstack((difference_matrix, (features_values[i] - mean_prediction) / features_values[i]))

    # C = 100*np.linalg.norm(difference_matrix, axis=0)
    C = 100 * np.linalg.norm(difference_matrix, ord=1, axis=0)

    # C = W.I



    cmap = "jet"
    vmin = 0
    vmax = max_threshold

    height = 9.36111
    width = 5.91667
    fig = plt.figure(figsize=(3 * width, 3 * height / 3))
    gs = grsp.GridSpec(
        wave.input_dim - 1,
        wave.input_dim,
        width_ratios=(wave.input_dim - 1) * [1] + [0.1],
    )

    xlabels_EP = read_labels(os.path.join(PROJECT_PATH,"anatomy", "EP_funct_labels_latex.txt"))
    xlabels_anatomy = read_labels(os.path.join(PROJECT_PATH,"anatomy", "modes_labels.txt"))
    xlabels = [lab for sublist in [xlabels_anatomy, xlabels_EP] for lab in sublist]

    path_match = os.path.join(PROJECT_PATH, "match")

    param_ranges_lower_anatomy = np.loadtxt(os.path.join(path_match, "anatomy_input_range_lower.dat"), dtype=float)
    param_ranges_upper_anatomy = np.loadtxt(os.path.join(path_match, "anatomy_input_range_upper.dat"), dtype=float)

    param_ranges_lower_EP = np.loadtxt(os.path.join(path_match, "EP_input_range_lower.dat"), dtype=float)
    param_ranges_upper_EP = np.loadtxt(os.path.join(path_match, "EP_input_range_upper.dat"), dtype=float)

    param_ranges_lower = np.append(param_ranges_lower_anatomy, param_ranges_lower_EP)
    param_ranges_upper = np.append(param_ranges_upper_anatomy, param_ranges_upper_EP)

    param_ranges = np.column_stack((param_ranges_lower, param_ranges_upper))

    for k in range(wave.input_dim * wave.input_dim):
        i = k % wave.input_dim
        j = k // wave.input_dim

        if i > j:
            axis = fig.add_subplot(gs[i - 1, j])
            axis.set_facecolor("xkcd:light grey")

            if (xlabels[i] == "$\mathregular{k_{fibre}}$") or (xlabels[j] == "$\mathregular{k_{fibre}}$"):
                hexagon_size = 9
            else:
                hexagon_size = 25

            if reduction_function == "min":
                cbar_label = "Min discrepancy (%)"
                im = axis.hexbin(
                    X[:, j],
                    X[:, i],
                    C=C,
                    reduce_C_function=np.min,
                    gridsize=hexagon_size,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
            if reduction_function == "max":
                cbar_label = "Max discrepancy (%)"
                im = axis.hexbin(
                    X[:, j],
                    X[:, i],
                    C=C,
                    reduce_C_function=np.max,
                    gridsize=hexagon_size,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )

            axis.set_xlim([param_ranges[j, 0], param_ranges[j, 1]])
            axis.set_ylim([param_ranges[i, 0], param_ranges[i, 1]])

            if i == wave.input_dim - 1:
                axis.set_xlabel(xlabels[j], fontsize=12)
            else:
                axis.set_xticklabels([])
            if j == 0:
                axis.set_ylabel(xlabels[i], fontsize=12)
            else:
                axis.set_yticklabels([])

    cbar_axis = fig.add_subplot(gs[:, wave.input_dim - 1])
    cbar = fig.colorbar(im, cax=cbar_axis, format='%.2f')
    cbar.set_label(cbar_label, size=12)
    # fig.tight_layout()
    plt.suptitle(plot_title, fontsize=18)
    plt.savefig(filename + ".png", bbox_inches="tight", dpi=300)

def plot_cv_vs_fec_template(feature = "TAT"):

    wave = hm.Wave()
    wave.load(os.path.join("/data", "fitting", "EP_template", "experiment_7_unfiltered", "wave4", "wave_4"))
    whole_space = wave.reconstruct_tests()
    fec_height_idx = 4
    cv_idx = 2

    _, _, emul = fitting_hm.run_GPE(waveno=4, train=False, active_feature=[feature], n_samples=100,
                                    training_set_memory=2, subfolder="EP_template/experiment_7_unfiltered",
                                    only_feasible=False)

    whole_space[:,0] = len(whole_space[:,0])*[np.mean(whole_space[:,0])]
    whole_space[:,3] = len(whole_space[:,3])*[np.mean(whole_space[:,3])]
    whole_space[:,1] = len(whole_space[:,4])*[np.mean(whole_space[:,1])]

    mean_prediction, _ = emul.predict(whole_space)

    cmap = "jet"
    vmin = 30
    vmax = 130


    cbar_label = "Mean " + feature + " (ms)"
    im = plt.hexbin(
        whole_space[:, cv_idx],
        whole_space[:, fec_height_idx],
        C=mean_prediction,
        reduce_C_function=np.mean,
        gridsize=30,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    plt.xlabel("CV (m/s)", fontsize=12)
    plt.ylabel("FEC height (%)", fontsize=12)

    cbar = plt.colorbar(im, format='%.2f')
    cbar.set_label(cbar_label, size=12)

    plt.suptitle("Emulated mean " + feature + " in the template mesh", fontsize=18)
    plt.show()

def plot_param_pairs_fixing_rest(feature="TAT", scenario="EP_template", fix_others=True):

    subfolder = scenario

    if scenario == "EP_template":
        subfolder += "/experiment_7_unfiltered"
        emul_num = 4
        xlabels = read_labels(os.path.join("/data","fitting",subfolder,"EP_funct_labels_latex.txt"))
        n_samples = 100
    elif scenario == "anatomy":
        emul_num = 2
        xlabels_ep = read_labels(os.path.join("/data/fitting/anatomy", "EP_funct_labels_latex.txt"))
        xlabels_anatomy = read_labels(os.path.join("/data/fitting/anatomy", "modes_labels.txt"))
        xlabels = [lab for sublist in [xlabels_anatomy, xlabels_ep] for lab in sublist]
        n_samples = 280

    else:
        print("Scenario not accepted.")
        sys.exit()

    output_labels = read_labels(os.path.join("/data", "fitting", subfolder, "output_labels.txt"))
    units = read_labels(os.path.join("/data", "fitting", subfolder, "output_units.txt"))
    cbar_label = "Mean " + feature + " (" + units[output_labels.index(feature)] + ")"
    plot_title = "Mean pairwise emulation in the last NROY region"
    filename = os.path.join("/data", "fitting", subfolder, "figures", "pairwise_emulation_" + feature)

    if not fix_others:
        filename += "_not_fixing"

    wave = hm.Wave()
    wave.load(os.path.join("/data", "fitting", subfolder, "wave" + str(emul_num), "wave_" + str(emul_num)))

    height = 9.36111
    width = 5.91667
    fig = plt.figure(figsize=(3 * width, 3 * height / 3))
    gs = grsp.GridSpec(
        wave.input_dim - 1,
        wave.input_dim,
        width_ratios=(wave.input_dim - 1) * [1] + [0.1],
    )

    _, _, emul = fitting_hm.run_GPE(waveno=emul_num, train=False, active_feature=[feature], n_samples=n_samples,
                                    training_set_memory=2, subfolder=subfolder,
                                    only_feasible=False)

    for k in range(wave.input_dim * wave.input_dim):
        i = k % wave.input_dim
        j = k // wave.input_dim

        if i > j:
            axis = fig.add_subplot(gs[i - 1, j])
            axis.set_facecolor("xkcd:light grey")

            if (xlabels[i] == "$\mathregular{k_{fibre}}$") or (xlabels[j] == "$\mathregular{k_{fibre}}$"):
                hexagon_size = 9
            else:
                hexagon_size = 25

            whole_space = wave.reconstruct_tests()

            if fix_others:
                for third_index in range(wave.input_dim):
                    if third_index != i and third_index != j:
                        whole_space[:, third_index] = len(whole_space[:, third_index]) * [np.mean(whole_space[:, third_index])]

            mean_prediction, _ = emul.predict(whole_space)

            cmap = "turbo"
            vmin = min(mean_prediction)
            vmax = max(mean_prediction)

            im = plt.hexbin(
                whole_space[:, j],
                whole_space[:, i],
                C=mean_prediction,
                reduce_C_function=np.mean,
                gridsize=hexagon_size,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )

            if i == wave.input_dim - 1:
                axis.set_xlabel(xlabels[j], fontsize=12)
            else:
                axis.set_xticklabels([])
            if j == 0:
                axis.set_ylabel(xlabels[i], fontsize=12)
            else:
                axis.set_yticklabels([])

    cbar_axis = fig.add_subplot(gs[:, wave.input_dim - 1])
    cbar = fig.colorbar(im, cax=cbar_axis, format='%.2f')
    cbar.set_label(cbar_label, size=12)
    fig.tight_layout()
    plt.suptitle(plot_title, fontsize=18)
    plt.savefig(filename + ".png", bbox_inches="tight", dpi=300)


def plot_pairwise_emulation_all(scenario, fix_others):
    subfolder = scenario

    if scenario == "EP_template":
        subfolder += "/experiment_7_unfiltered"

    feature_labels = read_labels(os.path.join("/data/fitting",subfolder,"output_labels.txt"))

    for feature in feature_labels:
        plot_param_pairs_fixing_rest(feature=feature, scenario=scenario, fix_others=fix_others)
