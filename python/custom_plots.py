from datetime import date
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as grsp
from scipy.stats import iqr
import random
import torch
import os
import seaborn as sns
import pandas as pd

from Historia.shared.design_utils import read_labels
from gpytGPE.gpe import GPEmul
from Historia.history import hm
from Historia.shared.plot_utils import interp_col, get_col


def plot_wave(W, xlabels=None, filename="./wave_impl", waveno = "unespecified",
             reduction_function = "min", plot_title = "Implausibility space",
             param_ranges = None):

    X = W.reconstruct_tests()

    if xlabels is None:
        xlabels = [f"p{i+1}" for i in range(X.shape[1])]

    C = W.I
    cmap = "jet"
    vmin = 1.0
    vmax = W.cutoff
    cbar_label = "Implausibility measure"

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

def plot_wave_training_points(W, xlabels=None, filename="./wave_impl", waveno = "unespecified",
             reduction_function = "min", plot_title = "Implausibility space",
             param_ranges = None, subfolder = "."):

    X = W.reconstruct_tests()

    X_train_whole = np.loadtxt(os.path.join("/data","fitting",subfolder,"wave0","X_feasible.dat"), dtype=float)
    X_train = X_train_whole[:round(0.8*0.8*X_train_whole.shape[0])]

    if xlabels is None:
        xlabels = [f"p{i+1}" for i in range(X.shape[1])]

    C = W.I
    cmap = "jet"
    vmin = 1.0
    vmax = W.cutoff
    cbar_label = "Implausibility measure"

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
            points_x = [x_row[j] for x_row in X_train]
            points_y = [x_row[i] for x_row in X_train]
            axis.scatter(points_x, points_y, c='w', s = 10, marker = 'o')



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

def plot_var_quotient(first_wave = 0, last_wave = 9, subfolder = ".", plot_title = "Evolution of variance quotient"):

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

def plot_output_evolution_complete(first_wave = 0, last_wave = 9,
                                   subfolder = "."):
    
    SEED = 2
    # ----------------------------------------------------------------
    # Make the code reproducible
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    exp_means = np.loadtxt(os.path.join("/data","fitting","match", "exp_mean.txt"), dtype=float)
    exp_stds = np.loadtxt(os.path.join("/data","fitting","match", "exp_std.txt"), dtype=float)
    output_labels = read_labels(os.path.join("/data","fitting", "EP_output_labels.txt"))

    for i in range(len(output_labels)):

        
        fig, ax = plt.subplots()
        emulation_list = []
        simulation_list = []

        for w in range(first_wave, last_wave + 1):

            X_test = np.loadtxt(os.path.join("/data","fitting",subfolder,"wave" + str(w), "X_test.dat"),dtype=float)
            X_train = np.loadtxt(os.path.join("/data","fitting",subfolder,"wave" + str(w), "X_feasible.dat"),dtype=float)
            y_train = np.loadtxt(os.path.join("/data","fitting",subfolder,"wave" + str(w), output_labels[i] + "_feasible.dat"),dtype=float)
            emul = GPEmul.load(X_train, y_train, loadpath=os.path.join("/data","fitting",subfolder,"wave" + str(w) + "/"),filename = "wave" + str(w) + "_" + output_labels[i] + ".gpe")

            emulated_means, _ = emul.predict(X_test)
            emulation_list.append(emulated_means)
            simulation_list.append(y_train)

        emul_vp = ax.violinplot(emulation_list, positions = [w+0.75 for w in range(first_wave,last_wave+1)],widths=0.5, showextrema=False)
        simul_vp = ax.violinplot(simulation_list, positions = [w+1.25 for w in range(first_wave,last_wave+1)],widths=0.5, showextrema=False)

        for pc in emul_vp['bodies']:
            pc.set_facecolor('dodgerblue')
            pc.set_alpha(0.8)
        for pc in simul_vp['bodies']:
            pc.set_facecolor('blue')
            pc.set_alpha(0.8)
        
        ax.fill_between(np.array([0.5, len(range(first_wave,last_wave+1)) + 0.5]),
                        max(0,exp_means[i] - 3*exp_stds[i]),
                        exp_means[i] + 3*exp_stds[i],
                        facecolor='gray', alpha=0.2)


        ax.fill_between(np.array([0.5, len(range(first_wave,last_wave+1)) + 0.5]),
                        max(0,exp_means[i] - exp_stds[i]**2),
                        exp_means[i] + exp_stds[i]**2,
                        facecolor='palegreen', alpha=0.2)
        

        legend_3SD, = ax.fill(np.NaN, np.NaN, 'gray', alpha=0.2, linewidth=0)
        legend_var, = ax.fill(np.NaN, np.NaN, 'palegreen', alpha=0.2, linewidth=0)
        emul_box, = ax.fill(np.NaN, np.NaN, 'dodgerblue', alpha=0.8, linewidth=0)
        simul_box, = ax.fill(np.NaN, np.NaN, 'blue', alpha=0.8, linewidth=0)


        ax.legend([legend_3SD, legend_var, emul_box, simul_box],
                    [r'Exp. mean $\pm 3$SD', r'Exp. mean $\pm $ var.',"Emulation mean of the previous NROY", "Added training points"])

        plt.title("Distribution of the outputs for " + output_labels[i])
        plt.xlabel("Wave")
        plt.ylabel("ms")
        plt.xlim([0.5, len(range(first_wave,last_wave+1)) + 0.5])
        plt.xticks(np.arange(1, len(range(first_wave,last_wave+1)) + 1, 1.0))
        ax.set_xticklabels(np.arange(first_wave, last_wave + 1, 1.0))
        fig.tight_layout()
        plt.savefig(os.path.join("/data","fitting",subfolder,"figures", output_labels[i] + "_complete_evolution.png"), bbox_inches="tight", dpi=300)

def plot_output_evolution_seaborn(first_wave = 0, last_wave = 9,
                                   subfolder = "."):
    matplotlib.rcParams.update({'font.size': 12})
    SEED = 2
    # ----------------------------------------------------------------
    # Make the code reproducible
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    exp_means = np.loadtxt(os.path.join("/data","fitting","match", "exp_mean.txt"), dtype=float)
    exp_stds = np.loadtxt(os.path.join("/data","fitting","match", "exp_std.txt"), dtype=float)
    output_labels = read_labels(os.path.join("/data","fitting", "EP_output_labels.txt"))

    for i in range(len(output_labels)):

        
        fig, ax = plt.subplots()
        
        data_for_df = []

        for w in range(first_wave, last_wave + 1):

            X_test = np.loadtxt(os.path.join("/data","fitting",subfolder,"wave" + str(w), "X_test.dat"),dtype=float)
            X_train = np.loadtxt(os.path.join("/data","fitting",subfolder,"wave" + str(w), "X_feasible.dat"),dtype=float)
            y_train = np.loadtxt(os.path.join("/data","fitting",subfolder,"wave" + str(w), output_labels[i] + "_feasible.dat"),dtype=float)
            
            emul = GPEmul.load(X_train, y_train, loadpath=os.path.join("/data","fitting",subfolder,"wave" + str(w) + "/"),filename = "wave" + str(w) + "_" + output_labels[i] + ".gpe")
            emulated_means, _ = emul.predict(X_test)

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


        ax.fill_between(np.array([first_wave-0.5, len(range(first_wave,last_wave)) + 0.5]),
                        max(0,exp_means[i] - exp_stds[i]**2),
                        exp_means[i] + exp_stds[i]**2,
                        facecolor='palegreen', alpha=0.2)
        

        legend_3SD, = ax.fill(np.NaN, np.NaN, 'gray', alpha=0.2, linewidth=0)
        legend_var, = ax.fill(np.NaN, np.NaN, 'palegreen', alpha=0.2, linewidth=0)
        emul_box, = ax.fill(np.NaN, np.NaN, 'dodgerblue', alpha=0.75, linewidth=0)
        simul_box, = ax.fill(np.NaN, np.NaN, 'blue', alpha=0.75, linewidth=0)


        ax.legend([legend_3SD, legend_var, emul_box, simul_box],
                    [r'Exp. mean $\pm 3$SD', r'Exp. mean $\pm $ var.',"Emulation mean of the previous NROY", "Added training points"])

        plt.title("Distribution of the outputs for " + output_labels[i])
        plt.xlabel("Wave")
        plt.ylabel("ms")
        plt.xlim([first_wave-0.5, len(range(first_wave,last_wave)) + 0.5])
        plt.ylim([-10,200])
        plt.xticks(np.arange(first_wave, len(range(first_wave,last_wave)) + 1, 1.0))
        ax.set_xticklabels(np.arange(first_wave,len(range(first_wave,last_wave)) + 1, 1.0))
        fig.tight_layout()
        plt.savefig(os.path.join("/data","fitting",subfolder,"figures", output_labels[i] + "_complete_evolution.png"), bbox_inches="tight", dpi=300)

def plot_scores_training_size():

        matplotlib.rcParams.update({'font.size': 22})

        R2_TAT = np.loadtxt(os.path.join("/data","fitting","figures","R2_TAT.dat"),
                            dtype=float)
        R2_TATLV = np.loadtxt(os.path.join("/data","fitting","figures","R2_TATLV.dat"),
                            dtype=float)
        ISE_TAT = np.loadtxt(os.path.join("/data","fitting","figures","ISE_TAT.dat"),
                            dtype=float)
        ISE_TATLV = np.loadtxt(os.path.join("/data","fitting","figures","ISE_TAT.dat"),
                            dtype=float)

        height = 9.36111
        width = 5.91667
        fig = plt.figure(figsize=(3 * width, 3 * height / 3))

        axes1 = plt.gca()
        axes2 = axes1.twiny()

        X1 = np.arange(79,313,1)
        X2 = 0.8*0.8*X1

        axes1.set_xlabel("Total set size")
        axes2.set_xlabel("Training set size")


        plt.plot(X1,100*R2_TAT, '.k-',
                label = "R2 score for the TAT emulator", markersize = 16)
        plt.plot(X1,100*R2_TATLV, '.r-',
                label = "R2 score for the TATLV emulator", markersize = 16)
        plt.plot(X1,ISE_TAT, 'k--',
                label = "ISE score for the TAT emulator", markersize = 16)
        plt.plot(X1,ISE_TATLV, 'r--',
                label = "ISE score for the TATLV emulator", markersize = 16)

        plt.title("Scores with different set sizes")

        plt.ylabel("Percentage of score")
        plt.legend(loc='lower right')
        fig.tight_layout()
        plt.savefig("/data/fitting/figures/setsize.png", bbox_inches="tight", dpi=300)

def plot_dataset_modified(Xdata, xlabels, Impl, cutoff):
    """Plot X high-dimensional dataset by pairwise plotting its features against each other.
    Args:
            - Xdata: n*m matrix
            - xlabels: list of m strings representing the name of X dataset's features.
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

def plot_percentages_NROY(subfolder = ".", last_wave = 9):
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