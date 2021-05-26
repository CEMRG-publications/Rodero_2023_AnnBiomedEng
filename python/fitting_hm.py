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
import generate

from Historia.shared.design_utils import get_minmax, lhd, read_labels
from Historia.shared.plot_utils import plot_pairwise_waves, interp_col, get_col
from gpytGPE.gpe import GPEmul
from gpytGPE.utils.metrics import IndependentStandardError as ISE
from gpytGPE.utils.metrics import ISE_bounded
from Historia.history import hm
from gpytGPE.utils.plotting import gsa_box, gsa_donut

SEED = 2
# ----------------------------------------------------------------
# Make the code reproducible
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def plot_waves(wavesno = 3, path_lab = os.path.join("/data","fitting"), subfolder = "."):
    """Function to make the triangular plot showing waveno waves at once.

    Args:
        wavesno (int, optional): Number of waves to show, including wave0.
        Defaults to 3.
        path_lab (str, optional): Path where the labels file is. Defaults to
        os.path.join("/data","fitting").
    """
    labels = read_labels(os.path.join(path_lab, "EP_funct_labels_latex.txt"))
    XL = []

    for i in range(wavesno):
        path_gpes = os.path.join(path_lab, subfolder,"wave" + str(i))
        X_test = np.loadtxt(os.path.join(path_gpes, "X_test.dat"), dtype=float)
        XL.append(X_test)

    colors = interp_col([get_col("light_blue")[0],get_col("blue")[1]], wavesno)
    wnames = ["Wave " + str(i) for i in range(wavesno)]

    outpath = os.path.join(path_lab, subfolder, "figures")
    pathlib.Path(outpath).mkdir(parents=True, exist_ok=True)

    plot_pairwise_waves(XL,colors,labels, wnames = wnames, outpath = outpath)

def first_GPE(active_features = ["TAT","TATLV"], train = False, saveplot = True,
              start_size = np.inf, figure_name = "inference_on_testset",
              return_scores = False, subfolder = "."):
    """Function to run the GPE with the original training set and testing it.

    Args:
        active_features (list, optional): List of output features to train the
        GPEs. Defaults to ["TAT","TATLV"].
        train (bool, optional): If False, loads the GPEs, otherwise it trains
        them. Defaults to False.
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

    X = np.loadtxt(os.path.join(path_gpes, "X_feasible.dat"), dtype=float)

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
        y = np.loadtxt(os.path.join(path_gpes, output_name + "_feasible.dat"),dtype=float)
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

def run_GPE(waveno = 2, train = False, active_feature = ["TAT"], n_samples = 125, training_set_memory = 100, subfolder = "."):

    SEED = 2
    # ----------------------------------------------------------------
    # Make the code reproducible
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if waveno == 0:
        X_train, y_train, emul = first_GPE(active_features = active_feature, train = train, saveplot=False, start_size = n_samples, subfolder = subfolder)
    else:
        current_X = np.loadtxt(os.path.join("/data","fitting",subfolder,"wave" + str(waveno),"X_feasible.dat"), dtype=float)
        current_y = np.loadtxt(os.path.join("/data","fitting",subfolder,"wave" + str(waveno), active_feature[0] + "_feasible.dat"),dtype=float)
        
        if training_set_memory > 0:
            X_prev, y_prev, _ = run_GPE(waveno = waveno - 1, train = False,
                                        active_feature = active_feature,
                                        n_samples = n_samples, subfolder = subfolder,
                                        training_set_memory = training_set_memory - 1)

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
                generate_simul_pts = 10, subfolder = ".", training_set_memory = 100):

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
    if run_simulations:
        train_GPE = True
    if train_GPE:
        fill_wave_space = True

    xlabels = read_labels(os.path.join("/data","fitting", "EP_funct_labels_latex.txt"))
    idx_train = round(0.8*0.8*n_samples)

    X = np.loadtxt(os.path.join("/data","fitting",subfolder,"wave0","X_feasible.dat"), dtype=float)
    X_train = X[:idx_train]
    I = get_minmax(X_train)

    active_features = ["TAT","TATLV"]
    emulator = []

    n_tests = 100000
    path_match = os.path.join("/data","fitting", "match")
    exp_mean = np.loadtxt(os.path.join(path_match, "exp_mean.txt"), dtype=float)
    exp_std = np.loadtxt(os.path.join(path_match, "exp_std.txt"), dtype=float)
    exp_var = np.power(exp_std, 2)

    #================ Load training sets or run simulations =================#
    if run_simulations:
        n_simuls_done = generate.template_EP_parallel(line_from = 0, line_to = np.inf,
                                            waveno = num_wave, subfolder = subfolder)
        generate.EP_output(waveno = num_wave, subfolder = subfolder)
        generate.filter_output(waveno = num_wave, subfolder = subfolder)

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

    if fill_wave_space:
        print("Adding points...")
        if num_wave == 0:
            TESTS = lhd(I, n_tests, SEED)
        else:
            TESTS = W.add_points(n_tests)  # use the "cloud technique" to populate
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


    W.plot_wave(xlabels=xlabels, display="impl", filename = os.path.join("/data","fitting",subfolder,"figures","wave" + str(num_wave) + "_impl"))
    W.plot_wave(xlabels=xlabels, display="var", filename = os.path.join("/data","fitting",subfolder,"figures","wave" + str(num_wave) + "_var"))

    #=================== Generate data for next wave =====================#
    SIMULS = W.get_points(generate_simul_pts)  # actual matrix of selected points

    pathlib.Path(os.path.join("/data","fitting",subfolder,"wave" + str(num_wave + 1))).mkdir(parents=True, exist_ok=True)
    np.savetxt(os.path.join("/data","fitting",subfolder,"wave" + str(num_wave + 1),"X.dat"), SIMULS, fmt="%.2f")

    W.save(os.path.join("/data","fitting",subfolder,"wave" + str(num_wave),
                    "wave_" + str(num_wave)))
    np.savetxt(os.path.join("/data","fitting",subfolder,"wave" + str(num_wave),"variance_quotient.dat"), W.PV, fmt="%.2f")

def plot_var_quotient(first_wave = 0, last_wave = 9, subfolder = "."):

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
        plt.title("Evolution of variance quotient")
        plt.xlabel("wave")
        plt.ylabel("GPE variance / EXP. variance")
        plt.legend(loc='upper right')
        fig.tight_layout()
        plt.savefig(os.path.join("/data","fitting",subfolder,"figures","variance_quotient.png"), bbox_inches="tight", dpi=300)

def plot_output_evolution(first_wave = 1, last_wave = 9,
                          add_training_points = True, subfolder = "."):

    matplotlib.rcParams.update(matplotlib.rcParamsDefault)

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

        output_list = []
        fig, ax = plt.subplots()


        for w in range(first_wave, last_wave + 1):
            new_output = []

            X_test = np.loadtxt(os.path.join("/data","fitting",subfolder,"wave" + str(w), "X_test.dat"),dtype=float)
            X_train = np.loadtxt(os.path.join("/data","fitting",subfolder,"wave" + str(w), "X_feasible.dat"),dtype=float)
            y_train = np.loadtxt(os.path.join("/data","fitting",subfolder,"wave" + str(w), output_labels[i] + "_feasible.dat"),dtype=float)
            emul = GPEmul.load(X_train, y_train, loadpath=os.path.join("/data","fitting",subfolder,"wave" + str(w) + "/"),filename = "wave" + str(w) + "_" + output_labels[i] + ".gpe")

            new_output, _ = emul.predict(X_test)
            output_list.append(new_output)

            width = 0.4
            if add_training_points:
                x_scatter = np.ones(len(y_train))*(w+1) + (np.random.rand(len(y_train))*width-width/2.)
                ax.scatter(x_scatter, y_train, color='k', s=25)


        ax.fill_between(np.array([0.5, len(output_list) + 0.5]),
                        exp_means[i] - 3*exp_stds[i],
                        exp_means[i] + 3*exp_stds[i],
                        facecolor='gray', alpha=0.2)

        ax.fill_between(np.array([0.5, len(output_list) + 0.5]),
                        exp_means[i] - 2*exp_stds[i],
                        exp_means[i] + 2*exp_stds[i],
                        facecolor='gray', alpha=0.4)


        legend_3SD, = ax.fill(np.NaN, np.NaN, 'gray', alpha=0.2, linewidth=0)
        legend_2SD, = ax.fill(np.NaN, np.NaN, 'gray', alpha=0.4, linewidth=0)

        if add_training_points:
            black_dot = matplotlib.lines.Line2D([], [], color='black',
                                                marker = 'o', linestyle='None',
                                                markersize=6)

        ax.violinplot(output_list, showmedians = True)

        
        if add_training_points:
            ax.legend([legend_2SD, legend_3SD, black_dot],[r'Exp. mean $\pm 2$SD', r'Exp. mean $\pm 3$SD', "Added training set"])
        else:
            ax.legend([legend_2SD, legend_3SD],[r'Exp. mean $\pm 2$SD', r'Exp. mean $\pm 3$SD'])
        plt.title("Distribution of the emulation outputs\n for " + output_labels[i])
        plt.xlabel("wave")
        plt.ylabel("ms")
        plt.xlim([0.5, len(output_list) + 0.5])
        plt.xticks(np.arange(1, len(output_list) + 1, 1.0))
        ax.set_xticklabels(np.arange(first_wave, last_wave + 1, 1.0))
        fig.tight_layout()
        plt.savefig(os.path.join("/data","fitting",subfolder,"figures", output_labels[i] + "_evolution.png"), bbox_inches="tight", dpi=300)

def GSA(emul_num = 5, feature = "TAT", generate_Sobol = False, subfolder ="."):

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

    index_i = read_labels(os.path.join("/data","fitting","EP_funct_labels_latex.txt"))
    index_ij = [f"({c[0]}, {c[1]})" for c in combinations(index_i, 2)]

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


    # ================================================================
    # Plotting
    # ================================================================
    thre = 1e-6

    gsa_donut(in_out_path + "/", thre, index_i, feature, savefig=True, STname = "STi_" + feature + ".txt", S1name = "Si_" + feature + ".txt", discrete_colors = True)
    gsa_box(in_out_path + "/", thre, index_i, index_ij, feature, savefig=True, STname = "STi_" + feature + ".txt", S1name = "Si_" + feature + ".txt", S2name = "Sij_" + feature + ".txt", violin = True)

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

        # axes2.set_xlim(axes1.get_xlim())

        # axes2.set_xticks(np.arange(79,313,30))
        # axes2.set_xticklabels(X2)

        plt.title("Scores with different set sizes")

        plt.ylabel("Percentage of score")
        plt.legend(loc='lower right')
        fig.tight_layout()
        plt.savefig("/data/fitting/figures/setsize.png", bbox_inches="tight", dpi=300)

if __name__ == "__main__":

    # original_training_set_size = 400
    # generate.EP_funct_param(original_training_set_size)
    # generate.template_EP_parallel(line_from = 312, line_to = original_training_set_size - 1, waveno = 0)
    # generate.EP_output(at_least, waveno = 0)
    # generate.filter_output(waveno = 0)

    # for num_wave in range(1,11):
    #     run_new_wave(num_wave = num_wave, run_simulations = False, train_GPE = False, fill_wave_space = False, cutoff = 2.0)
    #-------------------------------------------------
    """
    run_new_wave(num_wave = 1, run_simulations = False, train_GPE = True,
                fill_wave_space = True, cutoff = 4, n_samples = 150,
                generate_simul_pts = 50, subfolder = "experiment_4")
    run_new_wave(num_wave = 2, run_simulations = True, train_GPE = False,
                fill_wave_space = True, cutoff = 3.8, n_samples = 150,
                generate_simul_pts = 50, subfolder = "experiment_4")
    run_new_wave(num_wave = 3, run_simulations = True, train_GPE = False,
                fill_wave_space = True, cutoff = 3.6, n_samples = 150,
                generate_simul_pts = 50, training_set_memory = 2,
                subfolder = "experiment_4")
    run_new_wave(num_wave = 4, run_simulations = True, train_GPE = False,
                fill_wave_space = True, cutoff = 3.4, n_samples = 150,
                generate_simul_pts = 50, training_set_memory = 2,
                subfolder = "experiment_4")
    run_new_wave(num_wave = 5, run_simulations = True, train_GPE = False,
                fill_wave_space = True, cutoff = 3.2, n_samples = 150,
                generate_simul_pts = 50, training_set_memory = 2,
                subfolder = "experiment_4")
    run_new_wave(num_wave = 6, run_simulations = True, train_GPE = False,
                fill_wave_space = True, cutoff = 3., n_samples = 150,
                generate_simul_pts = 50, training_set_memory = 2,
                subfolder = "experiment_4")
    run_new_wave(num_wave = 7, run_simulations = True, train_GPE = False,
                fill_wave_space = True, cutoff = 3., n_samples = 150,
                generate_simul_pts = 50, training_set_memory = 2,
                subfolder = "experiment_4")
                """
    #-------------------------------------------------
    plot_waves(wavesno = 7, subfolder = "experiment_4")
    # plot_var_quotient(first_wave = 0, last_wave = 6, subfolder = "experiment_4")
    # plot_output_evolution(first_wave = 0, last_wave = 6, subfolder = "experiment_4")
    # GSA(emul_num = 5, feature = "TAT", generate_Sobol = False)
    # GSA(emul_num = 5, feature = "TATLV", generate_Sobol = False)