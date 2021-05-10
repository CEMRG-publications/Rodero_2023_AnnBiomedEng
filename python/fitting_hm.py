from pickle import FALSE, TRUE
import random
import numpy as np
import torch
import torchmetrics
import os
import pathlib
import matplotlib.pyplot as plt


from Historia.shared.design_utils import get_minmax, lhd, read_labels
from Historia.shared.plot_utils import plot_pairwise_waves
from gpytGPE.gpe import GPEmul
from gpytGPE.utils.metrics import IndependentStandardError as ISE
from gpytGPE.utils.metrics import ISE_bounded
from Historia.history import hm
import generate


SEED = 2

# ----------------------------------------------------------------
# Flags

wave0_generate_input = False
wave0_run_simulations = False
wave0_generate_output = False
wave0_train_GPE = False
test_GPE_wave0 = False
wave0 = False

gpe_wave1 = False

wave1_generate_test = False
wave1_run_simulations = False
wave1_generate_output = False
wave1_train_GPE = False
wave1 = False

gpe_wave2 = False

wave2_generate_test = False
wave2_run_simulations = False
wave2_generate_output = False
wave2_train_GPE = False
wave2 = False

# ----------------------------------------------------------------
# Make the code reproducible
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------------------------------------------------
# Paths
waveno = 0

path_match = os.path.join("/data","fitting", "match")
path_lab = os.path.join("/data","fitting")
path_gpes = os.path.join(path_lab, "wave" + str(waveno))


# Create output features
n_samples = 156

idx_train = round(0.8*0.8*n_samples) # 100 approx (5 input x 20)
idx_val = idx_train + round(0.2*0.8*n_samples) # Therefore 25
idx_test = idx_val + round(0.2*n_samples) # Therefore 31

# ----------------------------------------------------------------

if wave0_generate_input:
    generate.EP_funct_param(n_samples)

if wave0_run_simulations:
    generate.template_EP_parallel(line_from = 100, line_to = n_samples - 1,
                                  waveno = waveno)

# ----------------------------------------------------------------
if wave0_generate_output:
    generate.EP_output(n_samples, waveno = 0)

# ----------------------------------------------------------------
# Load experimental values (mean +- std) you aim to match
exp_mean = np.loadtxt(os.path.join(path_match, "exp_mean.txt"), dtype=float)
exp_std = np.loadtxt(os.path.join(path_match, "exp_std.txt"), dtype=float)
exp_var = np.power(exp_std, 2)

# ----------------------------------------------------------------
# Load input parameters and output features' names
xlabels = read_labels(os.path.join(path_lab, "EP_funct_labels.txt"))
ylabels = read_labels(os.path.join(path_lab, "EP_output_labels.txt"))
features_idx_dict = {key: idx for idx, key in enumerate(ylabels)}
# ----------------------------------------------------------------

# Define the list of features to match (these would normally correspond to the 
# (sub)set of output features for which you have experimental values)
active_features = ["TAT","TATLV"]
active_idx = [features_idx_dict[key] for key in active_features]

if len(active_idx) > 1:
    exp_mean = exp_mean[active_idx]
    exp_var = exp_var[active_idx]
    ylabels = [ylabels[idx] for idx in active_idx]

# ----------------------------------------------------------------
# Load a pre-trained univariate Gaussian process emulator (GPE) for each output
# feature to match

emulator = []

X = np.loadtxt(os.path.join(path_gpes, "X.dat"), dtype=float)
X_train = X[:idx_train]

for output_name in active_features:
    y = np.loadtxt(os.path.join(path_gpes, output_name + ".dat"),dtype=float)
    y_train = y[:idx_train]
    X_val, y_val = X[idx_train:idx_val], y[idx_train:idx_val]
    
    if wave0_train_GPE:
        emul = GPEmul(X_train, y_train)
        emul.train(X_val, y_val, max_epochs = 100, n_restarts = 5,savepath = path_gpes + "/")
        emul.save("wave" + str(waveno) + "_" + output_name + ".gpe")
# ----------------------------------------------------------------
    emul = GPEmul.load(X_train, y_train, loadpath=path_gpes + "/",filename = "wave" + str(waveno) + "_" + output_name + ".gpe")
    emulator.append(emul)

if test_GPE_wave0:

    mean_list = []
    std_list = []
    score_list = []
    ise_list = []
    # ise_bounded_list = []
    min_bound = []
    max_bound = []
    
    X_test = X[idx_val:idx_test]
    METRICS_DCT = {
        "MSE": torchmetrics.MeanSquaredError(),
        "R2Score": torchmetrics.R2Score(),
        }  # you can expand this dictionary with other metrics from torchmetrics you are intrested in monitoring
        
    WATCH_METRIC = "R2Score"
    metric_name = WATCH_METRIC

    metric = METRICS_DCT[
        metric_name
    ]  # initialise the chosen regression metric from torchmetrics

if test_GPE_wave0:

    for i, emul in enumerate(emulator):
        y = np.loadtxt(os.path.join(path_gpes, active_features[i] + ".dat"),dtype=float)
        y_test =  y[idx_val:idx_test]

        y_pred_mean, y_pred_std = emul.predict(X_test)
        mean_list.append(y_pred_mean)
        std_list.append(y_pred_std)

        score = metric(emul.tensorize(y_pred_mean), emul.tensorize(y_test))
        score_list.append(score)
        ise = ISE(
            emul.tensorize(y_test),
            emul.tensorize(y_pred_mean),
            emul.tensorize(y_pred_std),
        )
        ise_list.append(ise)

        # min_bound.append(max(0,exp_mean[i]-5*exp_std[i]))
        # max_bound.append(exp_mean[i]+5*exp_std[i])

        # ise_bounded = ISE_bounded(
        #     emul.tensorize(y_test),
        #     emul.tensorize(y_pred_mean),
        #     emul.tensorize(y_pred_std),
        #     min_bound[i],
        #     max_bound[i]
        #     ,
        # )
        # ise_bounded_list.append(ise_bounded)
        print(f"\nStatistics on test set for GPE trained for the output {active_features[i]}:")
        print(f"  {metric_name} = {score:.4f}")
        print(f"  %ISE = {ise:.2f} %\n")
        # print(f"  ISE_bounded = {ise_bounded:.2f} %\n")

        # (7) Plotting mean predictions + uncertainty vs observations
        height = 9.36111
        width = 5.91667
        fig, axes = plt.subplots(1, 2, figsize=(2 * width, 2 * height / 4))

        ci = 2  # ~95% confidance interval

        inf_bound = []
        sup_bound = []
    for i, (m, s) in enumerate(zip(mean_list, std_list)):

        y = np.loadtxt(os.path.join(path_gpes, active_features[i] + ".dat"),dtype=float)
        y_test =  y[idx_val:idx_test]
        # l = np.argsort(m)  # for the sake of a better visualisation
        l = np.array(range(len(m)))
        inf_bound.append((m - ci * s).min())  # same
        sup_bound.append((m + ci * s).max())  # same

        axes[i].scatter(
            np.arange(1, len(l) + 1),
            y_test[l],
            facecolors="none",
            edgecolors="C0",
            label="simulated",
        )
        axes[i].scatter(
            np.arange(1, len(l) + 1),
            m[l],
            facecolors="C0",
            s=16,
            label="emulated",
        )
        axes[i].errorbar(
            np.arange(1, len(l) + 1),
            m[l],
            yerr=ci * s[l],
            c="C0",
            ls="none",
            lw=0.5,
            label=f"uncertainty ({ci} STD)",
        )

        axes[i].set_xticks([])
        axes[i].set_xticklabels([])
        axes[i].set_ylabel("ms", fontsize=12)
        axes[i].set_title(
            f"GPE {active_features[i]} | {metric_name} = {score_list[i]:.2f} | %ISE = {ise_list[i]:.2f}",
            fontsize=12,
        )
        axes[i].legend(loc="upper left")

        # axes[i].axhline(y=min_bound[i], color='r', linestyle='-')
        # axes[i].axhline(y=max_bound[i], color='r', linestyle='-')

    axes[0].set_ylim([np.min(inf_bound), np.max(sup_bound)])
    axes[1].set_ylim([np.min(inf_bound), np.max(sup_bound)])
    

    fig.tight_layout()
    plt.savefig(
        os.path.join(path_gpes, "inference_on_testset.png"), bbox_inches="tight", dpi=300
    )

if wave0 or wave1 or wave2:
    I = get_minmax(X_train)  # get the spanning range for each of the parameters 
                            # from the training dataset


    cutoff = 2.0  # threshold value for the implausibility criterion
    maxno = 1  # max implausibility will be taken across all the output feature till
            # the last worse impl. measure. If maxno=2 --> till the 
            # previous-to-last worse impl. measure and so on.
    W = hm.Wave(
        emulator=emulator,
        Itrain=I,
        cutoff=cutoff,
        maxno=maxno,
        mean=exp_mean,
        var=exp_var,
    )  # instantiate the wave object

    
    n_samples = 100000
    X = lhd(I, n_samples)  # initial wave is performed on a big Latin hypercube
                        # design using same parameter ranges of the training
                        # dataset

    W.find_regions(X)  # enforce the implausibility criterion to detect regions of
                    # non-implausible and of implausible points
    W.print_stats()  # show statistics about the two obtained spaces
    W.plot_impl(xlabels = xlabels,
                filename = os.path.join(path_gpes,"wave_" + str(waveno))
                )  # plot the current wave of history matching
# ----------------------------------------------------------------
if gpe_wave1:
    # To continue on the next wave:
    #
    # (1) Select points to be simulated from the current non-implausible region
    n_simuls = 10  # how many more simulations you want to run to augment the 
    # training dataset (this number must be < W.NIMP.shape[0])

    SIMULS = W.get_points(n_simuls)  # actual matrix of selected points

    waveno = 1

    path_gpes = os.path.join(path_lab, "wave" + str(waveno))
    pathlib.Path(path_gpes).mkdir(parents=True, exist_ok=True)

    np.savetxt(os.path.join(path_gpes,"X.dat"), SIMULS, fmt="%.6f")

    W.save(os.path.join(path_gpes,"wave_" + str(waveno)))  # this is a good moment 
    # to save the wave object if you need it later
    #  for other purposes (see Appendix)

    n_tests = int(n_samples)


    if wave1_generate_test:
        TESTS = W.add_points(
                n_tests
            )  # use the "cloud technique" to populate what is left from W.NIMP\SIMULS (set difference) if points left are < the chosen n_tests

        np.savetxt(os.path.join(path_gpes,"X_test.dat"), TESTS, fmt="%.6f")
    # ----------------------------------------------------------------

    # (2) Simulate the selected points


    if wave1_run_simulations:
        generate.template_EP_parallel(line_from = 0, line_to = n_simuls - 1,
                                        waveno = waveno)
    # ----------------------------------------------------------------

    # Create output features

    if wave1_generate_output:
        generate.EP_output(n_simuls, waveno = waveno)
    # ----------------------------------------------------------------

    # (3) Add the simulated points and respective results to the training dataset 
    # used in the previous wave

    X_train = np.vstack((X_train, SIMULS))

    emulator = []

    for output_name in active_features:
        y = np.loadtxt(os.path.join(path_lab, "wave" + str(waveno - 1), output_name + ".dat"),dtype=float)
        y_previous_train = y[:idx_train]
        y = np.loadtxt(os.path.join(path_lab, "wave" + str(waveno), output_name + ".dat"),dtype=float)
        y_train = np.append(y_previous_train,y)
    # ----------------------------------------------------------------

    # # (3) Train GPEs on the new, augmented training dataset

        if wave1_train_GPE:
            emul = GPEmul(X_train, y_train)
            emul.train(X_val = None, y_val = None, max_epochs = 100, n_restarts = 5,savepath = path_gpes + "/")
            emul.save("wave" + str(waveno) + "_" + output_name + ".gpe")
    # ----------------------------------------------------------------
        emul = GPEmul.load(X_train, y_train, loadpath=path_gpes + "/",filename = "wave" + str(waveno) + "_" + output_name + ".gpe")
        emulator.append(emul)

if wave1 or wave2:

    waveno = 1

    path_gpes = os.path.join(path_lab, "wave" + str(waveno))
    pathlib.Path(path_gpes).mkdir(parents=True, exist_ok=True)


    W = hm.Wave(
        emulator=emulator,
        Itrain=I,
        cutoff=cutoff,
        maxno=maxno,
        mean=exp_mean,
        var=exp_var,
    )  # instantiate the wave object

    X_test = np.loadtxt(os.path.join(path_gpes,"X_test.dat"),
                        dtype=float)

    W.find_regions(X_test)  # enforce the implausibility criterion to detect regions of
                    # non-implausible and of implausible points
    W.print_stats()  # show statistics about the two obtained spaces
    W.plot_impl(xlabels = xlabels,
                filename = os.path.join(path_gpes,"wave_" + str(waveno))
                )  # plot the current wave of history matching



    # W.save(os.path.join(path_gpes,"wave_" + str(waveno))) 
    # np.savetxt(f"./X_test_{waveno}.txt", TESTS, fmt="%.6f")
    # # NOTE: do not save the wave object after having called W.add_points(n_tests), otherwise you will loose the wave original structure

#================================================
# WAVE 2
#================================================

if gpe_wave2:
    # To continue on the next wave:
    #
    # (1) Select points to be simulated from the current non-implausible region
    n_simuls = 10  # how many more simulations you want to run to augment the 
    # training dataset (this number must be < W.NIMP.shape[0])

    SIMULS = W.get_points(n_simuls)  # actual matrix of selected points

    waveno = 2

    path_gpes = os.path.join(path_lab, "wave" + str(waveno))
    pathlib.Path(path_gpes).mkdir(parents=True, exist_ok=True)

    np.savetxt(os.path.join(path_gpes,"X.dat"), SIMULS, fmt="%.6f")

    W.save(os.path.join(path_gpes,"wave_" + str(waveno)))  # this is a good moment 
    # to save the wave object if you need it later
    #  for other purposes (see Appendix)

    n_tests = int(n_samples)


    if wave2_generate_test:
        TESTS = W.add_points(
                n_tests
            )  # use the "cloud technique" to populate what is left from W.NIMP\SIMULS (set difference) if points left are < the chosen n_tests

        np.savetxt(os.path.join(path_gpes,"X_test.dat"), TESTS, fmt="%.6f")
    # ----------------------------------------------------------------

    # (2) Simulate the selected points


    if wave2_run_simulations:
        generate.template_EP_parallel(line_from = 0, line_to = n_simuls - 1,
                                        waveno = waveno)
    # ----------------------------------------------------------------

    # Create output features

    if wave2_generate_output:
        generate.EP_output(n_simuls, waveno = waveno)
    # ----------------------------------------------------------------

    # (3) Add the simulated points and respective results to the training dataset 
    # used in the previous wave

    X_train = np.vstack((X_train, SIMULS))

    emulator = []

    for output_name in active_features:
        y = np.loadtxt(os.path.join(path_lab, "wave" + str(waveno - 2), output_name + ".dat"),dtype=float)
        y_previous_train = y[:idx_train]
        y = np.loadtxt(os.path.join(path_lab, "wave" + str(waveno - 1), output_name + ".dat"),dtype=float)
        y_train = np.append(y_previous_train,y)
        y = np.loadtxt(os.path.join(path_lab, "wave" + str(waveno), output_name + ".dat"),dtype=float)
        y_train = np.append(y_train,y)
    # ----------------------------------------------------------------

    # # (3) Train GPEs on the new, augmented training dataset

        if wave2_train_GPE:
            emul = GPEmul(X_train, y_train)
            emul.train(X_val = None, y_val = None, max_epochs = 100, n_restarts = 5,savepath = path_gpes + "/")
            emul.save("wave" + str(waveno) + "_" + output_name + ".gpe")
    # ----------------------------------------------------------------
        emul = GPEmul.load(X_train, y_train, loadpath=path_gpes + "/",filename = "wave" + str(waveno) + "_" + output_name + ".gpe")
        emulator.append(emul)

if wave2:
    W = hm.Wave(
        emulator=emulator,
        Itrain=I,
        cutoff=cutoff,
        maxno=maxno,
        mean=exp_mean,
        var=exp_var,
    )  # instantiate the wave object

    X_test = np.loadtxt(os.path.join(path_gpes,"X_test.dat"),
                        dtype=float)

    W.find_regions(X_test)  # enforce the implausibility criterion to detect regions of
                    # non-implausible and of implausible points
    W.print_stats()  # show statistics about the two obtained spaces
    W.plot_impl(xlabels = xlabels,
                filename = os.path.join(path_gpes,"wave_" + str(waveno))
                )  # plot the current wave of history matching


def plot_waves():
    
    wavesno = 3
    path_lab = os.path.join("/data","fitting")
    labels = read_labels(os.path.join(path_lab, "EP_funct_labels.txt"))
    XL = []
    
    for i in range(wavesno):
        path_gpes = os.path.join(path_lab, "wave" + str(i))
        if i == 0:
            X = np.loadtxt(os.path.join(path_gpes, "X.dat"), dtype=float)
            n_samples = 156
            idx_train = round(0.8*0.8*n_samples)
            X_test = X[:idx_train]
        else:
            X_test = np.loadtxt(os.path.join(path_gpes, "X_test.dat"), dtype=float)
        XL.append(X_test)
        
    plot_pairwise_waves(XL,["#BDDAFF","#80BAFF","#3D95FF"],labels)

if __name__ == "__main__":
    plot_waves()