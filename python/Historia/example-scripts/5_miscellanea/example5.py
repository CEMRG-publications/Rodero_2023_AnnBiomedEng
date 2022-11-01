import matplotlib.gridspec as grsp
import matplotlib.pyplot as plt
import numpy as np
from gpytGPE.gpe import GPEmul

from Historia.shared.design_utils import read_labels
from Historia.shared.plot_utils import get_col, interp_col
from Historia.shared.predict_utils import upvl


def f_emul(emulator, x):
    return np.array([emul.predict(x.reshape(1, -1))[0] for emul in emulator]).ravel(), np.array([emul.predict(x.reshape(1, -1))[1] for emul in emulator]).ravel()


def pvloop(axes, t, lvv, lvp, color, linestyle, linewidth, label, uc=[]):
    ax1, ax2, ax3 = axes

    ax1.plot(t, lvv, c=color, ls=linestyle, lw=linewidth, label=label)
    ax2.plot(t, lvp, c=color, ls=linestyle, lw=linewidth, label=label)

    if uc:
        ax3.fill_between([uc[1][0]+uc[1][1], uc[0][0]-uc[0][1]], uc[2][0]-uc[2][1], uc[2][0]+uc[2][1], color=color, alpha=0.2)
        ax3.fill_between([uc[1][0]-uc[1][1], uc[1][0]+uc[1][1]], np.min(lvp), uc[2][0]-uc[2][1], color=color, alpha=0.2)
        ax3.fill_between([uc[0][0]-uc[0][1], uc[0][0]+uc[0][1]], np.min(lvp), uc[2][0]-uc[2][1], color=color, alpha=0.2)

    ax3.plot(lvv, lvp, c=color, ls=linestyle, lw=linewidth, label=label)

    return ax1, ax2, ax3


def main():
    # ----------------------------------------------------------------
    path = "data/"

    # ----------------------------------------------------------------
    # Load input parameters' and output features' names
    path_lab = path + "labels/"
    xlabels = read_labels(path_lab + "xlabels.txt")
    ylabels = read_labels(path_lab + "ylabels.txt")
    param_idx_dict = {key: idx for idx, key in enumerate(xlabels)}
    feat_idx_dict = {key: idx for idx, key in enumerate(ylabels)}

    # ----------------------------------------------------------------
    # Load a pre-trained univariate Gaussian process emulator (GPE) for each output feature
    path_gpes = path + "gpes/"
    emulator = []
    for idx in range(len(ylabels)):
        loadpath = path_gpes + str(idx) + "/"
        X_train = np.loadtxt(loadpath + "X_train.txt", dtype=float)
        y_train = np.loadtxt(loadpath + "y_train.txt", dtype=float)
        emul = GPEmul.load(
            X_train, y_train, loadpath=loadpath
        )  # NOTICE: GPEs must have been trained using gpytGPE library (https://github.com/stelong/gpytGPE)
        emulator.append(emul)

    # ----------------------------------------------------------------
    # Mapping parameters to "uncertain" PV-loops (i.e. obtained as a combination of emulators' predictions)
    param = "Ca50"  # choose one among xlabels
    idx = param_idx_dict[param]

    gs = grsp.GridSpec(2, 2, width_ratios=[1, 1.2])
    width = 5.91667
    height = 9.36111
    figsize = (2 * width, 2 * height / 3)
    fig = plt.figure(figsize=figsize)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[:, 1])

    edp = 0.3  # just a reference end-diastolic pressure for plotting
    x = np.ones(
        (len(xlabels),), dtype=float
    )  # a vector of ones gives back the reference, unperturbed PV-loop
    y,_ = f_emul(
        emulator, x
    )  # the provided emulators in this example have fractions of the unit as input parameter values
    t, lvv, lvp = upvl(
        edp, y
    )  # this function returns an uncertain PV-loop given a set of predicted LV features
    ax1, ax2, ax3 = pvloop(
        (ax1, ax2, ax3), t, lvv, lvp, "k", "--", 1.0, "control"
    )  # plotting

    perturbation_list = [0.8, 0.9, 1.1, 1.2]

    lsc = interp_col(get_col("blue"), len(perturbation_list))

    for i, p in enumerate(perturbation_list):
        x[idx] = p
        y, s = f_emul(emulator, x)
        t, lvv, lvp = upvl(p*edp if param == "p" else edp, y)
        ax1, ax2, ax3 = pvloop(
            (ax1, ax2, ax3),
            t,
            lvv,
            lvp,
            lsc[i],
            "-",
            1.5,
            r"{:g}$\times${}".format(p, param),
            uc=[(y[feat_idx_dict[key]], s[feat_idx_dict[key]]) for key in ["EDV", "ESV", "PeakP"]],  # comment this line to plot without uncertainty
        )

    ax1.set_xlabel("Time (ms)", fontsize=12)
    ax1.set_ylabel("LVV (μL)", fontsize=12)

    ax2.set_xlabel("Time (ms)", fontsize=12)
    ax2.set_ylabel("LVP (kPa)", fontsize=12)

    ax3.set_xlabel("LVV (μL)", fontsize=12)
    ax3.set_ylabel("LVP (kPa)", fontsize=12)
    ax3.legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
