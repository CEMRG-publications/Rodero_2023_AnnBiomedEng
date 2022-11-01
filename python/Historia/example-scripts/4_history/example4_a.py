import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from gpytGPE.gpe import GPEmul
from gpytGPE.utils.metrics import R2Score

from Historia.history import hm
from Historia.shared.design_utils import get_minmax, lhd

SEED = 8


def f(x):  # elliptic paraboloid: y = x1^2/a^2 + x2^2/b^2, a=b=1
    return np.sum(np.power(x, 2), axis=1)


def main():
    # ----------------------------------------------------------------
    # Make the code reproducible
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ----------------------------------------------------------------
    # Generate syntetic datum to match by visually exploring the parameter space
    N = 100
    x = np.linspace(-3, 3, N)
    x1, x2 = np.meshgrid(x, x)

    X = np.hstack((x1.reshape(N * N, 1), x2.reshape(N * N, 1)))
    y = f(X).reshape(N, N)

    fig, axis = plt.subplots(1, 1)

    PC = axis.pcolormesh(x1, x2, y, cmap="coolwarm", shading="auto")
    C = axis.contour(x1, x2, y, levels=[8.0])
    cbar = fig.colorbar(PC, ax=axis)
    cbar.add_lines(C)

    labels = ["$p_1$", "$p_2$"]
    axis.set_xlabel(labels[0], fontsize=12)
    axis.set_ylabel(labels[1], fontsize=12)

    fig.tight_layout()
    plt.show()  # we know that there exists a specific region in the parameter space where the function is equal to 8

    exp_mean = np.array(
        [8.0]
    )  # we generate a syntetic experimental value we aim to match by using Bayesian History Matching technique
    exp_var = 0.1 * exp_mean

    # ----------------------------------------------------------------
    # Train a univariate Gaussian process emulator (GPE) to replace the mapping f: X -> y
    I = np.array(
        [[-5, 5], [-5, 5]]
    )  # we use a broader space for the training dataset
    n_samples = 100
    X = lhd(I, n_samples)
    y = f(X)

    X_train, y_train = (
        X[:60],
        y[:60],
    )  # let's split the dataset in train(60%), val(10%) and test(30%) sets
    X_val, y_val = X[60:70], y[60:70]
    X_test, y_test = X[70:], y[70:]

    emulator = GPEmul(X_train, y_train)
    emulator.train(X_val, y_val, max_epochs=100, n_restarts=3)
    emulator.save()

    y_mean_pred, y_std_pred = emulator.predict(X_test)
    r2s = R2Score(emulator.tensorize(y_test), emulator.tensorize(y_mean_pred))
    print(r2s)  # check the GPE accuracy on the testing set

    # ----------------------------------------------------------------
    # Run first wave (iteration) of History matching (see "example4_b.py" for more details)
    emulator = [GPEmul.load(X_train, y_train)]

    waveno = 1
    cutoff = 3.0
    maxno = 1

    I = get_minmax(X_train)

    W = hm.Wave(
        emulator=emulator,
        Itrain=I,
        cutoff=cutoff,
        maxno=maxno,
        mean=exp_mean,
        var=exp_var,
    )

    n_samples = 100000
    X = lhd(I, n_samples)

    W.find_regions(X)
    W.print_stats()
    W.plot_wave(xlabels=labels, filename=f"./wave_{waveno}")

    # ----------------------------------------------------------------
    # Check that the found non-implausible parameter space is actually compatible with matching the syntatic experimental datum
    y_actual = f(W.NIMP)

    fig, axis = plt.subplots(1, 1)

    axis.boxplot(y_actual)
    axis.axhline(exp_mean, c="r", ls="--")
    xmin, xmax = axis.get_xlim()
    inf_conf = exp_mean - cutoff * np.sqrt(exp_var)
    sup_conf = exp_mean + cutoff * np.sqrt(exp_var)
    axis.fill_between([xmin, xmax], inf_conf, sup_conf, color="r", alpha=0.1)
    axis.set_xticks([])

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
