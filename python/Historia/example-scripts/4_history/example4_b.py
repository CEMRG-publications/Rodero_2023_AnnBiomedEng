import random

import numpy as np
import torch
from gpytGPE.gpe import GPEmul

from Historia.history import hm
from Historia.shared.design_utils import get_minmax, lhd, read_labels

SEED = 8


def main():
    # ----------------------------------------------------------------
    # Make the code reproducible
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ----------------------------------------------------------------
    path = "data/"

    # ----------------------------------------------------------------
    # Load experimental values (mean +- std) you aim to match
    path_match = path + "match/"
    exp_mean = np.loadtxt(path_match + "exp_mean.txt", dtype=float)
    exp_std = np.loadtxt(path_match + "exp_std.txt", dtype=float)
    exp_var = np.power(exp_std, 2)

    # ----------------------------------------------------------------
    # Load input parameters and output features' names
    path_lab = path + "labels/"
    xlabels = read_labels(path_lab + "xlabels.txt")
    ylabels = read_labels(path_lab + "ylabels.txt")
    features_idx_dict = {key: idx for idx, key in enumerate(ylabels)}

    # ----------------------------------------------------------------
    # Define the list of features to match (these would normally correspond to the (sub)set of output features for which you have experimental values)
    active_features = ["EDV", "ESV", "ET", "IVRT", "PeakP", "maxdP", "mindP"]
    active_idx = [features_idx_dict[key] for key in active_features]

    exp_mean = exp_mean[active_idx]
    exp_var = exp_var[active_idx]
    ylabels = [ylabels[idx] for idx in active_idx]

    # ----------------------------------------------------------------
    # Load a pre-trained univariate Gaussian process emulator (GPE) for each output feature to match
    path_gpes = path + "gpes/"
    emulator = []
    for idx in active_idx:
        loadpath = path_gpes + str(idx) + "/"
        X_train = np.loadtxt(loadpath + "X_train.txt", dtype=float)
        y_train = np.loadtxt(loadpath + "y_train.txt", dtype=float)
        emul = GPEmul.load(
            X_train, y_train, loadpath=loadpath
        )  # NOTICE: GPEs must have been trained using gpytGPE library (https://github.com/stelong/gpytGPE)
        emulator.append(emul)

    I = get_minmax(
        X_train
    )  # get the spanning range for each of the parameters from the training dataset

    # ----------------------------------------------------------------

    waveno = 1  # wave id number
    cutoff = 3.0  # threshold value for the implausibility criterion
    maxno = 1  # max implausibility will be taken across all the output feature till the last worse impl. measure. If maxno=2 --> till the previous-to-last worse impl. measure and so on.

    W = hm.Wave(
        emulator=emulator,
        Itrain=I,
        cutoff=cutoff,
        maxno=maxno,
        mean=exp_mean,
        var=exp_var,
    )  # instantiate the wave object

    n_samples = 100000
    X = lhd(
        I, n_samples
    )  # initial wave is performed on a big Latin hypercube design using same parameter ranges of the training dataset

    W.find_regions(
        X
    )  # enforce the implausibility criterion to detect regions of non-implausible and of implausible points
    W.print_stats()  # show statistics about the two obtained spaces
    
    W.plot_wave(xlabels=xlabels, display="impl", filename=f"./wave_{waveno}_impl")  # plot the current wave of history matching (implausibility measure plot)
    W.plot_wave(xlabels=xlabels, display="var", filename=f"./wave_{waveno}_var")  # we can also check the accuracy of the GPEs for the current wave

    # ----------------------------------------------------------------
    # To continue on the next wave:
    #
    # (1) Select points to be simulated from the current non-implausible region
    n_simuls = 128  # how many more simulations you want to run to augment the training dataset (this number must be < W.NIMP.shape[0])
    SIMULS = W.get_points(n_simuls)  # actual matrix of selected points
    np.savetxt(f"./X_simul_{waveno}.txt", SIMULS, fmt="%.6f")

    W.save(
        f"./wave_{waveno}"
    )  # this is a good moment to save the wave object if you need it later for other purposes (see Appendix)

    # (2) Simulate the selected points
    # (3) Add the simulated points and respective results to the training dataset used in the previous wave
    # (3) Train GPEs on the new, augmented training dataset
    # (4) Start a new wave of history matching, where the initial parameter space to be split into non-implausible and implausible regions is no more a Latin hypercube design but is now the non-implausible region obtained in the previous wave and saved as:
    n_tests = 100000  # number of test points we want for the next wave (from the current non-implausible region)
    TESTS = W.add_points(
        n_tests,
        scale=0.1,
    )  # use the "cloud technique" to populate what is left from W.NIMP\SIMULS (set difference) if points left are < the chosen n_tests. scale parameter regulate how far from the current NIMP we want to sample new tests points: it can be lowered till 0.01 if the algorithm is too slow
    np.savetxt(f"./X_test_{waveno}.txt", TESTS, fmt="%.6f")
    # NOTE: do not save the wave object after having called W.add_points(n_tests), otherwise you will loose the wave original structure

    # ----------------------------------------------------------------
    # Appendix - Wave object loading
    # You can load a wave object by providing the same data used to instantiate the wave: emulator, Itrain, cutoff, maxno, mean, var. This is normally done when you need to re-run the wave differently.
    # Alternatively, you can load the wave object by providing no data at all, just to better examine its internal structure:
    W = hm.Wave()
    W.load(f"./wave_{waveno}")
    W.print_stats()

    # This is the list of the loaded wave object attributes:
    print(W.__dict__.keys())

    # Noteworthy attributes are:
    # W.I = implausibility measure obtained for each point in the test set
    # W.NIMP = non-implausible region
    # W.nimp_idx = indices of the initial test set which resulted to be non-implausible
    # W.IMP = implausible region
    # W.imp_idx = indices of the initial test set which resulted to be implausible
    # W.simul_idx = indices of W.NIMP that were selected to be simulated for the next wave
    # W.nsimul_idx = indices of W.NIMP which were not selected for simulations (the respective points will appear in the test set of the next wave instead)

    # The original test set is not stored as an attribute to save space. However, this information can still be retrieved from stored attributes as:
    # X_test = W.reconstruct_tests()


if __name__ == "__main__":
    main()
