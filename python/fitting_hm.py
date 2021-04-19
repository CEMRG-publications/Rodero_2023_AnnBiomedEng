import random
import numpy as np
import torch
import os

from Historia.shared.design_utils import get_minmax, lhd, read_labels
from gpytGPE.gpe import GPEmul

SEED = 2

def template_EP_hm(train_now = True):
    # ----------------------------------------------------------------
    # Make the code reproducible
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ----------------------------------------------------------------
    
    
    path_data = os.path.join("/data","fitting","emul_data")
    path_match = os.path.join("/data","fitting", "match")
    path_lab = os.path.join("/data","fitting")
    path_gpes = os.path.join(path_lab, "gpes")


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
    # Define the list of features to match (these would normally correspond to the (sub)set of output features for which you have experimental values)
    active_features = ["QRS"]
    active_idx = [features_idx_dict[key] for key in active_features]
    
    if len(active_idx) > 1:
        exp_mean = exp_mean[active_idx]
        exp_var = exp_var[active_idx]
        ylabels = [ylabels[idx] for idx in active_idx]

# ----------------------------------------------------------------
    # Load a pre-trained univariate Gaussian process emulator (GPE) for each output feature to match
    emulator = []
    for idx in active_idx:
        loadpath = os.path.join(path_gpes, str(idx))

        X = np.loadtxt(os.path.join(loadpath, "X.txt"), dtype=float)
        y = np.loadtxt(os.path.join(loadpath, "y.txt"), dtype=float)

        n_samples = X.shape[0]
        if train_now is True:
            # let's split the dataset in train(60%), val(10%) and test(30%) sets
            idx_train = round(0.6*n_samples)
            idx_val = idx_train + round(0.1*n_samples)
            idx_test = round(0.3*n_samples)
            X_train, y_train = (
                                X[:idx_train],
                                y[:idx_train],
                            )  
            X_val, y_val = X[idx_train:idx_val], y[idx_train:idx_val]
            X_test, y_test = X[idx_val:], y[idx_val:]
            emulator = GPEmul(X_train, y_train)
            emulator.train(X_val, y_val, max_epochs=100, n_restarts=3, savepath = os.path.join(path_gpes, str(idx) + "/"))
            emulator.save("first_try.gpe")

    #     emul = GPEmul.load(
    #         X_train, y_train, loadpath=loadpath
    #     )  # NOTICE: GPEs must have been trained using gpytGPE library (https://github.com/stelong/gpytGPE)
    #     emulator.append(emul)

    # I = get_minmax(
    #     X_train
    # )  # get the spanning range for each of the parameters from the training dataset

if __name__ == "__main__":
    template_EP_hm()