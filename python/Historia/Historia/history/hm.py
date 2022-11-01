import diversipy as dp
import matplotlib.gridspec as grsp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import iqr

from Historia.shared.design_utils import get_minmax
from Historia.shared.indices_utils import diff, whereq_whernot
from Historia.shared.jsonfiles_utils import load_json, save_json


class Wave:
    def __init__(
        self,
        emulator=None,
        Itrain=None,
        cutoff=None,
        maxno=None,
        mean=None,
        var=None,
    ):
        self.emulator = emulator
        self.Itrain = Itrain
        self.output_dim = len(emulator) if emulator is not None else 0
        self.cutoff = cutoff
        self.maxno = maxno
        self.mean = mean
        self.var = var
        self.I = None
        self.PV = None
        self.NIMP = None
        self.nimp_idx = None
        self.IMP = None
        self.imp_idx = None

    def compute_impl(self, X):
        M = np.zeros((self.n_samples, self.output_dim), dtype=float)
        V = np.zeros((self.n_samples, self.output_dim), dtype=float)
        for j, emul in enumerate(self.emulator):
            mean, std = emul.predict(X)
            var = np.power(std, 2)
            M[:, j] = mean
            V[:, j] = var

        I = np.zeros((self.n_samples,), dtype=float)
        PV = np.zeros((self.n_samples,), dtype=float)
        for i in range(self.n_samples):
            In = np.sqrt(
                (np.power(M[i, :] - self.mean, 2)) / (V[i, :] + self.var)
            )
            PVn = V[i, :] / self.var

            I[i] = np.sort(In)[-self.maxno]
            PV[i] = np.sort(PVn)[-self.maxno]

        return I, PV

    def find_regions(self, X):
        self.n_samples = X.shape[0]
        self.input_dim = X.shape[1]

        I, PV = self.compute_impl(X)
        l = np.where(I < self.cutoff)[0]
        nl = diff(range(self.n_samples), l)

        self.I = I
        self.PV = PV
        self.nimp_idx = l
        self.NIMP = X[l]
        self.imp_idx = nl
        self.IMP = X[nl]

    def print_stats(self):
        nimp = len(self.nimp_idx)
        imp = len(self.imp_idx)
        tests = nimp + imp
        perc = 100 * nimp / tests

        stats = pd.DataFrame(
            index=["TESTS", "IMP", "NIMP", "PERC"],
            columns=["#POINTS"],
            data=[tests, imp, nimp, f"{perc:.4f} %"],
        )
        print(stats)

    def reconstruct_tests(self):
        X = np.zeros((self.n_samples, self.input_dim), dtype=float)
        X[self.nimp_idx] = self.NIMP
        X[self.imp_idx] = self.IMP
        return X

    def save(self, filename):
        dct = vars(self)
        excluded_keys = ["emulator"]
        obj_dct = {}
        obj_dct.update(
            {k: dct[k] for k in set(list(dct.keys())) - set(excluded_keys)}
        )
        save_json(obj_dct, filename)

    def load(self, filename):
        obj_dict = load_json(filename)
        for k, v in obj_dict.items():
            setattr(self, k, v)

    def get_points(self, n_simuls):
        NROY = np.copy(self.NIMP)
        if n_simuls >= NROY.shape[0] - 1:
            raise ValueError(
                "Not enough points for simulations! n_simuls must be strictly less than W.NIMP.shape[0] - 1."
            )
        else:
            SIMULS = dp.subset.psa_select(NROY, n_simuls)

        self.simul_idx, self.nsimul_idx = whereq_whernot(NROY, SIMULS)

        return SIMULS

    def add_points(
        self,
        n_tests,
        scale=0.1,
    ):  # NOTE: the Wave object instance internal structure will be compromised after calling this method: we recommend calling self.save() beforehand!
        nsidx = self.nsimul_idx
        NROY = np.copy(self.NIMP[nsidx])
        lbounds = self.Itrain[:, 0]
        ubounds = self.Itrain[:, 1]

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

            self.find_regions(temp)
            NROY = np.vstack((NROY, self.NIMP))

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

    def plot_wave(self, xlabels=None, display="impl", filename="./wave_impl"):
        X = self.reconstruct_tests()

        if xlabels is None:
            xlabels = [f"p{i+1}" for i in range(X.shape[1])]

        if display == "impl":
            C = self.I
            cmap = "jet"
            vmin = 1.0
            vmax = self.cutoff
            cbar_label = "Implausibility measure"

        elif display == "var":
            C = self.PV
            cmap = "bone_r"
            vmin = np.max(
                [
                    np.percentile(self.PV, 25) - 1.5 * iqr(self.PV),
                    self.PV.min(),
                ]
            )
            vmax = np.min(
                [
                    np.percentile(self.PV, 75) + 1.5 * iqr(self.PV),
                    self.PV.max(),
                ]
            )
            cbar_label = "GPE variance / EXP. variance"

        else:
            raise ValueError(
                "Not a valid display option! Can only display implausibilty maps ('impl') or proportion-of-exp.variance maps ('var')."
            )

        height = 9.36111
        width = 5.91667
        fig = plt.figure(figsize=(3 * width, 3 * height / 3))
        gs = grsp.GridSpec(
            self.input_dim - 1,
            self.input_dim,
            width_ratios=(self.input_dim - 1) * [1] + [0.1],
        )

        for k in range(self.input_dim * self.input_dim):
            i = k % self.input_dim
            j = k // self.input_dim

            if i > j:
                axis = fig.add_subplot(gs[i - 1, j])
                axis.set_facecolor("xkcd:light grey")

                im = axis.hexbin(
                    X[:, j],
                    X[:, i],
                    C=C,
                    reduce_C_function=np.min,
                    gridsize=20,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )

                axis.set_xlim([self.Itrain[j, 0], self.Itrain[j, 1]])
                axis.set_ylim([self.Itrain[i, 0], self.Itrain[i, 1]])

                if i == self.input_dim - 1:
                    axis.set_xlabel(xlabels[j], fontsize=12)
                else:
                    axis.set_xticklabels([])
                if j == 0:
                    axis.set_ylabel(xlabels[i], fontsize=12)
                else:
                    axis.set_yticklabels([])

        cbar_axis = fig.add_subplot(gs[:, self.input_dim - 1])
        cbar = fig.colorbar(im, cax=cbar_axis)
        cbar.set_label(cbar_label, size=12)
        fig.tight_layout()
        plt.savefig(filename + ".png", bbox_inches="tight", dpi=300)
