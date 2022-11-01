import matplotlib.pyplot as plt
import numpy as np

from Historia.cellcontr import solver as solcontr
from Historia.ep import solver as solep
from Historia.shared.plot_utils import get_col, interp_col


def main():
    rat = "sham"

    # ---------------------------------------------------------------
    # Electrophysiology - needed for generating the calcium transient
    hz = 6
    nbeats = 1
    ep_params_path = "../1_ep/data/parameters.json"

    E = solep.EPSolution(rat, hz, ep_params_path)
    E.solver_sol(nbeats)

    t = E.t
    calcium = E.ca

    # ---------------------------------------------------------------
    # Cellular contraction
    contr_params_path = "data/parameters.json"

    C = solcontr.CONTRSolution(rat, calcium, contr_params_path)
    C.solver_sol()

    t = C.t
    tension = C.T

    Cai = np.logspace(-1, 1, 1000)

    C_ss = solcontr.CONTRSolution(rat, Cai, contr_params_path)
    C_ss.steadystate_sol()

    Tref = C_ss.constant["Tref"]

    pCai = -np.log10(1e-6 * Cai)
    forcepCa = C_ss.F["F"]

    print(list(C.constant.keys()))  # list of available constants to be treated as parameters (same as C_ss.constant)

    param = "Ca50"  # let's consider e.g. the thin filament Ca2+ sensitivity as a target for perturbations
    param_ref_val = C.constant[
        param
    ]  # store the reference value for the selected parameter
    N = 6  # number of perturbations to make
    p_inf = 0.8  # percentage of maximum allowed decrease
    p_sup = 1.2  # percentage of maximum allowed increase
    param_vals = np.linspace(
        p_inf * param_ref_val, p_sup * param_ref_val, N
    )  # vector of perturbed parameter values

    T_list = []
    FpCa_list = []
    for i in range(N):
        # solve the CONTR model and calculate the steady-state solution by providing a new dictionary of altered parameter/s and its/their new value/s:
        new_params_dict = {
            param: param_vals[i]
        }  # here you can add more altered parameters if you want, e.g. "lambda": 1.2

        C = solcontr.CONTRSolution(rat, calcium, contr_params_path)
        C.solver_sol(p_dict=new_params_dict)

        T = C.T
        T_list.append(T)

        C_ss = solcontr.CONTRSolution(rat, Cai, contr_params_path)
        C_ss.steadystate_sol(p_dict=new_params_dict)

        Tref = C_ss.constant["Tref"]

        FpCa = C_ss.F["F"]
        FpCa_list.append(FpCa)

    # ---------------------------------------------------------------
    # Plot
    width = 5.91667
    height = 9.36111
    figsize = (2 * width, 2 * height / 4)
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    if rat == "sham":
        color = get_col("blue")
    else:
        color = get_col("red")
    lsc = interp_col(color, N)

    axes[0].plot(pCai, forcepCa, c="k", lw=2, ls="--", label="unperturbed")
    axes[1].plot(t, tension, c="k", lw=2, ls="--", label="unperturbed")
    for i, (FpCa, T) in enumerate(zip(FpCa_list, T_list)):
        axes[0].plot(
            pCai, FpCa, c=lsc[i], lw=2, label=f"{param} = {param_vals[i]:.6f}"
        )
        axes[1].plot(
            t, T, c=lsc[i], lw=2, label=f"{param} = {param_vals[i]:.6f}"
        )

    axes[0].legend(loc="upper left")
    axes[0].set_xlabel("pCa (-log$_{10}$[M])", fontsize=12)
    axes[0].set_ylabel("Force (kPa)", fontsize=12)
    axes[0].invert_xaxis()

    axes[1].legend(loc="upper right")
    axes[1].set_xlabel("Time (ms)", fontsize=12)
    axes[1].set_ylabel("Force (kPa)", fontsize=12)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
