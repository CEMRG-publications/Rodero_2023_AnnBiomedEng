import matplotlib.pyplot as plt
import numpy as np

from Historia.cellcontr import solver as solcontr
from Historia.ep import solver as solep
from Historia.shared.plot_utils import get_col


def fun(axes, rat, color):
    # ---------------------------------------------------------------
    # Electrophysiology
    hz = 6  # pacing frequency (either 1 or 6)
    nbeats = 1  # number of heart beats to simulate
    ep_params_path = "../1_ep/data/parameters.json"  # path to parameter file

    E = solep.EPSolution(
        rat, hz, ep_params_path
    )  # instantiate the class of EP model solution
    E.solver_sol(nbeats)  # solve the model

    t = E.t  # time vector used
    actionpotential = E.v  # action potential solution
    calcium = E.ca  # calcium transient solution
    # print(E.Y.shape) # all the other solved variables are stored in each row of matrix E.Y

    # ---------------------------------------------------------------
    # Cellular contraction
    contr_params_path = "data/parameters.json"  # path to parameter file

    C = solcontr.CONTRSolution(
        rat, calcium, contr_params_path
    )  # instantiate the class of CONTR model solution
    C.solver_sol()  # solve the model using the simulated calcium transient provided above

    t = C.t  # time vector used
    tension = C.T  # twitch force over time (this is an algebraic)
    # print(C.Y.shape) # all the solved variables are stored in each row of matrix C.Y

    Cai = np.logspace(-1, 1, 1000)  # define equally spaced log calcium values

    C_ss = solcontr.CONTRSolution(
        rat, Cai, contr_params_path
    )  # instantiate the class of CONTR model steady-state solution
    C_ss.steadystate_sol()  # calculate the steady-state solution using the given equally spaced log calcium values

    Tref = C_ss.constant[
        "Tref"
    ]  # used constants can be retrieved this way by knowing the constant name (see "example_3.py" for more details)

    pCai = -np.log10(1e-6 * Cai)  # x-vector for plotting
    forcepCa = C_ss.F["F"] / Tref  # normalised force
    # print(C_ss.F['pCa50'], C_ss.F['h'], C_ss.F['hl']) # other properties of the steady-state force are the pCa50, the Hill coefficient and the length-dependence factor
    # print(C_ss.TRPN['TRPN'].shape, C_ss.TRPN['EC50']) # steady-state solutions and their effective half-maximal concentrations are TRPN - TRPN_EC50
    # print(C_ss.XB['XB'].shape, C_ss.XB['EC50']) # and XB - XB_EC50
    # print(C_ss.F['pCa50'], -np.log10(1e-6*C_ss.XB['EC50'])) # since F = Tref * XB * h(l) * g(Q), notice that pCa50 = -log(1e-6 * XB_EC50)

    # ---------------------------------------------------------------
    # Plot
    axes[0, 0].plot(t, actionpotential, c=color, lw=2)
    axes[0, 1].plot(t, calcium, c=color, lw=2)
    axes[1, 0].plot(pCai, forcepCa, c=color, lw=2)
    axes[1, 1].plot(t, tension, c=color, lw=2)

    return axes


def main():
    rats = ["sham", "ab"]  # available rat phenotypes

    width = 5.91667
    height = 9.36111
    figsize = (2 * width, 2 * height / 2)
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    for rat in rats:
        if rat == "sham":
            color = get_col("blue")[1]
        else:
            color = get_col("red")[1]

        axes = fun(axes, rat, color)

    axes[0, 0].set_xlabel("Time (ms)", fontsize=12)
    axes[0, 0].set_ylabel("Action Potential (mV)", fontsize=12)
    axes[0, 1].set_xlabel("Time (ms)", fontsize=12)
    axes[0, 1].set_ylabel("[Ca$^{2+}$]$_i$ (μM)", fontsize=12)
    axes[1, 0].set_xlabel("pCa", fontsize=12)
    axes[1, 0].set_ylabel("Normalised Force", fontsize=12)
    axes[1, 0].invert_xaxis()
    axes[1, 1].set_xlabel("Time (ms)", fontsize=12)
    axes[1, 1].set_ylabel("Force (kPa)", fontsize=12)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
