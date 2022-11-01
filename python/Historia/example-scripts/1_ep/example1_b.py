import matplotlib.pyplot as plt
import numpy as np

from Historia.ep import solver as solep
from Historia.shared.plot_utils import get_col, interp_col


def main():
    # ---------------------------------------------------------------
    # Electrophysiology
    rat = "sham"
    hz = 6
    nbeats = 1
    ep_params_path = "data/parameters.json"

    E = solep.EPSolution(rat, hz, ep_params_path)
    E.solver_sol(nbeats)

    t = E.t
    actionpotential = E.v
    calcium = E.ca

    print(list(E.constant.keys()))  # list of available constants to be treated as parameters

    param = "g_Na"  # let's consider e.g. the sodium conductance as a target for perturbations
    param_ref_val = E.constant[
        param
    ]  # store the reference value for the selected parameter
    N = 6  # number of perturbations to make
    p_inf = 0.8  # percentage of maximum allowed decrease
    p_sup = 1.2  # percentage of maximum allowed increase
    param_vals = np.linspace(
        p_inf * param_ref_val, p_sup * param_ref_val, N
    )  # vector of perturbed parameter values

    V_list = []
    Ca_list = []
    for i in range(N):
        # solve the EP model by providing a new dictionary of altered parameter/s and its/their new value/s:
        new_params_dict = {
            param: param_vals[i]
        }  # here you can add more altered parameters if you want

        E = solep.EPSolution(rat, hz, ep_params_path)
        E.solver_sol(nbeats, p_dict=new_params_dict)

        V = E.v
        V_list.append(V)

        Ca = E.ca
        Ca_list.append(Ca)

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

    axes[0].plot(t, actionpotential, c="k", lw=2, ls="--", label="unperturbed")
    axes[1].plot(t, calcium, c="k", lw=2, ls="--", label="unperturbed")
    for i, (V, Ca) in enumerate(zip(V_list, Ca_list)):
        axes[0].plot(
            t, V, c=lsc[i], lw=2, label=f"{param} = {param_vals[i]:.6f}"
        )
        axes[1].plot(
            t, Ca, c=lsc[i], lw=2, label=f"{param} = {param_vals[i]:.6f}"
        )

    axes[0].legend(loc="upper right")
    axes[0].set_xlabel("Time (ms)", fontsize=12)
    axes[0].set_ylabel("Action Potential (mV)", fontsize=12)

    axes[1].legend(loc="upper right")
    axes[1].set_xlabel("Time (ms)", fontsize=12)
    axes[1].set_ylabel("[Ca$^{2+}$]$_i$ (μM)", fontsize=12)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
