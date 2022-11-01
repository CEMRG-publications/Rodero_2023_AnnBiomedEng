import json

import numpy as np
from scipy.integrate import solve_ivp

from Historia.ep.model import Gattoni2016


class EPSolution:
    """
    This class implements the solution of the Gattoni et al. (2016) electrophysiology model*.
    *https://physoc.onlinelibrary.wiley.com/doi/full/10.1113/JP273879
    """

    def __init__(self, rat, hz, paramspath):
        self.rat = rat
        self.hz = hz
        self.paramspath = paramspath

        with open(self.paramspath, "r") as f:
            dct = json.load(f)

        self.constant = dct[self.rat][str(self.hz)]

    def solver_sol(self, nbeats, p_dict=None):
        self.nbeats = nbeats

        if p_dict is not None:
            for key in p_dict.keys():
                self.constant[key] = p_dict[key]

        self.t = np.arange(int(self.nbeats * self.constant["stim_period"]) + 1)
        tspan = [0, self.t[-1]]

        Y0 = Gattoni2016.initStates(self.rat, self.hz)
        Y = solve_ivp(
            lambda t, y: Gattoni2016.computeRates(t, y, self.constant),
            t_span=tspan,
            y0=Y0,
            method="BDF",
            t_eval=self.t,
            max_step=1.0,
        )

        self.Y = Y.y
        self.v = Y.y[0, :]
        self.ca = 1e3 * Y.y[12, :]
