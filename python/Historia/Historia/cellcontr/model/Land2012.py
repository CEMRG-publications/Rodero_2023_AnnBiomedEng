import numpy as np


def initStates():
    sizeStates = 4
    states = np.zeros((sizeStates,), dtype=float)

    states[0] = 8e-02
    states[1] = 2e-03
    states[2] = 0.0
    states[3] = 0.0

    return states


def initConsts(c_dict):
    sizeConstants = 17
    constants = np.zeros((sizeConstants,), dtype=float)

    for i, key in enumerate(c_dict.keys()):
        constants[i] = c_dict[key]

    return constants


def computeAlgebraics(t, y, Cai, constants):
    sizeAlgebraics = 10
    algebraics = np.zeros((sizeAlgebraics,), dtype=float)

    algebraics[0] = t - len(Cai) * np.floor(t / len(Cai))

    if not t // len(Cai):
        algebraics[1] = Cai[int(algebraics[0])]
    else:
        algebraics[1] = Cai[-1]

    algebraics[2] = constants[2] * (
        1.0 + constants[11] * (constants[0] - 1.0)
    )  # safe lambda = np.min([1.2, np.max([0.8, constants[0]])])
    algebraics[3] = np.sqrt(np.power(y[0] / constants[4], constants[8]))
    algebraics[4] = (
        1.0 / algebraics[3]
    )  # safe 1/permtot = np.min([100.0, 1.0/algebraics[3]])
    algebraics[5] = y[2] + y[3]

    if algebraics[5] <= 0.0:
        algebraics[6] = (constants[12] * algebraics[5] + 1.0) / (
            1.0 - algebraics[5]
        )
    else:
        algebraics[6] = (1.0 + (constants[12] + 2.0) * algebraics[5]) / (
            1.0 + algebraics[5]
        )

    algebraics[7] = np.max(
        [
            0.0,
            1.0
            + constants[10]
            * (
                np.min([constants[0], 1.2])
                + np.min([np.min([constants[0], 1.2]), 0.87])
                - 1.87
            ),
        ]
    )
    algebraics[8] = algebraics[6] * algebraics[7] * y[1]
    algebraics[9] = constants[3] * algebraics[8]

    return algebraics


def computeRates(t, y, Cai, c_dict):
    constants = initConsts(c_dict)
    algebraics = computeAlgebraics(t, y, Cai, constants)

    sizeRates = 4
    rates = np.zeros((sizeRates,), dtype=float)

    rates[0] = (
        constants[6]
        * np.power(algebraics[1] / algebraics[2], constants[5])
        * (1.0 - y[0])
        - constants[7] * y[0]
    )
    rates[1] = constants[9] * (
        algebraics[3] * (1.0 - y[1]) - algebraics[4] * y[1]
    )
    rates[2] = constants[13] * constants[1] - constants[15] * y[2]
    rates[3] = constants[14] * constants[1] - constants[16] * y[3]

    return rates


def computeSteadystate(Cai, c_dict):
    constants = initConsts(c_dict)
    constants[2] = constants[2] * (1.0 + constants[11] * (constants[0] - 1.0))

    r = np.power(Cai / constants[2], constants[5])
    TRPN_ss = r / (constants[7] / constants[6] + r)
    trpn_EC50 = (
        np.power(constants[7] / constants[6], 1.0 / constants[5])
        * constants[2]
    )

    s = np.power(TRPN_ss / constants[4], constants[8])
    XB_ss = s / (1.0 + s)
    xb_EC50 = constants[2] * np.power(
        (constants[7] * constants[4]) / (constants[6] * (1.0 - constants[4])),
        1.0 / constants[5],
    )

    hl = np.max(
        [
            0.0,
            1.0
            + constants[10]
            * (
                np.min([constants[0], 1.2])
                + np.min([np.min([constants[0], 1.2]), 0.87])
                - 1.87
            ),
        ]
    )

    F_ss = constants[3] * hl * XB_ss
    pCa50 = -np.log10(
        1e-6
        * constants[2]
        * np.power(
            (constants[7] * constants[4])
            / (constants[6] * (1.0 - constants[4])),
            1.0 / constants[5],
        )
    )
    nH = constants[8] * constants[5] * (1.0 - constants[4])

    return TRPN_ss, trpn_EC50, XB_ss, xb_EC50, F_ss, pCa50, nH, hl
