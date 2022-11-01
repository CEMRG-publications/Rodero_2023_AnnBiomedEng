import numpy as np


def piecewise(condition, case1, case2):
    if condition:
        return case1
    else:
        return case2


def initStates(rat, hz):
    sizeStates = 18
    states = np.zeros((sizeStates,), dtype=float)

    if rat == "sham":
        if hz == 1:
            states[0] = -85.1221681609219
            states[1] = 8.46899983583716
            states[2] = 0.00208137744708665
            states[3] = 0.815520320018128
            states[4] = 0.815471795073686
            states[5] = 142.919492013701
            states[6] = 0.00146331830465093
            states[7] = 0.996934138278418
            states[8] = 0.78841193673441
            states[9] = 0.0019683140031203
            states[10] = 0.416987850222633
            states[11] = 0.00566123148325894
            states[12] = 0.000103020385969363
            states[13] = 0.96268028201207
            states[14] = 0.990016532916529
            states[15] = 0.00845823628523856
            states[16] = 0.00151233172289407
            states[17] = 0.0633670056927004
        elif hz == 6:
            states[0] = -85.5696059816548
            states[1] = 12.960204722764
            states[2] = 0.00194320513549252
            states[3] = 0.824266840236936
            states[4] = 0.722995966352439
            states[5] = 139.994789812443
            states[6] = 0.00140717973440432
            states[7] = 0.954569649396715
            states[8] = 0.300568843000913
            states[9] = 0.00189620611861644
            states[10] = 0.310616913661423
            states[11] = 0.00395453931879583
            states[12] = 0.000463188750506148
            states[13] = 1.0663135643117
            states[14] = 0.915759670501361
            states[15] = 0.0116308821918943
            states[16] = 0.0716996181298885
            states[17] = 0.0462772817360074

    elif rat == "ab":
        if hz == 1:
            states[0] = -84.8179335690707
            states[1] = 7.86198765026574
            states[2] = 0.00218089269066671
            states[3] = 0.807799685437556
            states[4] = 0.807494936350446
            states[5] = 139.964416466575
            states[6] = 0.00150276538930736
            states[7] = 0.996791125715503
            states[8] = 0.625815901322428
            states[9] = 0.0020195578095622
            states[10] = 0.368560872566697
            states[11] = 0.00484325258805042
            states[12] = 4.53106928940201e-5
            states[13] = 1.22566116672694
            states[14] = 0.989618266628688
            states[15] = 0.00828851292530188
            states[16] = 0.00207588449544264
            states[17] = 0.0669207270187171
        elif hz == 6:
            states[0] = -86.2742001770196
            states[1] = 10.83563050735
            states[2] = 0.00174392847776228
            states[3] = 0.831855538514125
            states[4] = 0.585528587217805
            states[5] = 140.1638657529
            states[6] = 0.00132309740749641
            states[7] = 0.913059928712858
            states[8] = 0.215036691884835
            states[9] = 0.00178817500396492
            states[10] = 0.209855226174927
            states[11] = 0.0025539544569868
            states[12] = 0.000145080891755077
            states[13] = 1.78499523287115
            states[14] = 0.650490049579104
            states[15] = 0.00570189155987283
            states[16] = 0.340821224038804
            states[17] = 0.0590608556435992

    return states


def initParams(p_dict):
    sizeParams = 17
    parameters = np.zeros((sizeParams,), dtype=float)

    for i, key in enumerate(p_dict.keys()):
        parameters[i] = p_dict[key]

    return parameters


def initConsts(parameters):
    sizeConsts = 71
    constants = np.zeros((sizeConsts,), dtype=float)

    constants[0] = 8314
    constants[1] = 310
    constants[2] = 96487
    constants[3] = 0.0001
    constants[4] = parameters[16]
    constants[5] = 3
    constants[6] = -0.0012
    constants[7] = 25850
    constants[8] = 2.585e-5
    constants[9] = 2.098e-6
    constants[10] = parameters[0]
    constants[11] = 140
    constants[12] = parameters[1]
    constants[13] = 0.883
    constants[14] = 0.117
    constants[15] = 5.4
    constants[16] = parameters[2]
    constants[17] = parameters[3]
    constants[18] = parameters[4]
    constants[19] = 0.2
    constants[20] = 8.015e-8
    constants[21] = 1.38e-7
    constants[22] = parameters[5]
    constants[23] = 3.6
    constants[24] = 19
    constants[25] = 22
    constants[26] = 880
    constants[27] = 0.3
    constants[28] = 1.8
    constants[29] = 1.8
    constants[30] = 0.1
    constants[31] = 0.02
    constants[32] = parameters[6]
    constants[33] = parameters[7]
    constants[34] = parameters[8]
    constants[35] = 7
    constants[36] = 11.5
    constants[37] = 1
    constants[38] = 1550
    constants[39] = 1.17
    constants[40] = 2.4
    constants[41] = 0.05
    constants[42] = 0.012
    constants[43] = 0.065
    constants[44] = parameters[9]
    constants[45] = 0.0625
    constants[46] = 14
    constants[47] = 0.01
    constants[48] = 100
    constants[49] = 87.5
    constants[50] = 1.38
    constants[51] = 0.35
    constants[52] = 0.1
    constants[53] = parameters[10]
    constants[54] = parameters[11]
    constants[55] = parameters[12]
    constants[56] = parameters[13]
    constants[57] = 0.00035
    constants[58] = parameters[14]
    constants[59] = parameters[15]
    constants[60] = 0.04
    constants[61] = 40
    constants[62] = 0.07
    constants[63] = 0.002382
    constants[64] = 0.05
    constants[65] = 0
    constants[66] = 0.00015
    constants[67] = 2100.00
    constants[68] = 1.00000 - constants[19]
    constants[69] = constants[36] / constants[37]
    constants[70] = constants[41] / constants[39]

    return constants


def computeAlgebraics(voi, states, constants):
    sizeAlgebraics = 85
    algebraics = np.zeros((sizeAlgebraics,), dtype=float)

    algebraics[8] = 1.00000 / (
        1.00000 + np.exp((states[0] + 87.5000) / 10.3000)
    )
    algebraics[1] = 1.00000 / (
        1.00000 + np.exp((states[0] + 45.0000) / -6.50000)
    )
    algebraics[11] = 1.36000 / (
        (0.320000 * (states[0] + 47.1300))
        / (1.00000 - np.exp(-0.100000 * (states[0] + 47.1300)))
        + 0.0800000 * np.exp(-states[0] / 11.0000)
    )
    algebraics[2] = 1.00000 / (
        1.00000 + np.exp((states[0] + 76.1000) / 6.07000)
    )
    algebraics[12] = piecewise(
        states[0] >= -40.0000,
        0.453700 * (1.00000 + np.exp(-(states[0] + 10.6600) / 11.1000)),
        3.49000
        / (
            0.135000 * np.exp(-(states[0] + 80.0000) / 6.80000)
            + 3.56000 * np.exp(0.0790000 * states[0])
            + 310000.0 * np.exp(0.350000 * states[0])
        ),
    )
    algebraics[3] = 1.00000 / (
        1.00000 + np.exp((states[0] + 76.1000) / 6.07000)
    )
    algebraics[13] = piecewise(
        states[0] >= -40.0000,
        (11.6300 * (1.00000 + np.exp(-0.100000 * (states[0] + 32.0000))))
        / np.exp(-2.53500e-07 * states[0]),
        3.49000
        / (
            (
                (states[0] + 37.7800)
                / (1.00000 + np.exp(0.311000 * (states[0] + 79.2300)))
            )
            * (
                -127140.0 * np.exp(0.244400 * states[0])
                - 3.47400e-05 * np.exp(-0.0439100 * states[0])
            )
            + (0.121200 * np.exp(-0.0105200 * states[0]))
            / (1.00000 + np.exp(-0.137800 * (states[0] + 40.1400)))
        ),
    )
    algebraics[14] = 100.000 / (
        45.1600 * np.exp(0.0357700 * (states[0] + 50.0000))
        + 98.9000 * np.exp(-0.100000 * (states[0] + 38.0000))
    )
    algebraics[4] = 1.00000 / (
        1.00000 + np.exp((states[0] + 10.6000) / -11.4200)
    )
    algebraics[15] = (
        20.0000 * np.exp(-(np.power((states[0] + 70.0000) / 25.0000, 2.00000)))
        + 35.0000
    )
    algebraics[5] = 1.00000 / (
        1.00000 + np.exp((states[0] + 45.3000) / 6.88410)
    )
    algebraics[16] = (
        1300.00 * np.exp(-(np.power((states[0] + 70.0000) / 30.0000, 2.00000)))
        + 35.0000
    )
    algebraics[6] = 1.00000 / (
        1.00000 + np.exp((states[0] + 45.3000) / 6.88410)
    )
    algebraics[17] = 10000.0 / (
        45.1600 * np.exp(0.0357700 * (states[0] + 50.0000))
        + 98.9000 * np.exp(-0.100000 * (states[0] + 38.0000))
    )
    algebraics[7] = 1.00000 / (
        1.00000 + np.exp((states[0] + 11.5000) / -11.8200)
    )
    algebraics[18] = 1000.00 / (
        0.118850 * np.exp((states[0] + 80.0000) / 28.3700)
        + 0.562300 * np.exp((states[0] + 80.0000) / -14.1900)
    )
    algebraics[9] = 1.00000 / (
        1.00000 + np.exp((states[0] + 138.600) / 10.4800)
    )
    algebraics[23] = ((constants[0] * constants[1]) / constants[2]) * np.log(
        constants[15] / states[5]
    )
    algebraics[24] = (
        constants[12]
        * states[6]
        * (constants[13] * states[7] + constants[14] * states[8])
        * (states[0] - algebraics[23])
    )
    algebraics[25] = (
        constants[16] * states[9] * states[10] * (states[0] - algebraics[23])
    )
    algebraics[26] = (
        (
            0.0480000
            / (
                np.exp((states[0] + 37.0000) / 25.0000)
                + np.exp((states[0] + 37.0000) / -25.0000)
            )
            + 0.0100000
        )
        * 0.00100000
    ) / (
        1.00000 + np.exp((states[0] - (algebraics[23] + 76.7700)) / -17.0000)
    ) + (
        constants[17] * (states[0] - (algebraics[23] + 1.73000))
    ) / (
        (
            1.00000
            + np.exp(
                (
                    1.61300
                    * constants[2]
                    * (states[0] - (algebraics[23] + 1.73000))
                )
                / (constants[0] * constants[1])
            )
        )
        * (1.00000 + np.exp((constants[15] - 0.998800) / -0.124000))
    )
    algebraics[31] = constants[21] * (states[0] - algebraics[23])
    algebraics[33] = constants[23] * (
        np.power(1.00000 + constants[24] / states[1], 2.00000)
    ) * (
        1.00000
        + (constants[25] / states[1])
        * np.exp(
            ((-constants[27] * constants[2] * states[0]) / constants[0])
            / constants[1]
        )
    ) + (
        np.power(1.00000 + constants[28] / constants[15], 2.00000)
    ) * (
        1.00000
        + (constants[11] / constants[26])
        * np.exp(
            (
                (-(1.00000 - constants[27]) * constants[2] * states[0])
                / constants[0]
            )
            / constants[1]
        )
    )
    algebraics[34] = (constants[22] * (constants[23] + 1.00000)) / algebraics[
        33
    ]
    algebraics[20] = piecewise(
        voi - np.floor(voi / constants[4]) * constants[4] >= 0.00000
        and voi - np.floor(voi / constants[4]) * constants[4] <= constants[5],
        constants[6],
        0.00000,
    )
    algebraics[28] = (
        constants[18]
        * states[11]
        * constants[68]
        * (states[0] - algebraics[23])
    )
    algebraics[0] = (constants[2] * states[0]) / (constants[0] * constants[1])
    algebraics[19] = 2.00000 * algebraics[0]
    algebraics[44] = piecewise(
        np.fabs(algebraics[19]) >= 1.00000e-09,
        (
            states[12]
            + (
                (constants[32] / constants[30])
                * constants[29]
                * algebraics[19]
                * np.exp(-algebraics[19])
            )
            / (1.00000 - np.exp(-algebraics[19]))
        )
        / (
            1.00000
            + ((constants[32] / constants[30]) * algebraics[19])
            / (1.00000 - np.exp(-algebraics[19]))
        ),
        (states[12] + (constants[32] / constants[30]) * constants[29])
        / (1.00000 + constants[32] / constants[30]),
    )
    algebraics[46] = (
        np.power(algebraics[44], 2.00000)
        + constants[47] * (np.power(constants[43], 2.00000))
    ) / (
        constants[40]
        * (
            np.power(algebraics[44], 2.00000)
            + np.power(constants[43], 2.00000)
        )
    )
    algebraics[40] = (
        np.power(states[12], 2.00000)
        + constants[47] * (np.power(constants[43], 2.00000))
    ) / (
        constants[40]
        * (np.power(states[12], 2.00000) + np.power(constants[43], 2.00000))
    )
    algebraics[35] = np.exp((states[0] - constants[34]) / constants[35])
    algebraics[36] = algebraics[35] / (
        constants[37] * (algebraics[35] + 1.00000)
    )
    algebraics[37] = (np.power(states[12], 2.00000)) / (
        constants[39]
        * (np.power(states[12], 2.00000) + np.power(constants[43], 2.00000))
    )
    algebraics[45] = (np.power(algebraics[44], 2.00000)) / (
        constants[39]
        * (
            np.power(algebraics[44], 2.00000)
            + np.power(constants[43], 2.00000)
        )
    )
    algebraics[52] = (algebraics[36] + constants[69]) * (
        (constants[69] + constants[70] + algebraics[45])
        * (constants[70] + algebraics[37])
        + algebraics[36] * (constants[70] + algebraics[45])
    )
    algebraics[53] = (
        algebraics[36]
        * constants[70]
        * (algebraics[36] + constants[69] + constants[70] + algebraics[37])
    ) / algebraics[52]
    algebraics[56] = (
        constants[69]
        * constants[70]
        * (constants[69] + algebraics[36] + constants[70] + algebraics[45])
    ) / algebraics[52]
    algebraics[57] = (
        algebraics[53] * algebraics[46] + algebraics[56] * algebraics[40]
    )
    algebraics[47] = (
        constants[42]
        * constants[48]
        * (
            np.power(algebraics[44], 2.00000)
            + constants[47] * (np.power(constants[43], 2.00000))
        )
    ) / (
        constants[40]
        * (
            constants[48] * (np.power(algebraics[44], 2.00000))
            + constants[47] * (np.power(constants[43], 2.00000))
        )
    )
    algebraics[41] = (
        constants[42]
        * constants[48]
        * (
            np.power(states[12], 2.00000)
            + constants[47] * (np.power(constants[43], 2.00000))
        )
    ) / (
        constants[40]
        * (
            constants[48] * (np.power(states[12], 2.00000))
            + constants[47] * (np.power(constants[43], 2.00000))
        )
    )
    algebraics[59] = (
        algebraics[36] * algebraics[47] + constants[69] * algebraics[41]
    ) / (algebraics[36] + constants[69])
    algebraics[42] = (
        states[12] + (constants[31] / constants[30]) * states[13]
    ) / (1.00000 + constants[31] / constants[30])
    algebraics[43] = (algebraics[42] * (algebraics[35] + constants[45])) / (
        constants[38] * constants[44] * (algebraics[35] + 1.00000)
    )
    algebraics[38] = (states[12] * (algebraics[35] + constants[45])) / (
        constants[38] * constants[44] * (algebraics[35] + 1.00000)
    )
    algebraics[54] = (
        constants[69]
        * (
            algebraics[37] * (constants[69] + constants[70] + algebraics[45])
            + algebraics[45] * algebraics[36]
        )
    ) / algebraics[52]
    algebraics[65] = (
        algebraics[54] * algebraics[43] + algebraics[56] * algebraics[38]
    )
    algebraics[39] = (constants[46] * (algebraics[35] + constants[45])) / (
        constants[38] * (constants[46] * algebraics[35] + constants[45])
    )
    algebraics[67] = algebraics[39]
    algebraics[69] = (constants[69] * algebraics[38]) / (
        algebraics[36] + constants[69]
    )
    algebraics[71] = algebraics[39]
    algebraics[73] = ((1.00000 - states[14]) - states[15]) - states[16]
    algebraics[61] = (constants[70] * algebraics[40]) / (
        constants[70] + algebraics[37]
    )
    algebraics[63] = algebraics[41]
    algebraics[21] = ((constants[0] * constants[1]) / constants[2]) * np.log(
        constants[11] / states[1]
    )
    algebraics[22] = (
        constants[10]
        * (np.power(states[2], 3.00000))
        * states[3]
        * states[4]
        * (states[0] - algebraics[21])
    )
    algebraics[30] = constants[20] * (states[0] - algebraics[21])
    algebraics[74] = (
        constants[53]
        * (
            np.exp(constants[51] * algebraics[0])
            * (np.power(states[1], 3.00000))
            * constants[29]
            - np.exp((constants[51] - 1.00000) * algebraics[0])
            * (np.power(constants[11], 3.00000))
            * states[12]
        )
    ) / (
        (np.power(constants[11], 3.00000) + np.power(constants[49], 3.00000))
        * (constants[29] + constants[50])
        * (
            1.00000
            + constants[52] * np.exp((constants[51] - 1.00000) * algebraics[0])
        )
    )
    algebraics[75] = algebraics[74] * constants[2] * constants[8]
    algebraics[27] = (
        constants[18]
        * states[11]
        * constants[19]
        * (states[0] - algebraics[21])
    )
    algebraics[29] = algebraics[27] + algebraics[28]
    algebraics[77] = (constants[56] * states[12]) / (
        constants[57] + states[12]
    )
    algebraics[78] = algebraics[77] * 2.00000 * constants[2] * constants[8]
    algebraics[51] = piecewise(
        np.fabs(algebraics[19]) > 1.00000e-05,
        (
            (
                (constants[32] * algebraics[19])
                / (1.00000 - np.exp(-algebraics[19]))
            )
            * (
                (constants[29] * np.exp(-algebraics[19]) - states[12])
                + (constants[31] / constants[30])
                * (constants[29] * np.exp(-algebraics[19]) - states[13])
            )
        )
        / (
            1.00000
            + constants[31] / constants[30]
            + ((constants[32] / constants[30]) * algebraics[19])
            / (1.00000 - np.exp(algebraics[19]))
        ),
        (
            ((constants[32] * 1.00000e-05) / (1.00000 - np.exp(-1.00000e-05)))
            * (
                (constants[29] * np.exp(-1.00000e-05) - states[12])
                + (constants[31] / constants[30])
                * (constants[29] * np.exp(-1.00000e-05) - states[13])
            )
        )
        / (
            1.00000
            + constants[31] / constants[30]
            + ((constants[32] / constants[30]) * 1.00000e-05)
            / (1.00000 - np.exp(-1.00000e-05))
        ),
    )
    algebraics[50] = piecewise(
        np.fabs(algebraics[19]) > 1.00000e-05,
        (
            (
                (constants[32] * algebraics[19])
                / (1.00000 - np.exp(-algebraics[19]))
            )
            * (constants[29] * np.exp(-algebraics[19]) - states[12])
        )
        / (
            1.00000
            + ((constants[32] / constants[30]) * algebraics[19])
            / (1.00000 - np.exp(-algebraics[19]))
        ),
        (
            ((constants[32] * 1.00000e-05) / (1.00000 - np.exp(-1.00000e-05)))
            * (constants[29] * np.exp(-1.00000e-05) - states[12])
        )
        / (
            1.00000
            + ((constants[32] / constants[30]) * 1.00000e-05)
            / (1.00000 - np.exp(-1.00000e-05))
        ),
    )
    algebraics[55] = (
        algebraics[36]
        * (
            algebraics[45] * (algebraics[36] + constants[70] + algebraics[37])
            + algebraics[37] * constants[69]
        )
    ) / algebraics[52]
    algebraics[66] = (
        algebraics[51] * algebraics[55] + algebraics[50] * algebraics[53]
    )
    algebraics[68] = (algebraics[50] * algebraics[36]) / (
        algebraics[36] + constants[69]
    )
    algebraics[70] = (
        (states[14] * algebraics[66] + states[15] * algebraics[68])
        * constants[33]
    ) / constants[7]
    algebraics[72] = -algebraics[70] * 2.00000 * constants[2] * constants[8]
    algebraics[79] = (
        (constants[0] * constants[1]) / (2.00000 * constants[2])
    ) * np.log(constants[29] / states[12])
    algebraics[80] = constants[58] * (algebraics[79] - states[0])
    algebraics[81] = -algebraics[80] * 2.00000 * constants[2] * constants[8]
    algebraics[48] = (constants[31] * (states[13] - states[12])) / (
        1.00000 + constants[31] / constants[30]
    )
    algebraics[49] = piecewise(
        np.fabs(algebraics[19]) > 1.00000e-05,
        (
            constants[31]
            * (
                (states[13] - states[12])
                + (
                    ((constants[32] / constants[30]) * algebraics[19])
                    / (1.00000 - np.exp(-algebraics[19]))
                )
                * (states[13] - constants[29] * np.exp(-algebraics[19]))
            )
        )
        / (
            1.00000
            + constants[31] / constants[30]
            + ((constants[32] / constants[30]) * algebraics[19])
            / (1.00000 - np.exp(-algebraics[19]))
        ),
        (
            constants[31]
            * (
                (states[13] - states[12])
                + (
                    ((constants[32] / constants[30]) * 1.00000e-05)
                    / (1.00000 - np.exp(-1.00000e-05))
                )
                * (states[13] - constants[29] * np.exp(-1.00000e-05))
            )
        )
        / (
            1.00000
            + constants[31] / constants[30]
            + ((constants[32] / constants[30]) * 1.00000e-05)
            / (1.00000 - np.exp(-1.00000e-05))
        ),
    )
    algebraics[58] = (
        algebraics[55] * algebraics[49] + algebraics[48] * algebraics[54]
    )
    algebraics[60] = (algebraics[48] * algebraics[37]) / (
        constants[70] + algebraics[37]
    )
    algebraics[62] = (
        (states[14] * algebraics[58] + states[16] * algebraics[60])
        * constants[33]
    ) / constants[7]
    algebraics[64] = algebraics[62]
    algebraics[76] = (constants[54] * (np.power(states[12], 2.00000))) / (
        np.power(constants[55], 2.00000) + np.power(states[12], 2.00000)
    )
    algebraics[82] = constants[59] * (states[13] - states[12])
    algebraics[83] = (
        constants[60] * (constants[62] - states[17])
        - constants[61] * states[17] * states[12]
    )
    algebraics[84] = np.power(
        1.00000
        + (constants[63] * constants[64])
        / (np.power(constants[63] + states[12], 2.00000))
        + (constants[65] * constants[66])
        / (np.power(constants[66] + states[12], 2.00000)),
        -1.00000,
    )
    algebraics[10] = states[12]
    algebraics[32] = algebraics[30] + algebraics[31]

    return algebraics


def computeRates(voi, states, p_dict):
    parameters = initParams(p_dict)
    constants = initConsts(parameters)
    algebraics = computeAlgebraics(voi, states, constants)

    sizeRates = 18
    rates = np.zeros((sizeRates,), dtype=float)

    rates[0] = (
        -(
            algebraics[22]
            + algebraics[24]
            + algebraics[25]
            + algebraics[29]
            + algebraics[26]
            + algebraics[30]
            + algebraics[31]
            + algebraics[34]
            + algebraics[81]
            + algebraics[75]
            + algebraics[78]
            + algebraics[72]
            + algebraics[20]
        )
        / constants[3]
    )
    rates[1] = (
        -(
            algebraics[22]
            + algebraics[30]
            + algebraics[75] * 3.00000
            + algebraics[34] * 3.00000
            + algebraics[27]
        )
        * 1.00000
    ) / (constants[8] * constants[2])
    rates[2] = (algebraics[1] - states[2]) / algebraics[11]
    rates[3] = (algebraics[2] - states[3]) / algebraics[12]
    rates[4] = (algebraics[3] - states[4]) / algebraics[13]
    rates[5] = (
        -(
            algebraics[20]
            + algebraics[25]
            + algebraics[31]
            + algebraics[24]
            + algebraics[26]
            + algebraics[28]
            + -2.00000 * algebraics[34]
        )
        * 1.00000
    ) / (constants[8] * constants[2])
    rates[6] = (algebraics[4] - states[6]) / algebraics[14]
    rates[7] = (algebraics[5] - states[7]) / algebraics[15]
    rates[8] = (algebraics[6] - states[8]) / algebraics[16]
    rates[9] = (algebraics[7] - states[9]) / algebraics[17]
    rates[10] = (algebraics[8] - states[10]) / constants[67]
    rates[11] = (algebraics[9] - states[11]) / algebraics[18]
    rates[12] = algebraics[84] * (
        ((algebraics[64] - algebraics[76]) + algebraics[82] + algebraics[83])
        - (
            -2.00000 * algebraics[75]
            + algebraics[72]
            + algebraics[78]
            + algebraics[81]
        )
        / (2.00000 * constants[8] * constants[2])
    )
    rates[13] = (constants[8] / constants[9]) * (
        (-algebraics[64] + algebraics[76]) - algebraics[82]
    )
    rates[14] = (
        -(algebraics[57] + algebraics[65]) * states[14]
        + algebraics[59] * states[15]
        + algebraics[67] * states[16]
    )
    rates[15] = (
        algebraics[57] * states[14]
        - (algebraics[59] + algebraics[69]) * states[15]
    ) + algebraics[71] * algebraics[73]
    rates[16] = (
        algebraics[65] * states[14]
        - (algebraics[67] + algebraics[61]) * states[16]
    ) + algebraics[63] * algebraics[73]
    rates[17] = algebraics[83]

    return rates
