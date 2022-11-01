import numpy as np
from scipy.integrate import simps, solve_ivp

from Historia.shared import design_utils as desu
from Historia.shared import mesh_utils as mesu


def save_Ca_Ta_lambda_dlambdadt(path_to_sim, path_to_out):
    """
    Works with 4 heart beats simulations performed with -exdata_init option on.
    Reads everything you need from *.exdata files to estimate XB(t) and further estimate ATP(t).
    Read data are Ca, Ta, lambda, dlambda_dt: they are stored in *.txt files, 4 files for each element.
    These files are saved into an existing folder specified by the user.
    Each A_elem_B.txt file will contain A variation in time at each of the 64 Gauss points in the ref. configuration of B element.
    """
    Ca = []
    Ta = []
    lambd = []
    dlambd = []

    for ti in range(664):
        if ti < 10:
            file = path_to_sim + "t_000" + str(ti) + "00.exdata"
        elif 10 <= ti and ti < 100:
            file = path_to_sim + "t_00" + str(ti) + "00.exdata"
        else:
            file = path_to_sim + "t_0" + str(ti) + "00.exdata"

        data = mesu.exdata(file)

        Ca.append(data[0])
        Ta.append(data[1])
        lambd.append(data[2])
        dlambd.append(data[3])

    for elem in range(108):
        Ca_v = np.zeros((64, 1), dtype=float)
        Ta_v = np.zeros((64, 1), dtype=float)
        lambd_v = np.zeros((64, 1), dtype=float)
        dlambd_v = np.zeros((64, 1), dtype=float)

        for ti in range(664):
            Ca_v = np.hstack((Ca_v, Ca[ti][elem]))
            Ta_v = np.hstack((Ta_v, Ta[ti][elem]))
            lambd_v = np.hstack((lambd_v, lambd[ti][elem]))
            dlambd_v = np.hstack((dlambd_v, dlambd[ti][elem]))

        desu.write_txt(
            Ca_v[:, 1:], "%f", path_to_out + "Ca_elem_{}".format(elem + 1)
        )
        desu.write_txt(
            Ta_v[:, 1:], "%f", path_to_out + "Ta_elem_{}".format(elem + 1)
        )
        desu.write_txt(
            lambd_v[:, 1:],
            "%f",
            path_to_out + "lambda_elem_{}".format(elem + 1),
        )
        desu.write_txt(
            dlambd_v[:, 1:],
            "%f",
            path_to_out + "dlambda_dt_elem_{}".format(elem + 1),
        )

    return


def qfac(Q1, Q2, c):
    a = c[10]
    Q = Q1 + Q2
    if Q < 0.0:
        return (a * Q + 1.0) / (1.0 - Q)
    else:
        return (1.0 + (a + 2.0) * Q) / (1.0 + Q)


def overlap(l, c):
    beta0 = c[6]
    return np.max(
        [
            0,
            1.0
            + beta0
            * (np.min([l, 1.2]) + np.min([np.min([l, 1.2]), 0.87]) - 1.87),
        ]
    )


def tension(qfac, overlap, XB, c):
    Tref = c[-1]
    return Tref * qfac * overlap * XB


def rhs(t, y, calcium, lambd, dlambd, c):
    dy = np.zeros((5,), dtype=float)

    ca50ref = c[0]
    perm50 = c[1]
    kxb = c[2]
    ktrpn = c[3]
    nperm = c[4]
    ntrpn = c[5]
    beta1 = c[7]
    alpha1 = c[8]
    alpha2 = c[9]
    A1 = c[11]
    A2 = c[12]

    Cai = calcium[int(t)]
    lambdi = lambd[int(t)]
    dlambdi = dlambd[int(t)]

    ca50 = ca50ref * (
        1.0 + beta1 * (np.min([1.2, np.max([0.8, lambdi])]) - 1.0)
    )
    permtot = np.power((y[1] / perm50), nperm / 2.0)
    inprmt = np.min([100.0, 1.0 / permtot])

    dy[0] = kxb * (permtot * (1.0 - y[0]) - inprmt * y[0])
    dy[1] = ktrpn * (np.power(Cai / ca50, ntrpn) * (1.0 - y[1]) - y[1])
    dy[2] = A1 * dlambdi - alpha1 * y[2]
    dy[3] = A2 * dlambdi - alpha2 * y[3]
    dy[4] = kxb * inprmt * y[0]

    return dy


def ind_list(C):
    xi = [0.069, 0.330, 0.670, 0.931]
    CN = []
    for i in range(64):
        v = []
        for j in range(3):
            if C[i, j] == xi[0]:
                v.append(0)
            elif C[i, j] == xi[1]:
                v.append(1)
            elif C[i, j] == xi[2]:
                v.append(2)
            else:
                v.append(3)
        CN.append(v)

    return CN


def integrate_elem(C, JX, f):
    wi = np.array([0.17392742, 0.32607258, 0.32607258, 0.17392742])
    I = 0.0
    for i in range(64):
        I += wi[C[i]][0] * wi[C[i]][1] * wi[C[i]][2] * f[i] * JX[i]

    return I


def cell_contraction(path_to_in, elem, node):
    t = np.arange(664)
    tspan = [0.0, 663]

    Y0 = [0.01, 0.1, 0.0, 0.0, 0.01]

    # hard-coded (change them according to your specific simulation setup)
    ca50ref = 0.572458  # 1.45837
    perm50 = 0.35
    kxb = 0.01125  # 0.019156
    ktrpn = 0.106601  # 0.052712
    nperm = 5.0
    ntrpn = 2.0
    beta0 = 1.65
    beta1 = -1.5
    alpha1 = 0.15
    alpha2 = 0.5
    a = 0.35
    A1 = -29.0
    A2 = 116.0
    Tref = 126.12  # 136.242

    c = [
        ca50ref,
        perm50,
        kxb,
        ktrpn,
        nperm,
        ntrpn,
        beta0,
        beta1,
        alpha1,
        alpha2,
        a,
        A1,
        A2,
        Tref,
    ]

    Ca = desu.read_txt(path_to_in + "Ca_elem_{}".format(elem + 1), "float64")[
        node
    ]
    Ta = desu.read_txt(path_to_in + "Ta_elem_{}".format(elem + 1), "float64")[
        node
    ]
    lambd = desu.read_txt(
        path_to_in + "lambda_elem_{}".format(elem + 1), "float64"
    )[node]
    dlambd = desu.read_txt(
        path_to_in + "dlambda_dt_elem_{}".format(elem + 1), "float64"
    )[node]

    Y = solve_ivp(
        fun=lambda t, y: rhs(t, y, Ca, lambd, dlambd, c),
        t_span=tspan,
        y0=Y0,
        method="BDF",
        t_eval=t,
        max_step=1.0,
    )

    gq = np.array([qfac(Y.y[2, i], Y.y[3, i], c) for i in range(664)])
    hl = np.array([overlap(lambd[i], c) for i in range(664)])
    Ta_sim = tension(gq, hl, Y.y[0, :], c)

    return t, Y.y, Ta, Ta_sim


def save_XB(path_to_in, path_to_out):
    for elem in range(108):
        XB = np.zeros((1, 664), dtype=float)
        for node in range(64):
            _, Y, _, _ = cell_contraction(path_to_in, elem, node)
            XB = np.vstack((XB, Y[0, :].reshape(1, -1)))

        desu.write_txt(
            XB[1:, :], "%f", path_to_out + "XB_elem_{}".format(elem + 1)
        )

    return


def compute_ATP(path_to_in):
    kxb = 0.01125  # 0.019156
    t = np.arange(664)

    xifile = path_to_in + "xi.exdata"
    Ct, _, JX = mesu.exdata_xi(xifile)
    C = ind_list(Ct[0])

    XB = []
    for elem in range(108):
        XB.append(
            desu.read_txt(
                path_to_in + "XB_elem_{}".format(elem + 1), "float64"
            )
        )

    It = []
    for ti in range(664):
        I = 0.0
        for elem in range(108):
            I += integrate_elem(C, JX[elem], XB[elem][:, ti])
        It.append(kxb * I)
    It = np.array(It).ravel()

    ATP = []
    for ti in range(664):
        ATP.append(simps(It[: ti + 1], t[: ti + 1]))
    ATP = np.array(ATP).ravel()

    return t, It, ATP
