import itertools

import numpy as np


def ell(i, x):
    r = np.linspace(0, 1, 4)
    ind = list(set(range(4)) - set([i]))

    num = (x - r[ind[0]]) * (x - r[ind[1]]) * (x - r[ind[2]])
    den = (r[i] - r[ind[0]]) * (r[i] - r[ind[1]]) * (r[i] - r[ind[2]])

    return num / den


def psi(i, x):
    ind = [0, 1, 2, 3]
    L_ = [p for p in itertools.product(ind, repeat=3)]
    L_split = [l for l in np.vsplit(np.array(L_), 4)]
    L = []
    for j in range(16):
        for k in range(4):
            L.append(list(L_split[k][4 * j // 16 + 4 * j % 16]))

    return ell(L[i][0], x[0]) * ell(L[i][1], x[1]) * ell(L[i][2], x[2])


def PSI(c, x):
    val = 0
    for i in range(64):
        val += c[i, :] * psi(i, x)
    return val


def T(c, xhat):
    return PSI(c, xhat)


def exnode(file):
    A = []
    with open(file, "r") as f:
        for _ in range(7):
            next(f)
        for i, line in enumerate(f):
            if i % 2:
                A.append([float(line.split()[i]) for i in range(3)])
    A = np.array(A)

    return A


def exelem(file):
    B = []
    with open(file, "r") as f:
        for _ in range(588):
            next(f)
        for i, line in enumerate(f):
            if i % 2 == 1:
                B.append([int(line.split()[1:][j]) for j in range(64)])
    B = np.array(B)

    return B


def exdata_xi(file):
    Ct = []
    Ft = []
    JXt = []
    with open(file, "r") as f:
        for _ in range(17):
            next(f)
        for i, line in enumerate(f):
            if (i - 5) % 4 == 0:
                Ct.append([float(line.split()[j]) for j in range(3, 6)])
            elif (i - 6) % 4 == 0:
                Ft.append([float(line.split()[j]) for j in range(3)])
            elif (i - 7) % 4 == 0:
                JXt.append(float(line))
    Ct = np.array(Ct)
    Ft = np.array(Ft)
    JXt = np.array(JXt).reshape(-1, 1)

    C = [c for c in np.vsplit(Ct, 108)]
    F = [f for f in np.vsplit(Ft, 108)]
    JX = [j for j in np.vsplit(JXt, 108)]

    return C, F, JX


def exdata(file):
    Cat = []
    Tat = []
    lambdt = []
    dlambdt = []
    with open(file, "r") as f:
        for _ in range(22):
            next(f)
        for i, line in enumerate(f):
            if (i - 4) % 3 == 0:
                v = [float(line.split()[j]) for j in range(1, 5)]
                Cat.append(v[0])
                Tat.append(v[1])
                lambdt.append(v[2])
                dlambdt.append(v[3])

    Cat = np.array(Cat).reshape(-1, 1)
    Tat = np.array(Tat).reshape(-1, 1)
    lambdt = np.array(lambdt).reshape(-1, 1)
    dlambdt = np.array(dlambdt).reshape(-1, 1)

    Ca = [c for c in np.vsplit(Cat, 108)]
    Ta = [t for t in np.vsplit(Tat, 108)]
    lambd = [l for l in np.vsplit(lambdt, 108)]
    dlambd = [dl for dl in np.vsplit(dlambdt, 108)]

    return Ca, Ta, lambd, dlambd


def connect(B, n_nodes, n_elements):
    L = []
    for i in range(n_nodes):
        l = []
        for j in range(n_elements):
            if any(B[j, :] == i + 1):
                l.append(j + 1)
        L.append(l)
    return L
