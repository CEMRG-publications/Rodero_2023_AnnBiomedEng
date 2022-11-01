import numpy as np
from scipy import interpolate


def init_header(rat, n_stims, filename):
    with open("headers/" + rat + "/MeshFlatBase.in", "r") as f:
        with open(filename + ".in", "w") as fh:
            for line in f:
                fh.write(line)
            fh.write("\n")
            fh.write("stims: {}".format(n_stims))
            fh.write("\n")
    return


def append_to_header(f, stim, ca):
    f.write(f"ca{stim} {len(ca)-1} 1 ")
    np.savetxt(f, ca.reshape(1, -1), fmt="%.6f")
    return


def calculate_bio(t, Ca):
    DCA = Ca[0]
    imax = np.argmax(Ca)

    idx = [imax - 1, imax, imax + 1]
    density = 10 * (len(idx) - 1)
    tck = interpolate.splrep(
        [t[i] for i in idx], [Ca[i] for i in idx], s=0, k=2
    )
    t_new = np.linspace(t[idx[0]], t[idx[-1]], density)
    Ca_new = interpolate.splev(t_new, tck, der=0)
    imax_new = np.argmax(Ca_new)

    PCA = Ca_new[imax_new]
    TP = t_new[imax_new]

    idx = list(range(imax + 1, len(t)))
    density = 10 * (len(idx) - 1)
    tck = interpolate.splrep(
        [t[i] for i in idx], [Ca[i] for i in idx], s=0, k=1
    )
    t_new = np.linspace(t[idx[0]], t[idx[-1]], density)
    Ca_new = interpolate.splev(t_new, tck, der=0)

    RT50 = -1
    i = 0
    while i < density:
        if Ca_new[i] <= 0.5 * (DCA + PCA):
            RT50 = t_new[i] - TP
            break
        i += 1

    return np.array([DCA, PCA - DCA, TP, RT50])


def build_ca(t, Ca, p):
    A = p[0]
    B = p[1]
    C = p[2]
    D = p[3]

    DCA = Ca[0]
    Ca_new = A * DCA + B * (Ca - DCA)

    imax = np.argmax(Ca)
    T = t[-1]
    valid = False

    if C * (t[imax] - t[0]) + D * (t[-1] - t[imax]) <= T:
        valid = True
        t_tmp = np.array(
            list(C * t[: imax + 1])
            + list(C * t[imax] + D * (t[imax + 1 : -1] - t[imax]))
            + [t[-1]]
        )
        f = interpolate.interp1d(t_tmp, Ca_new)
        Ca_new = f(t)

    return valid, Ca_new
