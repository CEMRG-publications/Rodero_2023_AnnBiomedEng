import numpy as np

from Historia.mech import mech_out as glv
from Historia.mech import scan_logfile as slf


def extract_features(path_in, dim, path_out, nc):
    XA = np.zeros((0, 12), dtype=float)
    X = np.zeros((0, 12), dtype=float)
    YA = []
    Y = np.zeros((0, 14), dtype=float)
    l = []
    for i in range(dim):
        tag = path_in + str(i + 1) + "/output_log.txt"
        S = slf.MECHSolution(tag)
        try:
            S.extract_loginfo()
        except FileNotFoundError:
            print("\n=== [Index: {}] Logfile not found!".format(i + 1))
            continue

        XA = np.vstack((XA, np.array(S.p)))

        RS = glv.LeftVentricle()
        RS.get_lvfeatures(S, nc)

        YA.append(RS.conv)
        if RS.conv:
            l.append(i + 1)
            X = np.vstack((X, np.array(S.p)))
            Y = np.vstack((Y, np.array(RS.f)))

    np.savetxt(path_out + "_inputs.txt", XA, fmt="%.6f")
    np.savetxt(path_out + "_conly_inputs.txt", X, fmt="%.6f")
    np.savetxt(path_out + "_outputs.txt", YA, fmt="%d")
    np.savetxt(path_out + "_conly_outputs.txt", Y, fmt="%.6f")
    np.savetxt(path_out + "_lconv.txt", l, fmt="%d")

    return None
