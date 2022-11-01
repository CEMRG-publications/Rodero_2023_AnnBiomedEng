import numpy as np


class LeftVentricle:
    """This class implements the left ventricular (LV) pump function."""

    def __init__(self):
        self.conv = 0
        self.t = []
        self.phase = []
        self.lv_v = []
        self.lv_p = []
        self.f = []

    def get_lvfeatures(self, S, nc):
        """Extract the LV features from the last heart beat.
        Args:
                S: mechanics solution object obtained through the dedicated module 'scan_logfile.py'
                nc: number of simulated heart beats.
        """
        if S.conv:
            v = []
            if S.phase[0] != S.phase[1]:
                v = S.phase[1:]
            elif S.phase[-2] != S.phase[-1]:
                v = S.phase[:-1]
            else:
                v = S.phase

            cp = ph_counter(v)

            if cp[3] >= nc + 1:
                self.conv = 1

            if self.conv:
                M = []
                for i in range(4):
                    M.append(
                        isl_ranges(np.where(np.array(v) == i + 1)[0], cp[i])
                    )

                ind = np.sort(
                    np.concatenate(
                        [np.reshape(M[i], cp[i] * 2) for i in range(4)], axis=0
                    )
                )

                last = np.argmax([M[i][-1, 1] for i in range(4)])
                if last == 3:
                    rmv = 8
                elif last == 2:
                    rmv = 6
                elif last == 1:
                    rmv = 4
                else:
                    rmv = 2

                ind = ind[:-rmv]
                ind_r = ind[-8:]
                interval = range(ind_r[0], ind_r[-1] + 1)

                self.t = [S.t[i] for i in interval]
                self.phase = [S.phase[i] for i in interval]
                self.lv_p = [S.lv_p[i] for i in interval]
                self.lv_v = [S.lv_v[i] for i in interval]

                time = [S.t[i] for i in ind_r]

                ps1 = list(np.where(np.array(self.phase) == 1)[0])
                lvv1 = [self.lv_v[i] for i in ps1]

                p1 = np.max(lvv1)  # EDV (end-diastolic volume)
                p2 = S.lv_v[ind_r[3]]  # ESV (end-systolic volume)
                p3 = p1 - p2  # SV (stroke volume)
                p4 = 100 * p3 / p1  # EF (ejection fraction)
                p5 = time[1] - time[0]  # IVCT (isovolumetric contraction time)
                p6 = time[3] - time[2]  # ET (ejection time)
                p7 = time[5] - time[4]  # IVRT (isovolumetric relaxation time)
                p8 = time[7] - time[4]  # Tdiast (diastolic time)

                ind_max = np.argmax(self.lv_p)

                q1 = self.lv_p[ind_max]  # PeakP (peak pressure)
                q2 = (
                    self.t[ind_max] - self.t[0]
                )  # Tpeak (time to peak pressure)
                q3 = S.lv_p[ind_r[3]]  # ESP (end-systolic pressure)

                dP = np.gradient(self.lv_p, self.t)
                dP = check_der(dP)

                idx_relax = np.where(np.array(self.lv_p) == q3)[0][0]
                idx_min = np.argmin(dP[idx_relax:])
                idx_min += idx_relax

                PA = 2 * (self.lv_p[idx_min] - np.min(self.lv_p[idx_min:]))
                tau = -0.25 * PA / dP[idx_min]

                q4 = np.max(dP)  # maxdP (maximum pressure rise rate)
                q5 = dP[idx_min]  # mindP (maximum pressure decay rate)
                q6 = tau  # Tau logistic (isovolumetric pressure relaxation constant)

                self.f = [
                    p1,
                    p2,
                    p3,
                    p4,
                    p5,
                    p6,
                    p7,
                    p8,
                    q1,
                    q2,
                    q3,
                    q4,
                    q5,
                    q6,
                ]


def ph_counter(phase):
    n = len(phase)

    cp = [0, 0, 0, 0]
    for i in range(4):
        j = 0
        while j < n - 1:
            if phase[j] == i + 1:
                cp[i] = cp[i] + 1
                j = j + 1
                while phase[j] == i + 1 and j < n - 1:
                    j = j + 1
            else:
                j = j + 1

    return cp


def isl_ranges(l, n_isl):
    len_l = len(l)
    islands = 0
    i = 1

    M = np.zeros((n_isl, 2), dtype=int)
    M[0, 0] = l[0]
    M[-1, -1] = l[-1]

    while i < len_l:
        if l[i] != l[i - 1] + 1:
            M[islands, 1] = l[i - 1]
            islands = islands + 1
            M[islands, 0] = l[i]
        i = i + 1

    return M


def check_der(y):
    delta = y[:-1] - y[1:]
    mean, std = np.mean(delta), np.std(delta)
    l = list(np.where(np.abs(delta) > mean + 3 * std)[0])
    if l:
        for idx in l:
            if delta[idx] > 0:
                y[idx + 1] = y[idx]
            elif delta[idx] < 0:
                y[idx] = y[idx + 1]
    return y
