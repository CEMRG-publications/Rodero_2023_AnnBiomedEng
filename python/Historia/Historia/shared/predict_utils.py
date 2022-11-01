import numpy as np
from scipy import interpolate
from scipy.optimize import brentq


def poli2(x, p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    a = -(y1 - y2) / np.power(x1 - x2, 2)
    b = -2 * a * x1
    c = y2 - a * np.power(x2, 2) - b * x2

    return a * np.power(x, 2) + b * x + c


def csigma(x, p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    z = 0.5 * (x1 + x2)
    Delta = 0.1 * (y2 - y1)
    k = -np.log((y2 - y1) / Delta + 1) / (x1 - z)

    return y1 - Delta + (y2 - y1 + 2 * Delta) / (1 + np.exp(-k * (x - z)))


def kfun(k, p1, p2, MAXMINDP):
    x1, y1 = p1
    x2, y2 = p2
    z = 0.5 * (x1 + x2)

    return k * np.exp(-k * z) * (y2 - y1) * (1 + np.exp(-k * (x1 - z))) * (
        1 + np.exp(-k * (x2 - z))
    ) - MAXMINDP * (4 * (np.exp(-k * x1) - np.exp(-k * x2)))


def csigmadp(x, p1, p2, MAXMINDP):
    x1, y1 = p1
    x2, y2 = p2

    z = 0.5 * (x1 + x2)

    K = brentq(lambda k: kfun(k, p1, p2, MAXMINDP), -1, 1)

    A = (
        np.exp(-K * z)
        * (y1 * (1 + np.exp(-K * (x1 - z))) - y2 * (1 + np.exp(-K * (x2 - z))))
        / (np.exp(-K * x1) - np.exp(-K * x2))
    )
    B = (
        np.exp(-K * z)
        * (y2 - y1)
        * (1 + np.exp(-K * (x1 - z)))
        * (1 + np.exp(-K * (x2 - z)))
        / (np.exp(-K * x1) - np.exp(-K * x2))
    )

    return A + B * np.power(1 + np.exp(-K * (x - z)), -1)


def exponential(t, p1, p2, tau):
    t_max, y_max = p1
    t_f, y_f = p2

    C1 = (1 - np.exp(-t / tau)) / (1 - np.exp(-t_f / tau))
    C2 = (np.exp(-t / tau) - np.exp(-t_f / tau)) / (1 - np.exp(-t_f / tau))

    return C1 * y_f + C2 * y_max


def upvl(edp, y_mean):
    ylab = [
        "EDV",
        "ESV",
        "SV",
        "EF",
        "IVCT",
        "ET",
        "IVRT",
        "Tdiast",
        "PeakP",
        "Tpeak",
        "ESP",
        "maxdP",
        "mindP",
        "Tau",
    ]
    lvfd = {key: i for i, key in enumerate(ylab)}

    # ================
    # LV Volume
    # ================
    ivct = int(y_mean[lvfd["IVCT"]])
    t_ivct = np.linspace(0, ivct, ivct + 1)

    et = int(y_mean[lvfd["ET"]])
    t_et = np.linspace(ivct, ivct + et, et + 1)

    ivrt = int(y_mean[lvfd["IVRT"]])
    t_ivrt = np.linspace(ivct + et, ivct + et + ivrt, ivrt + 1)

    tdiast = int(y_mean[lvfd["Tdiast"]]) - ivrt
    t_diast = np.linspace(
        ivct + et + ivrt, ivct + et + ivrt + tdiast, tdiast + 1
    )

    p1 = t_ivct[0], y_mean[lvfd["EDV"]]
    p2 = t_ivct[-1], y_mean[lvfd["EDV"]]
    xp12 = np.linspace(p1[0], p2[0])

    p3 = t_ivct[-1], y_mean[lvfd["EDV"]]
    p4 = t_et[-1], y_mean[lvfd["ESV"]]
    xp34 = np.linspace(p3[0], p4[0])

    p5 = t_ivrt[0], y_mean[lvfd["ESV"]]
    p6 = t_ivrt[-1], y_mean[lvfd["ESV"]]
    xp56 = np.linspace(p5[0], p6[0])

    p7 = t_ivrt[-1], y_mean[lvfd["ESV"]]
    p8 = t_diast[-1], y_mean[lvfd["EDV"]]
    xp78 = np.linspace(p7[0], p8[0])

    t1, lvv1 = xp12, np.linspace(p1[1], p2[1])
    t2, lvv2 = xp34, csigma(xp34, p3, p4)
    t3, lvv3 = xp56, np.linspace(p5[1], p6[1])
    t4, lvv4 = xp78, csigma(xp78, p7, p8)

    tc = np.concatenate((t1[:-1], t2[:-1], t3[:-1], t4))
    lvvc = np.concatenate((lvv1[:-1], lvv2[:-1], lvv3[:-1], lvv4))

    t = np.linspace(tc[0], tc[-1], int(tc[-1]) + 1)
    flvv = interpolate.interp1d(tc, lvvc)
    lvv = flvv(t)

    # ================
    # LV Pressure
    # ================
    tpeak = int(y_mean[lvfd["Tpeak"]])
    t_peak = np.linspace(0, tpeak, tpeak + 1)

    dtp = t_peak[1] - t_peak[0]
    tpn = [t_peak[-1] - 5 * dtp, t_peak[-1] + 5 * dtp]

    p1 = t_ivct[0], edp
    p2 = tpn[0], y_mean[lvfd["PeakP"]]
    xp12 = np.linspace(p1[0], p2[0])

    p3 = tpn[0], y_mean[lvfd["PeakP"]]
    p4 = tpn[-1], y_mean[lvfd["PeakP"]]
    xp34 = np.linspace(p3[0], p4[0])

    p5 = tpn[-1], y_mean[lvfd["PeakP"]]
    p6 = t_et[-1], y_mean[lvfd["ESP"]]
    xp56 = np.linspace(p5[0], p6[0])

    p7 = 0, y_mean[lvfd["ESP"]]
    p8 = t_ivrt[-1] - t_et[-1], edp
    xp78 = np.linspace(p7[0], p8[0])

    p9 = t_ivrt[-1], edp
    p10 = t_diast[-1], edp
    xp910 = np.linspace(p9[0], p10[0])

    t1, lvp1 = xp12, csigmadp(xp12, p1, p2, y_mean[lvfd["maxdP"]])
    t2, lvp2 = xp34, np.linspace(p3[1], p4[1])
    t3, lvp3 = xp56, poli2(xp56, p5, p6)
    t4, lvp4 = xp78 + t_et[-1], exponential(xp78, p7, p8, y_mean[lvfd["Tau"]])
    t5, lvp5 = xp910, np.linspace(p9[1], p10[1])

    tc = np.concatenate((t1[:-1], t2[:-1], t3[:-1], t4[:-1], t5))
    lvpc = np.concatenate((lvp1[:-1], lvp2[:-1], lvp3[:-1], lvp4[:-1], lvp5))

    flvp = interpolate.interp1d(tc, lvpc)
    lvp = flvp(t)

    # ================
    # PV loop
    # ================
    return t, lvv, lvp
