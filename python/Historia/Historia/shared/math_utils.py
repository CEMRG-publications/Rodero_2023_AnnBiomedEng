import numpy as np


def mare(Y_true, Y_pred):
    sample_dim = Y_true.shape[0]
    out_dim = Y_true.shape[1]
    if out_dim == 1:
        return np.linalg.norm((Y_pred - Y_true) / Y_true, ord=1) / sample_dim
    else:
        e = np.zeros((sample_dim,), dtype=float)
        for i in range(sample_dim):
            e[i] = (
                np.linalg.norm(
                    (Y_pred[i, :] - Y_true[i, :]) / Y_true[i, :], ord=1
                )
                / out_dim
            )
        return np.sum(e) / sample_dim


def centroid(M):
    c = []
    for j in range(M.shape[1]):
        c.append(np.sum(M[:, j]) / M.shape[0])
    return c


def der(t, y):
    N = len(t)
    dt = (t[-1] - t[0]) / N

    p2 = [-1.0 / 2.0, 1.0 / 2.0]
    p4 = [1.0 / 12.0, -2.0 / 3.0, 2.0 / 3.0, -1.0 / 12.0]
    p6 = [
        -1.0 / 60.0,
        3.0 / 20.0,
        -3.0 / 4.0,
        3.0 / 4.0,
        -3.0 / 20.0,
        1.0 / 60.0,
    ]
    p8 = [
        1.0 / 280.0,
        -4.0 / 105.0,
        1.0 / 5.0,
        -4.0 / 5.0,
        4.0 / 5.0,
        -1.0 / 5.0,
        4.0 / 105.0,
        -1.0 / 280.0,
    ]

    dy = N * [0.0]

    dy[0] = (y[1] - y[0]) / dt
    dy[-1] = (y[-1] - y[-2]) / dt

    dy[1] = (p2[0] * y[0] + p2[1] * y[2]) / dt
    dy[-2] = (p2[0] * y[-3] + p2[1] * y[-1]) / dt

    dy[2] = (p4[0] * y[0] + p4[1] * y[1] + p4[2] * y[3] + p4[3] * y[4]) / dt
    dy[-3] = (
        p4[0] * y[-5] + p4[1] * y[-4] + p4[2] * y[-2] + p4[3] * y[-1]
    ) / dt

    dy[3] = (
        p6[0] * y[0]
        + p6[1] * y[1]
        + p6[2] * y[2]
        + p6[3] * y[4]
        + p6[4] * y[5]
        + p6[5] * y[6]
    ) / dt
    dy[-4] = (
        p6[0] * y[-7]
        + p6[1] * y[-6]
        + p6[2] * y[-5]
        + p6[3] * y[-3]
        + p6[4] * y[-2]
        + p6[5] * y[-1]
    ) / dt

    for i in range(4, N - 4):
        dy[i] = (
            p8[0] * y[i - 4]
            + p8[1] * y[i - 3]
            + p8[2] * y[i - 2]
            + p8[3] * y[i - 1]
            + p8[4] * y[i + 1]
            + p8[5] * y[i + 2]
            + p8[6] * y[i + 3]
            + p8[7] * y[i + 4]
        ) / dt

    return dy


def der2(t, y):
    N = len(t)
    dt = (t[-1] - t[0]) / N

    p2 = [1.0, -2.0, 1.0]
    p4 = [-1.0 / 12.0, 4.0 / 3.0, -5.0 / 2.0, 4.0 / 3.0, -1.0 / 12.0]
    p6 = [
        1.0 / 90.0,
        -3.0 / 20.0,
        3.0 / 2.0,
        -49.0 / 18.0,
        3.0 / 2.0,
        -3.0 / 20.0,
        1.0 / 90.0,
    ]
    p8 = [
        -1.0 / 560.0,
        8.0 / 315.0,
        -1.0 / 5.0,
        8.0 / 5.0,
        -205.0 / 72.0,
        8.0 / 5.0,
        -1.0 / 5.0,
        8.0 / 315.0,
        -1.0 / 560.0,
    ]

    ddy = N * [0.0]

    ddy[1] = (p2[0] * y[0] + p2[1] * y[1] + p2[2] * y[2]) / (dt * dt)
    ddy[-2] = (p2[0] * y[-3] + p2[1] * y[-2] + p2[2] * y[-1]) / (dt * dt)

    ddy[0] = ddy[1]
    ddy[-1] = ddy[-2]

    ddy[2] = (
        p4[0] * y[0]
        + p4[1] * y[1]
        + p4[2] * y[2]
        + p4[3] * y[3]
        + p4[4] * y[4]
    ) / (dt * dt)
    ddy[-3] = (
        p4[0] * y[-5]
        + p4[1] * y[-4]
        + p4[2] * y[-3]
        + p4[3] * y[-2]
        + p4[4] * y[-1]
    ) / (dt * dt)

    ddy[3] = (
        p6[0] * y[0]
        + p6[1] * y[1]
        + p6[2] * y[2]
        + p6[3] * y[3]
        + p6[4] * y[4]
        + p6[5] * y[5]
        + p6[6] * y[6]
    ) / (dt * dt)
    ddy[-4] = (
        p6[0] * y[-7]
        + p6[1] * y[-6]
        + p6[2] * y[-5]
        + p6[3] * y[N - 4]
        + p6[4] * y[N - 3]
        + p6[5] * y[N - 2]
        + p6[6] * y[-1]
    ) / (dt * dt)

    for i in range(4, N - 4):
        ddy[i] = (
            p8[0] * y[i - 4]
            + p8[1] * y[i - 3]
            + p8[2] * y[i - 2]
            + p8[3] * y[i - 1]
            + p8[4] * y[i]
            + p8[5] * y[i + 1]
            + p8[6] * y[i + 2]
            + p8[7] * y[i + 3]
            + p8[8] * y[i + 4]
        ) / (dt * dt)

    return ddy
