import matplotlib.gridspec as grsp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def get_col(color_name=None):
    """Material Design color palettes (only '100' and '900' variants).
    Help: call with no arguments to see the list of available colors.
    Kwarg:
            - color_name: string representing the color's name
    Output:
            - color: list of two elements
                    [0] = lightest color '100'-variant (RGB-triplet in [0, 1])
                    [1] = darkest color '900'-variant (RGB-triplet in [0, 1])
    """
    colors = {
        "red": [[255, 205, 210], [183, 28, 28]],
        "pink": [[248, 187, 208], [136, 14, 79]],
        "purple": [[225, 190, 231], [74, 20, 140]],
        "deep_purple": [[209, 196, 233], [49, 27, 146]],
        "indigo": [[197, 202, 233], [26, 35, 126]],
        "blue": [[187, 222, 251], [13, 71, 161]],
        "light_blue": [[179, 229, 252], [1, 87, 155]],
        "cyan": [[178, 235, 242], [0, 96, 100]],
        "teal": [[178, 223, 219], [0, 77, 64]],
        "green": [[200, 230, 201], [27, 94, 32]],
        "light_green": [[220, 237, 200], [51, 105, 30]],
        "lime": [[240, 244, 195], [130, 119, 23]],
        "yellow": [[255, 249, 196], [245, 127, 23]],
        "amber": [[255, 236, 179], [255, 111, 0]],
        "orange": [[255, 224, 178], [230, 81, 0]],
        "deep_orange": [[255, 204, 188], [191, 54, 12]],
        "brown": [[215, 204, 200], [62, 39, 35]],
        "gray": [[245, 245, 245], [33, 33, 33]],
        "blue_gray": [[207, 216, 220], [38, 50, 56]],
    }
    if not color_name:
        print("\n=== Colors available are:")
        for key, _ in colors.items():
            print("- " + key)
        return
    else:
        color = [
            [colors[color_name][i][j] / 255 for j in range(3)]
            for i in range(2)
        ]
        return color


def interp_col(color, n):
    """Linearly interpolate a color.
    Args:
            - color: list with two elements:
                    color[0] = lightest color variant (get_col('color_name')[0])
                    color[1] = darkest color variant (get_col('color_name')[1]).
            - n: number of desired output colors (n >= 2).
    Output:
            - lsc: list of n linearly scaled colors.
    """
    c = [
        np.interp(list(range(1, n + 1)), [1, n], [color[0][i], color[1][i]])
        for i in range(3)
    ]
    lsc = [[c[0][i], c[1][i], c[2][i]] for i in range(n)]
    return lsc


def plot_pairwise(Xdata, xlabels, color="C0"):
    """Plot X high-dimensional dataset by pairwise plotting its features against each other.
    Args:
            - Xdata: n*m matrix
            - xlabels: list of m strings representing the name of X dataset's features.
    """
    sample_dim = Xdata.shape[0]
    in_dim = Xdata.shape[1]
    fig, axes = plt.subplots(
        nrows=in_dim,
        ncols=in_dim,
        sharex="col",
        sharey="row",
        figsize=(20, 11.3),
    )
    for i, axis in enumerate(axes.flatten()):
        axis.scatter(
            Xdata[:, i % in_dim], Xdata[:, i // in_dim], fc=color, ec=color
        )
        if i // in_dim == in_dim - 1:
            axis.set_xlabel(xlabels[i % in_dim])
        if i % in_dim == 0:
            axis.set_ylabel(xlabels[i // in_dim])
    for i in range(in_dim):
        for j in range(in_dim):
            if i < j:
                axes[i, j].set_visible(False)
    plt.suptitle(
        "Sample dimension = {} points".format(sample_dim),
        x=0.1,
        y=0.95,
        ha="left",
        va="top",
    )
    plt.show()
    return


def plot_dataset(Xdata, Ydata, xlabels, ylabels, color="C0"):
    """Plot Y high-dimensional dataset by pairwise plotting its features against each X dataset's feature.
    Args:
            - Xdata: n*m1 matrix
            - Ydata: n*m2 matrix
            - xlabels: list of m1 strings representing the name of X dataset's features
            - ylabels: list of m2 strings representing the name of Y dataset's features.
    """
    sample_dim = Xdata.shape[0]
    in_dim = Xdata.shape[1]
    out_dim = Ydata.shape[1]
    fig, axes = plt.subplots(
        nrows=out_dim,
        ncols=in_dim,
        sharex="col",
        sharey="row",
        figsize=(20, 11.3),
    )
    for i, axis in enumerate(axes.flatten()):
        axis.scatter(
            Xdata[:, i % in_dim], Ydata[:, i // in_dim], fc=color, ec=color
        )
        inf = min(Xdata[:, i % in_dim])
        sup = max(Xdata[:, i % in_dim])
        mean = 0.5 * (inf + sup)
        delta = sup - mean
        if i // in_dim == out_dim - 1:
            axis.set_xlabel(xlabels[i % in_dim])
            axis.set_xlim(left=inf - 0.3 * delta, right=sup + 0.3 * delta)
        if i % in_dim == 0:
            axis.set_ylabel(ylabels[i // in_dim])
    plt.suptitle("Sample dimension = {} points".format(sample_dim))
    plt.show()
    return


def plot_pairwise_waves(XL, colors, xlabels, wnames):
    """Plot a vector XL of overlapping high-dimensional datasets by means of pairwise components plotting.
    Args:
            - XL: list of L matrices with dimensions n*m_i, for i=1,...,L
            - colors: list of L colors
            - xlabels: list of n strings representing the name of X_is datasets' common features.
    """
    handles, labels = (0, 0)
    L = len(XL)
    in_dim = XL[0].shape[1]
    height = 9.36111
    width = 5.91667
    fig, axes = plt.subplots(
        nrows=in_dim,
        ncols=in_dim,
        sharex="col",
        sharey="row",
        figsize=(2 * width, 2 * height / 2),
    )
    for t, ax in enumerate(axes.flatten()):
        i = t % in_dim
        j = t // in_dim
        if j >= i:
            sns.scatterplot(
                ax=ax,
                x=XL[0][:, i],
                y=XL[0][:, j],
                color=colors[0],
                edgecolor=colors[0],
                label=wnames[0],
            )
            for k in range(1, L):
                sns.scatterplot(
                    ax=ax,
                    x=XL[k][:, i],
                    y=XL[k][:, j],
                    color=colors[k],
                    edgecolor=colors[k],
                    label=wnames[k],
                )
                handles, labels = ax.get_legend_handles_labels()
                ax.get_legend().remove()
        else:
            ax.set_axis_off()
        if i == 0:
            ax.set_ylabel(xlabels[j])
        if j == in_dim - 1:
            ax.set_xlabel(xlabels[i])
        if i == in_dim - 1 and j == 0:
            ax.legend(handles, labels, loc="center")
    plt.show()
    return


def plot_pvloop(S, RS):
    """Plot LV volume and pressure curves and pv-loop.
    Args:
            - S: mechanics simulation solution class
            - RS: last heart beat solution class
    """
    c = get_col("blue")[1]
    gs = grsp.GridSpec(2, 2, width_ratios=[1, 1.2])
    width = 5.91667
    height = 9.36111
    figsize = (2 * width, 2 * height / 3)
    fig = plt.figure(figsize=figsize)
    for i in range(2):
        ax = fig.add_subplot(gs[i, 0])
        if i == 0:
            ax.plot(RS.t, RS.lv_v, color=c, linewidth=1.0)
            ax.plot(S.t, S.lv_v, color=c, linewidth=2.5)
            plt.xlim(RS.t[0], S.t[-1])
            plt.xlabel("Time (ms)")
            plt.ylabel("Left ventricular volume (μL)")
        else:
            ax.plot(RS.t, RS.lv_p, color=c, linewidth=1.0)
            ax.plot(S.t, S.lv_p, color=c, linewidth=2.5)
            plt.xlim(RS.t[0], S.t[-1])
            plt.xlabel("Time (ms)")
            plt.ylabel("Left ventricular pressure (kPa)")

    ax = fig.add_subplot(gs[:, 1])
    ax.plot(S.lv_v, S.lv_p, color=c, linewidth=2.5)
    plt.xlabel("LVV (μL)")
    plt.ylabel("LVP (kPa)")
    fig.tight_layout()
    plt.show()
    return


def plot_linreg_surf(Xdata, Ydata, vis_dim_x, vis_dim_y, deg, n_points):
    X = Xdata[:, vis_dim_x]
    y = Ydata[:, vis_dim_y]

    poly = PolynomialFeatures(degree=deg)
    X_ = poly.fit_transform(X)

    lr = LinearRegression(n_jobs=-1)
    lr.fit(X_, y)

    mn = np.min(X, axis=0)
    mx = np.max(X, axis=0)
    x1grid, x2grid = np.meshgrid(
        np.linspace(mn[0], mx[0], n_points),
        np.linspace(mn[1], mx[1], n_points),
    )

    points = np.transpose(np.vstack([x1grid.ravel(), x2grid.ravel()]))
    points_ = poly.fit_transform(points)
    yp = lr.predict(points_)
    ygrid = yp.reshape(n_points, n_points)

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter3D(X[:, 0], X[:, 1], y, c=y, cmap="Reds")
    ax.plot_surface(
        x1grid, x2grid, ygrid, rstride=1, cstride=1, color="b", alpha=0.5
    )
    plt.title("Linear regression with " + str(deg) + "-degree polynomial")
    plt.show()
    return
