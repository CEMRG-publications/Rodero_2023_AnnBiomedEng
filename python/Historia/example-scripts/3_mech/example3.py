from Historia.mech import mech_out, scan_logfile
from Historia.shared.design_utils import read_labels
from Historia.shared.plot_utils import plot_pvloop

N_BEATS = 4


def main():
    path = "data/"

    S = scan_logfile.MECHSolution(
        path + "output_log.txt"
    )  # MECH model solution class, instantiate by providing the path to the log file of the simulation
    S.extract_loginfo()  # extract parameters used for the simulation, full solution curves and convergence info

    if (
        S.conv
    ):  # if the mechanics run successfully completed, we can further extract the final heart beat solution curves and calculate the LV features of interest
        RS = (
            mech_out.LeftVentricle()
        )  # instantiate the last heart beat solution curves class
        RS.get_lvfeatures(
            S, N_BEATS
        )  # get the last heart beat solution and its features from the full solution (S) by providing the number of beats the simulation was run for (N_BEATS)

        if (
            RS.conv
        ):  # mechanics might have converged but at same time producing non-physiological curves: discard this simulation if it did
            plot_pvloop(
                RS, S
            )  # plot the results by providing the full solution and the last heart beat solution classes

    # relevant classes' attributes:
    xlabels = read_labels(path + "xlabels.txt")
    ylabels = read_labels(path + "ylabels.txt")
    params_dict = {key: val for key, val in zip(xlabels, S.p)}
    features_dict = {key: val for key, val in zip(ylabels, RS.f)}
    print(params_dict)  # input parameters used to run the mechanics simulation
    print(features_dict)  # resulting output LV features

    # full simulation solution curves:
    # print(np.array(S.t).shape) # time vector
    # print(np.array(S.lv_v).shape) # LV volume curve
    # print(np.array(S.lv_p).shape) # LV pressure curve

    # last heart beat solution curves:
    # print(np.array(RS.t).shape) # time vector
    # print(np.array(RS.lv_v).shape) # LV volume curve
    # print(np.array(RS.lv_p).shape) # LV pressure curve


if __name__ == "__main__":
    main()
