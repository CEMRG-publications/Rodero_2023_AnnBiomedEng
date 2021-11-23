import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import pathlib
from functools import partial
from scipy.stats import pearsonr
import tqdm

from Historia.history import hm
from Historia.shared.design_utils import read_labels
from gpytGPE.gpe import GPEmul

import fitting_hm
import anatomy

# Global variables
from global_variables_config import *


def compute_impl_measure(wave, input_points):
    mean_matrix = np.zeros((input_points.shape[0], len(wave.emulator)), dtype=float)
    var_matrix = np.zeros((input_points.shape[0], len(wave.emulator)), dtype=float)
    # for j, emul in enumerate(wave.emulator):
    #     mean, std = emul.predict(input_points)
    #     var = np.power(std, 2)
    #     mean_matrix[:, j] = mean
    #     var_matrix[:, j] = var

    # mean_matrix = np.zeros((input_points.shape[0], 2), dtype=float)
    # var_matrix = np.zeros((input_points.shape[0], 2), dtype=float)

    for j, emul in enumerate(wave.emulator):
        mean, std = emul.predict(input_points)
        var = np.power(std, 2)
        mean_matrix[:, j] = mean
        var_matrix[:, j] = var

    impl_vector = np.zeros((input_points.shape[0],), dtype=float)
    for i in range(input_points.shape[0]):
        impl_point = np.sqrt(
            (np.power(mean_matrix[i, :] - wave.mean, 2)) / (var_matrix[i, :] + wave.var)
        )
        impl_vector[i] = np.sort(impl_point)[-wave.maxno]

    return impl_vector

def plot_relation_impl_l1dist_CT(CT_mesh_number=1, plot_and_save=True):
    matplotlib.rcParams.update({'font.size': 15})

    CT_anatomy_values = np.loadtxt(open(os.path.join(PROJECT_PATH, "CT_anatomy", "X_anatomy.csv")),
                                delimiter=',', skiprows=1)
    CT_points = np.hstack((CT_anatomy_values[0:19, 0:9], np.tile([80, 70, 0.8, 0.29, 7], (19, 1))))

    active_features = read_labels(os.path.join("/data","fitting","anatomy_max_range","output_labels.txt"))
    CT_results = []

    for feature_i in range(len(active_features)):
        CT_results.append(np.loadtxt(os.path.join(PROJECT_PATH, "CT_anatomy", active_features[feature_i] + ".dat"), dtype=float))

    nroy_input, impl_values, best_cases_idx = bhm_fill_gaps(LVV_mean=CT_results[0][CT_mesh_number-1],
                                            RVV_mean=CT_results[1][CT_mesh_number-1],
                                            LAV_mean=CT_results[2][CT_mesh_number-1],
                                            RAV_mean=CT_results[3][CT_mesh_number-1],
                                            LVOTdiam_mean=CT_results[4][CT_mesh_number-1],
                                            RVOTdiam_mean=CT_results[5][CT_mesh_number-1],
                                            LVmass_mean=CT_results[6][CT_mesh_number-1],
                                            LVWT_mean=CT_results[7][CT_mesh_number-1],
                                            LVEDD_mean=CT_results[8][CT_mesh_number-1],
                                            SeptumWT_mean=CT_results[9][CT_mesh_number-1],
                                            RVlongdiam_mean=CT_results[10][CT_mesh_number-1],
                                            TAT_mean=CT_results[11][CT_mesh_number-1],
                                            TATLVendo_mean=CT_results[12][CT_mesh_number-1],
                                                )

    l1_dists = [np.linalg.norm((CT_points[CT_mesh_number-1] - nroy_input[i]), ord=1) for i in range(len(nroy_input))]


    if plot_and_save:
        plt.plot(impl_values, l1_dists, 'k.')

        plt.title('Points in the last wave NROY region compared\n with mesh ' + str(CT_mesh_number) + ' of the CT cohort')
        plt.ylabel('l1 distance')
        plt.xlabel('Impl. measure')
        plt.xlim([0,10])

        plt.savefig(os.path.join('/data','fitting','anatomy_max_range','figures','rel_impl_l1_' + str(CT_mesh_number) + '.png'), bbox_inches="tight", dpi=300)

        plt.close()
    else:
        return pearsonr(impl_values, l1_dists)[0]


###### Pre-trained wave, changing \overline{T}

def bhm_fill_gaps(subfolder="anatomy_max_range",num_wave=2,
                  LVV_mean = None, LVV_var = None,
                  RVV_mean = None, RVV_var = None,
                  LAV_mean = None, LAV_var = None,
                  RAV_mean = None, RAV_var = None,
                  LVOTdiam_mean = None, LVOTdiam_var = None,
                  RVOTdiam_mean = None, RVOTdiam_var = None,
                  LVmass_mean = None, LVmass_var = None,
                  LVWT_mean = None, LVWT_var = None,
                  LVEDD_mean = None, LVEDD_var = None,
                  SeptumWT_mean = None, SeptumWT_var = None,
                  RVlongdiam_mean = None, RVlongdiam_var = None,
                  TAT_mean = None, TAT_var = None,
                  TATLVendo_mean = None, TATLVendo_var = None,
                  cutoff=3):

    SEED = 2
    # ----------------------------------------------------------------
    # Make the code reproducible
    np.random.seed(SEED)

    nroy_input = np.loadtxt(os.path.join(PROJECT_PATH, subfolder, "wave" + str(num_wave), "X_test.dat"),
                            dtype=float)
    wave_loaded = num_wave

    while wave_loaded > 0:
        previous_nroy_input = np.loadtxt(os.path.join(PROJECT_PATH, subfolder, "wave" + str(wave_loaded - 1), "X_test.dat"),
                            dtype=float)
        nroy_input = np.vstack((nroy_input,previous_nroy_input))

        wave_loaded -= 1

    nroy_input = np.unique(nroy_input, axis = 0)

    # ========== Constant variables =============#

    # xlabels_EP = read_labels(os.path.join(PROJECT_PATH, subfolder, "EP_funct_labels_latex.txt"))
    # xlabels_anatomy = read_labels(os.path.join(PROJECT_PATH, subfolder, "modes_labels.txt"))
    # xlabels = [lab for sublist in [xlabels_anatomy, xlabels_EP] for lab in sublist]
    #
    # idx_train = round(0.8 * 0.8 * n_samples)

    # default_mean = [92.8, 86., 45., 37.5, 20.3, 31.9, 126.8, 8.8, 44.3, 8.6, 67.8, 76.4, 31.3]
    default_var = [615.04, 441., 182.25, 182.25, 5.29, 22.09, 1398.76, 2.25, 23.04, 2.56, 64., 67.24, 125.6641]

    user_mean = [LVV_mean, RVV_mean, LAV_mean, RAV_mean, LVOTdiam_mean, RVOTdiam_mean, LVmass_mean, LVWT_mean,
                 LVEDD_mean, SeptumWT_mean, RVlongdiam_mean, TAT_mean, TATLVendo_mean]
    user_var = [LVV_var, RVV_var, LAV_var, RAV_var, LVOTdiam_var, RVOTdiam_var, LVmass_var, LVWT_var,
                 LVEDD_var, SeptumWT_var, RVlongdiam_var, TAT_var, TATLVendo_var]

    active_features_default = read_labels(os.path.join("/data","fitting",subfolder,"output_labels.txt"))

    active_features = [active_features_default[i] for i in range(len(user_mean)) if user_mean[i] is not None]


    emulator = []

    # n_tests = int(1e5)
    # path_match = os.path.join(PROJECT_PATH, "match")
    # exp_mean = np.loadtxt(os.path.join(path_match, "exp_mean_anatomy_EP.txt"), dtype=float)
    # exp_std = np.loadtxt(os.path.join(path_match, "exp_std_anatomy_EP.txt"), dtype=float)
    # exp_var = np.power(exp_std, 2)


    # X = np.loadtxt(os.path.join(PROJECT_PATH, subfolder, "wave0", "X.dat"), dtype=float)
    # X_train = X[:idx_train]
    # I = get_minmax(X_train)

    # =================== Train or load GPE =========================#
    for output_name in active_features:
        _, _, emul = fitting_hm.run_GPE(waveno=num_wave, train=False, active_feature=[output_name], n_training_pts = 280,
                                        training_set_memory=2, subfolder=subfolder, only_feasible=False)
        emulator.append(emul)
    # emulator now might be a list of list, we need to flatten it
    em_shape = (np.array(emulator)).shape
    if len(em_shape) > 1:
        emulator_flat = [item for sublist in emulator for item in sublist]
        emulator = emulator_flat
    # =================== Load or create the wave object ===================#
    wave = hm.Wave()
    wave.load(os.path.join(PROJECT_PATH, subfolder, "wave" + str(num_wave),"wave_" + str(num_wave)))

    wave.emulator = emulator
    # wave.emulator = emulator[0:2]
    wave.cutoff = cutoff

    wave.mean = [user_mean[i] for i in range(len(user_mean)) if user_mean[i] is not None]
    # wave.mean = [user_mean[0:2]]

    wave.var = []
    for i in range(len(user_mean)):
        if user_mean[i] is not None:
            if user_var[i] is None:
                wave.var.append(default_var[i])
            else:
                wave.var.append(user_var[i])
    # wave.var = [default_var[0:2]]

    wave.emulator = wave.emulator[0:2]
    wave.mean = wave.mean[0:2]
    wave.var = wave.var[0:2]

    impl_values = compute_impl_measure(wave, nroy_input)

    best_cases_idx = np.argsort(impl_values)[0:10]

    return nroy_input, impl_values, best_cases_idx

def validation_simulation(CT_mesh_number=1):

    CT_anatomy_values = np.loadtxt(open(os.path.join(PROJECT_PATH, "CT_anatomy", "X_anatomy.csv")),
                                   delimiter=',', skiprows=1)

    active_features = read_labels(os.path.join("/data", "fitting", "anatomy_max_range", "output_labels.txt"))
    CT_results = []

    for feature_i in range(len(active_features)):
        CT_results.append(
            np.loadtxt(os.path.join(PROJECT_PATH, "CT_anatomy", active_features[feature_i] + ".dat"), dtype=float))

    nroy_input, impl_values, best_cases_idx = bhm_fill_gaps(LVV_mean=CT_results[0][CT_mesh_number - 1],
                                                            RVV_mean=CT_results[1][CT_mesh_number - 1],
                                                            LAV_mean=CT_results[2][CT_mesh_number - 1],
                                                            RAV_mean=CT_results[3][CT_mesh_number - 1],
                                                            LVOTdiam_mean=CT_results[4][CT_mesh_number - 1],
                                                            RVOTdiam_mean=CT_results[5][CT_mesh_number - 1],
                                                            LVmass_mean=CT_results[6][CT_mesh_number - 1],
                                                            LVWT_mean=CT_results[7][CT_mesh_number - 1],
                                                            LVEDD_mean=CT_results[8][CT_mesh_number - 1],
                                                            SeptumWT_mean=CT_results[9][CT_mesh_number - 1],
                                                            RVlongdiam_mean=CT_results[10][CT_mesh_number - 1],
                                                            TAT_mean=CT_results[11][CT_mesh_number - 1],
                                                            TATLVendo_mean=CT_results[12][CT_mesh_number - 1],
                                                            )

    points_to_test = nroy_input[best_cases_idx[:10]]

    SEED = 2
    # ----------------------------------------------------------------
    # Make the code reproducible
    np.random.seed(SEED)

    subfolder = os.path.join("validation","CT_" + str(CT_mesh_number), "output_all_input_none")
    meshes_folder = os.path.join("/data","fitting",subfolder)

    pathlib.Path(meshes_folder).mkdir(parents=True, exist_ok=True)

    anatomy.preprocess_input(path_gpes = meshes_folder, anatomy_and_EP_values = points_to_test)
    anatomy.build_meshes(subfolder=subfolder, path_gpes = meshes_folder)
    anatomy.EP_setup(subfolder=subfolder, path_gpes = meshes_folder)
    anatomy.EP_simulations(subfolder=subfolder, path_gpes = meshes_folder)
    anatomy.write_output_casewise(subfolder=subfolder, outpath = meshes_folder)
    anatomy.collect_output(subfolder=subfolder, outpath = meshes_folder)

def plot_difference_BHM_groundtruth(CT_mesh_number = 1, num_input = 10):

    if num_input == "all":
        for i in range(1,11):
            plot_difference_BHM_groundtruth(CT_mesh_number=CT_mesh_number, num_input=i)
    else:
        output_labels = read_labels(os.path.join("/data", "fitting", "CT_anatomy", "output_labels.txt"))
        ep_labels_latex = read_labels(os.path.join("/data","fitting", "validation", "EP_funct_labels_latex.txt"))

        path_figures = os.path.join("/data", "fitting", "validation", "CT_" + str(CT_mesh_number), "output_all_input_none", "figures")

        pathlib.Path(path_figures).mkdir(parents=True, exist_ok=True)

        ct_modes_csv_path = os.path.join("/data", "fitting", "match", "CT_cohort_modes_weights.csv")
        all_ct_modes = np.genfromtxt(ct_modes_csv_path, delimiter=',', skip_header=True)
        wanted_ct_modes = all_ct_modes[CT_mesh_number-1, 1:10]
        ep_param_paper = np.array([80, 70, 0.8, 0.29, 7])

        bhm_modes_csv_path = os.path.join("/data", "fitting", "validation", "CT_" + str(CT_mesh_number), "output_all_input_none", "X_anatomy.csv")
        all_bhm_modes = np.genfromtxt(bhm_modes_csv_path, delimiter=',', skip_header=True)
        wanted_bhm_modes = all_bhm_modes[:, 0:9]
        ep_param_bhm = np.genfromtxt(os.path.join("/data", "fitting", "validation", "CT_" + str(CT_mesh_number), "output_all_input_none", "X_EP.dat"))

        y_simul = []

        for feature in output_labels:
            y_simul_current = np.genfromtxt(os.path.join("/data", "fitting", "CT_anatomy", feature + ".dat"))
            y_simul.append(y_simul_current[CT_mesh_number-1])

        y_bhm = []

        for feature in output_labels:
            y_bhm_current = np.genfromtxt(os.path.join("/data", "fitting",  "validation", "CT_" + str(CT_mesh_number), "output_all_input_none", feature + ".dat"))
            y_bhm = np.hstack((y_bhm,y_bhm_current))

        y_bhm =  y_bhm.reshape((len(output_labels),10)) # Each row is a feature
        y_bhm = list(map(list,zip(*y_bhm))) # Each row is a mesh

        y_bhm_normalised = 100*np.copy((np.array(y_bhm) - np.array(np.tile(y_simul,[10,1])))/(np.array(np.tile(y_simul,[10,1]))))

        # We plot the output

        height = 9.36111
        width = 5.91667
        fig, axes = plt.subplots(1, 1, figsize=(2 * width, 2 * height / 4))

        inf_bound = [np.nanmin(y_bhm_normalised),-100]
        sup_bound = [np.nanmax(y_bhm_normalised),100]


        axes.scatter(
            np.tile(np.arange(len(output_labels)),num_input),
            y_bhm_normalised[:num_input,],
            s=5
        )

        axes.axhline(y=0, color='r', linestyle='-')

        axes.set_xticks(np.arange(len(output_labels)))
        axes.set_xticklabels(output_labels)
        axes.set_xlabel("Biomarker", fontsize=12)
        axes.set_ylabel("Normalised difference (%)", fontsize=12)

        figure_name = "output_BHM_" + str(num_input)
        axes.set_ylim([min(inf_bound), max(sup_bound)])
        axes.set_title("Comparison of the first " + str(num_input) + " output from BHM (normalised) with the simulation ground truth (red line)")

        fig.tight_layout()

        plt.savefig(os.path.join(path_figures, figure_name + ".png"), bbox_inches="tight", dpi=300)
        plt.close()

        # We plot input modes
        modes_bhm_normalised = 100*np.copy((np.array(wanted_bhm_modes) - np.array(np.tile(wanted_ct_modes, [10, 1])))/(np.array(np.tile(wanted_ct_modes, [10, 1]))))

        height = 9.36111
        width = 5.91667
        fig, axes = plt.subplots(1, 1, figsize=(2 * width, 2 * height / 4))

        inf_bound = [np.nanmin(modes_bhm_normalised), -100]
        sup_bound = [np.nanmax(modes_bhm_normalised), 100]

        axes.scatter(
            np.tile(np.arange(len(wanted_ct_modes)), num_input),
            modes_bhm_normalised[:num_input, ],
            s=5
        )

        axes.axhline(y=1, color='r', linestyle='-')

        axes.set_xticks(np.arange(len(wanted_ct_modes)))
        axes.set_xticklabels(["Mode " + str(m) for m in range(1,10)])
        axes.set_ylabel("Normalised difference (%)", fontsize=12)

        figure_name = "input_modes_BHM_" + str(num_input)
        axes.set_ylim([min(inf_bound), max(sup_bound)])
        axes.set_title("Comparison of the first " + str(
            num_input) + " input modes from BHM (normalised) with the simulation ground truth (red line)")

        fig.tight_layout()

        plt.savefig(os.path.join(path_figures, figure_name + ".png"), bbox_inches="tight", dpi=300)
        plt.close()

        # We plot input EP parameters
        ep_param_bhm_normalised = 100 * np.copy(
            (np.array(ep_param_bhm) - np.array(np.tile(ep_param_paper, [10, 1]))) / (
                np.array(np.tile(ep_param_paper, [10, 1]))))

        height = 9.36111
        width = 5.91667
        fig, axes = plt.subplots(1, 1, figsize=(2 * width, 2 * height / 4))

        inf_bound = [np.nanmin(ep_param_bhm_normalised), -100]
        sup_bound = [np.nanmax(ep_param_bhm_normalised), 100]

        axes.scatter(
            np.tile(np.arange(len(ep_param_paper)), num_input),
            ep_param_bhm_normalised[:num_input, ],
            s=5
        )

        axes.axhline(y=1, color='r', linestyle='-')

        axes.set_xticks(np.arange(len(ep_param_paper)))
        axes.set_xticklabels(ep_labels_latex)
        axes.set_ylabel("Normalised difference (%)", fontsize=12)

        figure_name = "input_ep_param_BHM_" + str(num_input)
        axes.set_ylim([min(inf_bound), max(sup_bound)])
        axes.set_title("Comparison of the first " + str(
            num_input) + " input EP parameters from BHM (normalised) with the simulation ground truth (red line)")

        fig.tight_layout()

        plt.savefig(os.path.join(path_figures, figure_name + ".png"), bbox_inches="tight", dpi=300)
        plt.close()


def plot_wave_new_imp(CT_mesh_number = 1, numwave=0, reduction_function="all", subfolder="anatomy_max_range_limit_CT25percent",cutoff=3):

    if reduction_function == "all":
        plot_wave_new_imp(CT_mesh_number=1, numwave=0, reduction_function="prob",subfolder=subfolder,cutoff=cutoff)
        plot_wave_new_imp(CT_mesh_number=1, numwave=0, reduction_function="min",subfolder=subfolder,cutoff=cutoff)
    else:

        filename = os.path.join(PROJECT_PATH, "validation", "CT_" + str(CT_mesh_number), "output_all_input_none",
                                "figures", "wave_prob_imp")
        plot_title = "BHM in patient #" + str(CT_mesh_number) + ": percentage of implausible points"

        active_features = read_labels(os.path.join("/data", "fitting", subfolder, "output_labels.txt"))
        CT_results = []

        for feature_i in range(len(active_features)):
            CT_results.append(
                np.loadtxt(os.path.join(PROJECT_PATH, "CT_anatomy", active_features[feature_i] + ".dat"), dtype=float))

        CT_anatomy_values = np.loadtxt(open(os.path.join(PROJECT_PATH, "CT_anatomy", "X_anatomy.csv")), delimiter=',', skiprows=1)
        CT_mesh_modes = CT_anatomy_values[CT_mesh_number-1,0:9]
        CT_ep_param = np.array([80, 70, 0.8, 0.29, 7])

        CT_goal_point = np.append(CT_mesh_modes, CT_ep_param)

        print("Computing implausibility...")
        nroy_input, impl_values, _ = bhm_fill_gaps(LVV_mean=CT_results[0][CT_mesh_number - 1],
                                                                RVV_mean=CT_results[1][CT_mesh_number - 1],
                                                                LAV_mean=CT_results[2][CT_mesh_number - 1],
                                                                RAV_mean=CT_results[3][CT_mesh_number - 1],
                                                                LVOTdiam_mean=CT_results[4][CT_mesh_number - 1],
                                                                RVOTdiam_mean=CT_results[5][CT_mesh_number - 1],
                                                                LVmass_mean=CT_results[6][CT_mesh_number - 1],
                                                                LVWT_mean=CT_results[7][CT_mesh_number - 1],
                                                                LVEDD_mean=CT_results[8][CT_mesh_number - 1],
                                                                SeptumWT_mean=CT_results[9][CT_mesh_number - 1],
                                                                RVlongdiam_mean=CT_results[10][CT_mesh_number - 1],
                                                                TAT_mean=CT_results[11][CT_mesh_number - 1],
                                                                TATLVendo_mean=CT_results[12][CT_mesh_number - 1],
                                                                num_wave=numwave,
                                                                subfolder=subfolder,
                                                   cutoff=cutoff
                                                                )
        print("Done")

        X = nroy_input

        xlabels_EP = read_labels(os.path.join(PROJECT_PATH, "validation", "CT_" + str(CT_mesh_number),"output_all_input_none", "EP_funct_labels_latex.txt"))
        xlabels_anatomy = read_labels(os.path.join(PROJECT_PATH, "validation", "CT_" + str(CT_mesh_number),"output_all_input_none", "modes_labels.txt"))
        xlabels = [lab for sublist in [xlabels_anatomy, xlabels_EP] for lab in sublist]

        C = impl_values
        cmap = "jet"

        height = 9.36111
        width = 5.91667
        fig = plt.figure(figsize=(3 * width, 3 * height / 3))
        gs = matplotlib.gridspec.GridSpec(
            X.shape[1] - 1,
            X.shape[1],
            width_ratios=(X.shape[1] - 1) * [1] + [0.1],
        )

        path_match = os.path.join(PROJECT_PATH, "match")

        param_ranges_lower_anatomy = np.loadtxt(os.path.join(path_match, "anatomy_input_range_lower_max_range.dat"), dtype=float)
        param_ranges_upper_anatomy = np.loadtxt(os.path.join(path_match, "anatomy_input_range_upper_max_range.dat"), dtype=float)

        param_ranges_lower_EP = np.loadtxt(os.path.join(path_match, "EP_input_range_lower_max_range.dat"), dtype=float)
        param_ranges_upper_EP = np.loadtxt(os.path.join(path_match, "EP_input_range_upper_max_range.dat"), dtype=float)

        param_ranges_lower = np.append(param_ranges_lower_anatomy, param_ranges_lower_EP)
        param_ranges_upper = np.append(param_ranges_upper_anatomy, param_ranges_upper_EP)

        param_ranges = np.column_stack((param_ranges_lower, param_ranges_upper))

        if reduction_function == "prob":
            print("Plotting...")

            for k in tqdm.tqdm(range(X.shape[1] * X.shape[1])):
                i = k % X.shape[1]
                j = k // X.shape[1]

                if i > j:
                    axis = fig.add_subplot(gs[i - 1, j])
                    axis.set_facecolor("#800000")

                    if (xlabels[i] == "$\mathregular{k_{fibre}}$") or (xlabels[j] == "$\mathregular{k_{fibre}}$"):
                        hexagon_size = 9
                    else:
                        hexagon_size = 49

                    cbar_label = "Implausible points (%)"

                    def prob_IMP(x, threshold):
                        return 100 * sum(i >= threshold for i in x) / len(x)

                    reduce_function = partial(prob_IMP, threshold=cutoff)

                    im = axis.hexbin(
                        X[:, j],
                        X[:, i],
                        C=C,
                        reduce_C_function=reduce_function,
                        gridsize=hexagon_size,
                        cmap=cmap,
                        vmin=0,
                        vmax=100,
                    )

                    arrow_point_x = CT_goal_point[j]
                    arrow_point_y = CT_goal_point[i]

                    delta_x = 0.25 * (param_ranges[j, 1] - param_ranges[j, 0])
                    delta_y = 0.25 * (param_ranges[i, 1] - param_ranges[i, 0])

                    # plt.arrow(arrow_point_x - delta_x, arrow_point_y - delta_y, delta_x, delta_y, fc='k', ec='k', clip_on=False)
                    plt.plot(CT_goal_point[j], CT_goal_point[i], 'kx', clip_on=False)

                    axis.set_xlim([param_ranges[j, 0], param_ranges[j, 1]])
                    axis.set_ylim([param_ranges[i, 0], param_ranges[i, 1]])

                    if i == X.shape[1] - 1:
                        axis.set_xlabel(xlabels[j], fontsize=12)
                    else:
                        axis.set_xticklabels([])
                    if j == 0:
                        axis.set_ylabel(xlabels[i], fontsize=12)
                    else:
                        axis.set_yticklabels([])

            cbar_axis = fig.add_subplot(gs[:, X.shape[1] - 1])
            cbar = fig.colorbar(im, cax=cbar_axis, format='%.2f')
            cbar.set_label(cbar_label, size=8)
            fig.tight_layout()
            plt.suptitle(plot_title, fontsize=18)

            print("Saving HD plot...")
            plt.savefig(filename + ".png", bbox_inches="tight", dpi=300)
            plt.close()

        if reduction_function == "min":

            print("Plotting...")

            filename = os.path.join(PROJECT_PATH, "validation", "CT_" + str(CT_mesh_number), "output_all_input_none",
                                    "figures", "wave_min_imp")
            plot_title = "BHM in patient #" + str(CT_mesh_number) + ": minimum implausibility"

            height = 9.36111
            width = 5.91667
            fig = plt.figure(figsize=(3 * width, 3 * height / 3))
            gs = matplotlib.gridspec.GridSpec(
                X.shape[1] - 1,
                X.shape[1],
                width_ratios=(X.shape[1] - 1) * [1] + [0.1],
            )

            for k in tqdm.tqdm(range(X.shape[1] * X.shape[1])):
                i = k % X.shape[1]
                j = k // X.shape[1]

                if i > j:
                    axis = fig.add_subplot(gs[i - 1, j])
                    axis.set_facecolor("#800000")

                    if (xlabels[i] == "$\mathregular{k_{fibre}}$") or (xlabels[j] == "$\mathregular{k_{fibre}}$"):
                        hexagon_size = 9
                    else:
                        hexagon_size = 49

                    cbar_label = "Implausibility"

                    im = axis.hexbin(
                        X[:, j],
                        X[:, i],
                        C=C,
                        reduce_C_function=np.min,
                        gridsize=hexagon_size,
                        cmap=cmap,
                        vmin=1,
                        vmax=cutoff,
                    )

                    arrow_point_x = CT_goal_point[j]
                    arrow_point_y = CT_goal_point[i]

                    delta_x = 0.25*(param_ranges[j, 1] - param_ranges[j, 0])
                    delta_y = 0.25*(param_ranges[i, 1] - param_ranges[i, 0])

                    # plt.arrow(arrow_point_x-delta_x, arrow_point_y-delta_y, delta_x, delta_y, fc='k', ec='k', clip_on=False)
                    plt.plot(CT_goal_point[j], CT_goal_point[i], 'wx', clip_on=False)

                    axis.set_xlim([param_ranges[j, 0], param_ranges[j, 1]])
                    axis.set_ylim([param_ranges[i, 0], param_ranges[i, 1]])

                    if i == X.shape[1] - 1:
                        axis.set_xlabel(xlabels[j], fontsize=12)
                    else:
                        axis.set_xticklabels([])
                    if j == 0:
                        axis.set_ylabel(xlabels[i], fontsize=12)
                    else:
                        axis.set_yticklabels([])

            cbar_axis = fig.add_subplot(gs[:, X.shape[1] - 1])
            cbar = fig.colorbar(im, cax=cbar_axis, format='%.2f')
            cbar.set_label(cbar_label, size=8)
            fig.tight_layout()
            plt.suptitle(plot_title, fontsize=18)

            print("Saving HD plot...")
            plt.savefig(filename + ".png", bbox_inches="tight", dpi=300)

##### Pre-trained wave, using NROY region

def validation_simulation_minimising_emulation(CT_mesh_number=1):

    active_features = read_labels(os.path.join("/data", "fitting", "anatomy_max_range", "output_labels.txt"))

    CT_results = []

    for feature_i in range(len(active_features)):
        CT_results.append(
            np.loadtxt(os.path.join("/data", "fitting", "CT_anatomy", active_features[feature_i] + ".dat"), dtype=float))

    exp_goal = [x[CT_mesh_number-1] for x in CT_results]

    wave = hm.Wave()
    wave.load(os.path.join("/data/fitting", "anatomy_max_range", "wave2", "wave_2"))

    nroy_input = np.loadtxt(os.path.join("/data/fitting", "anatomy_max_range", "wave2", "X_test.dat"),
                            dtype=float)

    difference_matrix = np.empty((0,len(nroy_input)))

    for i, output_name in enumerate(active_features):
        _, _, emul = fitting_hm.run_GPE(waveno=2, train=False, active_feature=[output_name], n_samples=280,
                                        training_set_memory=2, subfolder="anatomy_max_range", only_feasible=False)
        mean_prediction, _ = emul.predict(nroy_input)
        difference_matrix = np.vstack((difference_matrix, (np.abs(exp_goal[i] - mean_prediction))/exp_goal[i]))

    vector_to_minimise = np.linalg.norm(difference_matrix, axis=0)

    best_cases_idx = np.argsort(vector_to_minimise)

    points_to_test = nroy_input[best_cases_idx[:10]]

    SEED = 2
    # ----------------------------------------------------------------
    # Make the code reproducible
    np.random.seed(SEED)

    subfolder = os.path.join("validation","CT_" + str(CT_mesh_number), "output_all_input_none_usingNROY")
    meshes_folder = os.path.join("/data","fitting",subfolder)

    pathlib.Path(meshes_folder).mkdir(parents=True, exist_ok=True)

    anatomy.preprocess_input(path_gpes = meshes_folder, anatomy_and_EP_values = points_to_test)
    anatomy.build_meshes(subfolder=subfolder, path_gpes = meshes_folder)
    anatomy.EP_setup(subfolder=subfolder, path_gpes = meshes_folder)
    anatomy.EP_simulations(subfolder=subfolder, path_gpes = meshes_folder)
    anatomy.write_output_casewise(subfolder=subfolder, outpath = meshes_folder)
    anatomy.collect_output(subfolder=subfolder, outpath = meshes_folder)

def plot_difference_mini_emul_groundtruth(CT_mesh_number = 1, num_input = 10):

    if num_input == "all":
        for i in range(1,11):
            plot_difference_mini_emul_groundtruth(CT_mesh_number=CT_mesh_number, num_input=i)
    else:
        output_labels = read_labels(os.path.join("/data", "fitting", "CT_anatomy", "output_labels.txt"))
        ep_labels_latex = read_labels(os.path.join("/data","fitting", "validation", "EP_funct_labels_latex.txt"))

        path_figures = os.path.join("/data", "fitting", "validation", "CT_" + str(CT_mesh_number), "output_all_input_none_usingNROY", "figures")

        pathlib.Path(path_figures).mkdir(parents=True, exist_ok=True)

        ct_modes_csv_path = os.path.join("/data", "fitting", "match", "CT_cohort_modes_weights.csv")
        all_ct_modes = np.genfromtxt(ct_modes_csv_path, delimiter=',', skip_header=True)
        wanted_ct_modes = all_ct_modes[CT_mesh_number-1, 1:10]
        ep_param_paper = np.array([80, 70, 0.8, 0.29, 7])

        bhm_modes_csv_path = os.path.join("/data", "fitting", "validation", "CT_" + str(CT_mesh_number), "output_all_input_none_usingNROY", "X_anatomy.csv")
        all_bhm_modes = np.genfromtxt(bhm_modes_csv_path, delimiter=',', skip_header=True)
        wanted_bhm_modes = all_bhm_modes[:, 0:9]
        ep_param_bhm = np.genfromtxt(os.path.join("/data", "fitting", "validation", "CT_" + str(CT_mesh_number), "output_all_input_none_usingNROY", "X_EP.dat"))

        y_simul = []

        for feature in output_labels:
            y_simul_current = np.genfromtxt(os.path.join("/data", "fitting", "CT_anatomy", feature + ".dat"))
            y_simul.append(y_simul_current[CT_mesh_number-1])

        y_bhm = []

        for feature in output_labels:
            y_bhm_current = np.genfromtxt(os.path.join("/data", "fitting",  "validation", "CT_" + str(CT_mesh_number), "output_all_input_none_usingNROY", feature + ".dat"))
            y_bhm = np.hstack((y_bhm,y_bhm_current))

        y_bhm =  y_bhm.reshape((len(output_labels),10)) # Each row is a feature
        y_bhm = list(map(list,zip(*y_bhm))) # Each row is a mesh

        y_bhm_normalised = 100*np.copy((np.array(y_bhm) - np.array(np.tile(y_simul,[10,1])))/(np.array(np.tile(y_simul,[10,1]))))

        # We plot the output

        height = 9.36111
        width = 5.91667
        fig, axes = plt.subplots(1, 1, figsize=(2 * width, 2 * height / 4))

        inf_bound = [np.nanmin(y_bhm_normalised),-100]
        sup_bound = [np.nanmax(y_bhm_normalised),100]


        axes.scatter(
            np.tile(np.arange(len(output_labels)),num_input),
            y_bhm_normalised[:num_input,],
            s=5
        )

        axes.axhline(y=0, color='r', linestyle='-')

        axes.set_xticks(np.arange(len(output_labels)))
        axes.set_xticklabels(output_labels)
        axes.set_xlabel("Biomarker", fontsize=12)
        axes.set_ylabel("Normalised difference (%)", fontsize=12)

        figure_name = "output_minim_emul_" + str(num_input)
        axes.set_ylim([min(inf_bound), max(sup_bound)])
        axes.set_title("First " + str(num_input) + " output from minimizing emulation of NROY region (normalised) compared with the simulation ground truth (red line)")

        fig.tight_layout()

        plt.savefig(os.path.join(path_figures, figure_name + ".png"), bbox_inches="tight", dpi=300)
        plt.close()

        # We plot input modes
        modes_bhm_normalised = 100*np.copy((np.array(wanted_bhm_modes) - np.array(np.tile(wanted_ct_modes, [10, 1])))/(np.array(np.tile(wanted_ct_modes, [10, 1]))))

        height = 9.36111
        width = 5.91667
        fig, axes = plt.subplots(1, 1, figsize=(2 * width, 2 * height / 4))

        inf_bound = [np.nanmin(modes_bhm_normalised), -100]
        sup_bound = [np.nanmax(modes_bhm_normalised), 100]

        axes.scatter(
            np.tile(np.arange(len(wanted_ct_modes)), num_input),
            modes_bhm_normalised[:num_input, ],
            s=5
        )

        axes.axhline(y=1, color='r', linestyle='-')

        axes.set_xticks(np.arange(len(wanted_ct_modes)))
        axes.set_xticklabels(["Mode " + str(m) for m in range(1,10)])
        axes.set_ylabel("Normalised difference (%)", fontsize=12)

        figure_name = "input_modes_minim_emul_" + str(num_input)
        axes.set_ylim([min(inf_bound), max(sup_bound)])
        axes.set_title("Comparison of the first " + str(
            num_input) + " input modes from minim. the emulation of the NROY region (normalised) with the simulation ground truth (red line)")

        fig.tight_layout()

        plt.savefig(os.path.join(path_figures, figure_name + ".png"), bbox_inches="tight", dpi=300)
        plt.close()

        # We plot input EP parameters
        ep_param_bhm_normalised = 100 * np.copy(
            (np.array(ep_param_bhm) - np.array(np.tile(ep_param_paper, [10, 1]))) / (
                np.array(np.tile(ep_param_paper, [10, 1]))))

        height = 9.36111
        width = 5.91667
        fig, axes = plt.subplots(1, 1, figsize=(2 * width, 2 * height / 4))

        inf_bound = [np.nanmin(ep_param_bhm_normalised), -100]
        sup_bound = [np.nanmax(ep_param_bhm_normalised), 100]

        axes.scatter(
            np.tile(np.arange(len(ep_param_paper)), num_input),
            ep_param_bhm_normalised[:num_input, ],
            s=5
        )

        axes.axhline(y=1, color='r', linestyle='-')

        axes.set_xticks(np.arange(len(ep_param_paper)))
        axes.set_xticklabels(ep_labels_latex)
        axes.set_ylabel("Normalised difference (%)", fontsize=12)

        figure_name = "input_ep_param_minim_emul_" + str(num_input)
        axes.set_ylim([min(inf_bound), max(sup_bound)])
        axes.set_title("Comparison of the first " + str(
            num_input) + " input EP parameters from minim. the emulation of the NROY region (normalised) with the simulation ground truth (red line)")

        fig.tight_layout()

        plt.savefig(os.path.join(path_figures, figure_name + ".png"), bbox_inches="tight", dpi=300)
        plt.close()

##### Extending emulator with other cases, changing \overline{T}
def add_simulation_to_emulator_and_wave(subfolder,CT_mesh_number = [1],num_wave=0,cutoff=3):

    ct_modes_csv_path = os.path.join("/data", "fitting", "match", "CT_cohort_modes_weights.csv")
    all_ct_modes = np.genfromtxt(ct_modes_csv_path, delimiter=',', skip_header=True)
    wanted_ct_modes = all_ct_modes[[i - 1 for i in CT_mesh_number], 1:10]
    ep_param_paper = np.array([80, 70, 0.8, 0.29, 7])

    CT_x = np.hstack((wanted_ct_modes, np.tile(ep_param_paper,(len(CT_mesh_number),1))))

    active_features = read_labels(os.path.join("/data", "fitting", subfolder, "output_labels.txt"))

    emulator = []

    emul_savepath = os.path.join("/data", "fitting", "validation", "CT_" + ''.join([str(i) for i in CT_mesh_number]), subfolder + "/")
    pathlib.Path(emul_savepath).mkdir(parents=True, exist_ok=True)

    for active_feature in active_features:
        all_CT_y = np.loadtxt(os.path.join("/data", "fitting", "CT_anatomy", active_feature + ".dat"), dtype=float)
        print(all_CT_y.shape)
        CT_y = all_CT_y[[i-1 for i in CT_mesh_number]]

        x_train_original, y_train_original, _ = fitting_hm.run_GPE(waveno=num_wave, train=False,
                                                                   active_feature=[active_feature], n_training_pts=280,
                                                                   training_set_memory=2, subfolder=subfolder,
                                                                   only_feasible=False)

        y_train_extended = np.append(y_train_original, CT_y)

        x_train_extended = np.vstack((x_train_original, CT_x))

        emul = GPEmul(x_train_extended, y_train_extended)

        emul.train(X_val=None, y_val=None, max_epochs=100, n_restarts=5, savepath=emul_savepath)

        # emul.save("wave2_" + active_feature + ".gpe")

        emulator.append(emul)

    # Now we update the previous wave or create one

    if num_wave > 0:
        wave_to_load = str(num_wave-1)
    else:
        wave_to_load = str(num_wave)

    wave = hm.Wave()
    wave.load(os.path.join("/data", "fitting", subfolder, "wave" + wave_to_load, "wave_" + wave_to_load))
    wave.emulator = emulator
    wave.cutoff = cutoff

    nroy_input = np.loadtxt(os.path.join("/data", "fitting", subfolder, "wave" + str(num_wave), "X_test.dat"),  dtype=float)

    wave.find_regions(nroy_input)

    # wave.save(os.path.join(emul_savepath,"wave_" + str(num_wave)))

    return emulator, wave

def bhm_fill_gaps_extended(subfolder="anatomy_max_range", num_wave=2,
                           LVV_mean = None, LVV_var = None,
                  RVV_mean = None, RVV_var = None,
                  LAV_mean = None, LAV_var = None,
                  RAV_mean = None, RAV_var = None,
                  LVOTdiam_mean = None, LVOTdiam_var = None,
                  RVOTdiam_mean = None, RVOTdiam_var = None,
                  LVmass_mean = None, LVmass_var = None,
                  LVWT_mean = None, LVWT_var = None,
                  LVEDD_mean = None, LVEDD_var = None,
                  SeptumWT_mean = None, SeptumWT_var = None,
                  RVlongdiam_mean = None, RVlongdiam_var = None,
                  TAT_mean = None, TAT_var = None,
                  TATLVendo_mean = None, TATLVendo_var = None,
                  CT_mesh_number=[1], cutoff=3):

    SEED = 2
    # ----------------------------------------------------------------
    # Make the code reproducible
    np.random.seed(SEED)

    nroy_input = np.loadtxt(os.path.join("/data", "fitting", subfolder, "wave" + str(num_wave), "X_test.dat"), dtype=float)

    wave_loaded = num_wave

    while wave_loaded > 0:
        previous_nroy_input = np.loadtxt(
            os.path.join(PROJECT_PATH, subfolder, "wave" + str(wave_loaded - 1), "X_test.dat"),
            dtype=float)
        nroy_input = np.vstack((nroy_input, previous_nroy_input))

        wave_loaded -= 1

    nroy_input = np.unique(nroy_input, axis=0)

    default_var = [615.04, 441., 182.25, 182.25, 5.29, 22.09, 1398.76, 2.25, 23.04, 2.56, 64., 67.24, 125.6641]

    user_mean = [LVV_mean, RVV_mean, LAV_mean, RAV_mean, LVOTdiam_mean, RVOTdiam_mean, LVmass_mean, LVWT_mean,
                 LVEDD_mean, SeptumWT_mean, RVlongdiam_mean, TAT_mean, TATLVendo_mean]
    user_var = [LVV_var, RVV_var, LAV_var, RAV_var, LVOTdiam_var, RVOTdiam_var, LVmass_var, LVWT_var,
                 LVEDD_var, SeptumWT_var, RVlongdiam_var, TAT_var, TATLVendo_var]

    emulator, wave = add_simulation_to_emulator_and_wave(subfolder=subfolder,CT_mesh_number=CT_mesh_number,num_wave=num_wave,cutoff=cutoff)
    wave.emulator = emulator

    wave.mean = [user_mean[i] for i in range(len(user_mean)) if user_mean[i] is not None]

    wave.var = []
    for i in range(len(user_mean)):
        if user_mean[i] is not None:
            if user_var[i] is None:
                wave.var.append(default_var[i])
            else:
                wave.var.append(user_var[i])

    impl_values = compute_impl_measure(wave, nroy_input)

    best_cases_idx = np.argsort(impl_values)[0:10]

    return nroy_input, impl_values, best_cases_idx

def validation_simulation_extended(CT_mesh_number=2, using_CT_mesh_number=1):

    CT_anatomy_values = np.loadtxt(open(os.path.join(PROJECT_PATH, "CT_anatomy", "X_anatomy.csv")),
                                   delimiter=',', skiprows=1)

    active_features = read_labels(os.path.join("/data", "fitting", "anatomy_max_range", "output_labels.txt"))
    CT_results = []

    for feature_i in range(len(active_features)):
        CT_results.append(
            np.loadtxt(os.path.join(PROJECT_PATH, "CT_anatomy", active_features[feature_i] + ".dat"), dtype=float))

    nroy_input, impl_values, best_cases_idx = bhm_fill_gaps_extended(LVV_mean=CT_results[0][CT_mesh_number - 1],
                                                            RVV_mean=CT_results[1][CT_mesh_number - 1],
                                                            LAV_mean=CT_results[2][CT_mesh_number - 1],
                                                            RAV_mean=CT_results[3][CT_mesh_number - 1],
                                                            LVOTdiam_mean=CT_results[4][CT_mesh_number - 1],
                                                            RVOTdiam_mean=CT_results[5][CT_mesh_number - 1],
                                                            LVmass_mean=CT_results[6][CT_mesh_number - 1],
                                                            LVWT_mean=CT_results[7][CT_mesh_number - 1],
                                                            LVEDD_mean=CT_results[8][CT_mesh_number - 1],
                                                            SeptumWT_mean=CT_results[9][CT_mesh_number - 1],
                                                            RVlongdiam_mean=CT_results[10][CT_mesh_number - 1],
                                                            TAT_mean=CT_results[11][CT_mesh_number - 1],
                                                            TATLVendo_mean=CT_results[12][CT_mesh_number - 1],
                                                                     CT_mesh_number = using_CT_mesh_number
                                                            )

    points_to_test = nroy_input[best_cases_idx[:10]]

    SEED = 2
    # ----------------------------------------------------------------
    # Make the code reproducible
    np.random.seed(SEED)

    subfolder = os.path.join("validation","CT_" + str(CT_mesh_number), "output_all_input_none_extended" + str(using_CT_mesh_number))
    meshes_folder = os.path.join("/data","fitting",subfolder)

    pathlib.Path(meshes_folder).mkdir(parents=True, exist_ok=True)

    anatomy.preprocess_input(path_gpes = meshes_folder, anatomy_and_EP_values = points_to_test)
    anatomy.build_meshes(subfolder=subfolder, path_gpes = meshes_folder)
    anatomy.EP_setup(subfolder=subfolder, path_gpes = meshes_folder)
    anatomy.EP_simulations(subfolder=subfolder, path_gpes = meshes_folder)
    anatomy.write_output_casewise(subfolder=subfolder, outpath = meshes_folder)
    anatomy.collect_output(subfolder=subfolder, outpath = meshes_folder)

def plot_difference_BHM_extended_groundtruth(CT_mesh_number = 2, num_input = 10, using_CT_mesh_number=1):

    if num_input == "all":
        for i in range(1,11):
            plot_difference_BHM_extended_groundtruth(CT_mesh_number=CT_mesh_number, num_input=i, using_CT_mesh_number=using_CT_mesh_number)
    else:
        output_labels = read_labels(os.path.join("/data", "fitting", "CT_anatomy", "output_labels.txt"))
        ep_labels_latex = read_labels(os.path.join("/data","fitting", "validation", "EP_funct_labels_latex.txt"))

        path_figures = os.path.join("/data", "fitting", "validation", "CT_" + str(CT_mesh_number), "output_all_input_none_extended" + str(using_CT_mesh_number), "figures")

        pathlib.Path(path_figures).mkdir(parents=True, exist_ok=True)

        ct_modes_csv_path = os.path.join("/data", "fitting", "match", "CT_cohort_modes_weights.csv")
        all_ct_modes = np.genfromtxt(ct_modes_csv_path, delimiter=',', skip_header=True)
        wanted_ct_modes = all_ct_modes[CT_mesh_number-1, 1:10]
        ep_param_paper = np.array([80, 70, 0.8, 0.29, 7])

        bhm_modes_csv_path = os.path.join("/data", "fitting", "validation", "CT_" + str(CT_mesh_number), "output_all_input_none_extended" + str(using_CT_mesh_number), "X_anatomy.csv")
        all_bhm_modes = np.genfromtxt(bhm_modes_csv_path, delimiter=',', skip_header=True)
        wanted_bhm_modes = all_bhm_modes[:, 0:9]
        ep_param_bhm = np.genfromtxt(os.path.join("/data", "fitting", "validation", "CT_" + str(CT_mesh_number), "output_all_input_none_extended" + str(using_CT_mesh_number), "X_EP.dat"))

        y_simul = []

        for feature in output_labels:
            y_simul_current = np.genfromtxt(os.path.join("/data", "fitting", "CT_anatomy", feature + ".dat"))
            y_simul.append(y_simul_current[CT_mesh_number-1])

        y_bhm = []

        for feature in output_labels:
            y_bhm_current = np.genfromtxt(os.path.join("/data", "fitting",  "validation", "CT_" + str(CT_mesh_number), "output_all_input_none_extended" + str(using_CT_mesh_number), feature + ".dat"))
            y_bhm = np.hstack((y_bhm,y_bhm_current))

        y_bhm =  y_bhm.reshape((len(output_labels),10)) # Each row is a feature
        y_bhm = list(map(list,zip(*y_bhm))) # Each row is a mesh

        y_bhm_normalised = 100*np.copy((np.array(y_bhm) - np.array(np.tile(y_simul,[10,1])))/(np.array(np.tile(y_simul,[10,1]))))

        # We plot the output

        height = 9.36111
        width = 5.91667
        fig, axes = plt.subplots(1, 1, figsize=(2 * width, 2 * height / 4))

        inf_bound = [np.nanmin(y_bhm_normalised),-100]
        sup_bound = [np.nanmax(y_bhm_normalised),100]


        axes.scatter(
            np.tile(np.arange(len(output_labels)),num_input),
            y_bhm_normalised[:num_input,],
            s=5
        )

        axes.axhline(y=0, color='r', linestyle='-')

        axes.set_xticks(np.arange(len(output_labels)))
        axes.set_xticklabels(output_labels)
        axes.set_xlabel("Biomarker", fontsize=12)
        axes.set_ylabel("Normalised difference (%)", fontsize=12)

        figure_name = "output_BHM_extendedwithcase_" + str(using_CT_mesh_number) + "_" + str(num_input)
        axes.set_ylim([min(inf_bound), max(sup_bound)])
        axes.set_title("Comparison of the first " + str(num_input) + " output from BHM extended with case " + str(using_CT_mesh_number) + " (normalised) with the simulation ground truth (red line)")

        fig.tight_layout()

        plt.savefig(os.path.join(path_figures, figure_name + ".png"), bbox_inches="tight", dpi=300)
        plt.close()

        # We plot input modes
        modes_bhm_normalised = 100*np.copy((np.array(wanted_bhm_modes) - np.array(np.tile(wanted_ct_modes, [10, 1])))/(np.array(np.tile(wanted_ct_modes, [10, 1]))))

        height = 9.36111
        width = 5.91667
        fig, axes = plt.subplots(1, 1, figsize=(2 * width, 2 * height / 4))

        inf_bound = [np.nanmin(modes_bhm_normalised), -100]
        sup_bound = [np.nanmax(modes_bhm_normalised), 100]

        axes.scatter(
            np.tile(np.arange(len(wanted_ct_modes)), num_input),
            modes_bhm_normalised[:num_input, ],
            s=5
        )

        axes.axhline(y=1, color='r', linestyle='-')

        axes.set_xticks(np.arange(len(wanted_ct_modes)))
        axes.set_xticklabels(["Mode " + str(m) for m in range(1,10)])
        axes.set_ylabel("Normalised difference (%)", fontsize=12)

        figure_name = "input_modes_BHM_extendedwithcase_" + str(using_CT_mesh_number) + "_" + str(num_input)
        axes.set_ylim([min(inf_bound), max(sup_bound)])
        axes.set_title("Comparison of the first " + str(
            num_input) + " input modes from BHM extended with case " + str(using_CT_mesh_number) + " (normalised) with the simulation ground truth (red line)")

        fig.tight_layout()

        plt.savefig(os.path.join(path_figures, figure_name + ".png"), bbox_inches="tight", dpi=300)
        plt.close()

        # We plot input EP parameters
        ep_param_bhm_normalised = 100 * np.copy(
            (np.array(ep_param_bhm) - np.array(np.tile(ep_param_paper, [10, 1]))) / (
                np.array(np.tile(ep_param_paper, [10, 1]))))

        height = 9.36111
        width = 5.91667
        fig, axes = plt.subplots(1, 1, figsize=(2 * width, 2 * height / 4))

        inf_bound = [np.nanmin(ep_param_bhm_normalised), -100]
        sup_bound = [np.nanmax(ep_param_bhm_normalised), 100]

        axes.scatter(
            np.tile(np.arange(len(ep_param_paper)), num_input),
            ep_param_bhm_normalised[:num_input, ],
            s=5
        )

        axes.axhline(y=1, color='r', linestyle='-')

        axes.set_xticks(np.arange(len(ep_param_paper)))
        axes.set_xticklabels(ep_labels_latex)
        axes.set_ylabel("Normalised difference (%)", fontsize=12)

        figure_name = "input_ep_param_BHM_extendedwithcase_" + str(using_CT_mesh_number) + "_" + str(num_input)
        axes.set_ylim([min(inf_bound), max(sup_bound)])
        axes.set_title("Comparison of the first " + str(
            num_input) + " input EP parameters from BHM extended with case " + str(using_CT_mesh_number) + " (normalised) with the simulation ground truth (red line)")

        fig.tight_layout()

        plt.savefig(os.path.join(path_figures, figure_name + ".png"), bbox_inches="tight", dpi=300)
        plt.close()


def plot_wave_new_imp_extended(CT_mesh_number = 2, using_CT_mesh_number=[1], subfolder="anatomy_max_range_limit_CT25percent", waveno=0,cutoff=3):

    output_folder = os.path.join(PROJECT_PATH, "validation", "CT_" + str(CT_mesh_number),
                                 "output_all_input_none_extended" + ''.join([str(i) for i in using_CT_mesh_number]),
                                "figures")
    pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)

    filename =  os.path.join(output_folder,"wave_prob_imp")

    if len(using_CT_mesh_number) == 1:
        plot_title = "BHM extended with patient #" + str(using_CT_mesh_number) + " in patient #" + str(CT_mesh_number) + ": percentage of implausible points"
    else:
        plot_title = "BHM extended with patients #" + ', #'.join([str(i) for i in using_CT_mesh_number]) + " in patient #" + str(CT_mesh_number) + ": percentage of implausible points"


    active_features = read_labels(os.path.join("/data", "fitting", subfolder, "output_labels.txt"))
    CT_results = []

    for feature_i in range(len(active_features)):
        CT_results.append(
            np.loadtxt(os.path.join(PROJECT_PATH, "CT_anatomy", active_features[feature_i] + ".dat"), dtype=float))

    CT_anatomy_values = np.loadtxt(open(os.path.join(PROJECT_PATH, "CT_anatomy", "X_anatomy.csv")), delimiter=',', skiprows=1)
    CT_mesh_modes = CT_anatomy_values[CT_mesh_number-1,0:9]
    CT_ep_param = np.array([80, 70, 0.8, 0.29, 7])

    CT_goal_point = np.append(CT_mesh_modes, CT_ep_param)

    print("Computing implausibility...")
    nroy_input, impl_values, _ = bhm_fill_gaps_extended(LVV_mean=CT_results[0][CT_mesh_number - 1],
                                                            RVV_mean=CT_results[1][CT_mesh_number - 1],
                                                            LAV_mean=CT_results[2][CT_mesh_number - 1],
                                                            RAV_mean=CT_results[3][CT_mesh_number - 1],
                                                            LVOTdiam_mean=CT_results[4][CT_mesh_number - 1],
                                                            RVOTdiam_mean=CT_results[5][CT_mesh_number - 1],
                                                            LVmass_mean=CT_results[6][CT_mesh_number - 1],
                                                            LVWT_mean=CT_results[7][CT_mesh_number - 1],
                                                            LVEDD_mean=CT_results[8][CT_mesh_number - 1],
                                                            SeptumWT_mean=CT_results[9][CT_mesh_number - 1],
                                                            RVlongdiam_mean=CT_results[10][CT_mesh_number - 1],
                                                            TAT_mean=CT_results[11][CT_mesh_number - 1],
                                                            TATLVendo_mean=CT_results[12][CT_mesh_number - 1],
                                                            CT_mesh_number = using_CT_mesh_number,
                                                        subfolder=subfolder,
                                                        num_wave=waveno,
                                                        cutoff=cutoff
                                                            )
    print("Done")

    X = nroy_input

    xlabels_EP = read_labels(os.path.join(PROJECT_PATH, subfolder, "EP_funct_labels_latex.txt"))
    xlabels_anatomy = read_labels(os.path.join(PROJECT_PATH, subfolder, "modes_labels.txt"))
    xlabels = [lab for sublist in [xlabels_anatomy, xlabels_EP] for lab in sublist]

    C = impl_values
    cmap = "jet"

    height = 9.36111
    width = 5.91667
    fig = plt.figure(figsize=(3 * width, 3 * height / 3))
    gs = matplotlib.gridspec.GridSpec(
        X.shape[1] - 1,
        X.shape[1],
        width_ratios=(X.shape[1] - 1) * [1] + [0.1],
    )

    path_match = os.path.join(PROJECT_PATH, "match")

    param_ranges_lower_anatomy = np.loadtxt(os.path.join(path_match, "anatomy_input_range_lower_max_range.dat"), dtype=float)
    param_ranges_upper_anatomy = np.loadtxt(os.path.join(path_match, "anatomy_input_range_upper_max_range.dat"), dtype=float)

    param_ranges_lower_EP = np.loadtxt(os.path.join(path_match, "EP_input_range_lower_max_range.dat"), dtype=float)
    param_ranges_upper_EP = np.loadtxt(os.path.join(path_match, "EP_input_range_upper_max_range.dat"), dtype=float)

    param_ranges_lower = np.append(param_ranges_lower_anatomy, param_ranges_lower_EP)
    param_ranges_upper = np.append(param_ranges_upper_anatomy, param_ranges_upper_EP)

    param_ranges = np.column_stack((param_ranges_lower, param_ranges_upper))
    print("Plotting...")

    for k in tqdm.tqdm(range(X.shape[1] * X.shape[1])):
        i = k % X.shape[1]
        j = k // X.shape[1]

        if i > j:
            axis = fig.add_subplot(gs[i - 1, j])
            axis.set_facecolor("#800000")

            if (xlabels[i] == "$\mathregular{k_{fibre}}$") or (xlabels[j] == "$\mathregular{k_{fibre}}$"):
                hexagon_size = 9
            else:
                hexagon_size = 49

            cbar_label = "Implausible points (%)"

            def prob_IMP(x, threshold):
                return 100 * sum(i >= threshold for i in x) / len(x)

            reduce_function = partial(prob_IMP, threshold=cutoff)

            im = axis.hexbin(
                X[:, j],
                X[:, i],
                C=C,
                reduce_C_function=reduce_function,
                gridsize=hexagon_size,
                cmap=cmap,
                vmin=0,
                vmax=100,
            )

            plt.plot(CT_goal_point[j], CT_goal_point[i], 'kx')

            axis.set_xlim([param_ranges[j, 0], param_ranges[j, 1]])
            axis.set_ylim([param_ranges[i, 0], param_ranges[i, 1]])

            if i == X.shape[1] - 1:
                axis.set_xlabel(xlabels[j], fontsize=12)
            else:
                axis.set_xticklabels([])
            if j == 0:
                axis.set_ylabel(xlabels[i], fontsize=12)
            else:
                axis.set_yticklabels([])

    cbar_axis = fig.add_subplot(gs[:, X.shape[1] - 1])
    cbar = fig.colorbar(im, cax=cbar_axis, format='%.2f')
    cbar.set_label(cbar_label, size=8)
    fig.tight_layout()
    plt.suptitle(plot_title, fontsize=18)

    print("Saving HD plot...")
    plt.savefig(filename + ".png", bbox_inches="tight", dpi=900)
    plt.close()


    print("Plotting...")

    filename =  os.path.join(output_folder,"wave_min_imp")

    if len(using_CT_mesh_number) == 1:
        plot_title = "BHM extended with patient #" + str(using_CT_mesh_number) + " in patient #" + str(CT_mesh_number) + ": minimum implausibility"
    else:
        plot_title = "BHM extended with patients #" + ', #'.join([str(i) for i in using_CT_mesh_number]) + " in patient #" + str(CT_mesh_number) + ": minimum implausibility"

    height = 9.36111
    width = 5.91667
    fig = plt.figure(figsize=(3 * width, 3 * height / 3))
    gs = matplotlib.gridspec.GridSpec(
        X.shape[1] - 1,
        X.shape[1],
        width_ratios=(X.shape[1] - 1) * [1] + [0.1],
    )

    for k in tqdm.tqdm(range(X.shape[1] * X.shape[1])):
        i = k % X.shape[1]
        j = k // X.shape[1]

        if i > j:
            axis = fig.add_subplot(gs[i - 1, j])
            axis.set_facecolor("#800000")

            if (xlabels[i] == "$\mathregular{k_{fibre}}$") or (xlabels[j] == "$\mathregular{k_{fibre}}$"):
                hexagon_size = 9
            else:
                hexagon_size = 49

            cbar_label = "Implausibility"

            im = axis.hexbin(
                X[:, j],
                X[:, i],
                C=C,
                reduce_C_function=np.min,
                gridsize=hexagon_size,
                cmap=cmap,
                vmin=1,
                vmax=cutoff,
            )

            plt.plot(CT_goal_point[j], CT_goal_point[i], 'wx')

            axis.set_xlim([param_ranges[j, 0], param_ranges[j, 1]])
            axis.set_ylim([param_ranges[i, 0], param_ranges[i, 1]])

            if i == X.shape[1] - 1:
                axis.set_xlabel(xlabels[j], fontsize=12)
            else:
                axis.set_xticklabels([])
            if j == 0:
                axis.set_ylabel(xlabels[i], fontsize=12)
            else:
                axis.set_yticklabels([])

    cbar_axis = fig.add_subplot(gs[:, X.shape[1] - 1])
    cbar = fig.colorbar(im, cax=cbar_axis, format='%.2f')
    cbar.set_label(cbar_label, size=8)
    fig.tight_layout()
    plt.suptitle(plot_title, fontsize=18)

    print("Saving HD plot...")
    plt.savefig(filename + ".png", bbox_inches="tight", dpi=900)




##### Extending emulator with other cases, using NROY_region

def validation_simulation_minimising_extended_emulation(CT_mesh_number=2, using_CT_mesh_number=1):

    active_features = read_labels(os.path.join("/data", "fitting", "anatomy_max_range", "output_labels.txt"))

    CT_results = []

    for feature_i in range(len(active_features)):
        CT_results.append(
            np.loadtxt(os.path.join("/data", "fitting", "CT_anatomy", active_features[feature_i] + ".dat"), dtype=float))

    exp_goal = [x[using_CT_mesh_number-1] for x in CT_results]

    nroy_input = np.loadtxt(os.path.join("/data/fitting", "anatomy_max_range", "wave2", "X_test.dat"),
                            dtype=float)

    difference_matrix = np.empty((0,len(nroy_input)))

    emulator, _ = add_simulation_to_emulator_and_wave(CT_mesh_number=CT_mesh_number)

    for i, output_name in enumerate(active_features):

        emul = emulator[i]

        mean_prediction, _ = emul.predict(nroy_input)
        difference_matrix = np.vstack((difference_matrix, (np.abs(exp_goal[i] - mean_prediction))/exp_goal[i]))

    vector_to_minimise = np.linalg.norm(difference_matrix, axis=0)

    best_cases_idx = np.argsort(vector_to_minimise)

    points_to_test = nroy_input[best_cases_idx[:10]]

    SEED = 2
    # ----------------------------------------------------------------
    # Make the code reproducible
    np.random.seed(SEED)

    subfolder = os.path.join("validation","CT_" + str(CT_mesh_number), "output_all_input_none_usingNROY_extended" + str(using_CT_mesh_number))
    meshes_folder = os.path.join("/data","fitting",subfolder)

    pathlib.Path(meshes_folder).mkdir(parents=True, exist_ok=True)

    anatomy.preprocess_input(path_gpes = meshes_folder, anatomy_and_EP_values = points_to_test)
    anatomy.build_meshes(subfolder=subfolder, path_gpes = meshes_folder)
    anatomy.EP_setup(subfolder=subfolder, path_gpes = meshes_folder)
    anatomy.EP_simulations(subfolder=subfolder, path_gpes = meshes_folder)
    anatomy.write_output_casewise(subfolder=subfolder, outpath = meshes_folder)
    anatomy.collect_output(subfolder=subfolder, outpath = meshes_folder)

def plot_difference_mini_extended_emul_groundtruth(CT_mesh_number = 2, num_input = 10, using_CT_mesh_number=1):

    if num_input == "all":
        for i in range(1,11):
            plot_difference_mini_extended_emul_groundtruth(CT_mesh_number=CT_mesh_number, num_input=i, using_CT_mesh_number=using_CT_mesh_number)
    else:
        output_labels = read_labels(os.path.join("/data", "fitting", "CT_anatomy", "output_labels.txt"))
        ep_labels_latex = read_labels(os.path.join("/data","fitting", "validation", "EP_funct_labels_latex.txt"))

        path_figures = os.path.join("/data", "fitting", "validation", "CT_" + str(CT_mesh_number), "output_all_input_none_usingNROY_extended" + str(using_CT_mesh_number), "figures")

        pathlib.Path(path_figures).mkdir(parents=True, exist_ok=True)

        ct_modes_csv_path = os.path.join("/data", "fitting", "match", "CT_cohort_modes_weights.csv")
        all_ct_modes = np.genfromtxt(ct_modes_csv_path, delimiter=',', skip_header=True)
        wanted_ct_modes = all_ct_modes[CT_mesh_number-1, 1:10]
        ep_param_paper = np.array([80, 70, 0.8, 0.29, 7])

        bhm_modes_csv_path = os.path.join("/data", "fitting", "validation", "CT_" + str(CT_mesh_number), "output_all_input_none_usingNROY_extended" + str(using_CT_mesh_number), "X_anatomy.csv")
        all_bhm_modes = np.genfromtxt(bhm_modes_csv_path, delimiter=',', skip_header=True)
        wanted_bhm_modes = all_bhm_modes[:, 0:9]
        ep_param_bhm = np.genfromtxt(os.path.join("/data", "fitting", "validation", "CT_" + str(CT_mesh_number), "output_all_input_none_usingNROY_extended" + str(using_CT_mesh_number), "X_EP.dat"))

        y_simul = []

        for feature in output_labels:
            y_simul_current = np.genfromtxt(os.path.join("/data", "fitting", "CT_anatomy", feature + ".dat"))
            y_simul.append(y_simul_current[CT_mesh_number-1])

        y_bhm = []

        for feature in output_labels:
            y_bhm_current = np.genfromtxt(os.path.join("/data", "fitting",  "validation", "CT_" + str(CT_mesh_number), "output_all_input_none_usingNROY_extended" + str(using_CT_mesh_number), feature + ".dat"))
            y_bhm = np.hstack((y_bhm,y_bhm_current))

        y_bhm =  y_bhm.reshape((len(output_labels),10)) # Each row is a feature
        y_bhm = list(map(list,zip(*y_bhm))) # Each row is a mesh

        y_bhm_normalised = 100*np.copy((np.array(y_bhm) - np.array(np.tile(y_simul,[10,1])))/(np.array(np.tile(y_simul,[10,1]))))

        # We plot the output

        height = 9.36111
        width = 5.91667
        fig, axes = plt.subplots(1, 1, figsize=(2 * width, 2 * height / 4))

        inf_bound = [np.nanmin(y_bhm_normalised),-100]
        sup_bound = [np.nanmax(y_bhm_normalised),100]


        axes.scatter(
            np.tile(np.arange(len(output_labels)),num_input),
            y_bhm_normalised[:num_input,],
            s=5
        )

        axes.axhline(y=0, color='r', linestyle='-')

        axes.set_xticks(np.arange(len(output_labels)))
        axes.set_xticklabels(output_labels)
        axes.set_xlabel("Biomarker", fontsize=12)
        axes.set_ylabel("Normalised difference (%)", fontsize=12)

        figure_name = "output_minim_extendedwithcase_" + str(using_CT_mesh_number) + "emul_" + str(num_input)
        axes.set_ylim([min(inf_bound), max(sup_bound)])
        axes.set_title("First " + str(num_input) + " output from minimizing emulation extended with case " + str(using_CT_mesh_number) + " of NROY region (normalised) compared with the simulation ground truth (red line)")

        fig.tight_layout()

        plt.savefig(os.path.join(path_figures, figure_name + ".png"), bbox_inches="tight", dpi=300)
        plt.close()

        # We plot input modes
        modes_bhm_normalised = 100*np.copy((np.array(wanted_bhm_modes) - np.array(np.tile(wanted_ct_modes, [10, 1])))/(np.array(np.tile(wanted_ct_modes, [10, 1]))))

        height = 9.36111
        width = 5.91667
        fig, axes = plt.subplots(1, 1, figsize=(2 * width, 2 * height / 4))

        inf_bound = [np.nanmin(modes_bhm_normalised), -100]
        sup_bound = [np.nanmax(modes_bhm_normalised), 100]

        axes.scatter(
            np.tile(np.arange(len(wanted_ct_modes)), num_input),
            modes_bhm_normalised[:num_input, ],
            s=5
        )

        axes.axhline(y=1, color='r', linestyle='-')

        axes.set_xticks(np.arange(len(wanted_ct_modes)))
        axes.set_xticklabels(["Mode " + str(m) for m in range(1,10)])
        axes.set_ylabel("Normalised difference (%)", fontsize=12)

        figure_name = "input_modes_minim_extendedwithcase_" + str(using_CT_mesh_number) + "_emul_" + str(num_input)
        axes.set_ylim([min(inf_bound), max(sup_bound)])
        axes.set_title("Comparison of the first " + str(
            num_input) + " input modes from minim. the emulation extended with case " + str(using_CT_mesh_number) + "of the NROY region (normalised) with the simulation ground truth (red line)")

        fig.tight_layout()

        plt.savefig(os.path.join(path_figures, figure_name + ".png"), bbox_inches="tight", dpi=300)
        plt.close()

        # We plot input EP parameters
        ep_param_bhm_normalised = 100 * np.copy(
            (np.array(ep_param_bhm) - np.array(np.tile(ep_param_paper, [10, 1]))) / (
                np.array(np.tile(ep_param_paper, [10, 1]))))

        height = 9.36111
        width = 5.91667
        fig, axes = plt.subplots(1, 1, figsize=(2 * width, 2 * height / 4))

        inf_bound = [np.nanmin(ep_param_bhm_normalised), -100]
        sup_bound = [np.nanmax(ep_param_bhm_normalised), 100]

        axes.scatter(
            np.tile(np.arange(len(ep_param_paper)), num_input),
            ep_param_bhm_normalised[:num_input, ],
            s=5
        )

        axes.axhline(y=1, color='r', linestyle='-')

        axes.set_xticks(np.arange(len(ep_param_paper)))
        axes.set_xticklabels(ep_labels_latex)
        axes.set_ylabel("Normalised difference (%)", fontsize=12)

        figure_name = "input_ep_param_minim_extendedwithcase_" + str(using_CT_mesh_number) + "_emul_" + str(num_input)
        axes.set_ylim([min(inf_bound), max(sup_bound)])
        axes.set_title("Comparison of the first " + str(
            num_input) + " input EP parameters from minim. the emulation extended with case " + str(using_CT_mesh_number) + "of the NROY region (normalised) with the simulation ground truth (red line)")

        fig.tight_layout()

        plt.savefig(os.path.join(path_figures, figure_name + ".png"), bbox_inches="tight", dpi=300)
        plt.close()

