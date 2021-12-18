import csv
import diversipy
import numpy as np
import os
import pathlib

import Historia

import postprocessing

from global_variables_config import *

def save_patient_implausibility(emulators_vector, input_folder, patient_number, sd_magnitude):
    """ Function to save a text file with the implausibility of the actual input values of a patient. Each line
    corresponds to the implausibility value of each biomarker.

    @param emulators_vector: array with emulators that we'll use to calculate the implausibility.
    @param input_folder: path where we will save the implausibility file.
    @param patient_number: Number of the patient for which we will calculate their implausibility.
    @param sd_magnitude: Percentage of the biomarker value of the patient to use as standard variation.
    """

    pathlib.Path(os.path.join(PROJECT_PATH, input_folder)).mkdir(parents=True, exist_ok=True)

    patients_simulation_output = np.loadtxt(open(os.path.join(PROJECT_PATH, "anatomy_EP_patients.csv"), "rb"),
                                            delimiter=",", skiprows=1)
    exp_mean = patients_simulation_output[patient_number - 1,]
    exp_std = sd_magnitude / 100. * exp_mean
    exp_var = np.power(exp_std, 2)

    anatomy_values = np.loadtxt(open(os.path.join(PROJECT_PATH, "X_anatomy.csv")),
                                delimiter=',', skiprows=1)

    patient_input_points = np.hstack((anatomy_values[0:19, 0:9], np.tile([80, 70, 0.8, 0.29, 7], (19, 1))))

    point_to_emulate = patient_input_points[patient_number-1,]

    numerator_vector = abs([emul.predict([point_to_emulate])[0][0] for emul in emulators_vector] - exp_mean)
    denominator_vector = np.sqrt([np.power(emul.predict([point_to_emulate])[1][0], 2) for emul in emulators_vector] + exp_var)

    impl_value_vector = numerator_vector/denominator_vector

    filename = os.path.join(PROJECT_PATH, input_folder,
                            "implausibilities_patient" + str(patient_number) + "_sd_" + str(sd_magnitude) + ".dat")
    with open(filename, "w") as f:
        f.write("\n".join([str(round(i,2)) for i in impl_value_vector]))


def compute_nroy_region(emulators_vector, implausibility_threshold, literature_data, input_folder, previous_wave_name=None,
                        patient_number=None,
                        sd_magnitude=None, first_time=False):
    """Function to compute the not-ruled-out-yet (NROY) region using specified emulators and biomarkers. It also saves
    the points to emulate and the percentage of NROY region compared to the previous wave.

    @param emulators_vector: array with emulator objects that will be used to compute implausibility.
    @param implausibility_threshold: Threshold to decide if a point is implausible or not.
    @param literature_data: If True, uses literature data instead of patient.
    @param input_folder: Folder name where the results will be saved.
    @param previous_wave_name: If it's not the first run, name of the previous wave to start from the previous NROY
    region.
    @param patient_number: If not using literature value, number of the patient to be used.
    @param sd_magnitude: Percentage of the value of the biomarker to be used as standard deviation.
    @param first_time: If true, it doesn't load a previous wave.

    @return The new wave object.
    """

    pathlib.Path(os.path.join(PROJECT_PATH, input_folder)).mkdir(parents=True, exist_ok=True)

    SEED = 2
    # ----------------------------------------------------------------
    # Make the code reproducible
    np.random.seed(SEED)

    param_ranges_lower_anatomy = np.loadtxt(os.path.join(PROJECT_PATH, "anatomy_input_range_lower.dat"),
                                            dtype=float)
    param_ranges_upper_anatomy = np.loadtxt(os.path.join(PROJECT_PATH, "anatomy_input_range_upper.dat"),
                                            dtype=float)

    param_ranges_lower_ep = np.loadtxt(os.path.join(PROJECT_PATH, "EP_input_range_lower.dat"), dtype=float)
    param_ranges_upper_ep = np.loadtxt(os.path.join(PROJECT_PATH, "EP_input_range_upper.dat"), dtype=float)

    param_ranges_lower = np.append(param_ranges_lower_anatomy, param_ranges_lower_ep)
    param_ranges_upper = np.append(param_ranges_upper_anatomy, param_ranges_upper_ep)

    param_ranges = np.column_stack((param_ranges_lower, param_ranges_upper))

    if literature_data:
        exp_mean = np.loadtxt(os.path.join(PROJECT_PATH, "anatomy_EP_lit_mean.txt"), dtype=float)
        exp_std = np.loadtxt(os.path.join(PROJECT_PATH, "anatomy_EP_lit_sd.txt"), dtype=float)
        exp_var = np.power(exp_std, 2)
    else:
        patients_simulation_output = np.loadtxt(open(os.path.join(PROJECT_PATH, "anatomy_EP_patients.csv"), "rb"),
                                                   delimiter=",", skiprows=1)
        exp_mean = patients_simulation_output[patient_number-1,]
        exp_std = sd_magnitude/100.*exp_mean
        exp_var = np.power(exp_std, 2)

    if first_time:
        wave = Historia.history.hm.Wave(emulator=emulators_vector,
                                        Itrain=param_ranges,
                                        cutoff=implausibility_threshold,
                                        maxno=1,
                                        mean=exp_mean,
                                        var=exp_var)
    else:
        wave = Historia.history.hm.Wave()
        wave.load(previous_wave_name)
        wave.emulator = emulators_vector
        wave.Itrain = param_ranges
        wave.cutoff = implausibility_threshold
        wave.maxno = 1
        wave.mean = exp_mean
        wave.var = exp_var


    if not os.path.isfile(os.path.join(PROJECT_PATH,input_folder, "points_to_emulate.dat")):
        if first_time:
            points_to_emulate = Historia.shared.design_utils.lhd(param_ranges, int(1e5), SEED)
        else:
            points_to_emulate = add_points_to_nroy(input_wave=wave, param_ranges=param_ranges, n_pts=int(1e5),
                                                   scale=0.1)
        np.savetxt(os.path.join(PROJECT_PATH,input_folder, "points_to_emulate.dat"), points_to_emulate, fmt="%.2f")
    else:
        points_to_emulate = np.loadtxt(os.path.join(PROJECT_PATH,input_folder, "points_to_emulate.dat"), dtype=float)

    # ============= We finally print and show the wave we wanted =============#
    print("Computing implausibility of NROY...")
    wave.find_regions(points_to_emulate)  # enforce the implausibility criterion to detect regions of
    # non-implausible and of implausible points
    print("Finished")
    nimp = len(wave.nimp_idx)
    imp = len(wave.imp_idx)
    tests = nimp + imp
    perc = 100 * nimp / tests

    if literature_data:
        np.savetxt(os.path.join(PROJECT_PATH, input_folder, "NROY_rel_literature.dat"), [perc], fmt='%.2f')
    else:
        np.savetxt(os.path.join(PROJECT_PATH, input_folder, "NROY_rel_patient" + str(patient_number) + "_sd_" +
                                str(sd_magnitude) + ".dat"), [perc], fmt='%.2f')

    return wave

def plot_nroy(input_folder, wave, literature_data, patient_number=None, sd_magnitude=None, title=None):
    """Function to plot the NROY region projecting with the minimum implausibility and with the percentage of
    implausible points.

    @param input_folder: Name of the folder where the plots will be saved.
    @param wave: Wave object to plot
    @param literature_data: If true the name of the file with the plot is literature data.
    @param patient_number: If using patients, number of the patient to use.
    @param sd_magnitude: Percentage of the value of the biomarker to be used as standard deviation.
    @param title: First part of the title of the plot.
    """

    param_ranges_lower_anatomy = np.loadtxt(os.path.join(PROJECT_PATH, "anatomy_input_range_lower.dat"),
                                            dtype=float)
    param_ranges_upper_anatomy = np.loadtxt(os.path.join(PROJECT_PATH, "anatomy_input_range_upper.dat"),
                                            dtype=float)

    param_ranges_lower_ep = np.loadtxt(os.path.join(PROJECT_PATH, "EP_input_range_lower.dat"), dtype=float)
    param_ranges_upper_ep = np.loadtxt(os.path.join(PROJECT_PATH, "EP_input_range_upper.dat"), dtype=float)

    param_ranges_lower = np.append(param_ranges_lower_anatomy, param_ranges_lower_ep)
    param_ranges_upper = np.append(param_ranges_upper_anatomy, param_ranges_upper_ep)

    param_ranges = np.column_stack((param_ranges_lower, param_ranges_upper))

    xlabels = np.genfromtxt(os.path.join(PROJECT_PATH, "param_labels.txt"), dtype=str, delimiter='\n')

    pathlib.Path(os.path.join(PROJECT_PATH, input_folder, "figures")).mkdir(parents=True, exist_ok=True)

    if literature_data:
        plot_name = "literature"
    else:
        plot_name = "patient_" + str(patient_number) + "_sd_" + str(sd_magnitude)

    print("Plotting impl min...")
    postprocessing.plot_wave(W=wave, xlabels=xlabels,
                             filename=os.path.join(PROJECT_PATH, input_folder, "figures", "wave_" + plot_name + "_impl_min"),
                             reduction_function="min",
                             plot_title=title + ": taking the min of each slice",
                             param_ranges=param_ranges
                             )
    print("Plotting prob imp...")
    postprocessing.plot_wave(W=wave, xlabels=xlabels,
                             filename=os.path.join(PROJECT_PATH, input_folder, "figures", "wave_" + plot_name + "_prob_imp"),
                             reduction_function="prob_IMP",
                             plot_title=title + ": percentage of implausible points",
                             param_ranges=param_ranges
                             )

def add_points_to_nroy(input_wave, param_ranges, n_pts, scale=0.1):
    """Function to add points in the NROY region. It uses the "cloud technique", looking for points close to other
     points we know they are non-implausible. It's a copy from add_points in Stefano's function specifying the
     parameters range to sample in an adequate region.

    @param input_wave: Wave object to use.
    @param param_ranges: Initial ranges of the parameters so the plots are comparable.
    @param n_pts: Number of points to emulate.
    @param scale: How much to look for non implausible points in the neighborhood of the points we know they are
    non-implausible. The bigger, the farthest it reaches. The bigger, more points will it find but the borders
    will be less defined.

    @return The points found in the NROY region

    @warning The wave object instance internal structure will change after calling this function: we recommend calling
    self.save() beforehand.
    """

    wave=input_wave

    nsidx = np.copy(wave.nsimul_idx)
    NROY = np.copy(wave.NIMP[nsidx])

    lbounds = param_ranges[:, 0]
    ubounds = param_ranges[:, 1]

    print(
        f"\nRequested points: {n_pts}\nAvailable points: {NROY.shape[0]}\nStart searching..."
    )

    count = 0
    a, b = (
        NROY.shape[0] if NROY.shape[0] < n_pts else n_pts,
        n_pts - NROY.shape[0] if n_pts - NROY.shape[0] > 0 else 0,
    )
    print(
        f"\n[Iteration: {count:<2}] Found: {a:<{len(str(n_pts))}} ({'{:.2f}'.format(100 * a / n_pts):>6}%) | Missing: {b:<{len(str(n_pts))}}"
    )

    while NROY.shape[0] < n_pts:
        count += 1

        I = Historia.shared.design_utils.get_minmax(NROY)
        SCALE = scale * np.array(
            [I[i, 1] - I[i, 0] for i in range(NROY.shape[1])]
        )

        temp = np.random.normal(loc=NROY, scale=SCALE)
        while True:
            l = []
            for i in range(temp.shape[0]):
                d1 = temp[i, :] - lbounds
                d2 = ubounds - temp[i, :]
                if (
                        np.sum(np.sign(d1)) != temp.shape[1]
                        or np.sum(np.sign(d2)) != temp.shape[1]
                ):
                    l.append(i)
            if l:
                temp[l, :] = np.random.normal(loc=NROY[l, :], scale=SCALE)
            else:
                break

        wave.find_regions(temp)
        NROY = np.vstack((NROY, wave.NIMP))

        a, b = (
            NROY.shape[0] if NROY.shape[0] < n_pts else n_pts,
            n_pts - NROY.shape[0] if n_pts - NROY.shape[0] > 0 else 0,
        )
        print(
            f"[Iteration: {count:<2}] Found: {a:<{len(str(n_pts))}} ({'{:.2f}'.format(100 * a / n_pts):>6}%) | Missing: {b:<{len(str(n_pts))}}"
        )

    print("\nDone.")
    TESTS = np.vstack(
        (
            NROY[: len(nsidx)],
            diversipy.subset.psa_select(NROY[len(nsidx):], n_pts - len(nsidx)),
        )
    )
    return TESTS

def generate_new_training_pts(wave, num_pts, output_folder, input_folder, wave_name):
    """ Function to save new training points from the NROY region to use in the next wave. It also saves the variance
    quotient of the emulators.

    @param wave: Wave object to use
    @param num_pts: Number of points to generate.
    @param output_folder: Folder to save the new data generated.
    @param input_folder: Name of the folder to re-save the wave object.
    @param wave_name: Name of wave for the file.
    """

    if not os.path.isfile(os.path.join(PROJECT_PATH, output_folder, "input_space_training.dat")):
        if num_pts > 0:
            training_pts = wave.get_points(num_pts)
        else:
            training_pts = []

        pathlib.Path(os.path.join(PROJECT_PATH, output_folder)).mkdir(parents=True, exist_ok=True)
        np.savetxt(os.path.join(PROJECT_PATH, output_folder, "input_space_training.dat"), training_pts, fmt="%.2f")

        wave.save(os.path.join(PROJECT_PATH, input_folder, wave_name))

    if not os.path.isfile(os.path.join(PROJECT_PATH, input_folder, "variance_quotient" + wave_name + ".dat")):
        np.savetxt(os.path.join(PROJECT_PATH, input_folder, "variance_quotient_" + wave_name + ".dat"), wave.PV, fmt="%.2f")

    if not os.path.isfile(os.path.join(PROJECT_PATH, output_folder, "input_ep_training.dat")) or not os.path.isfile(os.path.join(PROJECT_PATH, output_folder, "input_anatomy_training.csv")):
        with open(os.path.join(PROJECT_PATH, output_folder, "input_space_training.dat")) as f:
            anatomy_and_ep_values = f.read().splitlines()

        x_anatomy = []
        x_ep = []

        for full_line in anatomy_and_ep_values:
            if type(anatomy_and_ep_values) is not np.ndarray:
                line = full_line.split(' ')
            else:
                line = full_line
            x_anatomy.append(line[0:9])
            x_ep.append(line[9:14])

        if not os.path.isfile(os.path.join(PROJECT_PATH, output_folder, "input_ep_training.dat")):
            f = open(os.path.join(PROJECT_PATH, output_folder, "input_ep_training.dat"), "w")
            if type(anatomy_and_ep_values) is not np.ndarray:
                f.writelines(' '.join(row) + '\n' for row in x_ep)
            else:
                f.writelines(' '.join(str(elem) for elem in row) + '\n' for row in x_ep)
            f.close()

        if not os.path.isfile(os.path.join(PROJECT_PATH, output_folder, "input_ep_training.csv")):
            with open(os.path.join(PROJECT_PATH, output_folder, "input_anatomy_training.csv"), mode='w') as f:
                f_writer = csv.writer(f, delimiter=',')

                f_writer.writerow(["Mode" + str(i) for i in range(1, 19)])

                for current_line in range(len(x_anatomy)):
                    output = np.zeros(18)
                    output[0:9] = x_anatomy[current_line]
                    f_writer.writerow(["{0:.2f}".format(round(i, 2)) for i in output])

            f.close()
