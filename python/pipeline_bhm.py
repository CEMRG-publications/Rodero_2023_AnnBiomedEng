import csv
import numpy as np
import os
import pathlib
import skopt

from global_variables_config import *

import biomarkers
import emulators
import ep_simulations
import generate_meshes
import history_matching
import preprocess_mesh

np.random.seed(SEED)

def write_input_files(subfolder="initial_sweep", n_training_pts = 280, n_validation_pts=70, n_test_pts=88):
    """Function to generate the initial space using sobol sequences.

    @param subfolder: Folder within PROJECT_PATH where it will be created. Defaults to initial sweep.
    @param n_training_pts: Number of training points to generate. Defaults to 280 (20x n_params)
    @param n_validation_pts: Number of validation points to generate. Defaults to 70.
    @param n_test_pts: Number of test points to generate. Defaults to 88.

    @returns Files with input points for the three type of sets.
    """

    pathlib.Path(os.path.join(PROJECT_PATH, subfolder)).mkdir(parents=True, exist_ok=True)

    param_ranges_lower_anatomy = np.loadtxt(os.path.join(PROJECT_PATH, "anatomy_input_range_lower.dat"), dtype=float)
    param_ranges_upper_anatomy = np.loadtxt(os.path.join(PROJECT_PATH, "anatomy_input_range_upper.dat"), dtype=float)

    param_ranges_lower_ep = np.loadtxt(os.path.join(PROJECT_PATH, "EP_input_range_lower.dat"), dtype=float)
    param_ranges_upper_ep = np.loadtxt(os.path.join(PROJECT_PATH, "EP_input_range_upper.dat"), dtype=float)

    param_ranges_lower = np.append(1.25*param_ranges_lower_anatomy, param_ranges_lower_ep)
    param_ranges_upper = np.append(1.25*param_ranges_upper_anatomy, param_ranges_upper_ep)

    param_ranges = [(param_ranges_lower[i], param_ranges_upper[i]) for i in range(len(param_ranges_lower))]

    space = skopt.space.Space(param_ranges)

    sobol = skopt.sampler.Sobol(min_skip=SEED, max_skip=SEED)

    ####### TRAINING

    input_space = sobol.generate(space.dimensions, int(n_training_pts), random_state=SEED)

    input_ep = []
    input_anatomy = []

    num_ep = len(param_ranges_lower_ep)
    num_anatomy = len(param_ranges_lower_anatomy)
    for row in input_space:
        input_anatomy.append(row[:num_anatomy])
        input_ep.append(row[num_anatomy:(num_anatomy + num_ep)])

    f = open(os.path.join(PROJECT_PATH, subfolder, "input_space_training.dat"), "w")
    [f.write('%s\n' % ' '.join(map(str, [format(i, '.2f') for i in lhs_array]))) for lhs_array in input_space]
    f.close()

    f = open(os.path.join(PROJECT_PATH, subfolder, "input_ep_training.dat"), "w")
    [f.write('%s\n' % ' '.join(map(str, [format(i, '.2f') for i in lhs_array]))) for lhs_array in input_ep]
    f.close()

    with open(os.path.join(PROJECT_PATH, subfolder,"input_anatomy_training.csv"), mode='w') as f:
        f_writer = csv.writer(f, delimiter=',')

        f_writer.writerow(["Mode" + str(i) for i in range(1,19)])

        for current_line in range(len(input_anatomy)):
            output = np.zeros(18)
            output[0:9] = input_anatomy[current_line]
            f_writer.writerow(["{0:.2f}".format(round(i,2)) for i in output])
    f.close()

    ######## TEST

    if n_test_pts > 0:

        input_space = sobol.generate(space.dimensions, int(n_test_pts), random_state=SEED)

        input_ep = []
        input_anatomy = []

        num_ep = len(param_ranges_lower_ep)
        num_anatomy = len(param_ranges_lower_anatomy)
        for row in input_space:
            input_anatomy.append(row[:num_anatomy])
            input_ep.append(row[num_anatomy:(num_anatomy + num_ep)])

        f = open(os.path.join(PROJECT_PATH, subfolder, "input_space_test.dat"), "w")
        [f.write('%s\n' % ' '.join(map(str, [format(i, '.2f') for i in lhs_array]))) for lhs_array in input_space]
        f.close()

        f = open(os.path.join(PROJECT_PATH, subfolder, "input_ep_test.dat"), "w")
        [f.write('%s\n' % ' '.join(map(str, [format(i, '.2f') for i in lhs_array]))) for lhs_array in input_ep]
        f.close()

        with open(os.path.join(PROJECT_PATH, subfolder, "input_anatomy_test.csv"), mode='w') as f:
            f_writer = csv.writer(f, delimiter=',')

            f_writer.writerow(["Mode" + str(i) for i in range(1, 19)])

            for current_line in range(len(input_anatomy)):
                output = np.zeros(18)
                output[0:9] = input_anatomy[current_line]
                f_writer.writerow(["{0:.2f}".format(round(i, 2)) for i in output])

        f.close()



    ######### VALIDATION

    if n_validation_pts > 0:

        input_space = sobol.generate(space.dimensions, int(n_validation_pts), random_state=SEED)

        input_ep = []
        input_anatomy = []

        num_ep = len(param_ranges_lower_ep)
        num_anatomy = len(param_ranges_lower_anatomy)
        for row in input_space:
            input_anatomy.append(row[:num_anatomy])
            input_ep.append(row[num_anatomy:(num_anatomy + num_ep)])

        f = open(os.path.join(PROJECT_PATH, subfolder, "input_space_validation.dat"), "w")
        [f.write('%s\n' % ' '.join(map(str, [format(i, '.2f') for i in lhs_array]))) for lhs_array in input_space]
        f.close()

        f = open(os.path.join(PROJECT_PATH, subfolder, "input_ep_validation.dat"), "w")
        [f.write('%s\n' % ' '.join(map(str, [format(i, '.2f') for i in lhs_array]))) for lhs_array in input_ep]
        f.close()

        with open(os.path.join(PROJECT_PATH, subfolder, "input_anatomy_validation.csv"), mode='w') as f:
            f_writer = csv.writer(f, delimiter=',')

            f_writer.writerow(["Mode" + str(i) for i in range(1, 19)])

            for current_line in range(len(input_anatomy)):
                output = np.zeros(18)
                output[0:9] = input_anatomy[current_line]
                f_writer.writerow(["{0:.2f}".format(round(i, 2)) for i in output])
        f.close()

def initial_sweep():
    """ Function to pipeline the generation of meshes and run simulations for the initial emulators. It will be common
    for multiple patients.
    """
    write_input_files(subfolder="initial_sweep", n_training_pts=280, n_validation_pts=70, n_test_pts=88)

    generate_meshes.sample_atlas(subfolder="initial_sweep", csv_filename="input_anatomy_training.csv")
    generate_meshes.sample_atlas(subfolder="initial_sweep", csv_filename="input_anatomy_validation.csv")
    generate_meshes.sample_atlas(subfolder="initial_sweep", csv_filename="input_anatomy_test.csv")

    preprocess_mesh.biv_setup(subfolder="initial_sweep", anatomy_csv_file="input_anatomy_training.csv",
                              ep_dat_file="input_ep_training.dat")
    preprocess_mesh.biv_setup(subfolder="initial_sweep", anatomy_csv_file="input_anatomy_validation.csv",
                              ep_dat_file="input_ep_validation.dat")
    preprocess_mesh.biv_setup(subfolder="initial_sweep", anatomy_csv_file="input_anatomy_test.csv",
                              ep_dat_file="input_ep_test.dat")

    ep_simulations.run(subfolder="initial_sweep", anatomy_csv_file="input_anatomy_training.csv",
                       ep_dat_file="input_ep_training.dat")
    ep_simulations.run(subfolder="initial_sweep", anatomy_csv_file="input_anatomy_validation.csv",
                       ep_dat_file="input_ep_validation.dat")
    ep_simulations.run(subfolder="initial_sweep", anatomy_csv_file="input_anatomy_test.csv",
                       ep_dat_file="input_ep_test.dat")

    biomarkers.extract(subfolder="initial_sweep", anatomy_csv_file="input_anatomy_training.csv",
                       ep_dat_file="input_ep_training.dat")
    biomarkers.extract(subfolder="initial_sweep", anatomy_csv_file="input_anatomy_validation.csv",
                       ep_dat_file="input_ep_validation.dat")
    biomarkers.extract(subfolder="initial_sweep", anatomy_csv_file="input_anatomy_test.csv",
                       ep_dat_file="input_ep_test.dat")

    emulators_vector = emulators.train(folders=["initial_sweep"])

    return emulators_vector

def first_wave(subfolder="literature/wave1"):
    """ Function to pipeline the generation of meshes and run simulations for the initial emulators. It will be common
    for multiple patients, but for the sake of code performance we separate in in different folders.
    """
    write_input_files(subfolder=subfolder, n_training_pts=280, n_validation_pts=70, n_test_pts=88)

    generate_meshes.sample_atlas(subfolder=subfolder, csv_filename="input_anatomy_training.csv")
    generate_meshes.sample_atlas(subfolder=subfolder, csv_filename="input_anatomy_validation.csv")
    generate_meshes.sample_atlas(subfolder=subfolder, csv_filename="input_anatomy_test.csv")

    preprocess_mesh.biv_setup(subfolder=subfolder, anatomy_csv_file="input_anatomy_training.csv",
                              ep_dat_file="input_ep_training.dat")
    preprocess_mesh.biv_setup(subfolder=subfolder, anatomy_csv_file="input_anatomy_validation.csv",
                              ep_dat_file="input_ep_validation.dat")
    preprocess_mesh.biv_setup(subfolder=subfolder, anatomy_csv_file="input_anatomy_test.csv",
                              ep_dat_file="input_ep_test.dat")

    ep_simulations.run(subfolder=subfolder, anatomy_csv_file="input_anatomy_training.csv",
                       ep_dat_file="input_ep_training.dat")
    ep_simulations.run(subfolder=subfolder, anatomy_csv_file="input_anatomy_validation.csv",
                       ep_dat_file="input_ep_validation.dat")
    ep_simulations.run(subfolder=subfolder, anatomy_csv_file="input_anatomy_test.csv",
                       ep_dat_file="input_ep_test.dat")

    biomarkers.extract(subfolder=subfolder, anatomy_csv_file="input_anatomy_training.csv",
                       ep_dat_file="input_ep_training.dat")
    biomarkers.extract(subfolder=subfolder, anatomy_csv_file="input_anatomy_validation.csv",
                       ep_dat_file="input_ep_validation.dat")
    biomarkers.extract(subfolder=subfolder, anatomy_csv_file="input_anatomy_test.csv",
                       ep_dat_file="input_ep_test.dat")

    emulators_vector = emulators.train(folders=[subfolder])

    return emulators_vector

def literature(run_wave0, run_wave1, run_wave2):
    """Function to pipeline the generation of meshes and the running of simulations when using values from the
    literature (for verification).

    @param run_wave0: If true, it runs the initial wave loading the emulators from initial_sweep.
    @param run_wave1: If true, it runs the wave 1 (second wave) loading the emulators from the previous wave.
    @param run_wave2: If true, it runs the wave 2 (third wave) loading the emulators from the previous wave.
    """
    if run_wave0:
        emulators_vector = initial_sweep()

        wave = history_matching.compute_nroy_region(emulators_vector=emulators_vector, implausibility_threshold=3.2,
                                                    literature_data=True, input_folder="initial_sweep",first_time=True)
        history_matching.plot_nroy(input_folder="initial_sweep", wave=wave, literature_data=True,title = "Initial literature wave")
        history_matching.generate_new_training_pts(wave=wave, num_pts=140, output_folder="literature/wave1",
                                                   input_folder="initial_sweep", wave_name="wave0_literature")
    if run_wave1:
        generate_meshes.sample_atlas(subfolder="literature/wave1", csv_filename="input_anatomy_training.csv")
        preprocess_mesh.biv_setup(subfolder="literature/wave1", anatomy_csv_file="input_anatomy_training.csv",
                                  ep_dat_file="input_ep_training.dat")
        ep_simulations.run(subfolder="literature/wave1", anatomy_csv_file="input_anatomy_training.csv",
                           ep_dat_file="input_ep_training.dat")
        biomarkers.extract(subfolder="literature/wave1", anatomy_csv_file="input_anatomy_training.csv",
                           ep_dat_file="input_ep_training.dat")

        emulators_vector = emulators.train(folders=["initial_sweep","literature/wave1"])
        wave = history_matching.compute_nroy_region(emulators_vector=emulators_vector, implausibility_threshold=3.2,
                                                    literature_data=True, input_folder="literature/wave1",
                                                    previous_wave_name=os.path.join(PROJECT_PATH,"initial_sweep","wave0_literature"))
        history_matching.plot_nroy(input_folder="literature/wave1", wave=wave, literature_data=True,title = "Second literature wave")
        history_matching.generate_new_training_pts(wave=wave, num_pts=140, output_folder="literature/wave2",
                                                   input_folder="literature/wave1", wave_name="wave1_literature")
    if run_wave2:
        generate_meshes.sample_atlas(subfolder="literature/wave2", csv_filename="input_anatomy_training.csv")
        preprocess_mesh.biv_setup(subfolder="literature/wave2", anatomy_csv_file="input_anatomy_training.csv",
                                  ep_dat_file="input_ep_training.dat")
        ep_simulations.run(subfolder="literature/wave2", anatomy_csv_file="input_anatomy_training.csv",
                           ep_dat_file="input_ep_training.dat")
        biomarkers.extract(subfolder="literature/wave2", anatomy_csv_file="input_anatomy_training.csv",
                           ep_dat_file="input_ep_training.dat")

        emulators_vector = emulators.train(folders=["initial_sweep","literature/wave1","literature/wave2"])
        wave = history_matching.compute_nroy_region(emulators_vector=emulators_vector, implausibility_threshold=3.0,
                                                    literature_data=True, input_folder="literature/wave2",
                                                    previous_wave_name=os.path.join(PROJECT_PATH,"literature/wave1","wave1_literature"))
        history_matching.plot_nroy(input_folder="literature/wave2", wave=wave, literature_data=True, title = "Third literature wave")
        history_matching.generate_new_training_pts(wave=wave, num_pts=140, output_folder="literature/wave3",
                                                   input_folder="literature/wave2", wave_name="wave2_literature")


def literature_convergence(perc_convergence=95.):
    """Function to pipeline the generation of meshes and the running of simulations using biomarkers from literature
    values. There will be two waves with implausibility of 3.2 and then the convergence starts with an implausibility
    threshold of 3.

    @param perc_convergence: threshold value for convergence. It is based on the percentage of new NROY (relative to
    the previous NROY region).
    """
    wave_number = 1
    converged = False

    while not converged:
        print("Running wave " + str(wave_number) + "...")

        if wave_number == 1:
            emulators_vector = first_wave(subfolder="literature/wave" + str(wave_number))
        else:
            generate_meshes.sample_atlas(subfolder="literature/wave" + str(wave_number),
                                         csv_filename="input_anatomy_training.csv")
            preprocess_mesh.biv_setup(subfolder="literature/wave" + str(wave_number),
                                      anatomy_csv_file="input_anatomy_training.csv",
                                      ep_dat_file="input_ep_training.dat")
            ep_simulations.run(subfolder="literature/wave" + str(wave_number),
                               anatomy_csv_file="input_anatomy_training.csv",
                               ep_dat_file="input_ep_training.dat")
            biomarkers.extract(subfolder="literature/wave" + str(wave_number),
                               anatomy_csv_file="input_anatomy_training.csv",
                               ep_dat_file="input_ep_training.dat")

            emulators_folders = []
            for waves in range(wave_number):
                emulators_folders.append("literature/wave" + str(waves+1))
            emulators_vector = emulators.train(folders=emulators_folders)

        if wave_number < 3:
            implausibility_threshold = 3.2
        else:
            implausibility_threshold = 3.

        if wave_number == 1:
            wave = history_matching.compute_nroy_region(emulators_vector=emulators_vector,
                                                    implausibility_threshold=implausibility_threshold,
                                                    literature_data=True,
                                                    input_folder="literature/wave" + str(wave_number), first_time=True)
        else:
            wave = history_matching.compute_nroy_region(emulators_vector=emulators_vector,
                                                        implausibility_threshold=implausibility_threshold,
                                                        literature_data=True,
                                                        input_folder="literature/wave" + str(wave_number),
                                                        previous_wave_name=os.path.join(PROJECT_PATH,
                                                                                        "literature/wave" +
                                                                                        str(wave_number-1),
                                                                                        "wave" + str(wave_number-1) +
                                                                                        "_literature"))

        history_matching.plot_nroy(input_folder="literature/wave" + str(wave_number), wave=wave, literature_data=True,
                                   title="Literature wave " + str(wave_number))
        np.savetxt(os.path.join(PROJECT_PATH, "literature/wave" + str(wave_number),
                                "variance_quotient_wave" + str(wave_number) + "_literature.dat"), wave.PV,
                   fmt="%.2f")

        nroy_rel = np.genfromtxt("/media/crg17/Seagate Expansion Drive/literature/wave" + str(wave_number) +
                                 "/NROY_rel_literature.dat", dtype=float)

        if (implausibility_threshold > 3.) or (nroy_rel < perc_convergence):
            history_matching.generate_new_training_pts(wave=wave, num_pts=140,
                                                       output_folder="literature/wave" + str(wave_number + 1),
                                                       input_folder="literature/wave" + str(wave_number),
                                                       wave_name="wave" + str(wave_number) + "_literature")
            wave_number += 1

        else:
            converged = True


def patient(patient_number, run_wave0, run_wave1, run_wave2, run_wave3, sd_magnitude):
    """Function to pipeline the generation of meshes and the running of simulations when using values from the
    simulations of a specific patient.

    @param patient_number: Number of the subject to test (1 to 19).
    @param run_wave0: If true, it runs the initial wave loading the emulators from initial_sweep.
    @param run_wave1: If true, it runs the wave 1 (second wave) loading the emulators from the previous wave.
    @param run_wave2: If true, it runs the wave 2 (third wave) loading the emulators from the previous wave.
    @param sd_magnitude: Percentage of the mean of the biomarker to apply as standard deviation.
    """
    if run_wave0:
        print("Running wave 0...")
        emulators_vector = initial_sweep()
        print("Computing implausibilities...")
        history_matching.save_patient_implausibility(emulators_vector=emulators_vector, input_folder="initial_sweep",
                                                     patient_number=patient_number, sd_magnitude=sd_magnitude)
        print("Computing NROY region...")
        wave = history_matching.compute_nroy_region(emulators_vector=emulators_vector, implausibility_threshold=3.2,
                                                    literature_data=False, input_folder="initial_sweep",
                                                    patient_number=patient_number, sd_magnitude=sd_magnitude,
                                                    first_time=True)
        history_matching.plot_nroy(input_folder="initial_sweep", wave=wave, literature_data=False,
                                   patient_number=patient_number, sd_magnitude=sd_magnitude,
                                   title = "Initial wave for #" + str(patient_number) + " with SD=" + str(sd_magnitude) + "%")
        history_matching.generate_new_training_pts(wave=wave, num_pts=140, output_folder="patient" + str(patient_number) +  "_sd_" + str(sd_magnitude)+ "/wave1",
                                                   input_folder="initial_sweep", wave_name="wave0_patient" + str(patient_number) +  "_sd_" + str(sd_magnitude))
    if run_wave1:
        print("Running wave 1...")

        generate_meshes.sample_atlas(subfolder="patient" + str(patient_number) +  "_sd_" + str(sd_magnitude)+ "/wave1", csv_filename="input_anatomy_training.csv")
        preprocess_mesh.biv_setup(subfolder="patient" + str(patient_number) +  "_sd_" + str(sd_magnitude)+ "/wave1", anatomy_csv_file="input_anatomy_training.csv",
                                  ep_dat_file="input_ep_training.dat")
        ep_simulations.run(subfolder="patient" + str(patient_number) +  "_sd_" + str(sd_magnitude)+ "/wave1", anatomy_csv_file="input_anatomy_training.csv",
                           ep_dat_file="input_ep_training.dat")
        biomarkers.extract(subfolder="patient" + str(patient_number) +  "_sd_" + str(sd_magnitude)+ "/wave1", anatomy_csv_file="input_anatomy_training.csv",
                           ep_dat_file="input_ep_training.dat")

        emulators_vector = emulators.train(folders=["initial_sweep","patient" + str(patient_number) +  "_sd_" + str(sd_magnitude)+ "/wave1"])
        history_matching.save_patient_implausibility(emulators_vector=emulators_vector, input_folder="patient" + str(patient_number) +  "_sd_" + str(sd_magnitude)+ "/wave1",
                                                     patient_number=patient_number, sd_magnitude=sd_magnitude)
        wave = history_matching.compute_nroy_region(emulators_vector=emulators_vector, implausibility_threshold=3.2,
                                                    literature_data=False, input_folder="patient" + str(patient_number) +  "_sd_" + str(sd_magnitude)+ "/wave1",
                                                    patient_number=patient_number, sd_magnitude=sd_magnitude,
                                                    previous_wave_name=os.path.join(PROJECT_PATH,"initial_sweep","wave0_patient" + str(patient_number) +  "_sd_" + str(sd_magnitude)))
        history_matching.plot_nroy(input_folder="patient" + str(patient_number) +  "_sd_" + str(sd_magnitude)+ "/wave1", wave=wave, literature_data=False,
                                   patient_number=patient_number, sd_magnitude=sd_magnitude, title = "Second wave for #" + str(patient_number) + " with SD=" + str(sd_magnitude) + "%")
        history_matching.generate_new_training_pts(wave=wave, num_pts=140, output_folder="patient" + str(patient_number) +  "_sd_" + str(sd_magnitude)+ "/wave2",
                                                   input_folder="patient" + str(patient_number) +  "_sd_" + str(sd_magnitude)+ "/wave1",
                                                   wave_name="wave1_patient" + str(patient_number) +  "_sd_" + str(sd_magnitude))

    if run_wave2:
        print("Running wave 2...")

        generate_meshes.sample_atlas(subfolder="patient" + str(patient_number) + "_sd_" + str(sd_magnitude) + "/wave2",
                                     csv_filename="input_anatomy_training.csv")
        preprocess_mesh.biv_setup(subfolder="patient" + str(patient_number) + "_sd_" + str(sd_magnitude) + "/wave2",
                                  anatomy_csv_file="input_anatomy_training.csv",
                                  ep_dat_file="input_ep_training.dat")
        ep_simulations.run(subfolder="patient" + str(patient_number) + "_sd_" + str(sd_magnitude) + "/wave2",
                           anatomy_csv_file="input_anatomy_training.csv",
                           ep_dat_file="input_ep_training.dat")
        biomarkers.extract(subfolder="patient" + str(patient_number) + "_sd_" + str(sd_magnitude) + "/wave2",
                           anatomy_csv_file="input_anatomy_training.csv",
                           ep_dat_file="input_ep_training.dat")

        emulators_vector = emulators.train(
            folders=["initial_sweep", "patient" + str(patient_number) + "_sd_" + str(sd_magnitude) + "/wave1","patient" + str(patient_number) + "_sd_" + str(sd_magnitude) + "/wave2"])
        history_matching.save_patient_implausibility(emulators_vector=emulators_vector,
                                                     input_folder="patient" + str(patient_number) + "_sd_" + str(
                                                        sd_magnitude) + "/wave2",
                                                     patient_number=patient_number, sd_magnitude=sd_magnitude)
        wave = history_matching.compute_nroy_region(emulators_vector=emulators_vector, implausibility_threshold=3.0,
                                                    literature_data=False,
                                                    input_folder="patient" + str(patient_number) + "_sd_" + str(
                                                        sd_magnitude) + "/wave2",
                                                    patient_number=patient_number, sd_magnitude=sd_magnitude,
                                                    previous_wave_name=os.path.join(PROJECT_PATH,"patient" + str(patient_number) +  "_sd_" + str(sd_magnitude)+ "/wave1","wave1_patient" + str(patient_number) +  "_sd_" + str(sd_magnitude)))
        history_matching.plot_nroy(input_folder="patient" + str(patient_number) + "_sd_" + str(sd_magnitude) + "/wave2",
                                   wave=wave, literature_data=False,
                                   patient_number=patient_number, sd_magnitude=sd_magnitude,
                                   title = "Third wave for #" + str(patient_number) + " with SD=" + str(sd_magnitude) + "%")
        history_matching.generate_new_training_pts(wave=wave, num_pts=140,
                                                   output_folder="patient" + str(patient_number) + "_sd_" + str(
                                                       sd_magnitude) + "/wave3",
                                                   input_folder="patient" + str(patient_number) + "_sd_" + str(
                                                       sd_magnitude) + "/wave2",
                                                   wave_name="wave2_patient" + str(patient_number) + "_sd_" + str(
                                                       sd_magnitude))

    if run_wave3:
        print("Running wave 3...")

        generate_meshes.sample_atlas(subfolder="patient" + str(patient_number) + "_sd_" + str(sd_magnitude) + "/wave3",
                                     csv_filename="input_anatomy_training.csv")
        preprocess_mesh.biv_setup(subfolder="patient" + str(patient_number) + "_sd_" + str(sd_magnitude) + "/wave3",
                                  anatomy_csv_file="input_anatomy_training.csv",
                                  ep_dat_file="input_ep_training.dat")
        ep_simulations.run(subfolder="patient" + str(patient_number) + "_sd_" + str(sd_magnitude) + "/wave3",
                           anatomy_csv_file="input_anatomy_training.csv",
                           ep_dat_file="input_ep_training.dat")
        biomarkers.extract(subfolder="patient" + str(patient_number) + "_sd_" + str(sd_magnitude) + "/wave3",
                           anatomy_csv_file="input_anatomy_training.csv",
                           ep_dat_file="input_ep_training.dat")

        emulators_vector = emulators.train(
            folders=["initial_sweep", "patient" + str(patient_number) + "_sd_" + str(sd_magnitude) + "/wave1",
                     "patient" + str(patient_number) + "_sd_" + str(sd_magnitude) + "/wave2",
                     "patient" + str(patient_number) + "_sd_" + str(sd_magnitude) + "/wave3"])
        history_matching.save_patient_implausibility(emulators_vector=emulators_vector,
                                                     input_folder="patient" + str(patient_number) + "_sd_" + str(
                                                        sd_magnitude) + "/wave3",
                                                     patient_number=patient_number, sd_magnitude=sd_magnitude)
        wave = history_matching.compute_nroy_region(emulators_vector=emulators_vector, implausibility_threshold=3.0,
                                                    literature_data=False,
                                                    input_folder="patient" + str(patient_number) + "_sd_" + str(
                                                        sd_magnitude) + "/wave3",
                                                    patient_number=patient_number, sd_magnitude=sd_magnitude,
                                                    previous_wave_name=os.path.join(PROJECT_PATH,
                                                                                    "patient" + str(patient_number) +  "_sd_" + str(sd_magnitude)+ "/wave2",
                                                                                    "wave2_patient" + str(patient_number) +  "_sd_" + str(sd_magnitude)))
        history_matching.plot_nroy(input_folder="patient" + str(patient_number) + "_sd_" + str(sd_magnitude) + "/wave3",
                                   wave=wave, literature_data=False,
                                   patient_number=patient_number, sd_magnitude=sd_magnitude,
                                   title = "Fourth wave for #" + str(patient_number) + " with SD=" + str(sd_magnitude) + "%")
        history_matching.generate_new_training_pts(wave=wave, num_pts=140,
                                                   output_folder="patient" + str(patient_number) + "_sd_" + str(
                                                       sd_magnitude) + "/wave4",
                                                   input_folder="patient" + str(patient_number) + "_sd_" + str(
                                                       sd_magnitude) + "/wave3",
                                                   wave_name="wave3_patient" + str(patient_number) + "_sd_" + str(
                                                       sd_magnitude))


def patient_convergence(patient_number, perc_convergence=95., fixed_sd=10):
    wave_number = 1
    converged = False

    while not converged:
        print("Running wave " + str(wave_number) + "...")
        subfolder_name = "patient" + str(patient_number) + "_sd_" + str(fixed_sd)

        if wave_number == 1:
            emulators_vector = first_wave(subfolder=subfolder_name + "/wave" + str(wave_number))
        else:

            generate_meshes.sample_atlas(subfolder=subfolder_name + "/wave" + str(wave_number),
                                         csv_filename="input_anatomy_training.csv")

            preprocess_mesh.biv_setup(subfolder=subfolder_name + "/wave" + str(wave_number),
                                      anatomy_csv_file="input_anatomy_training.csv",
                                      ep_dat_file="input_ep_training.dat")

            ep_simulations.run(subfolder=subfolder_name  + "/wave" + str(wave_number),
                               anatomy_csv_file="input_anatomy_training.csv",
                               ep_dat_file="input_ep_training.dat")

            biomarkers.extract(subfolder=subfolder_name + "/wave" + str(wave_number),
                               anatomy_csv_file="input_anatomy_training.csv",
                               ep_dat_file="input_ep_training.dat")

            emulators_folders = []
            for waves in range(wave_number):
                emulators_folders.append(subfolder_name + "/wave" + str(waves + 1))
            emulators_vector = emulators.train(folders=emulators_folders)

            history_matching.save_patient_implausibility(emulators_vector=emulators_vector,
                                                         input_folder=subfolder_name + "/wave" + str(wave_number),
                                                         patient_number=patient_number,
                                                         sd_magnitude=fixed_sd)

        if wave_number < 3:
            implausibility_threshold = 3.2
        else:
            implausibility_threshold = 3.

        if wave_number == 1:
            wave = history_matching.compute_nroy_region(emulators_vector=emulators_vector,
                                                        implausibility_threshold=implausibility_threshold,
                                                        literature_data=False,
                                                        input_folder=subfolder_name + "/wave" + str(wave_number),
                                                        first_time=True,
                                                        patient_number=patient_number, sd_magnitude=fixed_sd
                                                        )
        else:
            # WARNING: Probably will fail because of previous_wave_name
            wave = history_matching.compute_nroy_region(emulators_vector=emulators_vector,
                                                        implausibility_threshold=implausibility_threshold,
                                                        literature_data=False,
                                                        input_folder=subfolder_name + "/wave" + str(wave_number),
                                                        patient_number=patient_number, sd_magnitude=fixed_sd,
                                                        previous_wave_name=os.path.join(PROJECT_PATH,
                                                                                        subfolder_name + "/wave" +
                                                                                        str(wave_number - 1),
                                                                                        "wave" + str(wave_number - 1) +
                                                                                        "_" + subfolder_name))

        history_matching.plot_nroy(input_folder=subfolder_name + "/wave" + str(wave_number), wave=wave,
                                   literature_data=False,
                                   patient_number=patient_number, sd_magnitude=fixed_sd,
                                   title="Wave " + str(wave_number) + ", patient #" + str(patient_number) + ", SD=" + str(fixed_sd) + "%")
        np.savetxt(os.path.join(PROJECT_PATH, subfolder_name + "/wave" + str(wave_number),
                                "variance_quotient_wave" + str(wave_number) + "_" + subfolder_name + ".dat"), wave.PV,
                   fmt="%.2f")

        nroy_rel = np.genfromtxt(os.path.join(PROJECT_PATH, subfolder_name, "wave" + str(wave_number),
                                 "NROY_rel_" + subfolder_name + ".dat"), dtype=float)

        if (implausibility_threshold > 3.) or (nroy_rel < perc_convergence):
            history_matching.generate_new_training_pts(wave=wave, num_pts=140,
                                                       output_folder=subfolder_name + "/wave" + str(wave_number + 1),
                                                       input_folder=subfolder_name + "/wave" + str(wave_number),
                                                       wave_name="wave" + str(wave_number) + "_" + subfolder_name)
            wave_number += 1

        else:
            converged = True

def mix_patients(use_emulators_from_patient, new_patient, sd_magnitude, emulators_folders = None, last_wave="wave2"):
    """Function to use the emulators train for one patient but with the biomarkers of a different patient. Runs one
    wave for now.

    @param use_emulators_from_patient: Number of the patient whose simulations/emulators we are going to reuse.
    @param new_patient: Patient to test, we are going to use their biomarkers.
    @param sd_magnitude: Percentage of the mean of the biomarker to apply as standard deviation.
    """

    if emulators_folders is None:
        emulators_folders = ["initial_sweep",
                             "patient" + str(use_emulators_from_patient) + "_sd_" + str(sd_magnitude) + "/wave1",
                             "patient" + str(use_emulators_from_patient) + "_sd_" + str(sd_magnitude) + "/wave2"]
    if last_wave == "initial_sweep":
        previous_wave_name = None
        first_time = True
        implausibility_threshold = 3.2
    else:
        numbered_last_wave = last_wave
        previous_wave_name = os.path.join(PROJECT_PATH, "patient" + str(use_emulators_from_patient) + "_sd_" + str(
            sd_magnitude) + "/" + last_wave, numbered_last_wave + "_patient" + str(
            use_emulators_from_patient) + "_sd_" + str(
            sd_magnitude))
        first_time = False
        implausibility_threshold = 3.

    emulators_vector = emulators.train(folders=emulators_folders)

    if use_emulators_from_patient != new_patient:
        sampling_pts_lhd = 1e6
    else:
        sampling_pts_lhd = 0

    wave = history_matching.compute_nroy_region(emulators_vector=emulators_vector, implausibility_threshold=implausibility_threshold,
                                                literature_data=False,
                                                input_folder="using_patient" + str(use_emulators_from_patient) + "_sd_" + str(
                                                    sd_magnitude) + "/" + last_wave,
                                                patient_number=new_patient, sd_magnitude=sd_magnitude,
                                                previous_wave_name=previous_wave_name, sampling_pts_lhd=sampling_pts_lhd,
                                                first_time=first_time)
    history_matching.save_patient_implausibility(emulators_vector=emulators_vector,
                                                 input_folder="using_patient" + str(use_emulators_from_patient) + "_sd_" + str(
                                                    sd_magnitude) + "/" + last_wave,
                                                 patient_number=new_patient, sd_magnitude=sd_magnitude)
    history_matching.plot_nroy(input_folder="using_patient" + str(use_emulators_from_patient) + "_sd_" + str(
                                                    sd_magnitude) + "/" + last_wave,
                               wave=wave, literature_data=False,
                               patient_number=new_patient, sd_magnitude=sd_magnitude,
                               title = "Wave for #" + str(new_patient) + " using data from #" + str(use_emulators_from_patient) + "(SD=" + str(sd_magnitude) + "%)")
    wave.save(os.path.join(PROJECT_PATH, "using_patient" + str(use_emulators_from_patient) + "_sd_" + str(
                                                    sd_magnitude) + "/" + last_wave, last_wave + "_patient" +\
                           str(new_patient) + "_using_patient" + str(use_emulators_from_patient) +  "_sd_" + str(sd_magnitude)))
    history_matching.generate_new_training_pts(wave=wave, num_pts=0,
                                               output_folder="using_patient" + str(use_emulators_from_patient) + "_sd_" + str(
                                                    sd_magnitude) + "/wavex",
                                               input_folder="using_patient" + str(use_emulators_from_patient) + "_sd_" + str(
                                                    sd_magnitude) + "/" + last_wave,
                                               wave_name=last_wave + "_patient" + str(new_patient) + "_using_patient" +\
                                                         str(use_emulators_from_patient) +  "_sd_" + str(sd_magnitude))

def run_farthest_patients(patient_number,input_or_output="input"):

    distance_matrix = np.loadtxt(os.path.join(PROJECT_PATH,"CT_patients_distance_l1_" + input_or_output + ".dat"))

    farthest_patient = np.argmax(distance_matrix[patient_number-1]) + 1

    print("Running patient " + str(farthest_patient))

    # patient(patient_number=patient_number, run_wave0=True, run_wave1=True, run_wave2=True, sd_magnitude=10)
    patient(patient_number=farthest_patient, run_wave0=True, run_wave1=True, run_wave2=True, sd_magnitude=10)

    mix_patients(use_emulators_from_patient=patient_number, new_patient=farthest_patient, sd_magnitude=10)