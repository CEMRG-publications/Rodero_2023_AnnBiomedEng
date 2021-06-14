import numpy as np
import os
import sys

import custom_plots
import fitting_hm

def summary_plots(wave_to_plot, experiment_name):
    custom_plots.plot_var_quotient(first_wave = 1, last_wave = wave_to_plot,
                                    subfolder = experiment_name,
                                    plot_title = "Evolution of variance quotient for " + experiment_name)
    custom_plots.plot_output_evolution_seaborn(first_wave = 0, last_wave = wave_to_plot,
                                            subfolder = experiment_name)
    custom_plots.plot_percentages_NROY(subfolder = experiment_name, last_wave = wave_to_plot)

def summary_statistics(last_wave,experiment_name):

    mean_var_quotient_vec = []
    NROY_rel = []
    NROY_abs = []
    for i in range(last_wave + 1):
        W_PV = np.loadtxt(os.path.join("/data","fitting",experiment_name,"wave" + str(i),"variance_quotient.dat"),
                        dtype=float)

        mean_var_quot = np.mean(W_PV)

        mean_var_quotient_vec.append(mean_var_quot)
    
        NROY_perc = np.loadtxt(os.path.join("/data","fitting",experiment_name,"wave" + str(i), "NROY_rel.dat"),dtype=float)
        NROY_rel.append(float(NROY_perc))
        if i == 0:
            NROY_abs.append(float(NROY_perc))
        else:
            NROY_abs.append(1e-2*float(NROY_perc)*NROY_abs[i-1])
    
    min_var_wave = mean_var_quotient_vec.index(min(mean_var_quotient_vec))
    print("Min. variance: " + str(round(mean_var_quotient_vec[min_var_wave],2)))
    print("Wave with min. var.: " + str(min_var_wave))
    print("NROY size: " + str(round(NROY_abs[min_var_wave],2)))
    

def experiment_coveney(only_plot = True):
    
    experiment_name = "coveney"
    original_training_set_size = 50

    if only_plot:
        summary_plots(wave_to_plot = 9, experiment_name = experiment_name)
    else:

        wave_to_plot = -1

        fitting_hm.run_new_wave(num_wave = 0, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 4, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 4)
        wave_to_plot += 1

        fitting_hm.run_new_wave(num_wave = 1, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 4, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 4)
        wave_to_plot += 1

        fitting_hm.run_new_wave(num_wave = 2, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.8, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 4)
        wave_to_plot += 1

        fitting_hm.run_new_wave(num_wave = 3, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.6, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 4)
        wave_to_plot += 1

        fitting_hm.run_new_wave(num_wave = 4, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.4, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 4)
        wave_to_plot += 1

        fitting_hm.run_new_wave(num_wave = 5, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.2, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 4)
        wave_to_plot += 1

        fitting_hm.run_new_wave(num_wave = 6, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 4)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)
        
        fitting_hm.run_new_wave(num_wave = 7, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 4)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 8, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 4)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 9, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 4)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

def experiment_longobardi(only_plot = True):
    
    experiment_name = "longobardi"
    original_training_set_size = 150

    if only_plot:
        summary_plots(wave_to_plot = 5, experiment_name = experiment_name)
    else:

        wave_to_plot = -1

        fitting_hm.run_new_wave(num_wave = 0, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 5.5, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 40)
        wave_to_plot += 1

        fitting_hm.run_new_wave(num_wave = 1, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 5, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 40)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 2, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 4.5, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 40)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 3, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 4, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 40)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 4, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.5, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 40)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 5, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3., n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 440)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

def experiment_3(only_plot = True):
    
    original_training_set_size = 100
    experiment_name = "experiment_3"
    wave_to_plot = -1


    if only_plot:
        summary_plots(wave_to_plot = 6, experiment_name = experiment_name)
    else:

        fitting_hm.run_new_wave(num_wave = 0, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 4, n_samples = original_training_set_size,
                    generate_simul_pts = 10, subfolder = experiment_name, training_set_memory = 40)
        wave_to_plot += 1

        fitting_hm.run_new_wave(num_wave = 1, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.8, n_samples = original_training_set_size,
                    generate_simul_pts = 10, subfolder = experiment_name, training_set_memory = 40)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 2, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.6, n_samples = original_training_set_size,
                    generate_simul_pts = 10, subfolder = experiment_name, training_set_memory = 40)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 3, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.4, n_samples = original_training_set_size,
                    generate_simul_pts = 10, subfolder = experiment_name, training_set_memory = 40)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 4, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.2, n_samples = original_training_set_size,
                    generate_simul_pts = 10, subfolder = experiment_name, training_set_memory = 40)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 5, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3., n_samples = original_training_set_size,
                    generate_simul_pts = 10, subfolder = experiment_name, training_set_memory = 440)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 6, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3., n_samples = original_training_set_size,
                    generate_simul_pts = 10, subfolder = experiment_name, training_set_memory = 440)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

def experiment_4(only_plot = True):
    # Accurate initial emulator, slow descent (memory 2)

    original_training_set_size = 100
    experiment_name = "experiment_4"
    wave_to_plot = -1

    if only_plot:
        summary_plots(wave_to_plot = 6, experiment_name = experiment_name)
    else:

        fitting_hm.run_new_wave(num_wave = 0, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 4, n_samples = original_training_set_size,
                    generate_simul_pts = 20, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1

        fitting_hm.run_new_wave(num_wave = 1, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.8, n_samples = original_training_set_size,
                    generate_simul_pts = 20, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 2, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.6, n_samples = original_training_set_size,
                    generate_simul_pts = 20, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 3, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.4, n_samples = original_training_set_size,
                    generate_simul_pts = 20, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 4, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.2, n_samples = original_training_set_size,
                    generate_simul_pts = 20, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 5, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3., n_samples = original_training_set_size,
                    generate_simul_pts = 20, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 6, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3., n_samples = original_training_set_size,
                    generate_simul_pts = 20, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

def experiment_5(only_plot = True):
    # Balanced emulator, initial and pre-final convergence (memory 2)

    original_training_set_size = 50
    experiment_name = "experiment_5"
    wave_to_plot = -1

    if only_plot:
        summary_plots(wave_to_plot = 9, experiment_name = experiment_name)
    else:

        fitting_hm.run_new_wave(num_wave = 0, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 4, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1

        fitting_hm.run_new_wave(num_wave = 1, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 4, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 2, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 4, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 3, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.8, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 4, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.6, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 5, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.4, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 6, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.2, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 7, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.2, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 8, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.2, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 9, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3., n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

def experiment_6(only_plot = True):
    # Cut to the chase (total memory)

    original_training_set_size = 100
    experiment_name = "experiment_6"
    wave_to_plot = -1

    if only_plot:
        summary_plots(wave_to_plot = 5, experiment_name = experiment_name)
    else:

        fitting_hm.run_new_wave(num_wave = 0, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.2, n_samples = original_training_set_size,
                    generate_simul_pts = 20, subfolder = experiment_name, training_set_memory = 20)
        wave_to_plot += 1

        fitting_hm.run_new_wave(num_wave = 1, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.2, n_samples = original_training_set_size,
                    generate_simul_pts = 20, subfolder = experiment_name, training_set_memory = 20)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 2, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.2, n_samples = original_training_set_size,
                    generate_simul_pts = 20, subfolder = experiment_name, training_set_memory = 20)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 3, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3., n_samples = original_training_set_size,
                    generate_simul_pts = 20, subfolder = experiment_name, training_set_memory = 20)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 4, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3., n_samples = original_training_set_size,
                    generate_simul_pts = 20, subfolder = experiment_name, training_set_memory = 20)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 5, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3., n_samples = original_training_set_size,
                    generate_simul_pts = 20, subfolder = experiment_name, training_set_memory = 20)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

def experiment_7(only_plot = False):
     # Cut to the chase (memory 2)

    original_training_set_size = 100
    experiment_name = "experiment_7"
    wave_to_plot = -1

    if only_plot:
        summary_plots(wave_to_plot = 5, experiment_name = experiment_name)
    else:

        fitting_hm.run_new_wave(num_wave = 0, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.2, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1

        fitting_hm.run_new_wave(num_wave = 1, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.2, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 2, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3.2, n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 3, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3., n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 4, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3., n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

        fitting_hm.run_new_wave(num_wave = 5, run_simulations = True, train_GPE = True,
                    fill_wave_space = True, cutoff = 3., n_samples = original_training_set_size,
                    generate_simul_pts = 50, subfolder = experiment_name, training_set_memory = 2)
        wave_to_plot += 1
        summary_plots(wave_to_plot = wave_to_plot, experiment_name = experiment_name)

def run_experiment(experiment_name):
    experiment_name = str(experiment_name)

    if experiment_name == "1" or experiment_name == "coveney":
        experiment_coveney()
    elif experiment_name == "2" or experiment_name == "longobardi":
        experiment_longobardi()
    elif experiment_name == "3":
        experiment_3()
    elif experiment_name == "4":
        experiment_4()
    elif experiment_name == "5":
        experiment_5()
    elif experiment_name == "6":
        experiment_6()
    elif experiment_name == "7":
        experiment_7()
    elif experiment_name == "all":
        for i in range(1,8):
            run_experiment(i)
    
    

if __name__ == "__main__":
    run_experiment(sys.argv[1])
