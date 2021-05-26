import generate
import fitting_hm

if __name__ == "__main__":

    # Accurate initial emulator, slow descent (memory 2)

    original_training_set_size = 100
    experiment_name = "experiment_4"

    generate.EP_funct_param(n_samples = round(original_training_set_size/(0.8*0.8)), waveno = 0, subfolder = experiment_name)
    generate.template_EP_parallel(line_from = 0, line_to = round(original_training_set_size/(0.8*0.8)) - 1, waveno = 0, subfolder = experiment_name)
    generate.EP_output(waveno = 0, subfolder = experiment_name)
    generate.filter_output(waveno = 0, subfolder = experiment_name)
    fitting_hm.run_new_wave(num_wave = 0, run_simulations = False, train_GPE = True,
                fill_wave_space = True, cutoff = 4, n_samples = original_training_set_size,
                generate_simul_pts = 20, subfolder = experiment_name, training_set_memory = 2)
    fitting_hm.run_new_wave(num_wave = 1, run_simulations = True, train_GPE = False,
                fill_wave_space = True, cutoff = 3.8, n_samples = original_training_set_size,
                generate_simul_pts = 20, subfolder = experiment_name, training_set_memory = 2)
    fitting_hm.run_new_wave(num_wave = 2, run_simulations = True, train_GPE = False,
                fill_wave_space = True, cutoff = 3.6, n_samples = original_training_set_size,
                generate_simul_pts = 20, subfolder = experiment_name, training_set_memory = 2)
    fitting_hm.run_new_wave(num_wave = 3, run_simulations = True, train_GPE = False,
                fill_wave_space = True, cutoff = 3.4, n_samples = original_training_set_size,
                generate_simul_pts = 20, subfolder = experiment_name, training_set_memory = 2)
    fitting_hm.run_new_wave(num_wave = 4, run_simulations = True, train_GPE = False,
                fill_wave_space = True, cutoff = 3.2, n_samples = original_training_set_size,
                generate_simul_pts = 20, subfolder = experiment_name, training_set_memory = 2)
    fitting_hm.run_new_wave(num_wave = 5, run_simulations = True, train_GPE = False,
                fill_wave_space = True, cutoff = 3., n_samples = original_training_set_size,
                generate_simul_pts = 20, subfolder = experiment_name, training_set_memory = 2)
    fitting_hm.run_new_wave(num_wave = 6, run_simulations = True, train_GPE = False,
                fill_wave_space = True, cutoff = 3., n_samples = original_training_set_size,
                generate_simul_pts = 20, subfolder = experiment_name, training_set_memory = 2)