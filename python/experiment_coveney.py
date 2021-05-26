import generate
import fitting_hm

if __name__ == "__main__":

    generate.EP_funct_param(n_samples = round(50/(0.8*0.8)), waveno = 0, subfolder = "coveney")
    generate.template_EP_parallel(line_from = 0, line_to = round(50/(0.8*0.8)) - 1, waveno = 0, subfolder = "coveney")
    generate.EP_output(waveno = 0, subfolder = "coveney")
    generate.filter_output(waveno = 0, subfolder = "coveney")
    fitting_hm.run_new_wave(num_wave = 0, run_simulations = False, train_GPE = False,
                fill_wave_space = True, cutoff = 4, n_samples = 50,
                generate_simul_pts = 50, subfolder = "coveney", training_set_memory = 4)
    fitting_hm.run_new_wave(num_wave = 1, run_simulations = True, train_GPE = False,
                fill_wave_space = True, cutoff = 4, n_samples = 50,
                generate_simul_pts = 50, subfolder = "coveney", training_set_memory = 4)
    fitting_hm.run_new_wave(num_wave = 2, run_simulations = True, train_GPE = False,
                fill_wave_space = True, cutoff = 3.8, n_samples = 50,
                generate_simul_pts = 50, subfolder = "coveney", training_set_memory = 4)
    fitting_hm.run_new_wave(num_wave = 3, run_simulations = True, train_GPE = False,
                fill_wave_space = True, cutoff = 3.6, n_samples = 50,
                generate_simul_pts = 50, subfolder = "coveney", training_set_memory = 4)
    fitting_hm.run_new_wave(num_wave = 4, run_simulations = True, train_GPE = False,
                fill_wave_space = True, cutoff = 3.4, n_samples = 50,
                generate_simul_pts = 50, subfolder = "coveney", training_set_memory = 4)
    fitting_hm.run_new_wave(num_wave = 5, run_simulations = True, train_GPE = False,
                fill_wave_space = True, cutoff = 3.2, n_samples = 50,
                generate_simul_pts = 50, subfolder = "coveney", training_set_memory = 4)
    fitting_hm.run_new_wave(num_wave = 6, run_simulations = True, train_GPE = False,
                fill_wave_space = True, cutoff = 3, n_samples = 50,
                generate_simul_pts = 50, subfolder = "coveney", training_set_memory = 4)
    fitting_hm.run_new_wave(num_wave = 7, run_simulations = True, train_GPE = False,
                fill_wave_space = True, cutoff = 3, n_samples = 50,
                generate_simul_pts = 50, subfolder = "coveney", training_set_memory = 4)
    fitting_hm.run_new_wave(num_wave = 8, run_simulations = True, train_GPE = False,
                fill_wave_space = True, cutoff = 3, n_samples = 50,
                generate_simul_pts = 50, subfolder = "coveney", training_set_memory = 4)
    fitting_hm.run_new_wave(num_wave = 9, run_simulations = True, train_GPE = False,
                fill_wave_space = True, cutoff = 3, n_samples = 50,
                generate_simul_pts = 50, subfolder = "coveney", training_set_memory = 4)