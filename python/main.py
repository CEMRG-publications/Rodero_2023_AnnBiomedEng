#!/usr/bin/env python3

import fitting_hm

if __name__ == "__main__":
    fitting_hm.mechanics_new_wave(num_wave=0, run_simulations=True, train_gpe=False, fill_wave_space=False, cutoff=0,
                                  n_samples=1, generate_simul_pts=-1, subfolder="mechanics", training_set_memory=2,
                                  only_feasible=False
                                  )
