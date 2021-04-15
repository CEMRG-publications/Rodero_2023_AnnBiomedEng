import skopt
import os
import numpy as np

import prepare_mesh
import fibres
import UVC
import run_EP

def input_EP_param(n_samples = None):

    seed = 2

    param_names = ["mode_" + str(mode) for mode in range(1,19)]

    param_ranges = [
        (-64.624573989, 64.624573989),
        (-49.3066597463, 49.3066597463),
        (-41.573252807, 41.573252807),
        (-30.4268528225, 30.4268528225),
        (-25.7218779298, 25.7218779298),
        (-22.304645526, 22.304645526),
        (-21.5206199592, 21.5206199592),
        (-17.351659107, 17.351659107),
        (-17.0448821118, 17.0448821118),
        (-15.2509822608, 15.2509822608),
        (-14.4696049643, 14.4696049643),
        (-13.8879578509, 13.8879578509),
        (-12.6375573967, 12.6375573967),
        (-11.2271563098, 11.2271563098),
        (-11.0001216799, 11.0001216799),
        (-10.0484889853, 10.0484889853),
        (-9.2438333607, 9.2438333607),
        (-8.7797037167, 8.7797037167)
    ]

    param_names += [
        "alpha_endo",
        "alpha_epi",
        "lastFECtag",
        "CV_l",
        "k_fibre",
        "k_FEC"
    ]

    param_ranges += [
        (40.,90.), # degrees
        (-90., -40.), # degrees
        (26, 38),
        (0.01, 2.), # m/s
        (0.01, 1.), 
        (1., 10.)
    ]

    if(n_samples is None):
        n_samples = 10*len(param_ranges)

    space = skopt.space.Space(param_ranges)

    sobol = skopt.sampler.Sobol(min_skip = seed, max_skip = seed)
    x = sobol.generate(space.dimensions, n_samples, random_state = seed)

    f = open(os.path.join("/data", "fitting", "EP_param_labels.txt"), "w")
    [f.write('%s\n' % key) for key in param_names]
    f.close()

    f = open(os.path.join("/data", "fitting", "EP_param_values.txt"), "w")
    [f.write('%s\n' % ' '.join(map(str, lhs_array))) for lhs_array in x]
    f.close()

def synthetic_mesh(mesh_ID):

    with open("/data/fitting/EP_param_labels.txt") as f:
        param_names = f.read().splitlines()

    with open("/data/fitting/EP_param_values.txt") as f:
        param_values = f.read().splitlines()

    modes_idx = np.where([x[0:4] == "mode" for x in param_names])[0]

    line2extract = param_values[mesh_ID].split(' ')

    modes_weights = [float(w) for w in np.take(line2extract, modes_idx)]

#def param_hm()

def EP_pipeline(mesh_ID):

    with open("/data/fitting/EP_param_labels.txt") as f:
        param_names = f.read().splitlines()

    with open("/data/fitting/EP_param_values.txt") as f:
        param_values = f.read().splitlines()

    alpha_epi_idx = int(np.where([x == "alpha_epi" for x in param_names])[0])
    alpha_endo_idx = int(np.where([x == "alpha_endo" for x in param_names])[0])
    lastFECtag_idx = int(np.where([x == "lastFECtag" for x in param_names])[0])
    CV_l_idx = int(np.where([x == "CV_l" for x in param_names])[0])
    k_fibre_idx = int(np.where([x == "k_fibre" for x in param_names])[0])
    k_FEC_idx = int(np.where([x == "k_FEC" for x in param_names])[0])

    line2extract = param_values[mesh_ID].split(' ')

    alpha_epi = float(line2extract[alpha_epi_idx])
    alpha_endo = float(line2extract[alpha_endo_idx])
    lastFECtag = int(line2extract[lastFECtag_idx])
    CV_l = float(line2extract[CV_l_idx])
    k_fibre = float(line2extract[k_fibre_idx])
    k_FEC = float(line2extract[k_FEC_idx])


    # prepare_mesh.extract_LDRB_biv(mesh_ID)
    # prepare_mesh.extract_MVTV_base(mesh_ID)
    # fibres.run_laplacian(mesh_ID)
    # fibres.full_pipeline(mesh_ID, alpha_epi, alpha_endo)
    # UVC.create(1, "MVTV")
    # UVC.bottom_third(1, "MVTV")
    # UVC.create_FEC(1, "MVTV")
    run_EP.carp2init(mesh_ID, lastFECtag, CV_l, k_fibre, k_FEC)
    run_EP.launch_init(mesh_ID, alpha_endo, alpha_epi)