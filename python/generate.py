from os import sep
import skopt
import os
import numpy as np
from joblib import Parallel, delayed
import tqdm
import pathlib

import prepare_mesh
import fibres
import UVC
import run_EP
import files_manipulations
from Historia.shared.design_utils import read_labels

SEED = 2

def input_EP_param(n_samples = None):

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

    sobol = skopt.sampler.Sobol(min_skip = SEED, max_skip = SEED)
    x = sobol.generate(space.dimensions, n_samples, random_state = SEED)

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

def template_mesh_setup():
    prepare_mesh.extract_LDRB_biv("Template")
    prepare_mesh.extract_MVTV_base("Template")
    UVC.create("Full_Heart_Mesh_Template", "MVTV")
    UVC.bottom_third("Template", "MVTV")
    UVC.create_FEC("Template", "MVTV")
    fibres.run_laplacian("Template")

def EP_funct_param(n_samples = None, waveno = 0, subfolder = "."):

    path_lab = os.path.join("/data","fitting")
    path_gpes = os.path.join(path_lab, subfolder, "wave" + str(waveno))

    pathlib.Path(path_gpes).mkdir(parents = True, exist_ok = True)

    param_names = [
        "alpha",
        "FEC_height",
        "CV_l",
        "k_fibre",
        "k_FEC"
    ]

    param_ranges = [
        (40.,90.), # degrees
        (33, 100), # %
        (0.01, 2.), # m/s
        (0.01, 1.), 
        (1., 10.)
    ]

    if(n_samples is None):
        n_samples = 10*len(param_ranges)

    space = skopt.space.Space(param_ranges)

    sobol = skopt.sampler.Sobol(min_skip = SEED, max_skip = SEED)
    x = sobol.generate(space.dimensions, n_samples, random_state = SEED)

    f = open(os.path.join("/data", "fitting", "EP_funct_labels.txt"), "w")
    [f.write('%s\n' % key) for key in param_names]
    f.close()

    f = open(os.path.join(path_gpes, "X.dat"), "w")
    [f.write('%s\n' % ' '.join(map(str,[format(i, '.2f') for i in lhs_array]))) for lhs_array in x]
    f.close()

def template_EP(param_line):

    with open("/data/fitting/EP_funct_labels.txt") as f:
        param_names = f.read().splitlines()

    with open("/data/fitting/EP_funct_values.txt") as f:
        param_values = f.read().splitlines()

    alpha_idx = int(np.where([x == "alpha" for x in param_names])[0])
    lastFECtag_idx = int(np.where([x == "lastFECtag" for x in param_names])[0])
    CV_l_idx = int(np.where([x == "CV_l" for x in param_names])[0])
    k_fibre_idx = int(np.where([x == "k_fibre" for x in param_names])[0])
    k_FEC_idx = int(np.where([x == "k_FEC" for x in param_names])[0])

    line2extract = param_values[param_line].split(' ')

    alpha_endo = float(line2extract[alpha_idx])
    alpha_epi = -alpha_endo
    lastFECtag = int(line2extract[lastFECtag_idx])
    CV_l = float(line2extract[CV_l_idx])
    k_fibre = float(line2extract[k_fibre_idx])
    k_FEC = float(line2extract[k_FEC_idx])

    fibres.full_pipeline("Template", alpha_epi, alpha_endo)
    run_EP.carp2init("Template", lastFECtag, CV_l, k_fibre, k_FEC, suffix = param_line)
    run_EP.launch_init("Template", alpha_endo, alpha_epi, suffix = param_line)

def template_EP_parallel(line_from = 0, line_to = 10, waveno = 0, subfolder = "."):

    path_lab = os.path.join("/data","fitting")
    path_gpes = os.path.join(path_lab, subfolder, "wave" + str(waveno))
    path_EP = os.path.join("/data","fitting","Full_Heart_Mesh_Template","biv",
                            "EP_simulations")

    pathlib.Path(path_EP).mkdir(parents = True, exist_ok = True)

    def template_EP_explicit_arguments(alpha_endo_vec, lastFECtag_vec, CV_l_vec,
                                       k_fibre_vec, k_FEC_vec, param_line,
                                       path_EP,FEC_height_vec):

        simulation_file_name = os.path.join(path_EP,
                                str(format(alpha_endo_vec[param_line],'.2f')) +\
                                str(format(FEC_height_vec[param_line],'.2f')) +\
                                str(format(CV_l_vec[param_line],'.2f')) +\
                                str(format(k_fibre_vec[param_line],'.2f')) +\
                                str(format(k_FEC_vec[param_line],'.2f')) +\
                                ".dat"
                                )
        
        if not os.path.isfile(simulation_file_name):

            fibres.full_pipeline("Template",
                                alpha_epi = -alpha_endo_vec[param_line],
                                alpha_endo = alpha_endo_vec[param_line]
                                )

            run_EP.carp2init("Template",
                            lastFECtag_vec[param_line],
                            CV_l_vec[param_line],
                            k_fibre_vec[param_line],
                            k_FEC_vec[param_line],
                            simulation_file_name = simulation_file_name,
                            path_EP = path_EP
                            )

            run_EP.launch_init("Template", 
                            alpha_endo = alpha_endo_vec[param_line], 
                            alpha_epi = -alpha_endo_vec[param_line], 
                            simulation_file_name = simulation_file_name,
                            path_EP = path_EP
                            )

    with open(os.path.join(path_lab,"EP_funct_labels.txt")) as f:
        param_names = f.read().splitlines()

    with open(os.path.join(path_gpes,"X.dat")) as f:
        param_values = f.read().splitlines()

    line_to = min(line_to, len(param_values)-1)

    alpha_idx = int(np.where([x == "alpha" for x in param_names])[0])
    FEC_height_idx = int(np.where([x == "FEC_height" for x in param_names])[0])
    CV_l_idx = int(np.where([x == "CV_l" for x in param_names])[0])
    k_fibre_idx = int(np.where([x == "k_fibre" for x in param_names])[0])
    k_FEC_idx = int(np.where([x == "k_FEC" for x in param_names])[0])

    alpha_endo_vec = [float(line.split(' ')[alpha_idx]) for line in param_values]
    FEC_height_vec = [float(line.split(' ')[FEC_height_idx]) for line in param_values]
    CV_l_vec = [float(line.split(' ')[CV_l_idx]) for line in param_values]
    k_fibre_vec = [float(line.split(' ')[k_fibre_idx]) for line in param_values]
    k_FEC_vec = [float(line.split(' ')[k_FEC_idx]) for line in param_values]

    def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return array[idx]
        
    FEC_height_to_lastFECtag = {33:25,
                                35:26,
                                40:27,
                                45:28,
                                50:29,
                                55:30,
                                60:31,
                                65:32,
                                70:33,
                                75:34,
                                80:35,
                                85:36,
                                90:37,
                                95:38,
                                100:39}

    lastFECtag_vec = []
    for height in FEC_height_vec:
        height_key = find_nearest(list(FEC_height_to_lastFECtag.keys()),round(float(height)))
        lastFECtag = FEC_height_to_lastFECtag[height_key]
        lastFECtag_vec = np.append(lastFECtag_vec, lastFECtag)

    for param_line in tqdm.tqdm(range(line_from,line_to+1)):

        template_EP_explicit_arguments(alpha_endo_vec,
                                        lastFECtag_vec, 
                                        CV_l_vec, 
                                        k_fibre_vec, 
                                        k_FEC_vec, 
                                        param_line,
                                        path_EP,
                                        FEC_height_vec)
    return line_to + 1

def EP_output(waveno = 0, subfolder = "."):

    EP_dir = os.path.join("/data","fitting","Full_Heart_Mesh_Template","biv", 
                          "EP_simulations")
    labels_dir = os.path.join("/data","fitting")
    path2UVC = os.path.join("/data","fitting","Full_Heart_Mesh_Template","biv",
                            "UVC_MVTV","UVC")

    outpath = os.path.join("/data", "fitting",subfolder, "wave" + str(waveno))

    output_names = ["TAT","TATLV"]

    f = open(os.path.join(labels_dir,"EP_output_labels.txt"), "w")
    [f.write("%s\n" % phenotype) for phenotype in output_names]
    f.close()

    lvendo_vtx = files_manipulations.vtx.read(os.path.join("/data","fitting",
                                                      "Full_Heart_Mesh_Template",
                                                      "biv","biv.lvendo.surf.vtx"),
                                        "biv")

    UVC_Z = np.genfromtxt(os.path.join(path2UVC, "COORDS_Z.dat"),dtype = float)

    Z_90 = np.where(UVC_Z < 0.9)[0]

    Z_90_endo_mask = np.in1d(lvendo_vtx.indices, Z_90)
    Z_90_endo = lvendo_vtx.indices[Z_90_endo_mask]

    with open(os.path.join(outpath,"X.dat")) as f:
        param_values = f.read().splitlines()
    
    X_names = [line.replace(" ","") for line in param_values]

    TAT = [float("nan")]*len(X_names)
    TATLV = [float("nan")]*len(X_names)

    for i in tqdm.tqdm(range(len(X_names))):
        with open(os.path.join(EP_dir, X_names[i] + ".dat")) as g:
            AT_vec = g.read().splitlines()
        g.close()
        # AT_vec_float = []
        # for x in AT_vec:
        #     if i > 28:
        #         print(x)
        #         print(X_names[i])
        #     AT_vec_float.append(float(x))
        AT_vec_float = [float(x) for x in AT_vec]
        TAT[i] = max(AT_vec_float)
        TATLV[i] = max(np.array(AT_vec_float)[Z_90_endo.astype(int)])

    output_numbers = [TAT, TATLV]

    for i,varname in enumerate(output_names):
        np.savetxt(os.path.join(outpath, varname + ".dat"),
                    output_numbers[i],
                    fmt="%.2f")

def anatomical_output(heart_name):
    
    biv_path = os.path.join("/data","fitting",heart_name,"biv")
    path2UVC = os.path.join(biv_path, "UVC_MVTV", "UVC")

    septum_idx = files_manipulations.vtx.read(os.path.join(biv_path,"biv.rvsept.surf.vtx"),
                                            "biv")
    lvendo_idx = files_manipulations.vtx.read(os.path.join(biv_path,"biv.lvendo.surf.vtx"),
                                            "biv")
    bivepi_idx = files_manipulations.vtx.read(os.path.join(biv_path,"biv.epi.surf.vtx"),
                                            "biv")

    biv_elem = files_manipulations.elem.read(os.path.join(biv_path,"biv.elem"))

    UVC_Z = np.genfromtxt(os.path.join(path2UVC, "COORDS_Z.dat"),dtype = float)
    UVC_PHI = np.genfromtxt(os.path.join(path2UVC, "COORDS_PHI.dat"),dtype = float)

    septum_Z = UVC_Z[septum_idx.indices]
    lvendo_Z = UVC_Z[lvendo_idx.indices]
    bivepi_Z = UVC_Z[bivepi_idx.indices]
    
    septum_band_idx = septum_idx.indices[np.intersect1d(np.where(septum_Z > 0.6),
                                                np.where(septum_Z < 0.9)
                                                )
                                ]
    lvendo_band_idx = lvendo_idx.indices[np.intersect1d(np.where(lvendo_Z > 0.6),
                                                np.where(lvendo_Z < 0.9)
                                                )
                                ]
    bivepi_band_idx = bivepi_idx.indices[np.intersect1d(np.where(bivepi_Z > 0.6),
                                                np.where(bivepi_Z < 0.9)
                                                )
                                ]

    septum_band_PHI = UVC_PHI[septum_band_idx]
    lvendo_band_PHI = UVC_PHI[lvendo_band_idx]
    bivepi_band_PHI = UVC_PHI[bivepi_band_idx]

    # Now we get the most centred third

    bandwith_PHI = max(septum_band_PHI) - min(septum_band_PHI)
    midpoint_PHI = (max(septum_band_PHI) + min(septum_band_PHI))/2.
    min_PHI = midpoint_PHI-bandwith_PHI/6.
    max_PHI = midpoint_PHI+bandwith_PHI/6.

    # Wall thickness as the opposite side of the septum
    if min_PHI <= 0:
        wall_min_PHI = min_PHI + np.pi
    else:
        wall_min_PHI = min_PHI - np.pi
    
    
    if max_PHI <= 0:
        wall_max_PHI = max_PHI + np.pi
    else:
        wall_max_PHI = max_PHI - np.pi

    septum_ROI_idx = septum_band_idx[np.intersect1d(np.where(septum_band_PHI > min_PHI),
                                                np.where(septum_band_PHI < max_PHI)
                                                )
                                ]
    lvendo_septum_ROI_idx = lvendo_band_idx[np.intersect1d(np.where(lvendo_band_PHI > min_PHI),
                                                np.where(lvendo_band_PHI < max_PHI)
                                                )
                                ]
    lvendo_wall_ROI_idx = lvendo_band_idx[np.intersect1d(np.where(lvendo_band_PHI > wall_min_PHI),
                                                np.where(lvendo_band_PHI < wall_max_PHI)
                                                )
                                ]
    lvepi_ROI_idx = bivepi_band_idx[np.intersect1d(np.where(bivepi_band_PHI > wall_min_PHI),
                                                np.where(bivepi_band_PHI < wall_max_PHI)
                                                )
                                ]

    biv_pts = files_manipulations.pts.read(os.path.join(biv_path,"biv.pts"))

    septum_ROI_pts = biv_pts.extract(files_manipulations.vtx(septum_ROI_idx,"biv"))
    lvendo_septum_ROI_pts = biv_pts.extract(files_manipulations.vtx(lvendo_septum_ROI_idx,"biv"))
    lvendo_wall_ROI_pts = biv_pts.extract(files_manipulations.vtx(lvendo_wall_ROI_idx,"biv"))
    lvepi_ROI_pts = biv_pts.extract(files_manipulations.vtx(lvepi_ROI_idx,"biv"))

    septum_ROI_thickness = []
    wall_ROI_thickness = []
    ED_dimension_ROI = []

    def dist_from_septum_to_lvendo_septum(i):
        return lvendo_septum_ROI_pts.min_dist(np.array([septum_ROI_pts.p1[i], septum_ROI_pts.p2[i], septum_ROI_pts.p3[i]]))
    def dist_from_lvepi_to_lvendo_wall(i):
        return lvendo_wall_ROI_pts.min_dist(np.array([lvepi_ROI_pts.p1[i], lvepi_ROI_pts.p2[i], lvepi_ROI_pts.p3[i]]))
    def dist_from_lvendo_septum_to_lvendo_wall(i):
        return lvendo_wall_ROI_pts.min_dist(np.array([lvendo_septum_ROI_pts.p1[i], lvendo_septum_ROI_pts.p2[i], lvendo_septum_ROI_pts.p3[i]]))

    septum_ROI_thickness.append(Parallel(n_jobs=20)(delayed(dist_from_septum_to_lvendo_septum)(i) for i in range(septum_ROI_pts.size)))
    wall_ROI_thickness.append(Parallel(n_jobs=20)(delayed(dist_from_lvepi_to_lvendo_wall)(i) for i in range(lvepi_ROI_pts.size)))
    ED_dimension_ROI.append(Parallel(n_jobs=20)(delayed(dist_from_lvendo_septum_to_lvendo_wall)(i) for i in range(lvendo_septum_ROI_pts.size)))

    # Itraventricular septal wall thickness, in mm
    ISWT = 1e-3*np.median(septum_ROI_thickness)
    # Wall thickness, in mm
    WT = 1e-3*np.median(wall_ROI_thickness)
    # End-diastolic dimension, mm
    EDD = 1e-3*np.median(ED_dimension_ROI)

    # Now we compute the LV mass by adding up the volume of the elements if 
    # they are in the LV and if they are UVC_Z < 0.9

    UVC_Z_elem = np.genfromtxt(os.path.join(path2UVC, "COORDS_Z_elem_scaled.dat"),dtype = float)
    UVC_V_elem = np.genfromtxt(os.path.join(path2UVC, "COORDS_V_elem.dat"),dtype = float)

    LV_mass_idx = np.intersect1d(np.where(UVC_Z_elem < 0.9), np.where(UVC_V_elem < 0))

    six_vol_LV = []

    def six_vol_element_cm3(i):
        result =  np.linalg.det(np.array([np.array([1e-4*biv_pts.p1[biv_elem.i1[i]], 1e-4*biv_pts.p2[biv_elem.i1[i]], 1e-4*biv_pts.p3[biv_elem.i1[i]], 1.], dtype = float),
                              np.array([1e-4*biv_pts.p1[biv_elem.i2[i]], 1e-4*biv_pts.p2[biv_elem.i2[i]], 1e-4*biv_pts.p3[biv_elem.i2[i]], 1.], dtype = float),
                              np.array([1e-4*biv_pts.p1[biv_elem.i3[i]], 1e-4*biv_pts.p2[biv_elem.i3[i]], 1e-4*biv_pts.p3[biv_elem.i3[i]], 1.], dtype = float),
                              np.array([1e-4*biv_pts.p1[biv_elem.i4[i]], 1e-4*biv_pts.p2[biv_elem.i4[i]], 1e-4*biv_pts.p3[biv_elem.i4[i]], 1.], dtype = float)
                             ], dtype = float)
                            )
        return np.abs(result)

    six_vol_LV.append(Parallel(n_jobs=20)(delayed(six_vol_element_cm3)(i) for i in range(len(LV_mass_idx))))

    LV_mass = 1.05*sum(six_vol_LV[0])/6.

    return ISWT, WT, EDD, LV_mass

def mechanics_archer2_setup(fourch_name):
    
    path2fourch = os.path.join("/data","fitting",fourch_name)
    path2biv = os.path.join(path2fourch,"biv")

    prepare_mesh.extract_peri_base(fourch_name)
    UVC.create(fourch_name, "peri")
    
    # We take the maximum UVC_Z between the original UVC and the peri UVC

    UVC_Z_MVTV_elem = np.genfromtxt(os.path.join(path2biv, "UVC_MVTV", "UVC", "COORDS_Z_elem.dat"),dtype = float)
    UVC_Z_peri_elem = np.genfromtxt(os.path.join(path2biv, "UVC_peri", "UVC", "COORDS_Z_elem.dat"),dtype = float)

    UVC_Z_max = np.maximum(UVC_Z_MVTV_elem, UVC_Z_peri_elem)

    # The polinomial for the pericardium. Max penalty at the apex, nothing from where UVC >= 0.82

    penalty_biv = 1.5266*(0.82 - UVC_Z_max)**3 - 0.37*(0.82 - UVC_Z_max)**2 + 0.4964*(0.82 - UVC_Z_max)
    penalty_biv[UVC_Z_max > 0.82] = 0.0

    # All this is on the biv, we need to map it to the whole heart.

    fourch_elem = files_manipulations.elem.read(os.path.join(path2fourch, fourch_name + ".elem"))
    tags_vec = fourch_elem.tags

    penalty_fourch = np.zeros(fourch_elem.size)

def filter_output(waveno = 0, subfolder = "."):
    
    path_match = os.path.join("/data","fitting","match")
    path_lab = os.path.join("/data","fitting")
    path_gpes = os.path.join(path_lab, subfolder, "wave" + str(waveno))

    exp_mean = np.loadtxt(os.path.join(path_match, "exp_mean.txt"), dtype=float)
    exp_std = np.loadtxt(os.path.join(path_match, "exp_std.txt"), dtype=float)
    ylabels = read_labels(os.path.join(path_lab, "EP_output_labels.txt"))

    X = np.loadtxt(os.path.join(path_gpes, "X.dat"), dtype=float)

    idx_feasible = np.array([], dtype = int)

    for i, output_name in enumerate(ylabels):

        unfeas_lower = max(0, exp_mean[i] - 5 * exp_std[i])
        unfeas_upper = exp_mean[i] + 5 * exp_std[i]

        y = np.loadtxt(os.path.join(path_gpes, output_name + ".dat"),dtype=float)
        idx_no_lower = np.where(y > unfeas_lower)[0]
        idx_no_upper = np.where(y < unfeas_upper)[0]
        if  i == 0:
            idx_feasible = np.intersect1d(idx_no_lower, idx_no_upper)
        else:
            idx_feasible = np.intersect1d(idx_feasible,np.intersect1d(idx_no_lower, idx_no_upper))

    idx_feasible = np.unique(idx_feasible)

    np.savetxt(os.path.join(path_gpes, "X_feasible.dat"), X[idx_feasible],
                fmt="%.2f")

    for output_name in ylabels:
        y = np.loadtxt(os.path.join(path_gpes, output_name + ".dat"),dtype=float)

        np.savetxt(os.path.join(path_gpes, output_name + "_feasible.dat"),
                    y[idx_feasible], fmt="%.2f")

    print("A total number of " + str(len(idx_feasible)) + " points are feasible")
