from os import sep
import skopt
import os
import numpy as np
from joblib import Parallel, delayed
import tqdm
import pathlib
import shutil

import prepare_mesh
import fibres
import UVC
import run_EP
import files_manipulations
from Historia.shared.design_utils import read_labels

SEED = 2

def template_mesh_setup():
    prepare_mesh.extract_LDRB_biv("Template")
    prepare_mesh.extract_MVTV_base("Template")
    UVC.create("Full_Heart_Mesh_Template", "MVTV")
    UVC.bottom_third("Template", "MVTV")
    UVC.create_FEC("Template", "MVTV")
    fibres.run_laplacian("Template")

def EP_funct_param(n_samples = None, waveno = 0, subfolder = "."):
    """Function to generate the first points to run EP in the template.

    Args:
        n_samples (int, optional): Number of points to generate. Defaults to 
        None.
        waveno (int, optional): Wave number, specifies the folder name. Defaults 
        to 0.
        subfolder (str, optional): Subfolder name of /data/fitting to work on. 
        Defaults to ".".
    """

    path_lab = os.path.join("/data","fitting")
    path_match = os.path.join("/data","fitting","match")
    path_gpes = os.path.join(path_lab, subfolder, "wave" + str(waveno))

    pathlib.Path(path_gpes).mkdir(parents = True, exist_ok = True)
    
    param_names = read_labels(os.path.join(path_lab, "EP_funct_labels.txt"))


    param_ranges_lower = np.loadtxt(os.path.join(path_match, "input_range_lower.dat"), dtype=float)
    param_ranges_upper = np.loadtxt(os.path.join(path_match, "input_range_upper.dat"), dtype=float)

    param_ranges = [(param_ranges_lower[i],param_ranges_upper[i]) for i in range(len(param_ranges_lower))]

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

def template_EP_parallel(line_from = 0, line_to = 10, waveno = 0, subfolder = "."):
    """Function to prepare the mesh and run the EP simulation. It works in a 
    sequential way to improve debugging.

    Args:
        line_from (int, optional): Line from the input parameters file to start
        running the simulations. Defaults to 0.
        line_to (int, optional): Final line from the input parameters file to 
        run the simulations. Defaults to 10.
        waveno (int, optional): Wave number, defines the folder name. Defaults 
        to 0.
        subfolder (str, optional): Folder name in /data/fitting to work  on. 
        Defaults to ".".

    Returns:
        had_to_run_new (bool): If a new simulation was run, returns to True.
        Otherwise is False.
    """

    path_lab = os.path.join("/data","fitting")
    path_gpes = os.path.join(path_lab, subfolder, "wave" + str(waveno))
    path_EP = os.path.join("/data","fitting","Full_Heart_Mesh_Template","biv",
                            "EP_simulations")

    pathlib.Path(path_EP).mkdir(parents = True, exist_ok = True)

    def template_EP_explicit_arguments(alpha_endo_vec, lastFECtag_vec, CV_l_vec,
                                       k_fibre_vec, k_FEC_vec, param_line,
                                       path_EP,FEC_height_vec, had_to_run_new):
        """Function to prepare the mesh and run a simulation with specific 
        parameter values.

        Args:
            alpha_endo_vec (array): Array with fibre angle values.
            lastFECtag_vec (array): Array with the FEC tag that marks the 
            difference between normal CV and FEC CV.
            CV_l_vec (array): Array with the values of the fibre-direction CV.
            k_fibre_vec (array): Array with the values of the fibre anisotropy.
            k_FEC_vec (array): Array with the values of the FEC layer 
            anisotropy.
            param_line (int): Specific line from which to take the param values.
            path_EP (str): Path where to run and save the EP simulations.
            FEC_height_vec (array): Array of the % of FEC height marking the
            difference between normal CV and FEC CV.
            had_to_run_new (bool): Boolean to know if a simulation was already
            ran. If a new mesh or simulation is run, it changes to True.

        Returns:
            had_to_run_new
        """

        simulation_file_name = os.path.join(path_EP,
                                str(format(alpha_endo_vec[param_line],'.2f')) +\
                                str(format(FEC_height_vec[param_line],'.2f')) +\
                                str(format(CV_l_vec[param_line],'.2f')) +\
                                str(format(k_fibre_vec[param_line],'.2f')) +\
                                str(format(k_FEC_vec[param_line],'.2f')) +\
                                ".dat"
                                )
        if not os.path.isfile(simulation_file_name):
            had_to_run_new = True

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

        return had_to_run_new

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
        """Function to find the closest value in an array to a given value.

        Args:
            array (array): Array to look into to find the closest value.
            value (same as array): Value to find the closest number in array.

        Returns:
            Closest value to "value" in "array".
        """
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

    had_to_run_new = False

    for param_line in tqdm.tqdm(range(line_from,line_to+1)):
        print("Experiment " + subfolder + ", wave " + str(waveno), end = '\r')
        had_to_run_new = template_EP_explicit_arguments(alpha_endo_vec,
                                        lastFECtag_vec, 
                                        CV_l_vec, 
                                        k_fibre_vec, 
                                        k_FEC_vec, 
                                        param_line,
                                        path_EP,
                                        FEC_height_vec,
                                        had_to_run_new = had_to_run_new)
    return had_to_run_new

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
        if os.path.isfile(os.path.join(EP_dir, X_names[i] + ".dat")):
            with open(os.path.join(EP_dir, X_names[i] + ".dat")) as g:
                AT_vec = g.read().splitlines()
            g.close()
            
            AT_vec_float = [float(x) for x in AT_vec]
            TAT[i] = max(AT_vec_float)
            TATLV[i] = max(np.array(AT_vec_float)[Z_90_endo.astype(int)])
            
            if TAT[i] > 1e5:
                #Hard reset
                print("rm " + os.path.join(EP_dir, X_names[i] + ".*"))
                os.system("rm " + os.path.join(EP_dir, X_names[i] + ".*"))
                angle = param_values[i].split(' ')[0]
                print("rm " + os.path.join(EP_dir,"..","fibres","rb_-" + angle + "_" + angle + "*"))
                os.system("rm " + os.path.join(EP_dir,"..","fibres","rb_-" + angle + "_" + angle + "*"))
                template_EP_parallel(line_from = i, line_to = i, waveno = waveno, subfolder = subfolder)
                i = i - 1
        else:
            template_EP_parallel(line_from = i, line_to = i, waveno = waveno, subfolder = subfolder)
            i = i - 1

    output_numbers = [TAT, TATLV]

    for i,varname in enumerate(output_names):
        np.savetxt(os.path.join(outpath, varname + ".dat"),
                    output_numbers[i],
                    fmt="%.2f")

def filter_output(waveno = 0, subfolder = ".", skip = False):
    
    path_match = os.path.join("/data","fitting","match")
    path_lab = os.path.join("/data","fitting")
    path_gpes = os.path.join(path_lab, subfolder, "wave" + str(waveno))

    exp_mean = np.loadtxt(os.path.join(path_match, "exp_mean.txt"), dtype=float)
    exp_std = np.loadtxt(os.path.join(path_match, "exp_std.txt"), dtype=float)
    ylabels = read_labels(os.path.join(path_lab, "EP_output_labels.txt"))

    if skip:
        shutil.copy(os.path.join(path_gpes, "X.dat"),os.path.join(path_gpes, "X_feasible.dat"))
    else:

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
        if skip:
            shutil.copy(os.path.join(path_gpes, output_name + ".dat"),
                        os.path.join(path_gpes, output_name + "_feasible.dat"))
        else:
            y = np.loadtxt(os.path.join(path_gpes, output_name + ".dat"),dtype=float)

            np.savetxt(os.path.join(path_gpes, output_name + "_feasible.dat"),
                        y[idx_feasible], fmt="%.2f")
    if not skip:
        print("A total number of " + str(len(idx_feasible)) + " points are feasible")
