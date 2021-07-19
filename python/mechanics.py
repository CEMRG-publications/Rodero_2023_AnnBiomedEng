import csv
import glob
import numpy as np
import os
import pathlib
import skopt
import time
import tqdm

import fibres
import files_manipulations
import prepare_mesh
import run_EP
import UVC

PROJECT_PATH = "/data/fitting"
ANATOMY_CSV = "X_anatomy.csv"
PERICARDIUM_FILE = "pericardium_penalty.dat"
BIV_EPI_NAME = "biv.epi"
SURF_EXTENSION = ".surf"
X_EP_FILE = "x_ep.dat"
SEED = 2
np.random.seed(SEED)


def input_generation(n_samples=None, waveno=0, subfolder="mechanics"):
    """Function to generate the input parameter space using Sobol' semi random
    sequences.

    Args:
        n_samples (int, optional): Number of points to generate. Defaults to
        None.
        waveno (int, optional):  Wave number, specifies the folder name.
        Defaults to 0.
        subfolder (str, optional): Subfolder name of /data/fitting to work on.
        Defaults to "mechanics".
    """

    path_lab = os.path.join(PROJECT_PATH, subfolder)
    path_match = os.path.join(PROJECT_PATH, "match")
    path_gpes = os.path.join(path_lab, "wave" + str(waveno))

    pathlib.Path(path_gpes).mkdir(parents=True, exist_ok=True)

    param_ranges_lower_anatomy = np.loadtxt(os.path.join(path_match, "mechanics_anatomy_input_range_lower.dat"),
                                            dtype=float)
    param_ranges_upper_anatomy = np.loadtxt(os.path.join(path_match, "mechanics_anatomy_input_range_upper.dat"),
                                            dtype=float)

    param_ranges_lower_ep = np.loadtxt(os.path.join(path_match, "mechanics_EP_input_range_lower.dat"), dtype=float)
    param_ranges_upper_ep = np.loadtxt(os.path.join(path_match, "mechanics_EP_input_range_upper.dat"), dtype=float)

    param_ranges_lower_mechanics = np.loadtxt(os.path.join(path_match, "mechanics_input_range_lower.dat"), dtype=float)
    param_ranges_upper_mechanics = np.loadtxt(os.path.join(path_match, "mechanics_input_range_upper.dat"), dtype=float)

    param_ranges_lower = np.concatenate((param_ranges_lower_anatomy, param_ranges_lower_ep,
                                         param_ranges_lower_mechanics))
    param_ranges_upper = np.concatenate((param_ranges_upper_anatomy, param_ranges_upper_ep,
                                         param_ranges_upper_mechanics))

    param_ranges = [(param_ranges_lower[i], param_ranges_upper[i]) for i in range(len(param_ranges_lower))]

    if n_samples is None:
        n_samples = 10*len(param_ranges)

    space = skopt.space.Space(param_ranges)

    sobol = skopt.sampler.Sobol(min_skip=SEED, max_skip=SEED)
    x = sobol.generate(space.dimensions, int(n_samples), random_state=SEED)

    x_ep = []
    x_anatomy = []
    x_mechanics = []
    for row in x:
        anatomy_index = len(param_ranges_lower_anatomy)
        x_anatomy.append(row[:anatomy_index])

        ep_index = anatomy_index + len(param_ranges_lower_ep)
        x_ep.append(row[anatomy_index:ep_index])

        mechanics_index = ep_index + len(param_ranges_lower_mechanics)
        x_mechanics.append(row[ep_index:mechanics_index])
    f = open(os.path.join(path_gpes, "X.dat"), "w")
    [f.write('%s\n' % ' '.join(map(str, [format(i, '.2f') for i in lhs_array]))) for lhs_array in x]
    f.close()

    f = open(os.path.join(path_gpes, X_EP_FILE), "w")
    [f.write('%s\n' % ' '.join(map(str, [format(i, '.2f') for i in lhs_array]))) for lhs_array in x_ep]
    f.close()

    f = open(os.path.join(path_gpes, "X_mechanics.dat"), "w")
    [f.write('%s\n' % ' '.join(map(str, [format(i, '.2f') for i in lhs_array]))) for lhs_array in x_mechanics]
    f.close()

    with open(os.path.join(path_gpes, ANATOMY_CSV), mode='w') as f:
        f_writer = csv.writer(f, delimiter=',')

        f_writer.writerow(["Mode" + str(i) for i in range(1, 19)])

        for current_line in range(len(x_anatomy)):
            output = np.zeros(18)
            output[0:5] = x_anatomy[current_line][0:5]
            output[8] = x_anatomy[current_line][5]
            f_writer.writerow(["{0:.2f}".format(round(i, 2)) for i in output])

    f.close()


def preprocess_input(waveno=0, subfolder="mechanics"):
    """Function to split the input from a .dat file to a .csv file. This is
    needed for deformetrica.

    Args:
        waveno (int, optional): Wave number, specifies the folder name.
        Defaults to 0.
        subfolder (str, optional): Subfolder name of /data/fitting to work on.
        Defaults to "mechanics".
    """
    path_gpes = os.path.join(PROJECT_PATH, subfolder, "wave" + str(waveno))

    with open(os.path.join(path_gpes, "X.dat")) as f:
        anatomy_ep_mechanics_values = f.read().splitlines()

    x_anatomy = []
    x_ep = []
    x_mechanics = []

    for full_line in anatomy_ep_mechanics_values:
        line = full_line.split(' ')
        x_anatomy.append(line[0:6])
        x_ep.append(line[6:10])
        x_mechanics.append(line[10:21])

    f = open(os.path.join(path_gpes, X_EP_FILE), "w")
    f.writelines(' '.join(row) + '\n' for row in x_ep)
    f.close()

    f = open(os.path.join(path_gpes, "X_mechanics.dat"), "w")
    f.writelines(' '.join(row) + '\n' for row in x_mechanics)
    f.close()

    with open(os.path.join(path_gpes, ANATOMY_CSV), mode='w') as f:
        f_writer = csv.writer(f, delimiter=',')

        f_writer.writerow(["Mode" + str(i) for i in range(1, 19)])

        for current_line in range(len(x_anatomy)):
            output = np.zeros(18)
            output[0:5] = x_anatomy[current_line][0:5]
            output[8] = x_anatomy[current_line][5]
            f_writer.writerow(["{0:.2f}".format(round(i, 2)) for i in output])

    f.close()


def build_meshes(waveno=0, subfolder="mechanics", force_construction=False):
    """Function to generate meshes using deformetrica given the modes values.

    Args:
        waveno (int, optional): Wave number, defines the folder. Defaults to 0.
        subfolder (str, optional): Subfolder where to work on. Defaults to ".".
        force_construction (bool, optional): If True, it generates the meshes,
        even if they already exit. Defaults to False.

    Returns:
        had_to_run_new (bool): Boolean variable to know if a new mesh was
        created.
    """
    path_lab = os.path.join(PROJECT_PATH, subfolder)
    path_gpes = os.path.join(path_lab, "wave" + str(waveno))
    temp_outpath = os.path.join(path_lab, "temp_meshes")
    had_to_run_new = False

    with open(os.path.join(path_gpes, ANATOMY_CSV)) as f:
        anatomy_values = f.read().splitlines()

    pathlib.Path(temp_outpath).mkdir(parents=True, exist_ok=True)

    for i in tqdm.tqdm(range(len(anatomy_values)-1)):
        temp_base_name = "wave" + str(waveno) + "_" + str(i)

        separated_modes = anatomy_values[i+1].split(',')
        specific_modes = separated_modes[0:5]
        specific_modes.append(separated_modes[8])
        final_base_name = "heart_" + ''.join(specific_modes)
        mesh_path = os.path.join(path_lab, final_base_name)

        if (not os.path.isfile(os.path.join(mesh_path, final_base_name + ".elem")) and
                not os.path.isfile(os.path.join(mesh_path, final_base_name + "_default.elem"))) or force_construction:
            had_to_run_new = True
            if not os.path.isfile(os.path.join(temp_outpath, "wave" + str(waveno) + "_" + str(i) + ".vtk")):
                csv_file_name = os.path.join(temp_outpath, "wave" + str(waveno) + "_" + str(i) + ".csv")
                np.savetxt(csv_file_name, np.array([anatomy_values[0], anatomy_values[i+1]]), fmt='%s', delimiter='\n')

                print(final_base_name)
                print(time.strftime("%H:%M:%S", time.localtime()))

                os.chdir(os.path.join("/home", "crg17", "Desktop", "KCL_projects", "fitting", "python",
                                      "CardiacMeshConstruction_outside"))
                os.system("python3.6 ./pipeline.py "+csv_file_name+" "+temp_outpath+" "+str(waveno)+"_"+str(i))

            pathlib.Path(mesh_path).mkdir(parents=True, exist_ok=True)

            os.system("cp " + os.path.join(temp_outpath, temp_base_name + ".vtk") +
                      " " + os.path.join(mesh_path, final_base_name + ".vtk"))
            prepare_mesh.vtk_mm2carp_um(fourch_name=final_base_name, subfolder=subfolder)

    os.system("rm -rf " + temp_outpath)
    return had_to_run_new


def ep_setup(waveno=0, subfolder="mechanics"):
    """Function to prepare the mesh ready for an EP simulation.

    Args:
        waveno (int, optional): Wave number, defines the folder. Defaults to 0.
        subfolder (str, optional): Subfolder name of /data/fitting to work on.
        Defaults to "mechanics".

    Returns:
        had_to_run_new (bool): If a new simulation was run, returns to True.
        Otherwise, is False.
    """

    path_lab = os.path.join(PROJECT_PATH, subfolder)
    path_gpes = os.path.join(path_lab, "wave" + str(waveno))
    had_to_run_new = False
    with open(os.path.join(path_gpes, ANATOMY_CSV)) as f:
        anatomy_values = f.read().splitlines()

    for i in tqdm.tqdm(range(len(anatomy_values)-1)):

        separated_modes = anatomy_values[i + 1].split(',')
        specific_modes = separated_modes[0:5]
        specific_modes.append(separated_modes[8])
        final_base_name = "heart_" + ''.join(specific_modes)
        mesh_path = os.path.join(path_lab, final_base_name)

        if not os.path.isfile(os.path.join(mesh_path, "biv", "biv_noRVendo.surf.vtx")):
            prepare_mesh.extract_LDRB_biv(final_base_name, subfolder=subfolder)
        if not os.path.isfile(os.path.join(mesh_path, "biv", "MVTV_base.surf.vtx")):
            prepare_mesh.extract_MVTV_base(fourch_name=final_base_name, subfolder=subfolder)
        if (not os.path.isfile(os.path.join(mesh_path, "biv", "UVC_MVTV", "UVC", "COORDS_V_elem_scaled.dat"))) or\
           (not os.path.isfile(os.path.join(mesh_path, "biv", "UVC_MVTV", "UVC", "COORDS_PHI_elem_scaled.dat"))) or\
           (not os.path.isfile(os.path.join(mesh_path, "biv", "UVC_MVTV", "UVC", "COORDS_Z_elem_scaled.dat"))) or\
           (not os.path.isfile(os.path.join(mesh_path, "biv", "UVC_MVTV", "UVC", "COORDS_RHO_elem_scaled.dat"))):
            UVC.create(fourch_name=final_base_name, base="MVTV", subfolder=subfolder)
        if not os.path.isfile(os.path.join(mesh_path, "biv", "EP", "bottom_third.vtx")):
            UVC.bottom_third(fourch_name=final_base_name, UVC_base="MVTV", subfolder=subfolder)
        if not os.path.isfile(os.path.join(mesh_path, "biv", "biv_fec.elem")):
            UVC.create_fec(fourch_name=final_base_name, UVC_base="MVTV", subfolder=subfolder)
        if not os.path.isfile(os.path.join(mesh_path, "biv", "fibres", "endoRV", "phie.igb")):
            had_to_run_new = True
            fibres.run_laplacian(fourch_name=subfolder + "/" + final_base_name)

    return had_to_run_new


def ep_simulations(waveno=0, subfolder="mechanics", map_fibres_to_fourch=True):
    """Function to prepare the mesh and run the EP simulation. It works in a
    sequential way to improve debugging.

    Args:
        waveno (int, optional): Wave number, defines the folder name. Defaults
        to 0.
        subfolder (str, optional): [description]. Folder name in /data/fitting to work  on.
        Defaults to "mechanics".
        map_fibres_to_fourch (bool, optional): If True, maps the fibres from the
        biventricular mesh to the four chamber mesh. Defaults to True.

    Returns:
        had_to_run_new (bool): If a new simulation was run, returns to True.
        Otherwise, is False.
    """

    path_lab = os.path.join(PROJECT_PATH, subfolder)
    path_gpes = os.path.join(path_lab, "wave" + str(waveno))

    had_to_run_new = False

    with open(os.path.join(path_lab, "mechanics_EP_labels.txt")) as f:
        param_names = f.read().splitlines()

    with open(os.path.join(path_gpes, X_EP_FILE)) as f:
        param_values = f.read().splitlines()

    with open(os.path.join(path_gpes, ANATOMY_CSV)) as f:
        anatomy_values = f.read().splitlines()

    alpha_idx = int(np.where([x == "alpha" for x in param_names])[0])
    fec_height_idx = int(np.where([x == "FEC_height" for x in param_names])[0])
    cv_l_idx = int(np.where([x == "CV_l" for x in param_names])[0])
    k_fec_idx = int(np.where([x == "k_FEC" for x in param_names])[0])

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
    fec_height_to_lastfectag = {33: 25,
                                35: 26,
                                40: 27,
                                45: 28,
                                50: 29,
                                55: 30,
                                60: 31,
                                65: 32,
                                70: 33,
                                75: 34,
                                80: 35,
                                85: 36,
                                90: 37,
                                95: 38,
                                100: 39}

    for line_num in tqdm.tqdm(range(len(param_values))):
        fec_height = round(float(param_values[line_num].split(' ')[fec_height_idx]), 2)
        height_key = find_nearest(list(fec_height_to_lastfectag.keys()), round(float(fec_height)))

        alpha = round(float(param_values[line_num].split(' ')[alpha_idx]), 2)
        lastfectag = round(float(fec_height_to_lastfectag[height_key]), 2)
        cv_l = round(float(param_values[line_num].split(' ')[cv_l_idx]), 2)
        k_fec = round(float(param_values[line_num].split(' ')[k_fec_idx]), 2)

        separated_modes = anatomy_values[line_num + 1].split(',')
        specific_modes = separated_modes[0:5]
        specific_modes.append(separated_modes[8])
        fourch_name = "heart_" + ''.join(specific_modes)

        path_ep = os.path.join(PROJECT_PATH, subfolder, fourch_name, "biv",
                               "EP_simulations")

        pathlib.Path(path_ep).mkdir(parents=True, exist_ok=True)
        simulation_file_name = os.path.join(path_ep,
                                            '{0:.2f}'.format(alpha) +
                                            '{0:.2f}'.format(fec_height) +
                                            '{0:.2f}'.format(cv_l) +
                                            '{0:.2f}'.format(k_fec) +
                                            ".dat"
                                            )
        if not os.path.isfile(os.path.join(PROJECT_PATH, subfolder, fourch_name, "biv",
                                           "fibres",
                                           "rb_-" + str('{0:.2f}'.format(alpha)) + "_" +
                                           str('{0:.2f}'.format(alpha)) + ".elem")):
            had_to_run_new = True

            fibres.full_pipeline(fourch_name=fourch_name,
                                 subfolder=subfolder,
                                 alpha_epi='{0:.2f}'.format(-alpha),
                                 alpha_endo='{0:.2f}'.format(alpha),
                                 map_fibres_to_fourch=map_fibres_to_fourch
                                 )

        if not os.path.isfile(os.path.join(path_ep, simulation_file_name)):
            had_to_run_new = True

            run_EP.carp2init(fourch_name=subfolder + "/" + fourch_name,
                             lastfectag=lastfectag,
                             CV_l=cv_l,
                             k_fibre=0.25,
                             k_fec=k_fec,
                             simulation_file_name=simulation_file_name,
                             path_ep=path_ep
                             )

            run_EP.launch_init(fourch_name=subfolder + "/" + fourch_name,
                               alpha_endo=alpha,
                               alpha_epi=-alpha,
                               simulation_file_name=simulation_file_name,
                               path_ep=path_ep
                               )

    return had_to_run_new


def mechanics_setup(waveno=0, subfolder="mechanics"):

    path_lab = os.path.join(PROJECT_PATH, subfolder)
    path_gpes = os.path.join(path_lab, "wave" + str(waveno))

    with open(os.path.join(path_gpes, ANATOMY_CSV)) as f:
        anatomy_values = f.read().splitlines()

    with open(os.path.join(path_lab, "mechanics_EP_labels.txt")) as f:
        param_names = f.read().splitlines()

    with open(os.path.join(path_gpes, X_EP_FILE)) as f:
        param_values = f.read().splitlines()

    for i in tqdm.tqdm(range(len(anatomy_values) - 1)):
        separated_modes = anatomy_values[i + 1].split(',')
        specific_modes = separated_modes[0:5]
        specific_modes.append(separated_modes[8])
        mesh_name = "heart_" + ''.join(specific_modes)

        mesh_path = os.path.join(path_lab, mesh_name)
        path_ep = os.path.join(mesh_path, "biv", "EP_simulations")

        alpha_idx = int(np.where([x == "alpha" for x in param_names])[0])
        fec_height_idx = int(np.where([x == "FEC_height" for x in param_names])[0])
        cv_l_idx = int(np.where([x == "CV_l" for x in param_names])[0])
        k_fec_idx = int(np.where([x == "k_FEC" for x in param_names])[0])

        alpha = round(float(param_values[i].split(' ')[alpha_idx]), 2)
        cv_l = round(float(param_values[i].split(' ')[cv_l_idx]), 2)
        k_fec = round(float(param_values[i].split(' ')[k_fec_idx]), 2)

        fec_height = round(float(param_values[i].split(' ')[fec_height_idx]), 2)

        ep_file_name = os.path.join(path_ep,
                                    '{0:.2f}'.format(alpha) +
                                    '{0:.2f}'.format(fec_height) +
                                    '{0:.2f}'.format(cv_l) +
                                    '{0:.2f}'.format(k_fec) +
                                    ".dat"
                                    )
        if not os.path.isfile(os.path.join(mesh_path, PERICARDIUM_FILE)):
            penalty_map(fourch_name=mesh_name, subfolder=subfolder)
        if not os.path.isfile(os.path.join(mesh_path, BIV_EPI_NAME+SURF_EXTENSION)):
            boundary_surfaces(fourch_name=mesh_name, subfolder=subfolder)

        meshes_batch = str(int(i/50))
        at_name = '{0:.2f}'.format(alpha)+'{0:.2f}'.format(fec_height)+'{0:.2f}'.format(cv_l)+'{0:.2f}'.format(k_fec)
        full_simulation_name = mesh_name + at_name


        folder_simulations = os.path.join(path_gpes, "meshes_batch" + meshes_batch, full_simulation_name)
        pathlib.Path(folder_simulations).mkdir(parents=True, exist_ok=True)

        _, _, files_in_final_mesh = next(os.walk(folder_simulations))
        file_count = len(files_in_final_mesh)
        if file_count < 17:
            prepare_folder_supercomputer(path2finalmesh=folder_simulations, subfolder=subfolder, mesh_name=mesh_name,
                                         at_name=at_name)


def penalty_map(fourch_name, subfolder):

    path2fourch = os.path.join(PROJECT_PATH, subfolder, fourch_name)
    path2biv = os.path.join(path2fourch, "biv")

    if not os.path.isfile(os.path.join(path2biv, "peri_base.surf.vtx")):
        prepare_mesh.extract_peri_base(fourch_name=fourch_name, subfolder=subfolder)
    if not os.path.isfile(os.path.join(path2biv, "UVC_peri", "UVC", "COORDS_V_elem_scaled.dat")):
        UVC.create(subfolder + "/" + fourch_name, "peri")

    if not os.path.isfile(os.path.join(path2biv, PERICARDIUM_FILE)):
        # We take the maximum UVC_Z between the original UVC and the peri UVC
        uvc_z_mvtv_elem = np.genfromtxt(os.path.join(path2biv, "UVC_MVTV", "UVC", "COORDS_Z_elem.dat"), dtype=float)
        uvc_z_peri_elem = np.genfromtxt(os.path.join(path2biv, "UVC_peri", "UVC", "COORDS_Z_elem.dat"), dtype=float)

        uvc_z_max = np.maximum(uvc_z_mvtv_elem, uvc_z_peri_elem)

        # The polynomial for the pericardium. Max penalty at the apex, nothing from where UVC >= 0.82

        penalty_biv = 1.5266*(0.82 - uvc_z_max)**3 - 0.37*(0.82 - uvc_z_max)**2 + 0.4964*(0.82 - uvc_z_max)
        penalty_biv[uvc_z_max > 0.82] = 0.0

        # All this is on the biv, we need to map it to the whole heart.

        np.savetxt(os.path.join(path2biv, PERICARDIUM_FILE),
                   penalty_biv, fmt="%.2f")

    os.system("meshtool insert data -msh=" + os.path.join(path2fourch, fourch_name) +
              " -submsh=" + os.path.join(path2biv, "biv") +
              " -submsh_data=" + os.path.join(path2biv, PERICARDIUM_FILE) +
              " -odat=" + os.path.join(path2fourch, PERICARDIUM_FILE) +
              " -mode=1"
              )


def boundary_surfaces(fourch_name, subfolder):

    path2fourch = os.path.join(PROJECT_PATH, subfolder, fourch_name)

    # Veins
    if not os.path.isfile(os.path.join(path2fourch, "IVC.neubc")):
        os.system("meshtool extract surface -msh=" + os.path.join(path2fourch, fourch_name) +
                  " -surf=" + os.path.join(path2fourch, "LAApp") + "," +
                  os.path.join(path2fourch, "RIPV") + "," +
                  os.path.join(path2fourch, "LIPV") + "," +
                  os.path.join(path2fourch, "LSPV") + "," +
                  os.path.join(path2fourch, "RSPV") + "," +
                  os.path.join(path2fourch, "SVC") + "," +
                  os.path.join(path2fourch, "IVC") +
                  " -op=18\;19\;20\;21\;22\;23\;24 -ifmt=carp_txt -ofmt=carp_txt")

    # Endocardia

    if not os.path.isfile(os.path.join(path2fourch, "lvendo_closed.surf")):
        if not os.path.isfile(os.path.join(path2fourch, "lvendo_closed.elem")):
            prepare_mesh.close_LV_endo(fourch_name=fourch_name, subfolder=subfolder)
        chamber_elem = files_manipulations.surf.read(os.path.join(path2fourch, "lvendo_closed.elem"),
                                                     mesh_from=fourch_name)
        chamber_surf = files_manipulations.surf.tosurf(chamber_elem)
        chamber_surf.write(os.path.join(path2fourch, "lvendo_closed.surf"))
    if not os.path.isfile(os.path.join(path2fourch, "rvendo_closed.surf")):
        if not os.path.isfile(os.path.join(path2fourch, "rvendo_closed.elem")):
            prepare_mesh.close_RV_endo(fourch_name=fourch_name, subfolder=subfolder)
        chamber_elem = files_manipulations.surf.read(os.path.join(path2fourch, "rvendo_closed.elem"),
                                                     mesh_from=fourch_name)
        chamber_surf = files_manipulations.surf.tosurf(chamber_elem)
        chamber_surf.write(os.path.join(path2fourch, "rvendo_closed.surf"))
    if not os.path.isfile(os.path.join(path2fourch, "laendo_closed.surf")):
        if not os.path.isfile(os.path.join(path2fourch, "laendo_closed.elem")):
            prepare_mesh.close_LA_endo(fourch_name=fourch_name, subfolder=subfolder)
        chamber_elem = files_manipulations.surf.read(os.path.join(path2fourch, "laendo_closed.elem"),
                                                     mesh_from=fourch_name)
        chamber_surf = files_manipulations.surf.tosurf(chamber_elem)
        chamber_surf.write(os.path.join(path2fourch, "laendo_closed.surf"))
    if not os.path.isfile(os.path.join(path2fourch, "raendo_closed.surf")):
        if not os.path.isfile(os.path.join(path2fourch, "raendo_closed.elem")):
            prepare_mesh.close_RA_endo(fourch_name=fourch_name, subfolder=subfolder)
        chamber_elem = files_manipulations.surf.read(os.path.join(path2fourch, "raendo_closed.elem"),
                                                     mesh_from=fourch_name)
        chamber_surf = files_manipulations.surf.tosurf(chamber_elem)
        chamber_surf.write(os.path.join(path2fourch, "raendo_closed.surf"))
    # Epicardium
    os.system("meshtool extract surface -msh=" + os.path.join(path2fourch, fourch_name) +
              " -surf=" + os.path.join(path2fourch, "biv.epi_endo_noatria") +
              " -op=1,2-3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24" +
              " -ifmt=carp_txt -ofmt=vtk_bin")
    os.system("meshtool extract unreachable -msh=" + os.path.join(path2fourch, "biv.epi_endo_noatria.surfmesh") +
              " -submsh=" + os.path.join(path2fourch, BIV_EPI_NAME) + " -ifmt=vtk_bin -ofmt=vtk_bin")

    epi_files = glob.glob(os.path.join(path2fourch, "biv.epi.part*"))
    size_files = [os.path.getsize(f) for f in epi_files]

    idx_max = size_files.index(max(size_files))
    name_epi = epi_files[idx_max]

    os.system("meshtool convert -ifmt=vtk_bin -ofmt=carp_txt -imsh=" + name_epi +
              " -omsh=" + os.path.join(path2fourch, BIV_EPI_NAME))
    for filename in epi_files:
        os.system("rm " + filename)
    epi_elem = files_manipulations.surf.read(os.path.join(path2fourch, "biv.epi.elem"), mesh_from=fourch_name)
    epi_surf = files_manipulations.surf.tosurf(epi_elem)
    epi_surf.write(os.path.join(path2fourch, BIV_EPI_NAME+SURF_EXTENSION))


def prepare_folder_supercomputer(path2finalmesh, subfolder, mesh_name, at_name):

    path2fourch = os.path.join(PROJECT_PATH, subfolder, mesh_name)

    pathlib.Path(path2finalmesh).mkdir(parents=True, exist_ok=True)

    os.system("meshtool convert -ifmt=carp_txt -ofmt=carp_bin" +
              " -imsh=" + os.path.join(path2fourch, mesh_name) +
              " -omsh=" + os.path.join(path2finalmesh, mesh_name + "_ED"))

    os.system("cp " + os.path.join(path2fourch, "biv", "EP_simulations", at_name + ".dat") +
              " " + os.path.join(path2finalmesh, at_name + ".dat"))

    os.system("cp " + os.path.join(path2fourch, PERICARDIUM_FILE) +
              " " + os.path.join(path2finalmesh, PERICARDIUM_FILE))
    
    for surf_name in ["LAApp", "RIPV", "LIPV", "LSPV", "RSPV", "SVC", "IVC",
                      "lvendo_closed", "rvendo_closed", "laendo_closed",
                      "raendo_closed", BIV_EPI_NAME]:
        os.system("cp " + os.path.join(path2fourch, surf_name + SURF_EXTENSION) +
                  " " + os.path.join(path2finalmesh, surf_name + SURF_EXTENSION))