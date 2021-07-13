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

    path_lab = os.path.join("/data", "fitting", subfolder)
    path_match = os.path.join("/data", "fitting", "match")
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

    param_ranges_lower = np.append(param_ranges_lower_anatomy, param_ranges_lower_ep, param_ranges_lower_mechanics)
    param_ranges_upper = np.append(param_ranges_upper_anatomy, param_ranges_upper_ep, param_ranges_upper_mechanics)

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

    f = open(os.path.join(path_gpes, "x_ep.dat"), "w")
    [f.write('%s\n' % ' '.join(map(str, [format(i, '.2f') for i in lhs_array]))) for lhs_array in x_ep]
    f.close()

    f = open(os.path.join(path_gpes, "X_mechanics.dat"), "w")
    [f.write('%s\n' % ' '.join(map(str, [format(i, '.2f') for i in lhs_array]))) for lhs_array in x_mechanics]
    f.close()

    with open(os.path.join(path_gpes, "X_anatomy.csv"), mode='w') as f:
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
    path_gpes = os.path.join("/data", "fitting", subfolder, "wave" + str(waveno))

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

    f = open(os.path.join(path_gpes, "x_ep.dat"), "w")
    f.writelines(' '.join(row) + '\n' for row in x_ep)
    f.close()

    f = open(os.path.join(path_gpes, "X_mechanics.dat"), "w")
    f.writelines(' '.join(row) + '\n' for row in x_mechanics)
    f.close()

    with open(os.path.join(path_gpes, "X_anatomy.csv"), mode='w') as f:
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
    path_lab = os.path.join("/data", "fitting", subfolder)
    path_gpes = os.path.join(path_lab, "wave" + str(waveno))
    temp_outpath = os.path.join(path_lab, "temp_meshes")
    had_to_run_new = False

    with open(os.path.join(path_gpes, "X_anatomy.csv")) as f:
        anatomy_values = f.read().splitlines()

    pathlib.Path(temp_outpath).mkdir(parents=True, exist_ok=True)

    for i in tqdm.tqdm(range(len(anatomy_values)-1)):
        temp_base_name = "wave" + str(waveno) + "_" + str(i)
        final_base_name = "heart_" + anatomy_values[i+1].replace(", ", "")[:-24]
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
            prepare_mesh.vtk_mm2carp_um(fourch_name=final_base_name)

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

    path_lab = os.path.join("/data", "fitting", subfolder)
    path_gpes = os.path.join(path_lab, "wave" + str(waveno))
    had_to_run_new = False
    with open(os.path.join(path_gpes, "X_anatomy.csv")) as f:
        anatomy_values = f.read().splitlines()

    for i in tqdm.tqdm(range(len(anatomy_values)-1)):
        final_base_name = "heart_" + anatomy_values[i+1].replace(", ", "")[:-24]
        mesh_path = os.path.join(path_lab, final_base_name)

        if not os.path.isfile(os.path.join(mesh_path, "biv", "biv_noRVendo.surf.vtx")):
            prepare_mesh.extract_LDRB_biv(final_base_name)
        if not os.path.isfile(os.path.join(mesh_path, "biv", "MVTV_base.surf.vtx")):
            prepare_mesh.extract_MVTV_base(final_base_name)
        if (not os.path.isfile(os.path.join(mesh_path, "biv", "UVC_MVTV", "UVC", "COORDS_V_elem_scaled.dat"))) or\
           (not os.path.isfile(os.path.join(mesh_path, "biv", "UVC_MVTV", "UVC", "COORDS_PHI_elem_scaled.dat"))) or\
           (not os.path.isfile(os.path.join(mesh_path, "biv", "UVC_MVTV", "UVC", "COORDS_Z_elem_scaled.dat"))) or\
           (not os.path.isfile(os.path.join(mesh_path, "biv", "UVC_MVTV", "UVC", "COORDS_RHO_elem_scaled.dat"))):
            UVC.create(final_base_name, "MVTV")
        if not os.path.isfile(os.path.join(mesh_path, "biv", "EP", "bottom_third.vtx")):
            UVC.bottom_third(final_base_name, "MVTV")
        if not os.path.isfile(os.path.join(mesh_path, "biv", "biv_fec.elem")):
            UVC.create_fec(final_base_name, "MVTV")
        if not os.path.isfile(os.path.join(mesh_path, "biv", "fibres", "endoRV", "phie.igb")):
            had_to_run_new = True
            fibres.run_laplacian(final_base_name)

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

    path_lab = os.path.join("/data", "fitting", subfolder)
    path_gpes = os.path.join(path_lab, "wave" + str(waveno))

    had_to_run_new = False

    with open(os.path.join(path_lab, "EP_funct_labels.txt")) as f:
        param_names = f.read().splitlines()

    with open(os.path.join(path_gpes, "x_ep.dat")) as f:
        param_values = f.read().splitlines()

    with open(os.path.join(path_gpes, "X_anatomy.csv")) as f:
        anatomy_values = f.read().splitlines()

    alpha_idx = int(np.where([x == "alpha" for x in param_names])[0])
    fec_height_idx = int(np.where([x == "fec_height" for x in param_names])[0])
    cv_l_idx = int(np.where([x == "CV_l" for x in param_names])[0])
    k_fibre_idx = int(np.where([x == "k_fibre" for x in param_names])[0])
    k_fec_idx = int(np.where([x == "k_fec" for x in param_names])[0])

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
        k_fibre = round(float(param_values[line_num].split(' ')[k_fibre_idx]), 2)
        k_fec = round(float(param_values[line_num].split(' ')[k_fec_idx]), 2)

        fourch_name = "heart_" + anatomy_values[line_num+1].replace(", ", "")[:-36]

        path_ep = os.path.join("/data", "fitting", fourch_name, "biv",
                               "ep_simulations")

        pathlib.Path(path_ep).mkdir(parents=True, exist_ok=True)
        simulation_file_name = os.path.join(path_ep,
                                            '{0:.2f}'.format(alpha) +
                                            '{0:.2f}'.format(fec_height) +
                                            '{0:.2f}'.format(cv_l) +
                                            '{0:.2f}'.format(k_fibre) +
                                            '{0:.2f}'.format(k_fec) +
                                            ".dat"
                                            )

        if not os.path.isfile(os.path.join("/data", "fitting", fourch_name, "biv",
                                           "fibres",
                                           "rb_-" + str('{0:.2f}'.format(alpha)) + "_" +
                                           str('{0:.2f}'.format(alpha)) + ".elem")):
            had_to_run_new = True

            fibres.full_pipeline(fourch_name=fourch_name,
                                 alpha_epi=-alpha,
                                 alpha_endo=alpha,
                                 map_fibres_to_fourch=map_fibres_to_fourch
                                 )
        if not os.path.isfile(os.path.join(path_ep, simulation_file_name)):
            had_to_run_new = True

            run_EP.carp2init(fourch_name=fourch_name,
                             lastfectag=lastfectag,
                             CV_l=cv_l,
                             k_fibre=k_fibre,
                             k_fec=k_fec,
                             simulation_file_name=simulation_file_name,
                             path_ep=path_ep
                             )

            run_EP.launch_init(fourch_name=fourch_name,
                               alpha_endo=alpha,
                               alpha_epi=-alpha,
                               simulation_file_name=simulation_file_name,
                               path_ep=path_ep
                               )

    return had_to_run_new


def penalty_map(fourch_name):
    """The chambers closed are in independent scripts in prepare_mesh.
    - Here we make the penalty map. One value for each element in the whole mesh
    - Move the ATs.
    - For each vein, we need a surf.
    - For each endo, we make surf.
    - Surf of epicardium (where to apply the PM).
    - Binary mesh.

    Args:
        fourch_name ([type]): [description]
    """

    path2fourch = os.path.join("/data", "fitting", fourch_name)
    path2biv = os.path.join(path2fourch, "biv")

    if not os.path.isfile(os.path.join(path2biv, "peri_base.surf.vtx")):
        prepare_mesh.extract_peri_base(fourch_name)
    if not os.path.isfile(os.path.join(path2biv, "UVC_peri", "UVC", "COORDS_V_elem_scaled.dat")):
        UVC.create(fourch_name, "peri")

    if not os.path.isfile(os.path.join(path2biv, "pericardium_penalty.dat")):
        # We take the maximum UVC_Z between the original UVC and the peri UVC

        uvc_z_mvtv_elem = np.genfromtxt(os.path.join(path2biv, "UVC_MVTV", "UVC", "COORDS_Z_elem.dat"), dtype=float)
        uvc_z_peri_elem = np.genfromtxt(os.path.join(path2biv, "UVC_peri", "UVC", "COORDS_Z_elem.dat"), dtype=float)

        uvc_z_max = np.maximum(uvc_z_mvtv_elem, uvc_z_peri_elem)

        # The polynomial for the pericardium. Max penalty at the apex, nothing from where UVC >= 0.82

        penalty_biv = 1.5266*(0.82 - uvc_z_max)**3 - 0.37*(0.82 - uvc_z_max)**2 + 0.4964*(0.82 - uvc_z_max)
        penalty_biv[uvc_z_max > 0.82] = 0.0

        # All this is on the biv, we need to map it to the whole heart.

        np.savetxt(os.path.join(path2biv, "pericardium_penalty.dat"),
                   penalty_biv, fmt="%.2f")

    os.system("meshtool insert data -msh=" + os.path.join(path2fourch, fourch_name) +
              " -submsh=" + os.path.join(path2biv, "biv") +
              " -submsh_data=" + os.path.join(path2biv, "pericardium_penalty.dat") +
              " -odat=" + os.path.join(path2fourch, "pericardium_penalty.dat") +
              " -mode=1"
              )


def boundary_surfaces(fourch_name):

    path2fourch = os.path.join("/data", "fitting", fourch_name)

    # Veins
    if not os.path.isfile(os.path.join(path2fourch, "IVC.neubc")):
        os.system("meshtool extract surface -msh=" + os.path.join(path2fourch, fourch_name) +
                  " -surf=LAApp,RIPV,LIPV,LSPV,RSPV,SVC,IVC" +
                  " -op=18;19;20;21;22;23;24 -ifmt=carp_txt -ofmt=carp_txt")

    # Endocardia

    if not os.path.isfile(os.path.join(path2fourch, "lvendo_closed.surf")):
        if not os.path.isfile(os.path.join(path2fourch, "lvendo_closed.elem")):
            prepare_mesh.close_LV_endo(fourch_name=fourch_name)
        chamber_elem = files_manipulations.surf.read(os.path.join(path2fourch, "lvendo_closed.elem"),
                                                     mesh_from=fourch_name)
        chamber_surf = files_manipulations.surf.tosurf(chamber_elem)
        chamber_surf.write(os.path.join(path2fourch, "lvendo_closed.surf"))
    if not os.path.isfile(os.path.join(path2fourch, "rvendo_closed.surf")):
        if not os.path.isfile(os.path.join(path2fourch, "rvendo_closed.elem")):
            prepare_mesh.close_RV_endo(fourch_name=fourch_name)
        chamber_elem = files_manipulations.surf.read(os.path.join(path2fourch, "rvendo_closed.elem"),
                                                     mesh_from=fourch_name)
        chamber_surf = files_manipulations.surf.tosurf(chamber_elem)
        chamber_surf.write(os.path.join(path2fourch, "rvendo_closed.surf"))
    if not os.path.isfile(os.path.join(path2fourch, "laendo_closed.surf")):
        if not os.path.isfile(os.path.join(path2fourch, "laendo_closed.elem")):
            prepare_mesh.close_LA_endo(fourch_name=fourch_name)
        chamber_elem = files_manipulations.surf.read(os.path.join(path2fourch, "laendo_closed.elem"),
                                                     mesh_from=fourch_name)
        chamber_surf = files_manipulations.surf.tosurf(chamber_elem)
        chamber_surf.write(os.path.join(path2fourch, "laendo_closed.surf"))
    if not os.path.isfile(os.path.join(path2fourch, "raendo_closed.surf")):
        if not os.path.isfile(os.path.join(path2fourch, "raendo_closed.elem")):
            prepare_mesh.close_RA_endo(fourch_name=fourch_name)
        chamber_elem = files_manipulations.surf.read(os.path.join(path2fourch, "raendo_closed.elem"),
                                                     mesh_from=fourch_name)
        chamber_surf = files_manipulations.surf.tosurf(chamber_elem)
        chamber_surf.write(os.path.join(path2fourch, "raendo_closed.surf"))

    # Epicardium

    os.system("meshtool extract surface -msh=" + os.path.join(path2fourch, fourch_name) +
              " -surf=" + os.path.join(path2fourch, "biv.epi_endo_noatria") +
              " -op=1, 2-3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19, 20, 21, 22, 23, 24" +
              " -ifmt=carp_txt -ofmt=vtk_bin")
    os.system("meshtool extract unreachable -msh=" + os.path.join(path2fourch, "biv.epi_endo_noatria.surfmesh") +
              " -submsh=" + os.path.join(path2fourch, "biv.epi") + " -ifmt=vtk_bin -ofmt=vtk_bin")

    epi_files = glob.glob(os.path.join(path2fourch, "biv.epi.part*"))
    size_files = [os.path.getsize(f) for f in epi_files]

    idx_max = size_files.index(max(size_files))
    name_epi = epi_files[idx_max]

    os.system("meshtool convert -ifmt=vtk_bin -ofmt=carp_txt -imsh=" + name_epi +
              " -omsh=" + os.path.join(path2fourch, "biv.epi"))
    for filename in epi_files:
        os.system("rm " + filename)
    epi_elem = files_manipulations.surf.read(os.path.join(path2fourch, "biv.epi.elem"), mesh_from=fourch_name)
    epi_surf = files_manipulations.surf.tosurf(epi_elem)
    epi_surf.write(os.path.join(path2fourch, "biv.epi.surf"))


def prepare_folder_supercomputer(path2finalmesh, mesh_name, at_name):

    path2fourch = os.path.join("/data", "fitting", mesh_name)

    pathlib.Path(path2finalmesh).mkdir(parents=True, exist_ok=True)

    os.system("meshtool convert -ifmt=carp_txt -ofmt=carp_bin" +
              " -imsh=" + os.path.join(path2fourch, mesh_name) +
              " -omsh=" + os.path.join(path2finalmesh, mesh_name + "_ED"))
    
    os.system("cp " + os.path.join(path2fourch, mesh_name, "biv", "ep_simulations", at_name + ".dat") +
              " " + os.path.join(path2finalmesh, at_name + ".dat"))

    os.system("cp " + os.path.join(path2fourch, "pericardium_penalty.dat") +
              " " + os.path.join(path2finalmesh, "pericardium_penalty.dat"))
    
    for surf_name in ["LAApp", "RIPV", "LIPV", "LSPV", "RSPV", "SVC", "IVC",
                      "lvendo_closed", "rvendo_closed", "laendo_closed",
                      "raendo_closed", "biv.epi"]:
        os.system("cp " + os.path.join(path2fourch, surf_name + ".surf") +
                  " " + os.path.join(path2finalmesh, surf_name + ".surf"))
