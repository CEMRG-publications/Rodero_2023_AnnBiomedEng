import csv
import glob
import numpy as np
import os
import pathlib
from posixpath import join
import skopt
from sys import path
import time
import tqdm

import files_manipulations
import prepare_mesh
import UVC

SEED = 2
np.random.seed(SEED)


def input(n_samples = None, waveno = 0, subfolder = "mechanics"):
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

    path_lab = os.path.join("/data","fitting",subfolder)
    path_match = os.path.join("/data","fitting","match")
    path_gpes = os.path.join(path_lab, "wave" + str(waveno))

    pathlib.Path(path_gpes).mkdir(parents = True, exist_ok = True)
    
    param_ranges_lower_anatomy = np.loadtxt(os.path.join(path_match, "mechanics_anatomy_input_range_lower.dat"), dtype=float)
    param_ranges_upper_anatomy = np.loadtxt(os.path.join(path_match, "mechanics_anatomy_input_range_upper.dat"), dtype=float)

    param_ranges_lower_EP = np.loadtxt(os.path.join(path_match, "mechanics_EP_input_range_lower.dat"), dtype=float)
    param_ranges_upper_EP = np.loadtxt(os.path.join(path_match, "mechanics_EP_input_range_upper.dat"), dtype=float)

    param_ranges_lower_mechanics = np.loadtxt(os.path.join(path_match, "mechanics_input_range_lower.dat"), dtype=float)
    param_ranges_upper_mechanics = np.loadtxt(os.path.join(path_match, "mechanics_input_range_upper.dat"), dtype=float)

    param_ranges_lower = np.append(param_ranges_lower_anatomy, param_ranges_lower_EP, param_ranges_lower_mechanics)
    param_ranges_upper = np.append(param_ranges_upper_anatomy, param_ranges_upper_EP, param_ranges_upper_mechanics)

    param_ranges = [(param_ranges_lower[i],param_ranges_upper[i]) for i in range(len(param_ranges_lower))]

    if(n_samples is None):
        n_samples = 10*len(param_ranges)

    space = skopt.space.Space(param_ranges)

    sobol = skopt.sampler.Sobol(min_skip = SEED, max_skip = SEED)
    x = sobol.generate(space.dimensions, int(n_samples), random_state = SEED)

    x_EP = []
    x_anatomy = []
    x_mechanics = []
    for row in x:
        x_anatomy.append(row[:len(param_ranges_lower_anatomy)])
        x_EP.append(row[len(param_ranges_lower_anatomy):(len(param_ranges_lower_anatomy)+len(param_ranges_lower_EP))])
        x_mechanics.append(row[(len(param_ranges_lower_anatomy)+len(param_ranges_lower_EP)):(len(param_ranges_lower_anatomy)+len(param_ranges_lower_EP)+len(param_ranges_lower_mechanics))])


    f = open(os.path.join(path_gpes, "X.dat"), "w")
    [f.write('%s\n' % ' '.join(map(str,[format(i, '.2f') for i in lhs_array]))) for lhs_array in x]
    f.close()

    f = open(os.path.join(path_gpes, "X_EP.dat"), "w")
    [f.write('%s\n' % ' '.join(map(str,[format(i, '.2f') for i in lhs_array]))) for lhs_array in x_EP]
    f.close()

    f = open(os.path.join(path_gpes, "X_mechanics.dat"), "w")
    [f.write('%s\n' % ' '.join(map(str,[format(i, '.2f') for i in lhs_array]))) for lhs_array in x_mechanics]
    f.close()

    with open(os.path.join(path_gpes,"X_anatomy.csv"), mode='w') as f:
        f_writer = csv.writer(f, delimiter=',')

        f_writer.writerow(["Mode" + str(i) for i in range(1,19)])

        for current_line in range(len(x_anatomy)):
            output = np.zeros(18)
            output[0:5] = x_anatomy[current_line][0:5]
            output[8] = x_anatomy[current_line][5]
            f_writer.writerow(["{0:.2f}".format(round(i,2)) for i in output])

    f.close()
def preprocess_input(waveno = 0, subfolder = "mechanics"):
    """Function to split the input from a .dat file to a .csv file. This is
    needed for deformetrica.

    Args:
        waveno (int, optional): Wave number, specifies the folder name.
        Defaults to 0.
        subfolder (str, optional): Subfolder name of /data/fitting to work on. 
        Defaults to "mechanics".
    """
    path_gpes = os.path.join("/data","fitting", subfolder, "wave" + str(waveno))

    with open(os.path.join(path_gpes,"X.dat")) as f:
        anatomy_EP_mechanics_values = f.read().splitlines()

    x_anatomy = []
    x_EP = []
    x_mechanics = []
    
    for full_line in anatomy_EP_mechanics_values:
        line = full_line.split(' ')
        x_anatomy.append(line[0:6])
        x_EP.append(line[6:10])
        x_mechanics.append(line[10:21])

    f = open(os.path.join(path_gpes, "X_EP.dat"), "w")
    f.writelines(' '.join(row) + '\n' for row in x_EP)
    f.close()

    f = open(os.path.join(path_gpes, "X_mechanics.dat"), "w")
    f.writelines(' '.join(row) + '\n' for row in x_mechanics)
    f.close()

    with open(os.path.join(path_gpes,"X_anatomy.csv"), mode='w') as f:
        f_writer = csv.writer(f, delimiter=',')

        f_writer.writerow(["Mode" + str(i) for i in range(1,19)])

        for current_line in range(len(x_anatomy)):
            output = np.zeros(18)
            output[0:5] = x_anatomy[current_line][0:5]
            output[8] = x_anatomy[current_line][5]
            f_writer.writerow(["{0:.2f}".format(round(i,2)) for i in output])

    f.close()
def build_meshes(waveno = 0, subfolder = "mechanics", force_construction = False):
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
    path_lab = os.path.join("/data","fitting", subfolder)
    path_gpes = os.path.join(path_lab, "wave" + str(waveno))
    temp_outpath = os.path.join(path_lab,"temp_meshes")
    had_to_run_new = False

    with open(os.path.join(path_gpes,"X_anatomy.csv")) as f:
        anatomy_values = f.read().splitlines()

    pathlib.Path(temp_outpath).mkdir(parents = True, exist_ok = True)
    
    for i in tqdm.tqdm(range(len(anatomy_values)-1)):
        temp_base_name = "wave" + str(waveno) + "_" + str(i)
        final_base_name = "heart_" + anatomy_values[i+1].replace(",","")[:-24]
        mesh_path = os.path.join(path_lab, final_base_name)

        if (not os.path.isfile(os.path.join(mesh_path,final_base_name + ".elem")) and not os.path.isfile(os.path.join(mesh_path,final_base_name + "_default.elem"))) or force_construction:
            had_to_run_new = True
            if not os.path.isfile(os.path.join(temp_outpath,"wave" + str(waveno) + "_" + str(i) + ".vtk")):
                csv_file_name = os.path.join(temp_outpath,"wave" + str(waveno) + "_" + str(i) + ".csv")
                np.savetxt(csv_file_name,np.array([anatomy_values[0],anatomy_values[i+1]]),fmt='%s',delimiter='\n')
                
                print(final_base_name)
                print(time.strftime("%H:%M:%S", time.localtime()))

                os.chdir(os.path.join("/home","crg17","Desktop","KCL_projects","fitting","python","CardiacMeshConstruction_outside"))
                os.system("python3.6 ./pipeline.py " + csv_file_name + " " + temp_outpath + " " + str(waveno) + "_" + str(i))

            pathlib.Path(mesh_path).mkdir(parents = True, exist_ok = True)

            os.system("cp " + os.path.join(temp_outpath,temp_base_name + ".vtk") +\
                    " " + os.path.join(mesh_path, final_base_name + ".vtk"))
            prepare_mesh.vtk_mm2carp_um(fourch_name = final_base_name)
    
    os.system("rm -rf " + temp_outpath)
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
    
    path2fourch = os.path.join("/data","fitting",fourch_name)
    path2biv = os.path.join(path2fourch,"biv")

    if not os.path.isfile(os.path.join(path2biv,"peri_base.surf.vtx")):
        prepare_mesh.extract_peri_base(fourch_name)
    if not os.path.isfile(os.path.join(path2biv,"UVC_peri","UVC","COORDS_V_elem_scaled.dat")):
        UVC.create(fourch_name, "peri")
    
    if not os.path.isfile(os.path.join(path2biv, "pericardium_penalty.dat")):
        # We take the maximum UVC_Z between the original UVC and the peri UVC

        UVC_Z_MVTV_elem = np.genfromtxt(os.path.join(path2biv, "UVC_MVTV", "UVC", "COORDS_Z_elem.dat"),dtype = float)
        UVC_Z_peri_elem = np.genfromtxt(os.path.join(path2biv, "UVC_peri", "UVC", "COORDS_Z_elem.dat"),dtype = float)

        UVC_Z_max = np.maximum(UVC_Z_MVTV_elem, UVC_Z_peri_elem)

        # The polinomial for the pericardium. Max penalty at the apex, nothing from where UVC >= 0.82

        penalty_biv = 1.5266*(0.82 - UVC_Z_max)**3 - 0.37*(0.82 - UVC_Z_max)**2 + 0.4964*(0.82 - UVC_Z_max)
        penalty_biv[UVC_Z_max > 0.82] = 0.0

        # All this is on the biv, we need to map it to the whole heart.

        np.savetxt(os.path.join(path2biv, "pericardium_penalty.dat"),
                            penalty_biv, fmt="%.2f")
        
    os.system("meshtool insert data -msh=" + os.path.join(path2fourch,fourch_name) +\
              " -submsh=" + os.path.join(path2biv,"biv") +\
              " -submsh_data=" + os.path.join(path2biv,"pericardium_penalty.dat") +\
              " -odat=" + os.path.join(path2fourch, "pericardium_penalty.dat") +\
              " -mode=1"
            )

def boundary_surfaces(fourch_name):

    path2fourch = os.path.join("/data","fitting",fourch_name)

    # Veins
    if not os.path.isfile(os.path.join(path2fourch,"IVC.neubc")):
        os.system("meshtool extract surface -msh=" + os.path.join(path2fourch,fourch_name) +\
                " -surf=LAApp,RIPV,LIPV,LSPV,RSPV,SVC,IVC" +\
                " -op=18;19;20;21;22;23;24 -ifmt=carp_txt -ofmt=carp_txt")

    # Endocardia

    if not os.path.isfile(os.path.join(path2fourch,"lvendo_closed.surf")):
        if not os.path.isfile(os.path.join(path2fourch,"lvendo_closed.elem")):
            prepare_mesh.close_LV_endo(fourch_name = fourch_name)
        chamber_elem = files_manipulations.surf.read(os.path.join(path2fourch,"lvendo_closed.elem"), mesh_from=fourch_name)
        chamber_surf = files_manipulations.surf.tosurf(chamber_elem)
        chamber_surf.write(os.path.join(path2fourch,"lvendo_closed.surf"))
    if not os.path.isfile(os.path.join(path2fourch,"rvendo_closed.surf")):
        if not os.path.isfile(os.path.join(path2fourch,"rvendo_closed.elem")):
            prepare_mesh.close_RV_endo(fourch_name = fourch_name)
        chamber_elem = files_manipulations.surf.read(os.path.join(path2fourch,"rvendo_closed.elem"), mesh_from=fourch_name)
        chamber_surf = files_manipulations.surf.tosurf(chamber_elem)
        chamber_surf.write(os.path.join(path2fourch,"rvendo_closed.surf"))
    if not os.path.isfile(os.path.join(path2fourch,"laendo_closed.surf")):
        if not os.path.isfile(os.path.join(path2fourch,"laendo_closed.elem")):
            prepare_mesh.close_LA_endo(fourch_name = fourch_name)
        chamber_elem = files_manipulations.surf.read(os.path.join(path2fourch,"laendo_closed.elem"), mesh_from=fourch_name)
        chamber_surf = files_manipulations.surf.tosurf(chamber_elem)
        chamber_surf.write(os.path.join(path2fourch,"laendo_closed.surf"))
    if not os.path.isfile(os.path.join(path2fourch,"raendo_closed.surf")):
        if not os.path.isfile(os.path.join(path2fourch,"raendo_closed.elem")):
            prepare_mesh.close_RA_endo(fourch_name = fourch_name)
        chamber_elem = files_manipulations.surf.read(os.path.join(path2fourch,"raendo_closed.elem"), mesh_from=fourch_name)
        chamber_surf = files_manipulations.surf.tosurf(chamber_elem)
        chamber_surf.write(os.path.join(path2fourch,"raendo_closed.surf"))

    # Epicardium

    os.system("meshtool extract surface -msh=" + os.path.join(path2fourch,fourch_name) +\
              " -surf=" + os.path.join(path2fourch,"biv.epi_endo_noatria") +\
              " -op=1,2-3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24" +\
              " -ifmt=carp_txt -ofmt=vtk_bin")
    os.system("meshtool extract unreachable -msh=" + os.path.join(path2fourch,"biv.epi_endo_noatria.surfmesh") +\
              " -submsh=" + os.path.join(path2fourch,"biv.epi") + " -ifmt=vtk_bin -ofmt=vtk_bin")

    epi_files = glob.glob(os.path.join(path2fourch,"biv.epi.part*"))
    size_files = [os.path.getsize(f) for f in epi_files]

    idx_max = size_files.index(max(size_files))
    name_epi = epi_files[idx_max]

    os.system("meshtool convert -ifmt=vtk_bin -ofmt=carp_txt -imsh=" + name_epi +\
              " -omsh=" + os.path.join(path2fourch,"biv.epi"))
    for filename in epi_files:
        os.system("rm " + filename)
    epi_elem = files_manipulations.surf.read(os.path.join(path2fourch,"biv.epi.elem"), mesh_from=fourch_name)
    epi_surf = files_manipulations.surf.tosurf(epi_elem)
    epi_surf.write(os.path.join(path2fourch,"biv.epi.surf"))

def prepare_folder_supercomputer(path2finalmesh, mesh_name, AT_name):

    path2fourch = os.path.join("/data","fitting",mesh_name)

    pathlib.Path(path2finalmesh).mkdir(parents = True, exist_ok = True)

    os.system("meshtool convert -ifmt=carp_txt -ofmt=carp_bin" +\
            " -imsh=" + os.path.join(path2fourch,mesh_name) +\
            " -omsh=" + os.path(path2finalmesh,mesh_name + "_ED"))
    
    os.system("cp " + os.path.join(path2fourch,mesh_name,"biv","EP_simulations",AT_name + ".dat") +\
              " " + os.path.join(path2finalmesh,AT_name + ".dat"))

    os.system("cp " + os.path.join(path2fourch,"pericardium_penalty.dat") +\
              " " + os.path.join(path2finalmesh,"pericardium_penalty.dat"))
    
    for surf_name in ["LAApp","RIPV","LIPV","LSPV","RSPV","SVC","IVC",\
                      "lvendo_closed","rvendo_closed","laendo_closed",\
                      "raendo_closed","biv.epi"]:
        os.system("cp " + os.path.join(path2fourch,surf_name + ".surf") +\
              " " + os.path.join(path2finalmesh,surf_name + ".surf"))