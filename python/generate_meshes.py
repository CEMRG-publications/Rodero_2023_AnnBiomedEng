import csv
import glob
from joblib import Parallel, delayed
import numpy as np
import os
import pathlib
import shutil
import tqdm

from global_variables_config import *


def deformetrica_single_mesh(mesh_path, anatomy_values, i):
    """Function to create a mesh from the template using Deformetrica.

    @param mesh_path: Path where the meshes are going to be created. Currently is /data/fitting/meshes
    @param anatomy_values: Array where each line corresponds to one mesh. Each component is a single
    str with the values of the modes (1 to 18)
    @param i: Index of the mesh to create from anatomy values.

    @returns The mesh created will be named wavei in the meshes_path folder.
    """
    temp_base_name = anatomy_values[i].replace(",", "")[:-36]
    csv_file_name = os.path.join(PROJECT_PATH, "meshes", "meshing_files", temp_base_name + ".csv")
    os.system("python3.6 " + os.path.join("/home", "crg17", "Desktop", "KCL_projects", "fitting", "python",
                                          "CardiacMeshConstruction_" + temp_base_name.replace('.',''),
                                          "pipeline.py ") + csv_file_name + " " + mesh_path + " " + temp_base_name.replace('.',''))

def preprocess_for_carp(mesh_path, anatomy_values, i):
    """Function to convert to carp format, convert the pts to micrometre instead of millimetre and safe-copying the
    elem file.

    @param mesh_path: Path where the meshes are going to be created. Currently is /data/fitting/meshes
    @param anatomy_values: Array where each line corresponds to one mesh. Each component is a single
    str with the values of the modes (1 to 18)
    @param i: Index of the mesh to create from anatomy values.

    @returns The final mesh will be in the meshes_path under the name heart_xx where xx is the values of the first
    9 modes appended without commas.
    """

    mesh_name = "heart_" + anatomy_values[i].replace(",", "")[:-36]
    mesh_dir = os.path.join(mesh_path, mesh_name)

    pathlib.Path(mesh_dir).mkdir(parents=True, exist_ok=True)

    os.system("meshtool convert -imsh=" + os.path.join(mesh_path, "wave" + anatomy_values[i].replace(",", "")[:-36].replace('.','') + ".vtk") +
          " -omsh=" + os.path.join(mesh_dir, mesh_name) +
          " -ifmt=vtk -ofmt=carp_txt")

    os.rename(os.path.join(mesh_dir, mesh_name) + ".pts", os.path.join(mesh_dir, mesh_name) + "_mm.pts")

    os.system(os.path.join("/home", "common", "cm2carp", "bin", "return_carp2original_coord.pl ") +
              os.path.join(mesh_dir, mesh_name) + "_mm.pts 1000 0 0 0 > " +
              os.path.join(mesh_dir, mesh_name) + "_um.pts")

    shutil.copy(os.path.join(mesh_dir, mesh_name) + "_um.pts",
                os.path.join(mesh_dir, mesh_name) + ".pts")

    shutil.copy(os.path.join(mesh_dir, mesh_name + ".elem"),
                os.path.join(mesh_dir, mesh_name + "_default.elem"))

    os.system("rm " + os.path.join(mesh_path, "wave" + anatomy_values[i].replace(",", "")[:-36].replace('.','') + ".vtk"))

def sample_atlas(subfolder = "initial_sweep", csv_filename = "input_anatomy_training.csv"):
    """Function to generate meshes from the atlas, in parallel (in chunks of 18 meshes).

    @param subfolder: Folder within PROJECT_PATH where the input data will be read.
    @param csv_filename: Name of the csv file containing _all_ the meshes modes values.

    @returns The final mesh in the meshes folder.
    """

    mesh_path = os.path.join(PROJECT_PATH, "meshes")

    pathlib.Path(mesh_path, "meshing_files").mkdir(parents=True, exist_ok=True)

    with open(os.path.join(PROJECT_PATH, subfolder, csv_filename)) as f:
        anatomy_values = f.read().splitlines()
    i=0
    while i < (len(anatomy_values)-1):
        mesh_name = "heart_" + anatomy_values[i+1].replace(",", "")[:-36]
        mesh_dir = os.path.join(mesh_path, mesh_name)

        if os.path.isfile(os.path.join(mesh_dir, mesh_name + "_default.elem")):
            del anatomy_values[i+1]
            i-=1

        i+=1

    header_line = anatomy_values[0]
    del anatomy_values[0]

    for batch in tqdm.tqdm(range((len(anatomy_values)//18) + 1)):
        sub_dataset = anatomy_values[(18*batch):np.min([18*(batch+1),len(anatomy_values)])]

        for i in tqdm.tqdm(range(len(sub_dataset))):
            individual_name = sub_dataset[i].replace(",", "")[:-36]

            print("Couldn't find " + os.path.join(mesh_path, "heart_" + sub_dataset[i].replace(",", "")[:-36], "heart_" + sub_dataset[i].replace(",", "")[:-36] + "_default.elem"))
            np.savetxt(os.path.join(mesh_path, "meshing_files", individual_name + ".csv"),
                       np.array([header_line, sub_dataset[i]]), fmt='%s', delimiter='\n')
            os.system("rm -rf " + os.path.join("/home", "crg17", "Desktop", "KCL_projects", "fitting", "python",
                                               "CardiacMeshConstruction_" + individual_name.replace('.','')))
            os.system("cp -rf " + os.path.join("/home", "crg17", "Desktop", "KCL_projects", "fitting", "python",
                                               "CardiacMeshConstruction_outside ") + \
                      os.path.join("/home", "crg17", "Desktop", "KCL_projects", "fitting", "python",
                                   "CardiacMeshConstruction_" +individual_name.replace('.','')))
        print("Running deformetrica")
        Parallel(n_jobs=20)(delayed(deformetrica_single_mesh)(mesh_path,sub_dataset,i) for i in range(len(sub_dataset)))
        print("Deformetrica finished")
        for i in range(len(sub_dataset)):
            individual_name = sub_dataset[i].replace(",", "")[:-36]
            os.system("rm -rf " + os.path.join("/home","crg17","Desktop","KCL_projects","fitting","python","CardiacMeshConstruction_" + individual_name.replace('.','')))
            preprocess_for_carp(mesh_path=mesh_path, anatomy_values=sub_dataset, i=i)

def clean_meshes_dir():
    """Function to remove all the directories in the meshes directory that are not present in the .csv files of
    the experiments present in the /data/fitting folder. If they not appear is because they are probably from
    older experiments.
    """

    first_level_dirs = glob.glob(PROJECT_PATH + "/*/") # With slashes
    first_level_dirs.remove(PROJECT_PATH + "/meshes/")

    all_directories = []
    for subfolder in first_level_dirs:
        all_directories.append(glob.glob(subfolder + "*/"))

    all_directories = [item for sublist in all_directories for item in sublist]

    all_anatomy_values = []
    for subfolder in all_directories:
        append_flag = False
        if os.path.isfile(subfolder +  "input_anatomy_training.csv"):
            append_flag = True
            with open(subfolder +  "input_anatomy_training.csv") as f:
                anatomy_values = f.read().splitlines()
                anatomy_values.pop(0)
        if os.path.isfile(subfolder +  "input_anatomy_validation.csv"):
            append_flag = True
            with open(subfolder +  "input_anatomy_validation.csv") as f:
                anatomy_values = f.read().splitlines()
                anatomy_values.pop(0)
        if os.path.isfile(subfolder +  "input_anatomy_test.csv"):
            append_flag = True
            with open(subfolder +  "input_anatomy_test.csv") as f:
                anatomy_values = f.read().splitlines()
                anatomy_values.pop(0)

        if append_flag:
            all_anatomy_values.append(anatomy_values)

    all_anatomy_values = [item for sublist in all_anatomy_values for item in sublist]

    all_mesh_folder_names = []
    for i in range(len(all_anatomy_values)):
        all_mesh_folder_names.append(PROJECT_PATH + "/meshes/heart_" + all_anatomy_values[i].replace(",", "")[:-36])

    folder_names_to_keep = list(set(all_mesh_folder_names))

    folder_names_to_clean = glob.glob(os.path.join(PROJECT_PATH,"meshes/*"))
    folder_names_to_clean.remove(os.path.join(PROJECT_PATH, "meshes","meshing_files"))

    folders_to_remove = list(set(folder_names_to_clean) - set(folder_names_to_keep))

    for path_i in tqdm.tqdm(range(len(folders_to_remove))):
        os.system("gio trash " + folders_to_remove[path_i])

    print("Note: The folders have not been permanently deleted. If you want to do so, remove the .Trash folder in the /data directory.")