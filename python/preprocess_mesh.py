from joblib import Parallel, delayed

import os
import tqdm

from global_variables_config import *

import fibres
import prepare_mesh
import UVC

def extract_base(mesh_path, anatomy_values, i):
    """Function to extract the base of the mesh ready to be parallelized.

    @param mesh_path: Path where the meshes are going to be created. Currently is /data/fitting/meshes
    @param anatomy_values: Array where each line corresponds to one mesh. Each component is a single
    str with the values of the modes (1 to 18)
    @param i: Index of the mesh to create from anatomy values.

    @return The final file written is mvtv_base.surf.vtx
    """
    if not os.path.isfile(os.path.join(mesh_path, "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv",
                                       "biv_noRVendo.surf.vtx")):
        prepare_mesh.extract_ldrb_biv(fourch_name="heart_" + anatomy_values[i + 1].replace(",", "")[:-36],
                                      subfolder="meshes")
    if not os.path.isfile(os.path.join(mesh_path, "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv",
                                       "mvtv_base.surf.vtx")):
        prepare_mesh.extract_mvtv_base(fourch_name="heart_" + anatomy_values[i + 1].replace(",", "")[:-36],
                                      subfolder="meshes")

def create_uvc(mesh_path, anatomy_values, i):
    """Function to create the universal ventricular coordinates of the mesh ready to be parallelized.

        @param mesh_path: Path where the meshes are going to be created. Currently is /data/fitting/meshes
        @param anatomy_values: Array where each line corresponds to one mesh. Each component is a single
        str with the values of the modes (1 to 18)
        @param i: Index of the mesh to create from anatomy values.

        @return The final file written is COORDS_RHO_elem_scaled.dat
    """

    if (not os.path.isfile(os.path.join(mesh_path, "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv", "UVC_mvtv", "UVC", "COORDS_V_elem_scaled.dat"))) or \
            (not os.path.isfile(os.path.join(mesh_path, "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv", "UVC_mvtv", "UVC", "COORDS_PHI_elem_scaled.dat"))) or \
            (not os.path.isfile(os.path.join(mesh_path, "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv", "UVC_mvtv", "UVC", "COORDS_Z_elem_scaled.dat"))) or \
            (not os.path.isfile(os.path.join(mesh_path, "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv", "UVC_mvtv", "UVC", "COORDS_RHO_elem_scaled.dat"))):
        os.system("rm -rf " + os.path.join(os.path.join(mesh_path, "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv", "UVC_mvtv")))
        UVC.create(fourch_name="heart_" + anatomy_values[i + 1].replace(",", "")[:-36], base="mvtv", subfolder="meshes")

def endocardial_surfaces(mesh_path, anatomy_values, i):
    """Function to create the bottom third of the mesh to be activated and the FEC layer of the mesh ready to be
    parallelized.

        @param mesh_path: Path where the meshes are going to be created. Currently is /data/fitting/meshes
        @param anatomy_values: Array where each line corresponds to one mesh. Each component is a single
        str with the values of the modes (1 to 18)
        @param i: Index of the mesh to create from anatomy values.

        @return The final file written is biv_fec.elem
    """

    if not os.path.isfile(os.path.join(mesh_path, "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv", "EP", "bottom_third.vtx")):
        UVC.bottom_third(fourch_name = "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], UVC_base="mvtv", subfolder="meshes")
    if not os.path.isfile(os.path.join(mesh_path, "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv", "biv_fec.elem")):
        UVC.create_fec(fourch_name = "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], uvc_base="mvtv", subfolder="meshes")

def run_laplacian(mesh_path, anatomy_values, i):
    """Function to run the laplacian activation to create the fibres of the mesh ready to be parallelized.

        @param mesh_path: Path where the meshes are going to be created. Currently is /data/fitting/meshes
        @param anatomy_values: Array where each line corresponds to one mesh. Each component is a single
        str with the values of the modes (1 to 18)
        @param i: Index of the mesh to create from anatomy values.

        @return The final file written is phie.igb
    """
    if not os.path.isfile(os.path.join(mesh_path, "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv", "fibres", "endoRV", "phie.igb")):
        fibres.run_laplacian(os.path.join("meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36]))

def generate_fibres(mesh_path, anatomy_values, param_values, i):
    """Function to create the fibres of the mesh ready to be parallelized.

        @param mesh_path: Path where the meshes are going to be created. Currently is /data/fitting/meshes
        @param anatomy_values: Array where each line corresponds to one mesh. Each component is a single
        str with the values of the modes (1 to 18)
        @param param_values: Array where each line corresponds to one mesh. Each component is a single
        str with the values of EP parameters.
        @param i: Index of the mesh to create from anatomy values.

        @return The final file written is rb_-alpha_alpha.elem
    """

    # alpha = round(float(param_values[i].split(' ')[0]),2)
    if not os.path.isfile(os.path.join(mesh_path, "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv",
                                       "fibres", "rb_-" + str('{0:.2f}'.format(round(float(param_values[i].split(' ')[0]),2))) + "_" + str('{0:.2f}'.format(round(float(param_values[i].split(' ')[0]),2))) + ".elem")):
        fibres.full_pipeline(fourch_name= "heart_" + anatomy_values[i + 1].replace(",", "")[:-36],
                             subfolder="meshes",
                             alpha_epi=str('{0:.2f}'.format(-round(float(param_values[i].split(' ')[0]),2))),
                             alpha_endo=str('{0:.2f}'.format(round(float(param_values[i].split(' ')[0]),2))),
                             map_to_fourch=False
                             )

def biv_setup(subfolder = "initial_sweep", anatomy_csv_file = "input_anatomy_training.csv", ep_dat_file = "input_ep_training.dat"):
    """Function to prepare the biv mesh ready for simulations.

    @param subfolder: Folder within PROJECT_PATH where the input data will be read.
    @param anatomy_csv_file: Name of the csv file containing _all_ the meshes modes values.
    @param ep_dat_file: Name of the dat file containing the values of the EP parameters.

    @return Fibres, FEC and bottom third created.
    """

    mesh_path = os.path.join(PROJECT_PATH,"meshes")

    with open(os.path.join(PROJECT_PATH, subfolder, anatomy_csv_file)) as f:
        anatomy_values = f.read().splitlines()

    with open(os.path.join(PROJECT_PATH, subfolder, ep_dat_file)) as f:
        param_values = f.read().splitlines()

    Parallel(n_jobs=20)(delayed(extract_base)(mesh_path, anatomy_values, i) for i in range(len(anatomy_values)-1))

    #Create UVC does no good in parallel (CARP error of "standard input: bad file descriptor")
    for i in tqdm.tqdm(range(len(anatomy_values)-1)):
        create_uvc(mesh_path,anatomy_values,i)

    Parallel(n_jobs=20)(delayed(endocardial_surfaces)(mesh_path, anatomy_values, i) for i in range(len(anatomy_values)-1))

    for i in tqdm.tqdm(range(len(anatomy_values)-1)):
        if not os.path.isfile(os.path.join(mesh_path, "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv", "UVC_mvtv", "UVC", "COORDS_RHO_elem.dat")):

            create_uvc(mesh_path, anatomy_values, i)
        run_laplacian(mesh_path, anatomy_values, i)

    Parallel(n_jobs=20)(delayed(generate_fibres)(mesh_path, anatomy_values, param_values, i) for i in range(len(anatomy_values)-1))