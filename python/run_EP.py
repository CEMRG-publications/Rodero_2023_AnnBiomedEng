import numpy as np
import os
import pathlib
import shutil

import files_manipulations

from global_variables_config import *


def carp2init(fourch_name = "Full_Heart_Mesh_Template", lastfectag = None,
              CV_l = None, k_fibre = None, k_fec = None,
              simulation_file_name = None, path_ep = "",subfolder="."):
              

    """Function with CARP arguments that creates an init file to then run
    ekbatch.

    Args:
        fourch_name (str, optional): Name of the four-chamber mesh. Defaults to 
        "Full_Heart_Mesh_Template".
        lastfectag (int, optional): Last tag to include in the fec layer. The tag is the percentage of the FEC layer
        times 1e4. The rest of the tags is myocardium. Defaults to None.
        CV_l (float, optional): Conduction velocity in the fibre direction. 
        Defaults to None.
        k_fibre (float, optional): Fibre anisotropy. Defaults to None.
        k_fec (float, optional): fec layer anisotropy. Defaults to None.
        simulation_file_name (str, optional): Name for the file containing the 
        simulation results. Defaults to None.
        path_ep (str, optional): Path where to save the EP simulations. Defaults
        to "".
    """

    CV_t = float(CV_l)*float(k_fibre)
    CV_fec = float(CV_l)*float(k_fec)

    path2biv = os.path.join(PROJECT_PATH,subfolder,fourch_name, "biv")
    lastfectag = int(float(lastfectag))
    tags_myo = np.append([1, 2], range(lastfectag + 1, 10000))
    tags_fec = np.array(range(3300, lastfectag + 1))

    bottom_third = files_manipulations.vtx.read(os.path.join(path2biv,"EP",
                                                "bottom_third.vtx"),
                                                "biv")

    # write .init file
    outname = simulation_file_name.split('/')[-1][:-4] + ".init"

    f = open(os.path.join(path_ep, outname), "w")

    # header
    f.write('vf:0 vs:0 vn:0 vPS:0\n') # Default properties
    f.write('retro_delay:0 antero_delay:0\n') # If there's no PS, it's ignored.

    # number of stimuli and regions
    f.write('%d %d\n' % (bottom_third.size, int(len(tags_myo)) + len(tags_fec)))

    # stimulus
    for n in bottom_third.indices:
        f.write('%d %f\n' % (int(n),0))

    # ek regions
    for i,t in enumerate(tags_myo):
        f.write('%d %f %f %f\n' % (int(t), CV_l, CV_t, CV_t))
    for i,t in enumerate(tags_fec):
        f.write('%d %f %f %f\n' % (int(t), CV_fec, CV_fec, CV_fec))
    
    f.close()


def launch_init(fourch_name="Full_Heart_Mesh_Template", alpha_endo=None, alpha_epi=None, simulation_file_name=None,
                path_ep="", map_to_fourch=False, subfolder='.'):
    """Function to run an EP simulation using ekbatch from an init file.

    Args:
        fourch_name (str, optional): Name of the four chamber mesh file.
        Defaults to "Full_Heart_Mesh_Template".
        alpha_endo (float, optional): Angle value in the endocardium, in 
        degrees. Defaults to None.
        alpha_epi (float, optional): Angle value in the epicardium, in 
        degrees. Defaults to None.
        simulation_file_name (str, optional): Name for the file containing the 
        simulation results. Defaults to None.
        path_ep (str, optional): Path where to save the EP simulations. Defaults
        to "".
        map_to_fourch: If True, maps the resulting activation times file to the four chamber
        mesh, as long as it doesn't exist already.
    """
    path2fourch = os.path.join(PROJECT_PATH, subfolder, fourch_name)
    path2biv = os.path.join(path2fourch, "biv")
    path2sim = path_ep

    outname = simulation_file_name.split('/')[-1][:-4]

    shutil.copy(os.path.join(path2biv, "biv_fec.elem"),
                os.path.join(path2sim, outname + ".elem")
                )

    shutil.copy(os.path.join(path2biv, "fibres", "rb_" + "{0:.2f}".format(alpha_epi) + "_" +
                             "{0:.2f}".format(alpha_endo)+".lon"),
                os.path.join(path2sim, outname + ".lon")
                )

    shutil.copy(os.path.join(path2biv, "biv.pts"),
                os.path.join(path2sim, outname + ".pts")
                )

    os.system("ekbatch " + os.path.join(path2sim, outname) + " " + os.path.join(path2sim, outname))

    if map_to_fourch and not os.path.isfile(os.path.join(path2fourch, "EP_simulations", outname + ".dat")):
        pathlib.Path(os.path.join(path2fourch, "EP_simulations")).mkdir(parents=True, exist_ok=True)

        os.system("meshtool insert data -msh=" + os.path.join(path2fourch, fourch_name.split('/')[-1]) +
                  " -submsh=" + os.path.join(path2biv, "biv") +
                  " -submsh_data=" + os.path.join(path2sim, outname + ".dat") +
                  " -odat=" + os.path.join(path2fourch, "EP_simulations", outname + ".dat") +
                  " -mode=0")
