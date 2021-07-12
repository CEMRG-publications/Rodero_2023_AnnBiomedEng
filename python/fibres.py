import os
import numpy as np
import tqdm
import shutil
import pathlib

import time

import files_manipulations


def run_laplacian(fourch_name = "Full_Heart_Mesh_Template", experiment = None):
    """Function to run all the laplacian simulations of Bayer 2012.

    Args:
        heart (int or str): Number of the mesh.
        experiment (str, optional): Experiment to run. Option are apba for apex
        to base, epi for epicardium to endocardium, endoLV for everything except 
        LV endo to LV endo, endoRV for analogous with RV endo. If None, runs all
        of them sequentially. Defaults to None.
    """

    if experiment != None:
        experiment_vec = [experiment]
    else:
        experiment_vec = ["apba","epi","endoLV","endoRV"]

    scripts_folder = os.path.join("/home","crg17","Desktop","KCL_projects",
                                  "fitting", "python")

    for exp in experiment_vec:
        os.system(os.path.join(scripts_folder,"run_fibres.py") + \
            " --experiment " + exp + " --current_case " + fourch_name + \
            " --np 20 --overwrite-behaviour overwrite")

def rb_bayer(fourch_name = "Full_Heart_Mesh_Template", alpha_epi = -60, alpha_endo = 80):
    """Function to create the fibre file based on the laplacian solves.

    Args:
        heart (int or str): Number of the mesh.
        alpha_epi (int, optional): Angle of the fibre direction in the
        epicardium. Defaults to -60.
        alpha_endo (int, optional): Angle of the fibre direction in the
        endocardium. Defaults to 80.
    """

    biv_dir = "/data/fitting/" + fourch_name + "/biv"
    fibre_dir = biv_dir + "/fibres"
    outname = "rb" + "_" + str(alpha_epi) + "_" + str(alpha_endo)

    for exp in ["apba","epi","endoLV","endoRV"]:

        os.system("igbextract " + os.path.join(fibre_dir,exp,"phie.igb") + \
                  " -o ascii -O " + os.path.join(fibre_dir,exp,"phie.dat"))
        os.system("sed -i s/'\\s'/'\\n'/g " + os.path.join(fibre_dir,exp,"phie.dat"))

    os.system("GlRuleFibers -m " + os.path.join(biv_dir,"biv") + \
                            " --type biv"  + \
                            " -a " + os.path.join(fibre_dir,"apba","phie.dat") + \
                            " -e " + os.path.join(fibre_dir,"epi","phie.dat") + \
                            " -l " + os.path.join(fibre_dir,"endoLV","phie.dat") + \
                            " -r " + os.path.join(fibre_dir,"endoRV","phie.dat") + \
                            " --alpha_endo " + str(alpha_endo) + \
                            " --alpha_epi " + str(alpha_epi) + \
                            " -o " + os.path.join(fibre_dir,outname + "_withsheets.lon")
                            )

    biv_lon = files_manipulations.lon.read(os.path.join(fibre_dir,outname + "_withsheets.lon"))
    corrected_biv_lon = files_manipulations.lon(biv_lon.f1, biv_lon.f2, biv_lon.f3)

    corrected_biv_lon.write(os.path.join(fibre_dir,outname + ".lon"))

def FibreCorrection(fourch_name = "Full_Heart_Mesh_Template", alpha_epi = -60, alpha_endo = 80):
    """Function to correct the fibre direction. For some reason, some fibres are
    not created correctly, and the thing they have in common is having a 
    z-coordinate = 0. It finds those fibres and takes the average direction
    of the neighbouring elements.

    Args:
        fourch_name (str, optional): Name of the four chamber mesh. Defaults to 
        "Full_Heart_Mesh_Template".
        alpha_epi (int, optional): Angle value in the epicardium, in degrees.
        Defaults to -60.
        alpha_endo (int, optional): Angle value in the endocardium, in degrees. 
        Defaults to 80.

    Returns:
        biv_lon (fibres): Fibre values for the biventricular mesh.
    """
    path2biv = os.path.join("/data","fitting",fourch_name,
                            "biv")
    biv_lon = files_manipulations.lon.read(os.path.join(path2biv,"fibres",\
                                            "rb" + "_" + str(alpha_epi) + \
                                            "_" + str(alpha_endo) + ".lon"
                                            )
                                            )
    biv_pts = files_manipulations.pts.read(os.path.join(path2biv,"biv.pts"))
    biv_elem = files_manipulations.elem.read(os.path.join(path2biv,"biv.elem"))

    # Identify tets with wrong fibres
    # t0=time.perf_counter()

    tets_ind = np.where(np.abs(biv_lon.f3) < 1e-6)[0]
    if len(tets_ind) > 1e5:
        print("Too many fibres to correct. Skipping...")
    else:
        print("Correcting fibres...")
        for index_to_correct in tqdm.tqdm(tets_ind):

            # Compute centre of gravity of the element
            g0 = (biv_pts.p1[biv_elem.i1[index_to_correct]] + \
                biv_pts.p1[biv_elem.i2[index_to_correct]] + \
                biv_pts.p1[biv_elem.i3[index_to_correct]] + \
                biv_pts.p1[biv_elem.i4[index_to_correct]]) / 4.

            g1 = (biv_pts.p2[biv_elem.i1[index_to_correct]] + \
                biv_pts.p2[biv_elem.i2[index_to_correct]] + \
                biv_pts.p2[biv_elem.i3[index_to_correct]] + \
                biv_pts.p2[biv_elem.i4[index_to_correct]]) / 4.

            g2 = (biv_pts.p3[biv_elem.i1[index_to_correct]] + \
                biv_pts.p3[biv_elem.i2[index_to_correct]] + \
                biv_pts.p3[biv_elem.i3[index_to_correct]] + \
                biv_pts.p3[biv_elem.i4[index_to_correct]]) / 4.

            # Find neighbours points in the sphere centred in the centre of gravity
            def create_sphere(radius = 800):
                """Function to create the sphere around the last element.

                Args:
                    radius (int, optional): Radius of the sphere in micrometers.
                    Defaults to 800.

                Returns:
                    sphere (array of ints): Array with the indices of the 
                    tetrahedra belonging to the sphere.
                """
                sphere = np.intersect1d(np.where(np.abs(biv_pts.p1-g0) < radius)[0],
                                    np.intersect1d(np.where(np.abs(biv_pts.p2-g1) < radius)[0],
                                                np.where(np.abs(biv_pts.p3-g2) < radius)[0]
                                                )
                                        )
                if len(sphere) <= 4:
                    print("Increasing sphere radius...")
                    create_sphere(radius + 100)
                else:
                    return sphere

            sphere = create_sphere()

            tets_in_sphere_i1 = np.where(np.isin(biv_elem.i1, sphere))
            tets_in_sphere_i2 = np.where(np.isin(biv_elem.i2, sphere))
            tets_in_sphere_i3 = np.where(np.isin(biv_elem.i3, sphere))
            tets_in_sphere_i4 = np.where(np.isin(biv_elem.i4, sphere))

            tets_in_sphere = np.intersect1d(
                            np.intersect1d(
                            np.intersect1d(tets_in_sphere_i1, tets_in_sphere_i2),
                                            tets_in_sphere_i3), tets_in_sphere_i4
                                            )

            # We take the average direction

            biv_lon.f1[index_to_correct] = np.mean(biv_lon.f1[tets_in_sphere])
            biv_lon.f2[index_to_correct] = np.mean(biv_lon.f2[tets_in_sphere])
            biv_lon.f3[index_to_correct] = np.mean(biv_lon.f3[tets_in_sphere])

    # print(time.perf_counter())
    return biv_lon

def full_pipeline(fourch_name = "Full_Heart_Mesh_Template", alpha_epi = -60, alpha_endo = 80, map_fibres_to_fourch = False):
    """Function that automatizes the creation of fibres.

    Args:
        fourch_name (str, optional): Name of the mesh. Defaults to 
        "Full_Heart_Mesh_Template".
        alpha_epi (int, optional): Angle value in the epicardium, in degrees.
        Defaults to -60.
        alpha_endo (int, optional): Angle value in the endocardium, in degrees. 
        Defaults to 80.
        map_fibres_to_fourch (bool, optional): If True, maps the fibres from the
        biventricular mesh to the four chamber mesh. Defaults to False.
    """
    
    fourch_dir = os.path.join("/data/fitting",fourch_name)
    biv_dir =  os.path.join(fourch_dir,"biv")
    fibre_dir = biv_dir + "/fibres"
    outname = "rb" + "_" + str(alpha_epi) + "_" + str(alpha_endo)

    if not os.path.isfile(os.path.join(fibre_dir,outname + ".lon")):

        rb_bayer(fourch_name, alpha_epi, alpha_endo)
        biv_lon_corrected = FibreCorrection(fourch_name, alpha_epi, alpha_endo)
        biv_lon_corrected.normalise
        biv_lon_corrected.write(os.path.join(fibre_dir,outname + ".lon"))

    shutil.copy(os.path.join(biv_dir,"biv.pts"),os.path.join(fibre_dir, outname + ".pts"))
    shutil.copy(os.path.join(biv_dir,"biv.elem"),os.path.join(fibre_dir, outname + ".elem"))

    # pathlib.Path(os.path.join(fourch_dir,"fibres")).mkdir(parents=True, exist_ok=True)

    if map_fibres_to_fourch:
        os.system("meshtool insert meshdata -imsh=" + \
                os.path.join(fibre_dir,outname) + \
                " -msh=" + os.path.join(fourch_dir,fourch_name) + \
                " -op=1 -ifmt=carp_txt -ofmt=carp_txt" + \
                " -outmsh=" + os.path.join(fourch_dir,fourch_name))

