import os
import pathlib
import shutil

import files_manipulations

def extract_LDRB_biv(heart):
    """Function to visualize the output of extract_LDRB_biv. Surf and vtx of 
    "biv_epi", "biv_endo", "biv_noLVendo", "biv_noRVendo", "biv_lvendo" and
    "biv_rvendo"

    Args:
        heart (int or str): Number of the mesh, part of the path.
    """

    fourch_name = "Full_Heart_Mesh_" + str(heart)
    path2fourch = os.path.join("/data","fitting",fourch_name)
    path2biv = path2fourch + "/biv"
    path2debug = os.path.join(path2biv,"debug")

    pathlib.Path(path2debug).mkdir(parents=True, exist_ok=True)

    biv_pts = files_manipulations.pts.read(os.path.join(path2biv,"biv.pts"))

    for files2check in ["biv.epi", "biv_endo", "biv_noLVendo", "biv_noRVendo",
                        "biv.lvendo", "biv.rvendo", "biv.rvendo_nosept",
                        "biv.rvsept"]:

        surf_file = files_manipulations.surf.read(os.path.join(path2biv, files2check + ".surf"), "biv")
        surfmesh_file = files_manipulations.surf.tosurfmesh(surf_file)

        surfmesh_file.write(os.path.join(path2debug, files2check + ".elem"))

        shutil.copy(os.path.join(path2biv,"biv.pts"),
                    os.path.join(path2debug, files2check + ".pts")
                    )
        os.system("meshtool convert -imsh=" + os.path.join(path2debug,files2check) + \
                                " -omsh=" + os.path.join(path2debug,files2check) + \
                                " -ifmt=carp_txt" + \
                                " -ofmt=vtk")

        vtx_vec = files_manipulations.vtx.read(os.path.join(path2biv, files2check + ".surf.vtx"), "biv")
        sub_pts = biv_pts.extract(vtx_vec)

        sub_pts.write(os.path.join(path2debug, files2check + ".pts"))

def extract_MVTV_base(heart):
    """Function to visualize the output of extract_MVTV_base. Surf and vtx of 
    MVTV_base.

    Args:
        heart (int or str): Number of the mesh, part of the path.
    """
    fourch_name = "Full_Heart_Mesh_" + str(heart)
    path2fourch = os.path.join("/data","fitting",fourch_name)
    path2biv = path2fourch + "/biv"
    path2debug = os.path.join(path2biv,"debug")

    pathlib.Path(path2debug).mkdir(parents=True, exist_ok=True)

    biv_pts = files_manipulations.pts.read(os.path.join(path2biv,"biv.pts"))

    shutil.copy(os.path.join(path2biv,"MVTV_base.surf"),
                os.path.join(path2debug, "MVTV_base.elem")
                )
    shutil.copy(os.path.join(path2biv,"biv.pts"),
                os.path.join(path2debug, "MVTV_base.pts")
                )
    os.system("meshtool convert -imsh=" + os.path.join(path2debug,"MVTV_base") + \
                              " -omsh=" + os.path.join(path2debug,"MVTV_base") + \
                              " -ifmt=carp_txt" + \
                              " -ofmt=vtk")

    vtx_vec = files_manipulations.vtx.read(os.path.join(path2biv,"MVTV_base.surf.vtx"), "biv")
    sub_pts = biv_pts.extract(vtx_vec)

    sub_pts.write(os.path.join(path2debug,"MVTV_base.pts"))

