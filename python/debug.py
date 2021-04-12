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

    for files2check in ["biv.epi", "biv.endo", "biv_noLVendo", "biv_noRVendo",
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

def bottom_third(heart):
    path2biv = os.path.join("/data","fitting","Full_Heart_Mesh_" + str(heart),
                            "biv")
    bottom_third_vtx = files_manipulations.vtx.read(os.path.join(path2biv, "EP",
                                                                "bottom_third.vtx"),
                                                    "biv")
    biv_pts = files_manipulations.pts.read(os.path.join(path2biv,"biv.pts"))

    bottom_third_pts = biv_pts.extract(bottom_third_vtx)

    bottom_third_pts.write(os.path.join(path2biv, "EP", "bottom_third.pts"))

def create_FEC(heart, UVC_base):
    path2biv = os.path.join("/data","fitting","Full_Heart_Mesh_" + str(heart),
                            "biv")
    path2UVC = os.path.join(path2biv,"UVC_" + UVC_base, "UVC")

    pathlib.Path(path2biv,"debug").mkdir(parents=True, exist_ok=True)
    shutil.copy(os.path.join(path2biv,"biv_FEC.elem"),
                os.path.join(path2biv,"debug","biv_FEC.elem"))
    shutil.copy(os.path.join(path2biv,"biv.pts"),
                os.path.join(path2biv,"debug","biv_FEC.pts"))
    
    os.system("GlVTKConvert -m " + os.path.join(path2biv,"debug","biv_FEC") +\
              " -e " + os.path.join(path2UVC,"COORDS_Z_elem.dat") +\
              " -e " + os.path.join(path2UVC,"COORDS_RHO_elem_scaled.dat") +\
              " -e " + os.path.join(path2UVC,"COORDS_RHO_elem.dat") +\
              " -n " + os.path.join(path2UVC,"COORDS_RHO.dat") +\
              " -f txt -F bin -o " + os.path.join(path2biv,"debug","biv_FEC"))

    # os.system("meshtool convert -ifmt=carp_txt -ofmt=vtk_bin" +
    #           " -imsh=" + os.path.join(path2biv,"debug","biv_FEC") + \
    #           " -omsh=" + os.path.join(path2biv,"debug","biv_FEC"))