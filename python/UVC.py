import os
import shutil
import pathlib
import numpy as np
import collections

import files_manipulations

def create(fourch_name, base):
    """Function to create Universal Ventricular Coordinates

    Args:
        fourch_name (str): Name of the four chamber mesh.
        base (str): preffix of the surf of the base.
    """

    path2biv = os.path.join("/data","fitting",fourch_name,"biv")
    path2scripts = os.path.join("/home","crg17","Desktop",
                                "KCL_projects","fitting")
    exe = os.path.join(path2scripts, "python", "model_arch_ek.py")

    shutil.copy(os.path.join(path2biv, base + "_base.surf"),
                os.path.join(path2biv,"biv.base.surf"))
    shutil.copy(os.path.join(path2biv, base + "_base.surf.vtx"),
                os.path.join(path2biv,"biv.base.surf.vtx"))

    os.system(exe + " --uvc --ID=" + os.path.join(path2biv,"UVC_" + base) + \
             " --basename=" + os.path.join(path2biv,"biv") + \
             " --mode biv --np 20 --tags=" + \
             os.path.join(path2scripts,"sh/etags.sh") + \
             " --overwrite-behaviour overwrite")

    for coord_case in ["COORDS_RHO", "COORDS_PHI", "COORDS_Z", "COORDS_V"]:
        os.system("meshtool interpolate node2elem -omsh=" + \
                os.path.join(path2biv,"biv") + " -idat=" + \
                os.path.join(path2biv,"UVC_"+base, "UVC", coord_case+".dat") + \
                " -odat=" + os.path.join(path2biv,"UVC_"+base, "UVC", coord_case+"_elem.dat")
                )
        not_scaled = np.genfromtxt(os.path.join(path2biv,"UVC_"+base, "UVC",
                                   coord_case+"_elem.dat"), dtype=float)
        scaled = files_manipulations.reescale(not_scaled)

        with open(os.path.join(path2biv,"UVC_"+base, "UVC",
                                   coord_case+"_elem_scaled.dat"), 'w') as f:
            for item in scaled:
                f.write("%s\n" % item)
def bottom_third(fourch_name = "Full_Heart_Mesh_Template", UVC_base = "MVTV"):
    """Function to write the vertices belonging to the bottom third of the
    apico-basal coordinate of a given UVC.

    Args:
        fourch_name (str, optional): Name of the four chamber mesh. Defaults to 
        "Full_Heart_Mesh_Template".
        UVC_base (str, optional): Suffix of the base of the UVC (should be also
        the name of the base). Defaults to "MVTV".
    """

    path2biv = os.path.join("/data","fitting",fourch_name,
                            "biv")
    path2UVC = os.path.join(path2biv,"UVC_" + UVC_base, "UVC")

    UVC_Z = np.genfromtxt(os.path.join(path2UVC, "COORDS_Z.dat"),dtype = float)
    UVC_RHO = np.genfromtxt(os.path.join(path2UVC, "COORDS_RHO.dat"),dtype = float)
    septum_vtx = files_manipulations.vtx.read(os.path.join(path2biv,"biv.rvsept.surf.vtx"),"biv")

    bottom_third_all = np.where(UVC_Z < 0.33)[0]
    bottom_third_endo = np.intersect1d(bottom_third_all,np.where(UVC_RHO == 0)[0])
    bottom_third_septum = np.intersect1d(bottom_third_all, septum_vtx.indices)

    bottom_third_indices = np.append(bottom_third_endo, bottom_third_septum)


    bottom_third_vtx = files_manipulations.vtx(bottom_third_indices, "biv")

    # bottom third in the septum


    pathlib.Path(os.path.join(path2biv,"EP")).mkdir(parents=True, exist_ok=True)

    bottom_third_vtx.write(os.path.join(path2biv,"EP","bottom_third.vtx"))
def create_FEC(fourch_name = "Full_Heart_Mesh_Template", UVC_base = "MVTV"):
    """Function to create the labels for the FEC layer. It consists on different
    labels in the ventricular endocardium from 0.35 to 1 of the apico-basal
    coordinate.

    Args:
        fourch_name (str, optional): Name of the four chamber mesh. Defaults to 
        "Full_Heart_Mesh_Template".
        UVC_base (str, optional): Suffix of the base of the UVC (should be also
        the name of the base). Defaults to "MVTV".
    """
    path2biv = os.path.join("/data","fitting",fourch_name,"biv")
    path2UVC = os.path.join(path2biv,"UVC_" + UVC_base, "UVC")

    UVC_Z = np.genfromtxt(os.path.join(path2UVC,"COORDS_Z_elem.dat"), dtype = float)
    biv_elem = files_manipulations.elem.read(os.path.join(path2biv,"biv_default.elem"))
    biv_endo = files_manipulations.vtx.read(os.path.join(path2biv,"biv.endo.surf.vtx"),
                                            "biv")


    FEC_levels  = np.append(0.33,np.arange(0.35,1,0.05))
    FEC_tags = range(25,25 + len(FEC_levels))

    new_tags = np.copy(biv_elem.tags)

    endo_i1 = np.where(np.in1d(biv_elem.i1, biv_endo.indices))[0]
    endo_i2 = np.where(np.in1d(biv_elem.i2, biv_endo.indices))[0]
    endo_i3 = np.where(np.in1d(biv_elem.i3, biv_endo.indices))[0]
    endo_i4 = np.where(np.in1d(biv_elem.i4, biv_endo.indices))[0]

    all_indices = [endo_i1, endo_i2, endo_i3, endo_i4]
    filtered_indices = [np.unique(i) for i in all_indices]
    tidy_indices = np.sort(np.concatenate(filtered_indices))

    dict_count = collections.Counter(tidy_indices)

    endo_elem = [idxs for idxs,counts in dict_count.items() if counts == 3]

    for i in endo_elem:
        if(UVC_Z[i] <= FEC_levels[0]):
            new_tags[i] = FEC_tags[0]
        else:
            new_tags[i] = FEC_tags[np.where(UVC_Z[i] > FEC_levels)[0][-1]]

    biv_FEC_elem = files_manipulations.elem(biv_elem.i1, biv_elem.i2,
                                            biv_elem.i3, biv_elem.i4,
                                            new_tags)

    biv_FEC_elem.write(os.path.join(path2biv,"biv_FEC.elem"))

