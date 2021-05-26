import os
import numpy as np
import shutil

import files_manipulations


def carp2init(heart, lastFECtag, CV_l, k_fibre, k_FEC, simulation_file_name,
              path_EP = ""):
    """Function with CARP arguments that creates an init file to then run
    ekbatch.

    Args:
        lastFECtag (int): Last tag to include in the FEC layer. Minimum 25,
        maximum tag is 38. The rest of the tags is myocardium.
    """

    CV_t = float(CV_l)*float(k_fibre)
    CV_FEC = float(CV_l)*float(k_FEC)

    path2biv = os.path.join("/data","fitting","Full_Heart_Mesh_" + str(heart),
                            "biv")
    lastFECtag = int(lastFECtag)
    tags_myo = np.append([1, 2],range(lastFECtag + 1, 39))
    tags_FEC = np.array(range(25,lastFECtag + 1))

    bottom_third = files_manipulations.vtx.read(os.path.join(path2biv,
                                                "bottom_third.vtx"),
                                                "biv")

    # write .init file
    outname = simulation_file_name.split('/')[-1][:-4] + ".init"

    f = open(os.path.join(path_EP, outname), "w")

    # header
    f.write('vf:0 vs:0 vn:0 vPS:0\n') # Default properties
    f.write('retro_delay:0 antero_delay:0\n') # If there's no PS, it's ignored.

    # number of stimuli and regions
    f.write('%d %d\n' % (bottom_third.size, int(len(tags_myo)) + len(tags_FEC)))

    # stimulus
    for n in bottom_third.indices:
        f.write('%d %f\n' % (int(n),0))

    # ek regions
    for i,t in enumerate(tags_myo):
        f.write('%d %f %f %f\n' % (int(t), CV_l, CV_t, CV_t))
    for i,t in enumerate(tags_FEC):
        f.write('%d %f %f %f\n' % (int(t), CV_FEC, CV_FEC, CV_FEC))
    
    f.close()

def launch_init(heart, alpha_endo, alpha_epi, simulation_file_name, path_EP = ""):
    path2biv = os.path.join("/data","fitting","Full_Heart_Mesh_" + str(heart),
                            "biv")
    path2sim = path_EP

    outname = simulation_file_name.split('/')[-1][:-4]

    
    shutil.copy(os.path.join(path2biv,"biv_FEC.elem"),
        os.path.join(path2sim, outname + ".elem")
        )

    shutil.copy(os.path.join(path2biv,"fibres",
        "rb_"+str(alpha_epi)+"_"+str(alpha_endo)+".lon"
        ),
        os.path.join(path2sim,outname + ".lon")
        )

    shutil.copy(os.path.join(path2biv,"biv.pts"),
                os.path.join(path2sim,outname + ".pts")
                )

    os.system("ekbatch " + os.path.join(path2sim,outname) + " " + \
            os.path.join(path2sim,outname)
    )