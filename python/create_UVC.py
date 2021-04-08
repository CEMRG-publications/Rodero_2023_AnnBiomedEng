import os

import shutil

def create(heart, base):

    path2biv = os.path.join("/data","fitting","Full_Heart_Mesh_1","biv")
    path2scripts = os.path.join("/home","crg17","Desktop",
                                "KCL_projects","fitting")
    exe = os.path.join(path2scripts, "python", "model_arch_ek.py")

    shutil.copy(os.path.join(path2biv, base + "_base.surf"),
                os.path.join(path2biv,"biv.base.surf"))
    shutil.copy(os.path.join(path2biv, base + "_base.surf.vtx"),
                os.path.join(path2biv,"biv.base.surf.vtx"))

    os.system(exe + " --uvc --ID=/data/fitting/Full_Heart_Mesh_1/biv/UVC_" +  \
             base + " --basename=" + os.path.join(path2biv,"biv") + \
             " --mode biv --np 20 --tags=" + \
             os.path.join(path2scripts,"sh/etags.sh") + " --overwrite-behaviour overwrite") 