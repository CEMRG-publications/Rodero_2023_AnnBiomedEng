#!/usr/bin/env python3

import os

import prepare_mesh 
import files_manipulations as fm

def main():
    bundle_number = "01"
    heart=1
    scripts_folder = "/home/crg17/Desktop/KCL_projects/fitting/python"

    # prepare_mesh.download_zip(bundle_number)
    # prepare_mesh.unzip_meshes(bundle_number)
    # prepare_mesh.vtk_mm2carp_um(heart)
    # prepare_mesh.extract_bdry_bayer(heart)
    # prepare_mesh.map_biv(heart)

    for experiment in ["apba","epi","endoLV","endoRV"]:
        os.system(os.path.join(scripts_folder,"run_fibres.py") + \
            " --experiment " + experiment + " --current_case " + str(heart) + \
            " --np 20 --overwrite-behaviour overwrite")

if __name__ == "__main__":
    main()