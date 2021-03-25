#!/usr/bin/env python3

from prepare_mesh import *
import os
from os.path import join

def main():
    bundle_number = "01"
    heart=1
    experiment = "apba"
    scripts_folder = "/home/crg17/Desktop/KCL_projects/fitting/python"

    # download_zip(bundle_number)
    # unzip_meshes(bundle_number)
    # vtk_mm2carp_um(heart)
    # extract_bdry_bayer(heart)
    os.system(join(scripts_folder,"run_fibres.py") + " --experiment " + experiment + " --current_case " + str(heart))

if __name__ == "__main__":
    main()