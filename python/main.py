#!/usr/bin/env python3

import os

import prepare_mesh 
import files_manipulations
import fibres
import debug

def main():
    bundle_number = "01"
    heart=1
    scripts_folder = "/home/crg17/Desktop/KCL_projects/fitting/python"

    # prepare_mesh.download_zip(bundle_number)
    # prepare_mesh.unzip_meshes(bundle_number)
    # prepare_mesh.vtk_mm2carp_um(heart)

    prepare_mesh.extract_LDRB_biv(heart)
    prepare_mesh.extract_MVTV_base(heart)
    fibres.run_laplacian(heart)
    fibres.rb_bayer(heart)

    # debug.extract_LDRB_biv(heart)

if __name__ == "__main__":
    main()