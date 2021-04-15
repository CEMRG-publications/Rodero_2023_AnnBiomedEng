#!/usr/bin/env python3

import os

import prepare_mesh 
import files_manipulations
import fibres
import debug
import UVC
import run_EP
import generate

def main():
    bundle_number = "01"
    heart=1
    scripts_folder = "/home/crg17/Desktop/KCL_projects/fitting/python"
    lastFECtag=30
    CV_l = 0.5
    k_fibre = 0.6
    k_FEC = 6.
    alpha_endo = 50
    alpha_epi = -50
    mesh_ID = 1

    # generate.input_EP_param()
    # generate.synthetic_mesh(mesh_ID)
    # generate.EP_pipeline(mesh_ID)

    # prepare_mesh.download_zip(bundle_number)
    # prepare_mesh.unzip_meshes(bundle_number)
    # prepare_mesh.vtk_mm2carp_um(heart)

    # prepare_mesh.extract_LDRB_biv(heart)
    # prepare_mesh.extract_MVTV_base(heart)
    # fibres.run_laplacian(heart)
    # fibres.full_pipeline(heart, alpha_epi, alpha_endo)
    # UVC.create(1, "MVTV")
    UVC.bottom_third(1, "MVTV")
    debug.bottom_third(1)
    # UVC.create_FEC(1, "MVTV")
    # debug.create_FEC(1, "MVTV")
    # run_EP.carp2init(heart, lastFECtag, CV_l, k_fibre, k_FEC)
    # run_EP.launch_init(heart, alpha_endo, alpha_epi)

if __name__ == "__main__":
    main()