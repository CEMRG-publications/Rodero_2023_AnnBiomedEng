#!/usr/bin/env python3

import os
import numpy as np

import prepare_mesh 
import files_manipulations
import fibres
import debug
import UVC
import run_EP
import generate

def main():
    bundle_number = "01"
    heart="Template"
    scripts_folder = "/home/crg17/Desktop/KCL_projects/fitting/python"
    lastFECtag=30
    CV_l = 0.5
    k_fibre = 0.6
    k_FEC = 6.
    alpha_endo = 50
    alpha_epi = -50
    mesh_ID = "Template"
    UVC_base = "MVTV"

    # generate.template_mesh_setup()
    # goal_training_set_size = 5*40
    # at_least = int(goal_training_set_size/(0.8*0.8))
    # generate.EP_funct_param(at_least)
    # generate.template_EP(50)
    # generate.template_EP_parallel(line_from = 156, line_to = at_least-1, waveno = 0)
    # generate.EP_output(15, waveno = 1)
    # ISWT, WT, EDD, LV_mass = generate.anatomical_output("Full_Heart_Mesh_Template_backup")
    # print(ISWT)
    # print(WT)
    # print(EDD)
    # print(LV_mass)

    # prepare_mesh.download_zip(bundle_number)
    # prepare_mesh.unzip_meshes(bundle_number)
    # prepare_mesh.vtk_mm2carp_um(heart)

    # fibres.FibreCorrection(heart, alpha_epi, alpha_endo)

    # prepare_mesh.extract_LDRB_biv(heart)
    # prepare_mesh.extract_MVTV_base(heart)
    fibres.run_laplacian(heart)
    # fibres.full_pipeline(heart, alpha_epi, alpha_endo)
    # UVC.create(1, "MVTV")
    # UVC.bottom_third(1, "MVTV")
    # debug.bottom_third(1)
    # UVC.create_FEC(1, "MVTV")
    # debug.create_FEC(1, "MVTV")
    # run_EP.carp2init(heart, lastFECtag, CV_l, k_fibre, k_FEC)
    # run_EP.launch_init(heart, alpha_endo, alpha_epi)

if __name__ == "__main__":
    main()