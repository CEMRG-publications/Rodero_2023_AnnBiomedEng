#!/usr/bin/env python3

import os
import numpy as np
import time

import prepare_mesh 
import files_manipulations
import fibres
import debug
import UVC
import run_EP
import template_EP
import anatomy

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
    # anatomical_output = generate.anatomical_output("Full_Heart_Mesh_Template_backup",return_ISWT = False, return_WT = False,
    #                   return_EDD = False, return_LVmass = False,
    #                   return_LVendovol = False, close_LV = False,
    #                   return_LVOTdiam = False, close_RV = False,
    #                   return_RVOTdiam = False, close_LA = False,
    #                   return_LAendovol = False, close_RA = False,
    #                   return_RAendovol = False, return_RVlongdiam = False,
    #                   return_RVbasaldiam = False, return_RVendovol = True)
    # [print(key,':',value) for key, value in anatomical_output.items()]
    # print(WT)
    # print(EDD)
    # print(LV_mass)

    # prepare_mesh.download_zip(bundle_number)
    # prepare_mesh.unzip_meshes(bundle_number)
    # prepare_mesh.vtk_mm2carp_um(heart)

    # fibres.FibreCorrection(heart, alpha_epi, alpha_endo)

    # prepare_mesh.extract_LDRB_biv(heart)
    # prepare_mesh.extract_MVTV_base(heart)
    # fibres.run_laplacian(heart)
    # fibres.full_pipeline(heart, alpha_epi, alpha_endo)
    # UVC.create(1, "MVTV")
    # UVC.bottom_third(1, "MVTV")
    # debug.bottom_third(1)
    # UVC.create_FEC(1, "MVTV")
    # debug.create_FEC(1, "MVTV")
    # run_EP.carp2init(heart, lastFECtag, CV_l, k_fibre, k_FEC)
    # run_EP.launch_init(heart, alpha_endo, alpha_epi)
    time_dict = {}
    start = time.time()
    first_start = start

    anatomy.input(n_samples = 1, waveno = 0, subfolder = "anatomy")
    anatomy.build_meshes(waveno = 0, subfolder = "anatomy")

    end = time.time()
    time_dict["Mesh construction"] = end-start

    start = time.time()
    anatomy.EP_setup(waveno = 0, subfolder = "anatomy")
    anatomy.EP_simulations(waveno = 0, subfolder = "anatomy")
    end = time.time()

    time_dict["EP simulation"] = end-start

    start = time.time()
    anatomy.write_output(waveno = 0, subfolder = "anatomy")
    end = time.time()

    time_dict["Output calculation"] = end-start
    time_dict["Total time spent"] = end - first_start

    print(time_dict)


if __name__ == "__main__":
    main()