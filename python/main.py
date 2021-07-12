#!/usr/bin/env python3

import os
import pathlib
import numpy as np
import time
import tqdm

import prepare_mesh 
import files_manipulations
import fibres
import debug
import UVC
import run_EP
import template_EP
import anatomy
import custom_plots
import mechanics_pipeline

from Historia.shared.design_utils import read_labels


def main():

    output_names = ["LVV","RVV","LAV","RAV",
                    "LVOTdiam", "RVOTdiam",
                    "LVmass", "LVWT", "LVEDD", "SeptumWT",
                    "RVlongdiam", "RVbasaldiam",
                    "TAT","TATLVendo"]
    subfolder = "anatomy"
    waveno=0

    outpath = os.path.join("/data", "fitting",subfolder, "wave" + str(waveno))


    with open(os.path.join(outpath,"X_anatomy.csv")) as f:
        anatomy_values = f.read().splitlines()

    mesh_names = ["heart_" + anatomy_values[i+1].replace(",","")[:-27] for i in range(len(anatomy_values)-1)]

    
    for i in tqdm.tqdm(range(1,len(mesh_names))):
        EP_dir = os.path.join("/data","fitting",mesh_names[i],"biv", 
                          "EP_simulations")
        with open(os.path.join(EP_dir,"TAT.dat")) as f:
            output_numbers = f.read().splitlines()

        for var_i, varname in enumerate(output_names):
            np.savetxt(os.path.join(EP_dir, varname + ".dat"),
                        [output_numbers[var_i]],
                        fmt="%s")


if __name__ == "__main__":
    

    final_base_name = "Full_Heart_Mesh_Template_backup"
    # mesh_path = os.path.join("/data","fitting",final_base_name)

    # if not os.path.isfile(os.path.join(mesh_path,"biv","biv_noRVendo.surf.vtx")):
    #     prepare_mesh.extract_LDRB_biv(final_base_name)
    # if not os.path.isfile(os.path.join(mesh_path,"biv","MVTV_base.surf.vtx")):
    #     prepare_mesh.extract_MVTV_base(final_base_name)
    # if not os.path.isfile(os.path.join(mesh_path,"biv","UVC_MVTV", "UVC",
    #                             "COORDS_V_elem_scaled.dat")):
    #     UVC.create(final_base_name, "MVTV")
    # if not os.path.isfile(os.path.join(mesh_path,"biv","EP","bottom_third.vtx")):
    #     UVC.bottom_third(final_base_name, "MVTV")
    # if not os.path.isfile(os.path.join(mesh_path,"biv","biv_FEC.elem")):
    #     UVC.create_FEC(final_base_name, "MVTV")
    # if not os.path.isfile(os.path.join(mesh_path,"biv","fibres","endoRV","phie.igb")):
    #     fibres.run_laplacian(final_base_name)
    # if not os.path.isfile(os.path.join(mesh_path,"pericardium_penalty.dat")):
    #     mechanics_pipeline.penalty_map(fourch_name = final_base_name)
    
    # mechanics_pipeline.boundary_surfaces(fourch_name = final_base_name)

    wave_to_plot = 2
    experiment_name = "anatomy"
    output_labels_dir = os.path.join("/data","fitting",experiment_name,"output_labels.txt")
    xlabels_EP = read_labels(os.path.join("/data","fitting", "EP_funct_labels.txt"))
    xlabels_anatomy = read_labels(os.path.join("/data","fitting", "modes_labels.txt"))
    xlabels = [lab for sublist in [xlabels_anatomy,xlabels_EP] for lab in sublist]

    output_labels = ["TAT", "TATLVendo"]
    # output_labels = ["LVV","RVV","LAV","RAV","LVOTdiam","RVOTdiam","LVmass","LVWT","LVEDD","SeptumWT","RVlongdiam","RVbasaldiam"]
    # output_labels = ["LVV","RVV","LAV","RAV","LVOTdiam","RVOTdiam","LVmass","LVWT","LVEDD","SeptumWT","RVlongdiam","RVbasaldiam","TAT", "TATLVendo"]

    custom_plots.print_ranking(emul_num = wave_to_plot, subfolder = experiment_name,
                        output_labels = output_labels,
                        input_labels = xlabels)
