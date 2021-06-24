#!/usr/bin/env python3

import os
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
    # main()
    anatomy.input(n_samples = 2, waveno = 0, subfolder = "testing")
    anatomy.build_meshes(waveno = 0, subfolder = "testing", force_construction = True)