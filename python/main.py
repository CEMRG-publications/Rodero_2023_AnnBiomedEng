#!/usr/bin/env python3

import os

import prepare_mesh 
import files_manipulations
import fibres

def main():
    bundle_number = "01"
    heart=1
    scripts_folder = "/home/crg17/Desktop/KCL_projects/fitting/python"

    # prepare_mesh.download_zip(bundle_number)
    # prepare_mesh.unzip_meshes(bundle_number)
    # prepare_mesh.vtk_mm2carp_um(heart)
    # prepare_mesh.extract_bdry_bayer(heart)
    # prepare_mesh.map_biv(heart)
    prepare_mesh.extract_LDRB_biv(heart)
    prepare_mesh.extract_MVTV_base(heart)
    fibres.run_laplacian(heart)
    fibres.rb_bayer(heart)

    # whole_pts = files_manipulations.pts.read("/data/fitting/Full_Heart_Mesh_1/Full_Heart_Mesh_1.pts")
    # new_vtx = files_manipulations.read_vtx("/data/fitting/Full_Heart_Mesh_1/biv_epi.surf.vtx")
    # new_pts = files_manipulations.pts.extract(whole_pts,new_vtx)
    # new_pts.write("/data/fitting/Full_Heart_Mesh_1/biv/debug/biv_epi.surf.pts")

if __name__ == "__main__":
    main()