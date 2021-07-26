import wget
import os
import zipfile
import shutil
import glob
import numpy as np
import pathlib

import files_manipulations

PROJECT_PATH = "/data/fitting"


def vtk_mm2carp_um(fourch_name, subfolder="."):
    """Function to convert to carp format and convert the pts to micrometre
    instead of millimetre.

    Args:
        fourch_name (str): Name of the four chamber mesh.
        subfolder: Name of the folder in PROJECT_PATH where we work in.
    """

    mesh_name = fourch_name
    mesh_dir = os.path.join(PROJECT_PATH, subfolder, mesh_name)

    print("meshtool convert -imsh=" + os.path.join(mesh_dir, mesh_name) +
          " -omsh=" + os.path.join(mesh_dir, mesh_name) +
          " -ifmt=vtk -ofmt=carp_txt")

    os.system("meshtool convert -imsh=" + os.path.join(mesh_dir, mesh_name) +
              " -omsh=" + os.path.join(mesh_dir, mesh_name) +
              " -ifmt=vtk -ofmt=carp_txt")

    os.rename(os.path.join(mesh_dir, mesh_name) + ".pts", os.path.join(mesh_dir, mesh_name) + "_mm.pts")

    os.system(os.path.join("/home", "common", "cm2carp", "bin", "return_carp2original_coord.pl ") +
              os.path.join(mesh_dir, mesh_name) + "_mm.pts 1000 0 0 0 > " +
              os.path.join(mesh_dir, mesh_name) + "_um.pts")

    shutil.copy(os.path.join(mesh_dir, mesh_name) + "_um.pts",
                os.path.join(mesh_dir, mesh_name) + ".pts")

    shutil.copy(os.path.join(mesh_dir, mesh_name + ".elem"),
                os.path.join(mesh_dir, mesh_name + "_default.elem"))
    
    os.system("rm " + os.path.join(mesh_dir, mesh_name) + ".vtk")


def extract_ldrb_biv(fourch_name="Full_Heart_Mesh_Template", subfolder="."):
    """Function to extract the boundary conditions for the LDRB method from
    Bayer 2012, except for the base and the apex.

    Args:
        fourch_name (str): Name of the four chamber mesh.
        subfolder: Name of the folder in PROJECT PATH where we work in.
    """

    path2fourch = os.path.join(PROJECT_PATH, subfolder, fourch_name)
    path2biv = os.path.join(path2fourch, "biv")
    path2nomapped = os.path.join(path2biv, "no_mapped")

    os.system("meshtool extract surface -msh=" + os.path.join(path2fourch, fourch_name) +
              " -surf=" + os.path.join(path2fourch, "biv.epi_endo") +
              " -op=1,2-7,8,9,10" +
              " -ifmt=carp_txt -ofmt=vtk_bin")

    pathlib.Path(path2biv).mkdir(parents=True, exist_ok=True)

    os.system("meshtool extract mesh -msh=" + os.path.join(path2fourch, fourch_name) +
              " -submsh=" + os.path.join(path2biv, "biv") + " -tags=1,2 -ifmt=carp_txt -ofmt=carp_txt")

    shutil.copy(os.path.join(path2biv, "biv.elem"),
                os.path.join(path2biv, "biv_default.elem"))

    shutil.copy(os.path.join(path2biv, "biv.pts"),
                os.path.join(path2biv, "biv_default.pts"))

    os.system("meshtool map -submsh=" + os.path.join(path2biv, "biv") +
              " -files=" + os.path.join(path2fourch, "biv.epi_endo.surf") +
              " -outdir=" + path2biv)

    biv_epi_endo_surf = files_manipulations.surf.read(os.path.join(path2biv, "biv.epi_endo.surf"), "biv")

    biv_epi_endo_vtx = biv_epi_endo_surf.tovtx()
    biv_epi_endo_vtx.write(os.path.join(path2biv, "biv.epi_endo.surf.vtx"))

    os.system("meshtool extract tags -msh=" + os.path.join(path2biv, "biv") +
              " -odat=" + os.path.join(path2biv, "biv_tags.dat") +
              " -ifmt=carp_txt")

    shutil.copy(os.path.join(path2biv, "biv.epi_endo.surf"), os.path.join(path2biv, "biv.epi_endo.surf.elem"))
    shutil.copy(os.path.join(path2biv, "biv.pts"), os.path.join(path2biv, "biv.epi_endo.surf.pts"))

    os.system("meshtool interpolate elemdata -omsh=" + os.path.join(path2biv, "biv.epi_endo.surf") +
              " -imsh=" + os.path.join(path2biv, "biv") +
              " -idat=" + os.path.join(path2biv, "biv_tags.dat") +
              " -odat=" + os.path.join(path2biv, "biv.epi_endo_tags.dat"))

    biv_epi_endo_tags = np.loadtxt(os.path.join(path2biv, "biv.epi_endo_tags.dat"))

    biv_epi_endo_surfmesh = files_manipulations.surf(biv_epi_endo_surf.i1,
                                                     biv_epi_endo_surf.i2,
                                                     biv_epi_endo_surf.i3,
                                                     biv_epi_endo_surf.mesh_from,
                                                     biv_epi_endo_tags
                                                     )

    biv_epi_endo_surfmesh.write(os.path.join(path2biv, "biv.epi_endo.elem"))
    shutil.copy(os.path.join(path2biv, "biv.pts"), os.path.join(path2biv, "biv.epi_endo.pts"))
    
    pathlib.Path(path2nomapped).mkdir(parents=True, exist_ok=True)

    os.system("meshtool extract unreachable -msh=" + os.path.join(path2biv, "biv.epi_endo") +
              " -submsh=" + os.path.join(path2nomapped, "biv.epi_endo_split") +
              " -ifmt=carp_txt -ofmt=carp_txt")

    epi_or_endo_files = glob.glob(os.path.join(path2nomapped, "*part*elem"))

    # We want only the three biggest files. The biggest will be the epi, and 
    # depending on the tags, the others will be the LV or the RV.

    size_files = [os.path.getsize(f) for f in epi_or_endo_files]

    while len(size_files) > 3:
        idx_min = size_files.index(min(size_files))
        epi_or_endo_files.pop(idx_min)
        size_files.pop(idx_min)

    idx_max = size_files.index(max(size_files))

    shutil.copy(epi_or_endo_files[idx_max], os.path.join(path2nomapped, "biv.epi.surfmesh"))

    name = epi_or_endo_files[idx_max]
    name_no_ext = name.split(".elem")[0]

    shutil.copy(name, os.path.join(path2nomapped, "biv.epi.elem"))
    shutil.copy(name, os.path.join(path2nomapped, "biv.epi.surf"))
    shutil.copy(name_no_ext + ".lon", os.path.join(path2nomapped, "biv.epi.lon"))
    shutil.copy(name_no_ext + ".pts", os.path.join(path2nomapped, "biv.epi.pts"))
    shutil.copy(name_no_ext + ".nod", os.path.join(path2nomapped, "biv.epi.nod"))
    shutil.copy(name_no_ext + ".eidx", os.path.join(path2nomapped, "biv.epi.eidx"))
    
    epi_or_endo_files.pop(idx_max)
    size_files.pop(idx_max)

    for i in range(len(epi_or_endo_files)):
        name = epi_or_endo_files[i]
        name_no_ext = name.split(".elem")[0]
        file_name = name_no_ext.split("/")[-1]

        os.system("meshtool extract tags -msh=" + name_no_ext +
                  " -odat=" + os.path.join(path2nomapped, file_name) + ".tags" +
                  " -ifmt=carp_txt")
        tag_file = np.loadtxt(os.path.join(path2nomapped, file_name) + ".tags")

        if int(sum(tag_file)) != len(tag_file):
            shutil.copy(name, os.path.join(path2nomapped, "biv.rvendo.surf"))
            shutil.copy(name, os.path.join(path2nomapped, "biv.rvendo.surfmesh"))
            shutil.copy(name_no_ext + ".pts", os.path.join(path2nomapped, "biv.rvendo.pts"))
            shutil.copy(name_no_ext + ".nod", os.path.join(path2nomapped, "biv.rvendo.nod"))
            shutil.copy(name_no_ext + ".eidx", os.path.join(path2nomapped, "biv.rvendo.eidx"))

            epi_or_endo_files.pop(i)
            size_files.pop(i)
            break

    # We extract the biv_rvendo_nosept and the septum as well for the UVC

    biv_rvendo = files_manipulations.surf.read(os.path.join(path2nomapped, "biv.rvendo.surfmesh"), "biv.rvendo")

    biv_rvendo_nosept = files_manipulations.surf.extract(biv_rvendo, np.where(biv_rvendo.tags == 2))
    biv_rvsept = files_manipulations.surf.extract(biv_rvendo, np.where(biv_rvendo.tags == 1))

    biv_rvendo_nosept.write(os.path.join(path2nomapped, "biv.rvendo_nosept.surf"))
    shutil.copy(os.path.join(path2nomapped, "biv.rvendo.pts"),
                os.path.join(path2nomapped, "biv.rvendo_nosept.pts"))
    shutil.copy(os.path.join(path2nomapped, "biv.rvendo.nod"),
                os.path.join(path2nomapped, "biv.rvendo_nosept.nod"))
    shutil.copy(os.path.join(path2nomapped, "biv.rvendo.eidx"),
                os.path.join(path2nomapped, "biv.rvendo_nosept.eidx"))

    biv_rvsept.write(os.path.join(path2nomapped, "biv.rvsept.surf"))
    shutil.copy(os.path.join(path2nomapped, "biv.rvendo.pts"),
                os.path.join(path2nomapped, "biv.rvsept.pts"))
    shutil.copy(os.path.join(path2nomapped, "biv.rvendo.nod"),
                os.path.join(path2nomapped, "biv.rvsept.nod"))
    shutil.copy(os.path.join(path2nomapped, "biv.rvendo.eidx"),
                os.path.join(path2nomapped, "biv.rvsept.eidx"))

    ####

    idx_max = size_files.index(max(size_files))

    name = epi_or_endo_files[idx_max]
    name_no_ext = name.split(".elem")[0]

    shutil.copy(name, os.path.join(path2nomapped, "biv.lvendo.surf"))
    shutil.copy(name_no_ext + ".pts", os.path.join(path2nomapped, "biv.lvendo.pts"))
    shutil.copy(name_no_ext + ".nod", os.path.join(path2nomapped, "biv.lvendo.nod"))
    shutil.copy(name_no_ext + ".eidx", os.path.join(path2nomapped, "biv.lvendo.eidx"))

    #  Map back to the biv

    for surfname in ["biv.epi", "biv.lvendo", "biv.rvendo",
                     "biv.rvendo_nosept", "biv.rvsept"]:

        os.system("meshtool map -submsh=" + os.path.join(path2nomapped, surfname) +
                  " -files=" + os.path.join(path2nomapped, surfname + ".surf") +
                  " -outdir=" + path2biv + " -mode=s2m")
        
        file_surf = files_manipulations.surf.read(os.path.join(path2biv, surfname + ".surf"), mesh_from="biv")
        files_surfmesh = files_manipulations.surf.tosurfmesh(file_surf)
        files_surfmesh.write(os.path.join(path2biv, surfname + ".surfmesh"))

    # All the surfs

    biv_lvendo_surfmesh = files_manipulations.surf.read(os.path.join(path2biv, "biv.lvendo.surfmesh"), mesh_from="biv")
    biv_rvendo_surfmesh = files_manipulations.surf.read(os.path.join(path2biv, "biv.rvendo.surfmesh"), mesh_from="biv")
    biv_epi_surfmesh = files_manipulations.surf.read(os.path.join(path2biv, "biv.epi.surfmesh"), mesh_from="biv")
    biv_rvendo_nosept_surfmesh = files_manipulations.surf.read(os.path.join(path2biv, "biv.rvendo_nosept.surfmesh"),
                                                               mesh_from="biv")
    biv_rvsept_surfmesh = files_manipulations.surf.read(os.path.join(path2biv, "biv.rvsept.surfmesh"), mesh_from="biv")

    biv_lvendo_surf = files_manipulations.surf.tosurf(biv_lvendo_surfmesh)
    biv_rvendo_surf = files_manipulations.surf.tosurf(biv_rvendo_surfmesh)
    biv_epi_surf = files_manipulations.surf.tosurf(biv_epi_surfmesh)
    biv_rvendo_nosept_surf = files_manipulations.surf.tosurf(biv_rvendo_nosept_surfmesh)
    biv_rvsept_surf = files_manipulations.surf.tosurf(biv_rvsept_surfmesh)

    biv_endo_surf = files_manipulations.surf.merge(biv_lvendo_surf, biv_rvendo_surf)
    biv_nolvendo_surf = files_manipulations.surf.merge(biv_epi_surf, biv_rvendo_surf)
    biv_norvendo_surf = files_manipulations.surf.merge(biv_epi_surf, biv_lvendo_surf)

    # The corresponding vtx

    biv_lvendo_vtx = files_manipulations.surf.tovtx(biv_lvendo_surf)
    biv_rvendo_vtx = files_manipulations.surf.tovtx(biv_rvendo_surf)
    biv_rvendo_nosept_vtx = files_manipulations.surf.tovtx(biv_rvendo_nosept_surf)
    biv_rvsept_vtx = files_manipulations.surf.tovtx(biv_rvsept_surf)
    biv_endo_vtx = files_manipulations.surf.tovtx(biv_endo_surf)
    biv_epi_vtx = files_manipulations.surf.tovtx(biv_epi_surf)
    biv_nolvendo_vtx = files_manipulations.surf.tovtx(biv_nolvendo_surf)
    biv_norvendo_vtx = files_manipulations.surf.tovtx(biv_norvendo_surf)

    # Write everything

    biv_lvendo_surf.write(os.path.join(path2biv, "biv.lvendo.surf"))
    biv_rvendo_surf.write(os.path.join(path2biv, "biv.rvendo.surf"))
    biv_rvendo_nosept_surf.write(os.path.join(path2biv, "biv.rvendo_nosept.surf"))
    biv_rvsept_surf.write(os.path.join(path2biv, "biv.rvsept.surf"))
    biv_endo_surf.write(os.path.join(path2biv, "biv.endo.surf"))
    biv_epi_surf.write(os.path.join(path2biv, "biv.epi.surf"))
    biv_nolvendo_surf.write(os.path.join(path2biv, "biv_noLVendo.surf"))
    biv_norvendo_surf.write(os.path.join(path2biv, "biv_noRVendo.surf"))

    biv_lvendo_vtx.write(os.path.join(path2biv, "biv.lvendo.surf.vtx"))
    biv_rvendo_vtx.write(os.path.join(path2biv, "biv.rvendo.surf.vtx"))
    biv_rvendo_nosept_vtx.write(os.path.join(path2biv, "biv.rvendo_nosept.surf.vtx"))
    biv_rvsept_vtx.write(os.path.join(path2biv, "biv.rvsept.surf.vtx"))
    biv_endo_vtx.write(os.path.join(path2biv, "biv.endo.surf.vtx"))
    biv_epi_vtx.write(os.path.join(path2biv, "biv.epi.surf.vtx"))
    biv_nolvendo_vtx.write(os.path.join(path2biv, "biv_noLVendo.surf.vtx"))
    biv_norvendo_vtx.write(os.path.join(path2biv, "biv_noRVendo.surf.vtx"))


def extract_mvtv_base(fourch_name = "Full_Heart_Mesh_Template", subfolder="."):
    """Function to extract the base and the apex as boundary conditions for the
    LDRB method from Bayer 2012.

    Args:
        fourch_name (str): Name of the four chamber mesh.
    """
    path2fourch = os.path.join(PROJECT_PATH,subfolder,fourch_name)
    path2biv = path2fourch + "/biv"

    os.system("meshtool extract surface -msh=" + os.path.join(path2fourch, fourch_name) +
              " -surf=" + os.path.join(path2fourch, "mvtv_base") +
              " -op=1,2:7,8" +
              " -ifmt=carp_txt" +
              " -ofmt=carp_txt")

    os.system("meshtool extract surface -msh=" + os.path.join(path2fourch, fourch_name) +
              " -surf=" + os.path.join(path2fourch, "mv") +
              " -op=7-1,3" +
              " -ifmt=carp_txt" +
              " -ofmt=carp_txt")

    mitral_valve = np.loadtxt(os.path.join(path2fourch,"mv.surfmesh.pts"), skiprows = 1)

    num_pts = mitral_valve.shape[0]

    sum_x = np.sum(mitral_valve[:, 0])
    sum_y = np.sum(mitral_valve[:, 1])
    sum_z = np.sum(mitral_valve[:, 2])

    centroid = np.array([sum_x/num_pts, sum_y/num_pts, sum_z/num_pts])

    os.system("meshtool interpolate elem2node -omsh=" + os.path.join(path2biv, "biv.epi_endo") +
              " -idat=" + os.path.join(path2biv, "biv.epi_endo_tags.dat") +
              " -odat=" + os.path.join(path2biv, "biv.epi_endo_tags_pts.dat")
              )

    biv_epi_endo_pts = files_manipulations.pts.read(os.path.join(path2biv, "biv.epi_endo.pts"))
    biv_epi_endo_tags_pts = np.loadtxt(os.path.join(path2biv, "biv.epi_endo_tags_pts.dat"), skiprows=1)

    dist_vec = np.zeros(len(biv_epi_endo_tags_pts))

    for i in range(len(biv_epi_endo_tags_pts)):
        if biv_epi_endo_tags_pts[i] == 1:
            new_point = np.array([biv_epi_endo_pts.p1[i], biv_epi_endo_pts.p2[i], biv_epi_endo_pts.p3[i]])
            dist_vec[i] = np.linalg.norm(centroid - new_point)

    idx_apex_in_epi_vec = np.where(dist_vec == max(dist_vec))
    idx_apex_in_epi = idx_apex_in_epi_vec[0]

    apex_vtx = files_manipulations.vtx(idx_apex_in_epi, "biv")
    apex_vtx.write(os.path.join(path2biv,"LV_apex_epi.vtx"))

    os.system("meshtool map -submsh=" + os.path.join(path2biv,"biv") + \
                           " -files=" + os.path.join(path2fourch,"mvtv_base.surf") + "," + \
                                        os.path.join(path2fourch,"mvtv_base.surf.vtx") + "," + \
                            " -outdir=" + path2biv)

    for vtx_file in ["mvtv_base.surf.vtx"]:
        vtx = files_manipulations.vtx.read(os.path.join(path2biv, vtx_file), "biv")
        vtx.write(os.path.join(path2biv,vtx_file))


def extract_peri_base(fourch_name, subfolder="."):
    
    path2fourch = os.path.join(PROJECT_PATH, subfolder, fourch_name)
    path2biv = os.path.join(path2fourch, "biv")

    # The base is intersection of the ventricles with the aorta, MV, TV, PV and AV

    os.system("meshtool extract surface -msh=" + os.path.join(path2fourch, fourch_name) +
              " -surf=" + os.path.join(path2fourch, "peri_base") +
              " -op=1,2,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39:5,7,8,9,10" +
              " -ifmt=carp_txt" +
              " -ofmt=carp_txt")

    os.system("meshtool map -submsh=" + os.path.join(path2biv, "biv") +
              " -files=" + os.path.join(path2fourch, "peri_base.surf") + "," +
              os.path.join(path2fourch, "peri_base.surf.vtx") + "," +
              " -outdir=" + path2biv)

    for vtx_file in ["peri_base.surf.vtx"]:
        vtx = files_manipulations.vtx.read(os.path.join(path2biv, vtx_file), "biv")
        vtx.write(os.path.join(path2biv, vtx_file))


def close_lv_endo(fourch_name, subfolder="."):
    """Function to generate the closed surface of the LV endocardium.

    Args:
        fourch_name (str): Name of the four chamber mesh file.
        subfolder: Name of the folder in PROJECT_PATH where we work in.
    """

    path2fourch = os.path.join(PROJECT_PATH, subfolder, fourch_name)
    path2biv = os.path.join(path2fourch, "biv")
    path2nomapped = os.path.join(path2fourch, "no_mapped")

    pathlib.Path(path2nomapped).mkdir(parents=True, exist_ok=True)

    os.system("meshtool map -submsh=" + os.path.join(path2biv, "biv") +
              " -files=" + os.path.join(path2biv, "biv.lvendo.surf") +
              " -outdir=" + path2fourch + " -mode=s2m")

    biv_lvendo_surf = files_manipulations.surf.read(os.path.join(path2fourch, "biv.lvendo.surf"),
                                                    mesh_from=fourch_name)
    lvendo_open_surfmesh = files_manipulations.surf.tosurfmesh(biv_lvendo_surf)
    lvendo_open_surfmesh.write(os.path.join(path2fourch, "lvendo_open.surfmesh.elem"))

    shutil.copy(os.path.join(path2biv, "biv.pts"),
                os.path.join(path2fourch, "lvendo_open.surfmesh.pts"))

    shutil.copy(os.path.join(path2fourch, fourch_name + ".pts"),
                os.path.join(path2fourch, "mv_mapped.pts"))

    shutil.copy(os.path.join(path2fourch, "mv.surf"),
                os.path.join(path2fourch, "mv_mapped.elem"))

    mv_mapped_surf = files_manipulations.surf.read(os.path.join(path2fourch, "mv_mapped.elem"), fourch_name)
    mv_mapped = files_manipulations.surf.tosurfmesh(mv_mapped_surf)

    lvendo_floating_mv_surfmesh = files_manipulations.surf.merge(lvendo_open_surfmesh, mv_mapped)
    lvendo_floating_mv_surfmesh.write(os.path.join(path2fourch, "lvendo_floating_mv.surfmesh.elem"))

    shutil.copy(os.path.join(path2fourch, fourch_name + ".pts"),
                os.path.join(path2fourch, "lvendo_floating_mv.surfmesh.pts"))

    os.system("meshtool extract unreachable -msh=" + os.path.join(path2fourch, "lvendo_floating_mv.surfmesh") +
              " -submsh=" + os.path.join(path2nomapped, "lvendo_mv_split.surfmesh") +
              " -ifmt=carp_txt" +
              " -ofmt=vtk_bin")

    chamber_or_valve_files = glob.glob(os.path.join(path2nomapped, "lvendo_mv_split.surfmesh*part*vtk"))

    size_files = [os.path.getsize(f) for f in chamber_or_valve_files]

    idx_chamber = size_files.index(max(size_files))
    idx_valve = size_files.index(max(size_files[:idx_chamber] + size_files[(idx_chamber + 1):]))

    shutil.copy(chamber_or_valve_files[idx_chamber], os.path.join(path2nomapped, "lvendo_mv.surfmesh.vtk"))
    shutil.copy(chamber_or_valve_files[idx_chamber][:-3] + "nod", os.path.join(path2nomapped, "lvendo_mv.surfmesh.nod"))

    shutil.copy(chamber_or_valve_files[idx_valve], os.path.join(path2nomapped, "mv_la.surfmesh.vtk"))
    shutil.copy(chamber_or_valve_files[idx_valve][:-3] + "nod", os.path.join(path2nomapped, "mv_la.surfmesh.nod"))

    os.system("meshtool convert -imsh=" + os.path.join(path2nomapped, "lvendo_mv.surfmesh.vtk") +
              " -omsh=" + os.path.join(path2nomapped, "lvendo_mv.surfmesh") +
              " -ifmt=vtk_bin -ofmt=carp_txt")

    os.system("meshtool convert -imsh=" + os.path.join(path2nomapped, "mv_la.surfmesh.vtk") +
              " -omsh=" + os.path.join(path2nomapped, "mv_la.surfmesh") +
              " -ifmt=vtk_bin -ofmt=carp_txt")

    lvendo_mv_surfmesh = files_manipulations.surf.read(os.path.join(path2nomapped, "lvendo_mv.surfmesh.elem"),
                                                       "lvendo_floating_mv.surfmesh")
    lvendo_mv_surf = files_manipulations.surf.tosurf(lvendo_mv_surfmesh)
    lvendo_mv_surf.write(os.path.join(path2nomapped, "lvendo_mv.surf"))

    os.system("meshtool map -submsh=" + os.path.join(path2nomapped, "lvendo_mv.surfmesh") +
              " -files=" + os.path.join(path2nomapped, "lvendo_mv.surf") +
              " -outdir=" + path2fourch +
              " -mode=s2m")

    mv_la_surfmesh = files_manipulations.surf.read(os.path.join(path2nomapped, "mv_la.surfmesh.elem"),
                                                   "mv_la.surfmesh")
    mv_la_surf = files_manipulations.surf.tosurf(mv_la_surfmesh)
    mv_la_surf.write(os.path.join(path2nomapped, "mv_la.surf"))

    os.system("meshtool map -submsh=" + os.path.join(path2nomapped, "mv_la.surfmesh") +
              " -files=" + os.path.join(path2nomapped, "mv_la.surf") +
              " -outdir=" + path2fourch +
              " -mode=s2m")

    # We repeat the process for the aortic valve.

    os.system("meshtool extract surface -msh=" + os.path.join(path2fourch, fourch_name) +
              " -surf=" + os.path.join(path2fourch, "av") +
              " -op=9-1,5 -ifmt=carp_txt -ofmt=carp_txt")

    shutil.copy(os.path.join(path2fourch, fourch_name + ".pts"),
                os.path.join(path2fourch, "av_mapped.pts"))

    shutil.copy(os.path.join(path2fourch, "av.surf"),
                os.path.join(path2fourch, "av_mapped.elem"))

    av_mapped_surf = files_manipulations.surf.read(os.path.join(path2fourch, "av_mapped.elem"), fourch_name)
    av_mapped = files_manipulations.surf.tosurfmesh(av_mapped_surf)

    lvendo_mv_surf = files_manipulations.surf.read(os.path.join(path2fourch, "lvendo_mv.surf"), fourch_name)
    lvendo_mv_surfmesh = files_manipulations.surf.tosurfmesh(lvendo_mv_surf)

    lvendo_floating_av_surfmesh = files_manipulations.surf.merge(lvendo_mv_surfmesh, av_mapped)
    lvendo_floating_av_surfmesh.write(os.path.join(path2fourch, "lvendo_floating_av.surfmesh.elem"))

    shutil.copy(os.path.join(path2fourch, fourch_name + ".pts"),
                os.path.join(path2fourch, "lvendo_floating_av.surfmesh.pts"))

    os.system("meshtool extract unreachable -msh=" + os.path.join(path2fourch, "lvendo_floating_av.surfmesh") +
              " -submsh=" + os.path.join(path2nomapped, "lvendo_av_split.surfmesh") +
              " -ifmt=carp_txt" +
              " -ofmt=vtk_bin")

    chamber_or_valve_files = glob.glob(os.path.join(path2nomapped, "lvendo_av_split.surfmesh*part*vtk"))

    size_files = [os.path.getsize(f) for f in chamber_or_valve_files]

    idx_chamber = size_files.index(max(size_files))
    idx_valve = size_files.index(max(size_files[:idx_chamber] + size_files[(idx_chamber + 1):]))

    shutil.copy(chamber_or_valve_files[idx_chamber], os.path.join(path2nomapped, "lvendo_closed.surfmesh.vtk"))
    shutil.copy(chamber_or_valve_files[idx_chamber][:-3] + "nod", os.path.join(path2nomapped,
                                                                               "lvendo_closed.surfmesh.nod"))

    shutil.copy(chamber_or_valve_files[idx_valve], os.path.join(path2nomapped, "av_ao.surfmesh.vtk"))
    shutil.copy(chamber_or_valve_files[idx_valve][:-3] + "nod", os.path.join(path2nomapped, "av_ao.surfmesh.nod"))

    os.system("meshtool convert -imsh=" + os.path.join(path2nomapped, "lvendo_closed.surfmesh.vtk") +
              " -omsh=" + os.path.join(path2nomapped, "lvendo_closed.surfmesh") +
              " -ifmt=vtk_bin -ofmt=carp_txt")

    os.system("meshtool convert -imsh=" + os.path.join(path2nomapped, "av_ao.surfmesh.vtk") +
              " -omsh=" + os.path.join(path2nomapped, "av_ao.surfmesh") +
              " -ifmt=vtk_bin -ofmt=carp_txt")

    lvendo_closed_surfmesh = files_manipulations.surf.read(os.path.join(path2nomapped, "lvendo_closed.surfmesh.elem"),
                                                       "lvendo_floating_av.surfmesh")
    lvendo_closed_surf = files_manipulations.surf.tosurf(lvendo_closed_surfmesh)
    lvendo_closed_surf.write(os.path.join(path2nomapped, "lvendo_closed.surf"))

    os.system("meshtool map -submsh=" + os.path.join(path2nomapped, "lvendo_closed.surfmesh") +
              " -files=" + os.path.join(path2nomapped, "lvendo_closed.surf") +
              " -outdir=" + path2fourch +
              " -mode=s2m")

    av_ao_surfmesh = files_manipulations.surf.read(os.path.join(path2nomapped, "av_ao.surfmesh.elem"),
                                                   "av_ao.surfmesh")
    av_ao_surf = files_manipulations.surf.tosurf(av_ao_surfmesh)
    av_ao_surf.write(os.path.join(path2nomapped, "av_ao.surf"))

    os.system("meshtool map -submsh=" + os.path.join(path2nomapped, "av_ao.surfmesh") +
              " -files=" + os.path.join(path2nomapped, "av_ao.surf") +
              " -outdir=" + path2fourch +
              " -mode=s2m")


def close_rv_endo(fourch_name, subfolder="."):
    """Function to generate the closed surface of the RV endocardium.

    Args:
        fourch_name (str): Name of the four chamber mesh file.
        subfolder: Name of the folder in PROJECT_PATH where we work in.
    """

    path2fourch = os.path.join(PROJECT_PATH, subfolder, fourch_name)
    path2biv = os.path.join(path2fourch, "biv")
    path2nomapped = os.path.join(path2fourch, "no_mapped")

    pathlib.Path(path2nomapped).mkdir(parents=True, exist_ok=True)

    os.system("meshtool map -submsh=" + os.path.join(path2biv, "biv") +
              " -files=" + os.path.join(path2biv, "biv.rvendo.surf") +
              " -outdir=" + path2fourch + " -mode=s2m")

    biv_rvendo_surf = files_manipulations.surf.read(os.path.join(path2fourch, "biv.rvendo.surf"),
                                                    mesh_from=fourch_name)
    rvendo_open_surfmesh = files_manipulations.surf.tosurfmesh(biv_rvendo_surf)
    rvendo_open_surfmesh.write(os.path.join(path2fourch, "rvendo_open.surfmesh.elem"))

    shutil.copy(os.path.join(path2biv, "biv.pts"),
                os.path.join(path2fourch, "rvendo_open.surfmesh.pts"))

    os.system("meshtool extract surface -msh=" + os.path.join(path2fourch, fourch_name) +
              " -surf=" + os.path.join(path2fourch, "tv") +
              " -op=8-2,4" +
              " -ifmt=carp_txt -ofmt=carp_txt")

    shutil.copy(os.path.join(path2fourch, fourch_name + ".pts"),
                os.path.join(path2fourch, "tv_mapped.pts"))

    shutil.copy(os.path.join(path2fourch, "tv.surf"),
                os.path.join(path2fourch, "tv_mapped.elem"))

    tv_mapped_surf = files_manipulations.surf.read(os.path.join(path2fourch, "tv_mapped.elem"), fourch_name)
    tv_mapped = files_manipulations.surf.tosurfmesh(tv_mapped_surf)

    rvendo_floating_tv_surfmesh = files_manipulations.surf.merge(rvendo_open_surfmesh, tv_mapped)
    rvendo_floating_tv_surfmesh.write(os.path.join(path2fourch, "rvendo_floating_tv.surfmesh.elem"))

    shutil.copy(os.path.join(path2fourch, fourch_name + ".pts"),
                os.path.join(path2fourch, "rvendo_floating_tv.surfmesh.pts"))

    os.system("meshtool extract unreachable -msh=" + os.path.join(path2fourch, "rvendo_floating_tv.surfmesh") +
              " -submsh=" + os.path.join(path2nomapped, "rvendo_tv_split.surfmesh") +
              " -ifmt=carp_txt" +
              " -ofmt=vtk_bin")

    chamber_or_valve_files = glob.glob(os.path.join(path2nomapped, "rvendo_tv_split.surfmesh*part*vtk"))

    size_files = [os.path.getsize(f) for f in chamber_or_valve_files]

    idx_chamber = size_files.index(max(size_files))
    idx_valve = size_files.index(max(size_files[:idx_chamber] + size_files[(idx_chamber + 1):]))

    shutil.copy(chamber_or_valve_files[idx_chamber], os.path.join(path2nomapped, "rvendo_tv.surfmesh.vtk"))
    shutil.copy(chamber_or_valve_files[idx_chamber][:-3] + "nod", os.path.join(path2nomapped,
                                                                               "rvendo_tv.surfmesh.nod"))

    shutil.copy(chamber_or_valve_files[idx_valve], os.path.join(path2nomapped, "tv_ra.surfmesh.vtk"))
    shutil.copy(chamber_or_valve_files[idx_valve][:-3] + "nod", os.path.join(path2nomapped, "tv_ra.surfmesh.nod"))

    os.system("meshtool convert -imsh=" + os.path.join(path2nomapped, "rvendo_tv.surfmesh.vtk") +
              " -omsh=" + os.path.join(path2nomapped, "rvendo_tv.surfmesh") +
              " -ifmt=vtk_bin -ofmt=carp_txt")

    os.system("meshtool convert -imsh=" + os.path.join(path2nomapped, "tv_ra.surfmesh.vtk") +
              " -omsh=" + os.path.join(path2nomapped, "tv_ra.surfmesh") +
              " -ifmt=vtk_bin -ofmt=carp_txt")

    rvendo_tv_surfmesh = files_manipulations.surf.read(os.path.join(path2nomapped, "rvendo_tv.surfmesh.elem"),
                                                       "rvendo_floating_tv.surfmesh")
    rvendo_tv_surf = files_manipulations.surf.tosurf(rvendo_tv_surfmesh)
    rvendo_tv_surf.write(os.path.join(path2nomapped, "rvendo_tv.surf"))

    os.system("meshtool map -submsh=" + os.path.join(path2nomapped, "rvendo_tv.surfmesh") +
              " -files=" + os.path.join(path2nomapped, "rvendo_tv.surf") +
              " -outdir=" + path2fourch +
              " -mode=s2m")

    tv_ra_surfmesh = files_manipulations.surf.read(os.path.join(path2nomapped, "tv_ra.surfmesh.elem"),
                                                   "tv_ra.surfmesh")
    tv_ra_surf = files_manipulations.surf.tosurf(tv_ra_surfmesh)
    tv_ra_surf.write(os.path.join(path2nomapped, "tv_ra.surf"))

    os.system("meshtool map -submsh=" + os.path.join(path2nomapped, "tv_ra.surfmesh") +
              " -files=" + os.path.join(path2nomapped, "tv_ra.surf") +
              " -outdir=" + path2fourch +
              " -mode=s2m")

    # We repeat the process for the pulmonary valve.

    os.system("meshtool extract surface -msh=" + os.path.join(path2fourch, fourch_name) +
              " -surf=" + os.path.join(path2fourch, "pv") +
              " -op=10-2,6 -ifmt=carp_txt -ofmt=carp_txt")

    shutil.copy(os.path.join(path2fourch, fourch_name + ".pts"),
                os.path.join(path2fourch, "pv_mapped.pts"))

    shutil.copy(os.path.join(path2fourch, "pv.surf"),
                os.path.join(path2fourch, "pv_mapped.elem"))

    pv_mapped_surf = files_manipulations.surf.read(os.path.join(path2fourch, "pv_mapped.elem"), fourch_name)
    pv_mapped = files_manipulations.surf.tosurfmesh(pv_mapped_surf)

    rvendo_tv_surf = files_manipulations.surf.read(os.path.join(path2fourch, "rvendo_tv.surf"), fourch_name)
    rvendo_tv_surfmesh = files_manipulations.surf.tosurfmesh(rvendo_tv_surf)

    rvendo_floating_pv_surfmesh = files_manipulations.surf.merge(rvendo_tv_surfmesh, pv_mapped)
    rvendo_floating_pv_surfmesh.write(os.path.join(path2fourch, "rvendo_floating_pv.surfmesh.elem"))

    shutil.copy(os.path.join(path2fourch, fourch_name + ".pts"),
                os.path.join(path2fourch, "rvendo_floating_pv.surfmesh.pts"))

    os.system("meshtool extract unreachable -msh=" + os.path.join(path2fourch, "rvendo_floating_pv.surfmesh") +
              " -submsh=" + os.path.join(path2nomapped, "rvendo_pv_split.surfmesh") +
              " -ifmt=carp_txt" +
              " -ofmt=vtk_bin")

    chamber_or_valve_files = glob.glob(os.path.join(path2nomapped, "rvendo_pv_split.surfmesh*part*vtk"))

    size_files = [os.path.getsize(f) for f in chamber_or_valve_files]

    idx_chamber = size_files.index(max(size_files))
    idx_valve = size_files.index(max(size_files[:idx_chamber] + size_files[(idx_chamber + 1):]))

    shutil.copy(chamber_or_valve_files[idx_chamber], os.path.join(path2nomapped, "rvendo_closed.surfmesh.vtk"))
    shutil.copy(chamber_or_valve_files[idx_chamber][:-3] + "nod", os.path.join(path2nomapped,
                                                                               "rvendo_closed.surfmesh.nod"))

    shutil.copy(chamber_or_valve_files[idx_valve], os.path.join(path2nomapped, "pv_pa.surfmesh.vtk"))
    shutil.copy(chamber_or_valve_files[idx_valve][:-3] + "nod", os.path.join(path2nomapped, "pv_pa.surfmesh.nod"))

    os.system("meshtool convert -imsh=" + os.path.join(path2nomapped, "rvendo_closed.surfmesh.vtk") +
              " -omsh=" + os.path.join(path2nomapped, "rvendo_closed.surfmesh") +
              " -ifmt=vtk_bin -ofmt=carp_txt")

    os.system("meshtool convert -imsh=" + os.path.join(path2nomapped, "pv_pa.surfmesh.vtk") +
              " -omsh=" + os.path.join(path2nomapped, "pv_pa.surfmesh") +
              " -ifmt=vtk_bin -ofmt=carp_txt")

    rvendo_closed_surfmesh = files_manipulations.surf.read(os.path.join(path2nomapped,
                                                                        "rvendo_closed.surfmesh.elem"),
                                                           "rvendo_floating_pv.surfmesh")
    rvendo_closed_surf = files_manipulations.surf.tosurf(rvendo_closed_surfmesh)
    rvendo_closed_surf.write(os.path.join(path2nomapped, "rvendo_closed.surf"))

    os.system("meshtool map -submsh=" + os.path.join(path2nomapped, "rvendo_closed.surfmesh") +
              " -files=" + os.path.join(path2nomapped, "rvendo_closed.surf") +
              " -outdir=" + path2fourch +
              " -mode=s2m")

    pv_pa_surfmesh = files_manipulations.surf.read(os.path.join(path2nomapped, "pv_pa.surfmesh.elem"),
                                                   "pv_pa.surfmesh")
    pv_pa_surf = files_manipulations.surf.tosurf(pv_pa_surfmesh)
    pv_pa_surf.write(os.path.join(path2nomapped, "pv_pa.surf"))

    os.system("meshtool map -submsh=" + os.path.join(path2nomapped, "pv_pa.surfmesh") +
              " -files=" + os.path.join(path2nomapped, "pv_pa.surf") +
              " -outdir=" + path2fourch +
              " -mode=s2m")


def close_LA_endo(fourch_name, subfolder="."):
    """WARNING: Mapping is not correct. Mimic what's done in close_lv_endo and close_rv_endo
    Function to generate the closed surface of the LA endocardium.

    Args:
        fourch_name (str): Name of the four chamber mesh file.
    """
    ## ATRIA: Extract the surfaces with the veins and the valve. The biggest
    # file will be the epi with the veins. The second will be the endo closed and
    # the third will be the valve in the ventricle.
    pass
    # path2fourch = os.path.join(PROJECT_PATH,subfolder,fourch_name)
    #
    # os.system("meshtool extract surface -msh=" + os.path.join(path2fourch,fourch_name) +\
    #           " -surf=" + os.path.join(path2fourch,"la_epi_endo_floating_mv") +\
    #           " -op=3,7,11,12,13,14,15,18,19,20,21,22-1 -ifmt=carp_txt -ofmt=carp_txt")
    #
    # os.system("meshtool extract unreachable -msh=" + os.path.join(path2fourch,"la_epi_endo_floating_mv.surfmesh") + \
    #                         " -submsh=" + os.path.join(path2fourch,"la_epi_endo_mv_split") + \
    #                         " -ifmt=carp_txt" + \
    #                         " -ofmt=carp_txt")
    #
    # chamber_or_valve_files = glob.glob(os.path.join(path2fourch,"la_epi_endo_mv_split*part*elem"))
    #
    # size_files = [os.path.getsize(f) for f in chamber_or_valve_files]
    #
    # idx_epi = np.argsort(size_files)[-1]
    # idx_endo = np.argsort(size_files)[-2]
    # idx_valve = np.argsort(size_files)[-3]
    #
    # shutil.copy(chamber_or_valve_files[idx_endo], os.path.join(path2fourch, "laendo_closed.elem"))
    # shutil.copy(chamber_or_valve_files[idx_endo][:-5] + ".pts", os.path.join(path2fourch, "laendo_closed.pts"))
    #
    # shutil.copy(chamber_or_valve_files[idx_epi], os.path.join(path2fourch, "laepi.elem"))
    # shutil.copy(chamber_or_valve_files[idx_epi][:-5] + ".pts", os.path.join(path2fourch, "laepi.pts"))
    #
    # shutil.copy(chamber_or_valve_files[idx_valve], os.path.join(path2fourch, "mv_lv.elem"))
    # shutil.copy(chamber_or_valve_files[idx_valve][:-5] + ".pts", os.path.join(path2fourch, "mv_lv.pts"))


def close_RA_endo(fourch_name, subfolder="."):
    """WARNING: Mapping is not correct. Mimic what's done in close_lv_endo and close_rv_endo
    Function to generate the closed surface of the RA endocardium.

    Args:
        fourch_name (str): Name of the four chamber mesh file.
    """
    pass
    # path2fourch = os.path.join(PROJECT_PATH, subfolder, fourch_name)
    #
    # os.system("meshtool extract surface -msh=" + os.path.join(path2fourch,fourch_name) +\
    #           " -surf=" + os.path.join(path2fourch,"ra_epi_endo_floating_tv") +\
    #           " -op=4,8,16,17,23,24-2 -ifmt=carp_txt -ofmt=carp_txt")
    #
    # os.system("meshtool extract unreachable -msh=" + os.path.join(path2fourch,"ra_epi_endo_floating_tv.surfmesh") + \
    #                         " -submsh=" + os.path.join(path2fourch,"ra_epi_endo_tv_split") + \
    #                         " -ifmt=carp_txt" + \
    #                         " -ofmt=carp_txt")
    #
    # chamber_or_valve_files = glob.glob(os.path.join(path2fourch,"ra_epi_endo_tv_split*part*elem"))
    #
    # size_files = [os.path.getsize(f) for f in chamber_or_valve_files]
    #
    # # idx_epi = np.argsort(size_files)[-1]
    # idx_epi_or_endo = np.argsort(size_files)[-2]
    # idx_valve = np.argsort(size_files)[-3]
    #
    # # The size criterion for the epi/endo doesn't seem to work for the RA. So
    # # we read the smallest of the two and check the tags.
    # is_this_epi_or_endo_elem = files_manipulations.surf.read(chamber_or_valve_files[idx_epi_or_endo],fourch_name)
    #
    # for tag_num in is_this_epi_or_endo_elem.tags:
    #     if tag_num == 23 or tag_num == 24:
    #         idx_epi = np.argsort(size_files)[-2]
    #         idx_endo = np.argsort(size_files)[-1]
    #         break
    #     elif tag_num == 8:
    #         idx_endo = np.argsort(size_files)[-2]
    #         idx_epi = np.argsort(size_files)[-1]
    #         break
    #
    # shutil.copy(chamber_or_valve_files[idx_endo], os.path.join(path2fourch, "raendo_closed.elem"))
    # shutil.copy(chamber_or_valve_files[idx_endo][:-5] + ".pts", os.path.join(path2fourch, "raendo_closed.pts"))
    #
    # shutil.copy(chamber_or_valve_files[idx_epi], os.path.join(path2fourch, "raepi.elem"))
    # shutil.copy(chamber_or_valve_files[idx_epi][:-5] + ".pts", os.path.join(path2fourch, "raepi.pts"))
    #
    # shutil.copy(chamber_or_valve_files[idx_valve], os.path.join(path2fourch, "tv_rv.elem"))
    # shutil.copy(chamber_or_valve_files[idx_valve][:-5] + ".pts", os.path.join(path2fourch, "tv_rv.pts"))