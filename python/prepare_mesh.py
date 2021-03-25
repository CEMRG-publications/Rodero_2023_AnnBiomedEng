import wget
import os
import zipfile
from os import listdir
from os.path import isfile, join, isdir, getsize
from shutil import copy
import glob
import numpy as np


def download_zip(bundle_number):

    def bar_custom(current, total, width = 80):
        os.system('clear')
        current_format = current / 1e6
        total_format = total / 1e6

        print("Downloading to %s: %d%% [%.1f / %d] Mb" % (out_name, current / total * 100, current_format, total_format))
    
    url = "https://zenodo.org/record/4506930/files/Final_models_" + bundle_number + ".zip?download=1"

    file_name = url.split("/")[-1]
    file_name = file_name.split("?")[0]

    out_name = "/data/fitting/" + file_name

    wget.download(url, out = out_name, bar = bar_custom)

def unzip_meshes(bundle_number):

    pathmeshes = "/data/fitting/Final_models_" + bundle_number

    print("Unzipping files...")
    with zipfile.ZipFile(pathmeshes + ".zip", 'r') as zip_ref:
        zip_ref.extractall(pathmeshes + "/..")
    print("Files unzipped.")
    
    files_in_dir = [f for f in listdir(pathmeshes) if isfile(join(pathmeshes, f))]

    for filename in files_in_dir:
        if filename.split(".")[-1] != "csv":
            if not isdir(join(pathmeshes, "..", filename.split(".")[0])):
                os.mkdir(join(pathmeshes, "..", filename.split(".")[0]))
            os.rename(join(pathmeshes,filename),join(pathmeshes,"..",filename.split(".")[0],filename))
        else:
            if not isdir(join(pathmeshes, "../Cases_weights")):
                os.mkdir(join(pathmeshes, "../Cases_weights"))
            os.rename(join(pathmeshes,filename),join(pathmeshes,"../Cases_weights",filename))

    os.rmdir(pathmeshes)

def vtk_mm2carp_um(heart):

    mesh_name = "Full_Heart_Mesh_" + str(heart)
    mesh_dir = join("/data","fitting",mesh_name)

    os.system("meshtool convert -imsh=" + join(mesh_dir,mesh_name) + \
                              " -omsh=" + join(mesh_dir,mesh_name) + \
                              " -ifmt=vtk" + \
                              " -ofmt=carp_txt")

    os.rename(join(mesh_dir, mesh_name) + ".pts", join(mesh_dir, mesh_name) + "_mm.pts")

    os.system(join("/home","common","cm2carp","bin","return_carp2original_coord.pl ") + \
              join(mesh_dir,mesh_name) + "_mm.pts 1000 0 0 0 > " + join(mesh_dir,mesh_name) + "_um.pts")

    copy(join(mesh_dir,mesh_name) + "_um.pts", join(mesh_dir,mesh_name) + ".pts")

def extract_bdry_bayer(heart):

    mesh_name = "Full_Heart_Mesh_" + str(heart)
    mesh_dir = join("/data","fitting",mesh_name)

    ####### Epi and endo

    os.system("meshtool extract surface -msh=" + join(mesh_dir,mesh_name) + \
                            " -surf=" + join(mesh_dir,"biv_epi_endo") + \
                            " -op=1,2-7,8,9,10" + \
                            " -ifmt=carp_txt" + \
                            " -ofmt=vtk_bin")

    os.system("meshtool extract unreachable -msh=" + join(mesh_dir,"biv_epi_endo.surfmesh") + \
                            " -submsh=" + join(mesh_dir,"biv_epi_endo") + \
                            " -op=1,2-7,8,9,10" + \
                            " -ifmt=vtk_bin" + \
                            " -ofmt=carp_txt")

    epi_or_endo_files = glob.glob(join(mesh_dir,"*part*elem"))

    # We want only the three biggest files. The biggest will be the epi, and 
    # depending on the tags, the others will be the LV or the RV.

    size_files = [getsize(f) for f in epi_or_endo_files]

    while(len(size_files) > 3):
        idx_min = size_files.index(min(size_files))
        epi_or_endo_files.pop(idx_min)
        size_files.pop(idx_min)

    idx_max = size_files.index(max(size_files))

    copy(epi_or_endo_files[idx_max], join(mesh_dir, "biv_epi.surf"))

    name = epi_or_endo_files[idx_max]
    name_no_ext = name.split(".elem")[0]
    file_name = name_no_ext.split("/")[-1]

    os.rename(name, join(mesh_dir, "biv_epi.elem"))
    os.rename(name_no_ext + ".lon", join(mesh_dir, "biv_epi.lon"))
    os.rename(name_no_ext + ".pts", join(mesh_dir, "biv_epi.pts"))

    epi_or_endo_files.pop(idx_max)
    size_files.pop(idx_max)

    for i in range(len(epi_or_endo_files)):
        name = epi_or_endo_files[i]
        name_no_ext = name.split(".elem")[0]
        file_name = name_no_ext.split("/")[-1]

        os.system("meshtool extract tags -msh=" + name_no_ext + \
                            " -odat=" + join(mesh_dir,file_name) + ".tags" + \
                            " -ifmt=carp_txt")
        tag_file = np.loadtxt(join(mesh_dir,file_name) + ".tags")

        if(int(sum(tag_file)) != len(tag_file)):
            copy(name, join(mesh_dir, "RV_endo.surf"))

            os.rename(name, join(mesh_dir, "RV_endo.elem"))
            os.rename(name_no_ext + ".lon", join(mesh_dir, "RV_endo.lon"))
            os.rename(name_no_ext + ".pts", join(mesh_dir, "RV_endo.pts"))

            epi_or_endo_files.pop(i)
            size_files.pop(i)
            break

    idx_max = size_files.index(max(size_files))

    copy(epi_or_endo_files[idx_max], join(mesh_dir, "LV_endo.surf"))

    name = epi_or_endo_files[idx_max]
    name_no_ext = name.split(".elem")[0]
    file_name = name_no_ext.split("/")[-1]

    os.rename(name, join(mesh_dir, "LV_endo.elem"))
    os.rename(name_no_ext + ".lon", join(mesh_dir, "LV_endo.lon"))
    os.rename(name_no_ext + ".pts", join(mesh_dir, "LV_endo.pts"))

    ####### Base and apex

    os.system("meshtool extract surface -msh=" + join(mesh_dir,mesh_name) + \
                        " -surf=" + join(mesh_dir,"base") + \
                        " -op=1,2:7,8" + \
                        " -ifmt=carp_txt" + \
                        " -ofmt=carp_txt")

    os.system("meshtool extract surface -msh=" + join(mesh_dir,mesh_name) + \
                        " -surf=" + join(mesh_dir,"MV") + \
                        " -op=7-1,3" + \
                        " -ifmt=carp_txt" + \
                        " -ofmt=carp_txt")

    MV = np.loadtxt(join(mesh_dir,"MV.surfmesh.pts"), skiprows = 1)

    num_pts =  MV.shape[0]

    sum_x = np.sum(MV[:, 0])
    sum_y = np.sum(MV[:, 1])
    sum_z = np.sum(MV[:, 2])

    centroid = np.array([sum_x/num_pts, sum_y/num_pts, sum_z/num_pts])

    os.system("meshtool extract tags -msh=" + join(mesh_dir,"biv_epi") + \
                    " -odat=" + join(mesh_dir,"biv_epi_tags.dat") + \
                    " -ifmt=carp_txt")

    os.system("meshtool interpolate elem2node -omsh=" + join(mesh_dir,"biv_epi") + \
                    " -idat=" + join(mesh_dir,"biv_epi_tags.dat") + \
                    " -odat=" + join(mesh_dir,"biv_epi_tags_pts.dat"))

    biv_tag_file = np.loadtxt(join(mesh_dir,"biv_epi_tags_pts.dat"))
    biv_epi_pts = np.loadtxt(join(mesh_dir,"biv_epi.pts"), skiprows = 1)

    dist_vec = np.zeros(len(biv_tag_file))

    for i in range(len(biv_tag_file)):
        if(biv_tag_file[i] == 1):
            dist_vec[i] = np.linalg.norm(centroid - biv_epi_pts[i,:])

    idx_max = np.where(dist_vec == max(dist_vec))

    apex_pts = np.copy(biv_epi_pts[idx_max,:])

    heart_pts = np.loadtxt(join(mesh_dir,mesh_name) + ".pts", skiprows = 1)

    os.system("meshtool extract tags -msh=" + join(mesh_dir,mesh_name) + \
                " -odat=" + join(mesh_dir, mesh_name + "_tags.dat") + \
                " -ifmt=carp_txt")

    os.system("meshtool interpolate elem2node -omsh=" + join(mesh_dir,mesh_name) + \
                    " -idat=" + join(mesh_dir, mesh_name + "_tags.dat") + \
                    " -odat=" + join(mesh_dir, mesh_name + "_tags_pts.dat"))

    heart_tag_file = np.loadtxt(join(mesh_dir, mesh_name + "_tags_pts.dat"))

    big_dist_vec = np.ones(len(heart_tag_file))

    for i in range(len(heart_tag_file)):
        if(heart_tag_file[i] == 1):
            big_dist_vec[i] = np.linalg.norm(apex_pts - heart_pts[i,:])

    apex_idx = np.where(big_dist_vec == min(big_dist_vec))

    write_vtx(join(mesh_dir,"LV_apex_epi.vtx"),apex_idx[0])

def write_vtx(pathname, array):

    header = [len(array), "intra"]
    str_to_write = np.append(header, array)

    with open(pathname, 'w') as f:
        for item in str_to_write:
            f.write("%s\n" % item)

def map_biv(heart):
     