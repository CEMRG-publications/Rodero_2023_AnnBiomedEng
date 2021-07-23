import os
import pathlib


def hpc_folder(folder_path):

    folder_name = folder_path.split('/')[-1]
    debug_folder = os.path.join('/'.join(folder_path.split('/')[0:-1]), folder_name + "_debug")

    pathlib.Path(debug_folder).mkdir(parents=True, exist_ok=True)

    mesh_name = '.'.join(folder_name.split('.')[:-4])[:-2]

    os.system("meshtool convert -imsh=" + os.path.join(folder_path, mesh_name) + "_ED" +
              " -omsh=" + os.path.join(debug_folder, mesh_name) + "_ED" +
              " -ifmt=carp_bin -ofmt=vtk_bin")

    os.system("meshtool convert -imsh=" + os.path.join(folder_path, mesh_name) + "_ED" +
              " -omsh=" + os.path.join(debug_folder, mesh_name) + "_ED" +
              " -ifmt=carp_bin -ofmt=carp_txt")

    at_name = '.'.join(folder_name.split('.')[-5:])[2:]

    os.system("GlVTKConvert -m " + os.path.join(debug_folder, mesh_name) + "_ED -n " +
              os.path.join(folder_path, at_name + ".dat") + " -F bin -o " + os.path.join(debug_folder, at_name))

    os.system("GlVTKConvert -m " + os.path.join(debug_folder, mesh_name) + "_ED -e " +
              os.path.join(folder_path, "pericardium_penalty.dat") + " -F bin -o " +
              os.path.join(debug_folder, "pericardium_penalty"))

    for surface_name in ["biv.epi", "LSPV", "lvendo_closed", "RSPV", "rvendo_closed", "SVC"]:
        os.system("cp " + os.path.join(debug_folder, mesh_name) + "_ED.pts " +
                  os.path.join(debug_folder, surface_name) + ".pts")
        os.system("cp " + os.path.join(folder_path, surface_name) + ".surf " +
                  os.path.join(debug_folder, surface_name) + ".elem")
        os.system("meshtool convert -imsh=" + os.path.join(debug_folder, surface_name) +
                  " -omsh=" + os.path.join(debug_folder, surface_name) + " -ifmt=carp_txt -ofmt=vtk_bin")
