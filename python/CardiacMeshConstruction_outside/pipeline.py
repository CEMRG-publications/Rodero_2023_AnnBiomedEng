import os
import pandas as pd
import numpy as np
import subprocess
import csv
import sys

from decomposition_pca import PcaWithScaling
from read_momenta import HandleMomenta
from create_model_and_dataset_files import ModelShooting
from MeshGeneration import MeshTetrahedralization


def pca_get_predefined_combinations(path_to_data=os.path.join(os.path.dirname(sys.argv[0]),'DataInput', 'PCA'),
                                    data_filename='OriginalDataSet.csv',
                                    weights_filename='Weights.csv',
                                    output_path=os.path.join(os.path.dirname(sys.argv[0]),'DataOutput', 'PCA', 'WeightedModes')):
    momenta_pca = PcaWithScaling(path_to_data, data_filename, output_path, number_of_components=18)
    momenta_pca.decompose_with_pca()
    weights = pd.read_csv(os.path.join(path_to_data, weights_filename))

    weighted_momenta = np.zeros((len(weights), momenta_pca.components.shape[1]))

    for single_set_of_weights in weights.iterrows():
        weighted_momenta[single_set_of_weights[0], :] = momenta_pca.get_n_main_modes(single_set_of_weights[1].values)

    momenta_pca.save_result('Weighted_momenta.csv', weighted_momenta)
    return(len(weights))


def generate_predefined_weighted_modes(path_to_momenta=os.path.join(os.path.dirname(sys.argv[0]),'DataOutput', 'PCA', 'WeightedModes'),
                                       output_path=os.path.join(os.path.dirname(sys.argv[0]),'DataInput', 'Deformetrica', 'Template')):

    momenta_vec = 'Weighted_momenta.csv'
    momenta_new = 'Weighted_momenta.txt'
    momenta = HandleMomenta(path_to_momenta, momenta_vec,  output_path=output_path, output_filename=momenta_new,
                            configuration='extreme')
    momenta.save_momenta_matrix_in_deformetrica_format()


def prep_predefined_cohort_for_reconstruction(source_path=os.path.join(os.path.dirname(sys.argv[0]),'DataInput', 'PCA'),
                                              template_path=os.path.join(os.path.dirname(sys.argv[0]),'DataInput', 'Deformetrica', 'Template'),
                                              output_path=os.path.join(os.path.dirname(sys.argv[0]),'DataInput', 'Deformetrica', 'Model')):

    mom_mdl = ModelShooting(source=source_path,
                            template_path=template_path,
                            output_path=output_path,
                            key_word='Template',  # Template copied to Decomposition folder
                            momenta_filename='Weighted_momenta.txt',
                            deformation_kernel_width=10)
    mom_mdl.build_with_lxml_tree()
    mom_mdl.write_xml(output_path)
    os.rename(os.path.join(output_path, 'model.xml'), os.path.join(output_path, 'predef_model_shooting.xml'))
    print("Model shooting in ",os.path.join(output_path, 'predef_model_shooting.xml'))


def perform_deformation(bash_file_path=os.path.join(os.path.dirname(sys.argv[0]),'DataInput', 'Deformetrica', 'Model'),
                        bash_file='build_mesh_surfaces.sh'):

    os.chdir(bash_file_path)
    subprocess.call(os.path.join(bash_file_path, bash_file))


def write_meshing_file(gmsh_exe_path,
                       tetra_dir_path=os.path.join(os.path.dirname(sys.argv[0]),'DataInput', 'Gmsh', 'tetra')):

    with open(os.path.join(os.path.dirname(sys.argv[0]),'DataInput', 'Gmsh', 'meshing.sh'), 'w') as f:
        # f.writelines(['#!/bin/bash\n',
        #               'cd "$(dirname "$0")"\n\n',
        #               'FILES="geofiles/*.geo"\n\n',
        #               'for f in $FILES\n',
        #               'do\n',
        #               '  d=${f##*/}\n',
        #               '  d=${d%_*_*}\n',
        #               '  echo "File: $f, chamber: $d"\n\n',
        #               '  {} "$f" -3 -v 2 -o {}/"$d"_tetra.vtk\n\n'.format(gmsh_exe_path, tetra_dir_path),
        #               'done'])

        geo_path = os.path.join(os.path.dirname(sys.argv[0]),'DataInput', 'Gmsh',"geofiles/*.geo")

        f.writelines(['#!/bin/bash\n',
                      'FILES='+geo_path+'\n\n',
                      'for f in $FILES\n',
                      'do\n',
                      '  d=${f##*/}\n',
                      '  d=${d%_*_*}\n',
                      '  echo "File: $f, chamber: $d"\n\n',
                      '  {} "$f" -3 -v 2 -o {}/"$d"_tetra.vtk\n\n'.format(gmsh_exe_path, tetra_dir_path),
                      'done'])


def merged_shapes_generation(final_mesh_preffix,
                             id_from=0,
                             id_to=20,
                             models_path=os.path.join(os.path.dirname(sys.argv[0]),'DataOutput', 'Deformetrica', 'MorphedModels'),
                             output_path='/media/mat/BEDC-845B/FullPipeline',
                             merged_type='tetra'):

    assert id_from < id_to, 'Insert proper range of IDs to create meshes from'
    assert os.path.exists(models_path), 'Provide proper path to models'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for j in range(id_from, id_to):

        if id_to == id_from + 1:
            k_model_value = ''
        else:
            k_model_value = j

        sup = MeshTetrahedralization(main_path=os.path.join(os.path.dirname(sys.argv[0]),'DataInput', 'Gmsh'),
                                     models_path=models_path,
                                     geo_dir='geofiles',
                                     temp_dir='temp',
                                     output_path=output_path,
                                     k_model=k_model_value,
                                     template=False)
        if merged_type == 'surface':
            sup.pipeline_aggr_surf_mesh(final_mesh_preffix = final_mesh_preffix)
        elif merged_type == 'tetra':
            sup.clean()
            sup.pipeline_surf_2_tetra_mesh(final_mesh_preffix = final_mesh_preffix)
        else:
            exit('Provide proper model generation type: "tetra" or "surface"')


def pipeline(gmsh_exe_path,
             weights_filename,
             final_mesh_preffix,
             tetrahedralized_mesh_output_path='/media/mat/BEDC-845B/FullPipeline',
             run_tetra=False):

    print('PCA on 19 cases')

    num_lines_file = pca_get_predefined_combinations(weights_filename =  weights_filename)

    print('Generating momenta')
    generate_predefined_weighted_modes()

    print('Prepare files necessary for deformation')
    prep_predefined_cohort_for_reconstruction()

    print('Generate tetrahedralization file')
    write_meshing_file(gmsh_exe_path)

    print('Run deformetrica')
    perform_deformation()

    if run_tetra:
        print('Perform tetrahedralization')
        merged_shapes_generation(id_from = 0, id_to = num_lines_file, merged_type='tetra',
                                output_path=tetrahedralized_mesh_output_path,
                                final_mesh_preffix = final_mesh_preffix)
    else:
        print('Perform surface construction')
        merged_shapes_generation(id_from = 0, id_to = num_lines_file, merged_type='surface',
                                output_path=tetrahedralized_mesh_output_path,
                                final_mesh_preffix = final_mesh_preffix)

def generate_weights(mode_number, lower_bound, upper_bound, num_lines,
                    output_dir = os.path.join(os.path.dirname(sys.argv[0]),"DataInput","PCA"),
                    output_name = "Weights.csv"):

    step = (upper_bound - lower_bound + 1)/float(num_lines)

    with open(os.path.join(output_dir,output_name), mode='w') as f:
        f_writer = csv.writer(f, delimiter=',')

        f_writer.writerow(["Mode" + str(i) for i in range(1,19)])

        for current_line in range(num_lines):
            output = np.zeros(18)
            output[mode_number-1] = lower_bound + current_line*step
            f_writer.writerow(output)

    f.close()

if __name__ == '__main__':

    weights_filename = sys.argv[1]
    temp_outpath = sys.argv[2]
    waveno = sys.argv[3]

    pipeline(gmsh_exe_path= os.path.join("/opt","anaconda3","bin","gmsh"),
             weights_filename =  weights_filename,
             final_mesh_preffix = "wave" + str(waveno),
             tetrahedralized_mesh_output_path=temp_outpath,
             run_tetra=True)