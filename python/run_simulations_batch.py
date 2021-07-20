#!/usr/bin/env python

import os

if __name__ == "__main__":

    type_of_simulation = 'unloading'
    input_path = '/work/e348/e348/crg17/wave0'
    script_path = '/work/e348/e348/crg17/scripts/run_mechanics_ARCHER.py'
    meshes_path = '/work/e348/e348/crg17/wave0/batch0'
    runtime = '00:01:00'
    nodes = 1

    with open(os.path.join(input_path, "X.dat")) as f:
        anatomy_ep_mechanics_values = f.read().splitlines()

    for simulation in anatomy_ep_mechanics_values:
        values = simulation.split(' ')

        sim_name = ''.join(values)
        mesh_name = ''.join(values[0:6])
        AT_name = ''.join(values[6:10])

        os.system(script_path +
                  ' --type_of_simulation ' + type_of_simulation +
                  ' --sim_name ' + sim_name +
                  ' --mesh_path ' + os.path.join(meshes_path, mesh_name) +
                  ' --mesh_name ' + mesh_name +
                  ' --AT_path ' + os.path.join(meshes_path, mesh_name) +
                  ' --AT_name ' + AT_name +
                  ' --LV_EDP ' + values[10] +
                  ' --RV_EDP ' + values[11] +
                  ' --Ao_EDP ' + values[12] +
                  ' --PA_EDP ' + values[13] +
                  ' --spring_BC ' + values[14] +
                  ' --scaling_Guccione ' + values[15] +
                  ' --scaling_neohookean ' + values[16] +
                  ' --AV_resistance ' + values[17] +
                  ' --PV_resistance ' + values[18] +
                  ' --peak_isometric_tension ' + values[19] +
                  ' --transient_dur ' + values[20] +
                  ' --runtime ' + runtime +
                  ' --np ' + str(int(nodes*128)) +
                  ' --dry'
                  )
