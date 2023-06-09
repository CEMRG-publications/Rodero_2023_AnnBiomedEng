#!/usr/bin/env python

import os

if __name__ == "__main__":

    type_of_simulation = 'contraction'
    batch=1
    input_path = '/work/e348/e348/crg17/wave0'
    script_path = '/work/e348/e348/crg17/scripts'
    scripts_name = 'run_mechanics_ARCHER2.py'
    meshes_path = '/work/e348/e348/crg17/wave0/batch' + str(batch)
    runtime = '23:59:00'
    nodes = 4
    force_restart = True

    with open(os.path.join(input_path, "X.dat")) as f:
        anatomy_ep_mechanics_values = f.read().splitlines()

    first_sim = int(10*batch)

    #for simulation_i in range(len(anatomy_ep_mechanics_values)):
    for simulation_i in range(first_sim,first_sim+9):
        
        simulation = anatomy_ep_mechanics_values[simulation_i]
        values = simulation.split(' ')

        sim_name = "heart_" + ''.join(values) + "_" + type_of_simulation
        mesh_name = "heart_" + ''.join(values[0:6])
        AT_name = ''.join(values[6:10])

        flag_run = True


        if os.path.exists(os.path.join(meshes_path, "simulations", sim_name)):
            if force_restart:
                os.system("rm -r " + os.path.join(meshes_path, "simulations", sim_name))
                os.system("rm " + os.path.join(meshes_path, "simulations", sim_name) + ".slrm")
                os.system("rm " + os.path.join(meshes_path, "simulations", "JOB_" + sim_name + ".out"))
            else:
                flag_run = False

        if type_of_simulation == 'contraction':
            if not os.path.exists(os.path.join(meshes_path, "simulations", "heart_" + ''.join(values) + "_unloading", "reference.pts")):
                    flag_run = False

        if flag_run:
            os.chdir(os.path.join(meshes_path, "simulations"))

            os.system(os.path.join(script_path, scripts_name) +
                      ' --type_of_simulation ' + type_of_simulation +
                      ' --sim_name ' + os.path.join(meshes_path, "simulations", sim_name) +
                      ' --mesh_path ' + os.path.join(meshes_path, "meshes", mesh_name + AT_name) +
                      ' --mesh_name ' + mesh_name +
                      ' --AT_path ' + os.path.join(meshes_path, "meshes", mesh_name + AT_name) +
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
                      ' --np ' + str(int(nodes*128))
                      )
