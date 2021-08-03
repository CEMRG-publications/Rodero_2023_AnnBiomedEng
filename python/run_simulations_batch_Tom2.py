#!/scratch/crg17/miniconda3/bin/python

import os

if __name__ == "__main__":

    type_of_simulation = 'unloading'
    input_path = '/scratch/crg17/wave0'
    script_path = '/scratch/crg17/scripts'
    scripts_name = 'create_submission_file_tom2.sh'
    meshes_path = '/scratch/crg17/wave0/batch0'
    template_path = '/scratch/crg17/templates'
    template_name = type_of_simulation + '_template.in'
    walltime = '0-00:05:00'
    nodes = 8
    force_restart = False
    mmHg_to_kPa = 0.133322387415

    with open(os.path.join(input_path, "X.dat")) as f:
        anatomy_ep_mechanics_values = f.read().splitlines()

    for simulation in anatomy_ep_mechanics_values:
        values = simulation.split(' ')
        mesh_path = "heart_" + ''.join(values[0:6])

        sim_name = "heart_" + ''.join(values) + "_" + type_of_simulation
        if type_of_simulation == "unloading":
            mesh_name = "heart_" + ''.join(values[0:6]) + "_ED"
        if type_of_simulation == "contraction":
            mesh_name = "heart_" + ''.join(values[0:6]) + "_unloaded"
        AT_name = ''.join(values[6:10])
        tanh_params = "t_emd=20.0,Tpeak=" + values[19] + ",tau_c0=50.0,tau_r=50.0,t_dur=" + values[20] + ",lambda_0=0.7,ld=6.0,ld_up=0.0,ldOn=1,VmThresh=-60.0"
        LV_EDP_kPa = str(float(values[10])*mmHg_to_kPa)
        RV_EDP_kPa = str(float(values[11])*mmHg_to_kPa)
        Ao_EDP_kPa = str(float(values[12])*mmHg_to_kPa)
        PA_EDP_kPa = str(float(values[13])*mmHg_to_kPa)
        guccione_params = "a=" + values[15] + ",b_f=8.0,b_fs=4.0,b_t=3.0,kappa=1000"
        neohookean_params = "c=" + values[16] +",kappa=1000"
        flag_run = True

        if os.path.exists(os.path.join(meshes_path, "simulations", sim_name)):
            if force_restart:
                os.system("rm -r " + os.path.join(meshes_path, "simulations", sim_name))
                os.system("rm " + os.path.join(meshes_path, "simulations", "JOB_" + sim_name) + ".in")
                os.system("rm " + os.path.join(script_path, "JOB_" + sim_name + ".out"))
            else:
                flag_run = False

        if flag_run:
            os.chdir(os.path.join(meshes_path, "simulations"))

            os.system(os.path.join(script_path, scripts_name) +
                      ' --type_of_simulation ' + type_of_simulation +
                      ' --simulation_path ' + os.path.join(meshes_path, "simulations") +
                      ' --sim_name ' + sim_name +
                      ' --mesh_path ' + os.path.join(meshes_path, "meshes", mesh_path + AT_name) +
                      ' --mesh_name ' + mesh_name +
                      ' --AT_path ' + os.path.join(meshes_path, "meshes", mesh_name + AT_name) +
                      ' --AT_name ' + AT_name +
                      ' --LV_EDP_mmHg ' + values[10] +
                      ' --LV_EDP_kPa ' + LV_EDP_kPa +
                      ' --RV_EDP_mmHg ' + values[11] +
                      ' --RV_EDP_kPa ' + RV_EDP_kPa +
                      ' --Ao_EDP_mmHg ' + values[12] +
                      ' --Ao_EDP_kPa ' + Ao_EDP_kPa +
                      ' --PA_EDP_mmHg ' + values[13] +
                      ' --PA_EDP_kPa ' + PA_EDP_kPa +
                      ' --spring_BC ' + values[14] +
                      ' --guccione_params ' + guccione_params +
                      ' --neohookean_params ' + neohookean_params +
                      ' --AV_resistance ' + values[17] +
                      ' --PV_resistance ' + values[18] +
                      ' --tanh_params ' + tanh_params +
                      ' --walltime ' + walltime +
                      ' --ncores ' + '64' +
                      ' --nnodes ' + str(nodes) +
                      ' --template_path ' + template_path +
                      ' --template_name ' + template_name
                      )

            os.system("sbatch " + os.path.join(meshes_path, "simulations", "JOB_" + sim_name + ".in"))