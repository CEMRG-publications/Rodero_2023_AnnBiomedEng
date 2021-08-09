#!/bin/bash

#SBATCH -p compute
#SBATCH --job-name=JOB_sim_name
#SBATCH --output=JOB_simname.out
#SBATCH -t 0-24:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=64
#SBATCH --mail-user=cristobal.rodero_gomez@kcl.ac.uk
#SBATCH --mail-type=ALL

NPROC=128
source "${HOME}"/.bashrc
export OMP_NUM_THREADS=1

solver_folder="/scratch/crg17/solver_files/resources"
sim_name="/scratch/crg17/simulations/test_unloading"
mesh_path="/scratch/crg17/meshes"
mesh_name="test_mesh"
AT_path="/scratch/crg17/meshes"
AT_name="test_AT.dat"
LV_EDP_mmHg=
LV_EDP_kPa=
RV_EDP_mmHg=
RV_EDP_kPa=
Ao_EDP_mmHg=
Ao_EDP_kPa=
PA_EDP_mmHg=
PA_EDP_kPa=
spring_BC=
guccione_params="a=1.720,b_f=8.0,b_fs=4.0,b_t=3.0,kappa=1000"
neohookean_params="c=8.375,kappa=1000"
AV_resistance=
PV_resistance=
tanh_params="t_emd=20.0,Tpeak=108.000,tau_c0=50.0,tau_r=50.0,t_dur=605.000,lambda_0=0.7,ld=6.0,ld_up=0.0,ldOn=1,VmThresh=-60.0"
################################################################################
# Execute shell commands

mkdir -p $sim_name

################################################################################
# Execute simulation

mpiexec -np $NPROC $CARPENTRYDIR/carp.pt \
  +F $solver_folder/petsc_options/amg_cg_opts \
  +F $solver_folder/petsc_options/ilu_cg_opts \
  +F $solver_folder/petsc_options/gamg_gmres_opts \
  -ellip_use_pt 0 \
  -parab_use_pt 0 \
  -purk_use_pt 0 \
  -mech_use_pt 0 \
  -ellip_options_file $solver_folder/petsc_options/amg_cg_opts \
  -parab_options_file $solver_folder/petsc_options/ilu_cg_opts \
  -purk_options_file $solver_folder/petsc_options/mumps_opts_nonsymmetric \
  -mechanics_options_file $solver_folder/petsc_options/gamg_gmres_opts \
  -mech_finite_element 0 \
  -mech_activate_inertia 0 \
  -simID $sim_name \
  -meshname $mesh_path"/"$mesh_name \
  -timedt 10.0 \
  -mechDT 1.0 \
  -spacedt 10.0 \
  -tend 1000.0 \
  -loadStepping 50.0 \
  -gridout_i 0 \
  -mech_output 1 \
  -vtk_output_mode 0 \
  -strain_value 0 \
  -stress_value 0 \
  -gzip_data 1 \
  -num_mregions 7 \
  -mregion[0].name "Ventricles" \
  -mregion[0].num_IDs 2 \
  -mregion[0].ID[0] 1 \
  -mregion[0].ID[1] 2 \
  -mregion[0].params $guccione_params \
  -mregion[0].type 9 \
  -mregion[1].name "Atria" \
  -mregion[1].num_IDs 2 \
  -mregion[1].ID[0] 3 \
  -mregion[1].ID[1] 4 \
  -mregion[1].params $neohookean_params \
  -mregion[1].type 2 \
  -mregion[2].name "Valve planes" \
  -mregion[2].num_IDs 4 \
  -mregion[2].ID[0] 7 \
  -mregion[2].ID[1] 8 \
  -mregion[2].ID[2] 9 \
  -mregion[2].ID[3] 10 \
  -mregion[2].params "c=1000.0,kappa=1000" \
  -mregion[2].type 2 \
  -mregion[3].name "Inlet planes" \
  -mregion[3].num_IDs 7 \
  -mregion[3].ID[0] 11 \
  -mregion[3].ID[1] 12 \
  -mregion[3].ID[2] 13 \
  -mregion[3].ID[3] 14 \
  -mregion[3].ID[4] 15 \
  -mregion[3].ID[5] 16 \
  -mregion[3].ID[6] 17 \
  -mregion[3].params "c=1000.0,kappa=1000" \
  -mregion[3].type 2 \
  -mregion[4].name "Aorta" \
  -mregion[4].num_IDs 1 \
  -mregion[4].ID[0] 5 \
  -mregion[4].params "c=26.66,kappa=1000" \
  -mregion[4].type 2 \
  -mregion[5].name "Pulmonary_Artery" \
  -mregion[5].num_IDs 1 \
  -mregion[5].ID[0] 6 \
  -mregion[5].params "c=3.7,kappa=1000" \
  -mregion[5].type 2 \
  -mregion[6].name "BC Veins" \
  -mregion[6].num_IDs 7 \
  -mregion[6].ID[0] 18 \
  -mregion[6].ID[1] 19 \
  -mregion[6].ID[2] 20 \
  -mregion[6].ID[3] 21 \
  -mregion[6].ID[4] 22 \
  -mregion[6].ID[5] 23 \
  -mregion[6].ID[6] 24 \
  -mregion[6].params $neohookean_params \
  -mregion[6].type 2 \
  -mech_vol_split_aniso 1 \
  -num_mechanic_nbc 6 \
  -num_mechanic_bs 4 \
  -mechanic_bs[0].value "$spring_BC" \
  -mechanic_bs[1].value "$spring_BC" \
  -mechanic_bs[2].value "$spring_BC" \
  -mechanic_nbc[0].name LSPV \
  -mechanic_nbc[0].surf_file $mesh_path"/LSPV" \
  -mechanic_nbc[0].spring_idx 0 \
  -mechanic_nbc[1].name RSPV \
  -mechanic_nbc[1].surf_file $mesh_path"/RSPV" \
  -mechanic_nbc[1].spring_idx 1 \
  -mechanic_nbc[2].name SVC \
  -mechanic_nbc[2].surf_file $mesh_path"/SVC" \
  -mechanic_nbc[2].spring_idx 2 \
  -mechanic_nbc[3].name lvendo_closed \
  -mechanic_nbc[3].surf_file $mesh_path"/lvendo_closed" \
  -mechanic_nbc[3].pressure $LV_EDP_kPa \
  -mechanic_nbc[4].name rvendo_closed \
  -mechanic_nbc[4].surf_file $mesh_path"/rvendo_closed" \
  -mechanic_nbc[4].pressure $RV_EDP_kPa \
  -num_mechanic_ed 1 \
  -mechanic_ed[0].ncomp 1 \
  -mechanic_ed[0].file $mesh_path"/pericardium_penalty" \
  -mechanic_bs[3].edidx 0 \
  -mechanic_bs[3].value "$spring_BC" \
  -mechanic_nbc[5].name "Pericardium" \
  -mechanic_nbc[5].surf_file $mesh_path"/biv.epi" \
  -mechanic_nbc[5].spring_idx 3 \
  -mechanic_nbc[5].nspring_idx 0 \
  -mechanic_nbc[5].nspring_config 1 \
  -num_stim 1 \
  -stimulus[0].stimtype 8 \
  -stimulus[0].data_file $AT_path"/"$AT_name".dat" \
  -diffusionOn 0 \
  -mech_deform_elec 0 \
  -mech_use_actStress 1 \
  -num_imp_regions 3 \
  -imp_region[0].name ventricles \
  -imp_region[0].im TT2 \
  -imp_region[0].num_IDs 2 \
  -imp_region[0].ID[0] 1 \
  -imp_region[0].ID[1] 2 \
  -imp_region[1].name atria \
  -imp_region[1].im COURTEMANCHE \
  -imp_region[1].num_IDs 2 \
  -imp_region[1].ID[0] 3 \
  -imp_region[1].ID[1] 4 \
  -imp_region[2].name others \
  -imp_region[2].im PASSIVE \
  -imp_region[2].num_IDs 20 \
  -imp_region[2].ID[0] 5 \
  -imp_region[2].ID[1] 6 \
  -imp_region[2].ID[2] 7 \
  -imp_region[2].ID[3] 8 \
  -imp_region[2].ID[4] 9 \
  -imp_region[2].ID[5] 10 \
  -imp_region[2].ID[6] 11 \
  -imp_region[2].ID[7] 12 \
  -imp_region[2].ID[8] 13 \
  -imp_region[2].ID[9] 14 \
  -imp_region[2].ID[10] 15 \
  -imp_region[2].ID[11] 16 \
  -imp_region[2].ID[12] 17 \
  -imp_region[2].ID[13] 18 \
  -imp_region[2].ID[14] 19 \
  -imp_region[2].ID[15] 20 \
  -imp_region[2].ID[16] 21 \
  -imp_region[2].ID[17] 22 \
  -imp_region[2].ID[18] 23 \
  -imp_region[2].ID[19] 24 \
  -imp_region[0].plugins TanhStress \
  -imp_region[0].plug_param $tanh_params \
  -veldep 0 \
  -volumeTracking 1 \
  -numElemVols 1 \
  -elemVols[0].name tissue \
  -elemVols[0].grid 8 \
  -pstrat 2 \
  -pstrat_i 2 \
  -krylov_tol_mech 0.0001 \
  -krylov_norm_mech 0 \
  -krylov_maxit_mech 1000.0 \
  -newton_atol_mech 0.001 \
  -newton_tol_mech 0.001 \
  -newton_adaptive_tol_mech 2 \
  -newton_tol_cvsys 0.001 \
  -newton_line_search 0 \
  -newton_maxit_mech 5 \
  -mech_activate_inertia 1 \
  -mass_lumping 1 \
  -mech_rho_inf 0.0 \
  -mech_stiffness_damping 0.1 \
  -mech_mass_damping 0.1 \
  -mech_lambda_upd 1 \
  -numSurfVols 2 \
  -surfVols[0].name lvendo_closed \
  -surfVols[0].surf_file $mesh_path"/lvendo_closed" \
  -surfVols[0].grid 8 \
  -surfVols[1].name rvendo_closed \
  -surfVols[1].surf_file $mesh_path"/rvendo_closed" \
  -surfVols[1].grid 8 \
  -CV_coupling 0 \
  -CV_FE_coupling 1 \
  -CVS_mode 0 \
  -num_cavities 2 \
  -cavities[0].cav_type 0 \
  -cavities[0].cavP 3 \
  -cavities[0].tube 2 \
  -cavities[0].cavVol 0 \
  -cavities[0].p0_cav $LV_EDP_mmHg \
  -cavities[0].p0_in $LV_EDP_mmHg \
  -cavities[0].p0_out $Ao_EDP_mmHg \
  -cavities[0].valve 0 \
  -cavities[0].state -1 \
  -lv_wk3.name Aorta \
  -lv_wk3.R1 $AV_resistance \
  -lv_wk3.R2 0.63 \
  -lv_wk3.C 5.16 \
  -mitral_valve.Rfwd 0.05 \
  -mitral_valve.Rbwd 1000.0 \
  -aortic_valve.Rfwd 0.0 \
  -cavities[1].cav_type 1 \
  -cavities[1].cavP 4 \
  -cavities[1].tube 2 \
  -cavities[1].cavVol 1 \
  -cavities[1].p0_cav $RV_EDP_mmHg \
  -cavities[1].p0_in $RV_EDP_mmHg \
  -cavities[1].p0_out $PA_EDP_mmHg \
  -cavities[1].valve 0 \
  -cavities[1].state -1 \
  -rv_wk3.name PA \
  -rv_wk3.R1 $PV_resistance \
  -rv_wk3.R2 0.1008 \
  -rv_wk3.C 21.156 \
  -tricuspid_valve.Rfwd 0.05 \
  -tricuspid_valve.Rbwd 1000.0 \
  -pulmonic_valve.Rfwd 0.0

