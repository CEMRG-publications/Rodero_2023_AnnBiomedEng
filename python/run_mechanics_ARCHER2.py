#!/usr/bin/env python
"""
Electromechanics benchmark for 4 chamber heart
"""

# --- import packages ---------------------------------------------------------
from __future__ import print_function
import os

from carputils import settings
import sys
from carputils import tools
from carputils import model

import itertools

MESHTOOL_EXE_PATH = "/work/e348/e348/shared/carpentry/bin/meshtool"


def parser_commands():

    parser = tools.standard_parser()
    parser.add_argument('--type_of_simulation',
                        choices=['unloading', 'contraction'],
                        help='Type of simulation to run.')
    parser.add_argument('--sim_name',
                        type=str,
                        default='test',
                        help='Name of the simulation. Recommendation is wave_batch_type_of_simulation.')
    parser.add_argument('--mesh_path',
                        type=str,
                        default='',
                        help='Path to the mesh.')
    parser.add_argument('--mesh_name',
                        type=str,
                        default='',
                        help='Name of the mesh as concatenation of all the modes.')
    parser.add_argument('--AT_path',
                        type=str,
                        default='',
                        help='Path to the activation time file.')
    parser.add_argument('--AT_name',
                        type=str,
                        default='',
                        help='Name of the EP simulation as concatenation of all the EP parameter values.')
    parser.add_argument('--LV_EDP',
                        type=str,
                        default='',
                        help='End-diastolic pressure of the LV, in mmHg')
    parser.add_argument('--RV_EDP',
                        type=str,
                        default='',
                        help='End-diastolic pressure of the RV, in mmHg')
    parser.add_argument('--Ao_EDP',
                        type=str,
                        default='',
                        help='End-diastolic pressure of the aorta, in mmHg')
    parser.add_argument('--PA_EDP',
                        type=str,
                        default='',
                        help='End-diastolic pressure of the pulmonary artery, in mmHg')
    parser.add_argument('--spring_BC',
                        type=str,
                        default='',
                        help='Spring stiffness for the boundary conditions, in kPa/um')
    parser.add_argument('--scaling_Guccione',
                        type=str,
                        default='',
                        help='Scaling factor for the Guccione material law, in kPa')
    parser.add_argument('--scaling_neohookean',
                        type=str,
                        default='',
                        help='Scaling factor for the neohookean material law, in kPa')
    parser.add_argument('--AV_resistance',
                        type=str,
                        default='',
                        help='Aortic valve resistance in the Windkessel model, in mmHg s/mL')
    parser.add_argument('--PV_resistance',
                        type=str,
                        default='',
                        help='Pulmonary valve resistance in the Windkessel model, in mmHg s/mL')
    parser.add_argument('--peak_isometric_tension',
                        type=str,
                        default='',
                        help='Peak isometric tension in the tanh model, in kPa')
    parser.add_argument('--transient_dur',
                        type=str,
                        default='',
                        help='Duration of tension transient in the tanh model, in ms')
    return parser


def job_id(args):
    """
    Generate name of top level output directory.
    """
    args = full_list_of_parameters(args)
    job_name = args.sim_name

    return job_name


def run(args, job):

    cmd = tools.carp_cmd()

    if args.type_of_simulation == "unloading":
        final_mesh_name = args.mesh_name + "_ED"
    else:
        final_mesh_name = args.mesh_name + "_unloaded"

    skip_execution = False

    if not os.path.isfile(os.path.join(args.mesh_path, final_mesh_name + ".blon")) and not os.path.isfile(os.path.join(args.mesh_path, final_mesh_name + ".belem")) and not os.path.isfile(os.path.join(args.mesh_path, final_mesh_name + ".bpts")):
        if os.path.isfile(os.path.join(args.sim_name[:-11] + "unloading", "reference.pts")):
            use_unloaded_mesh(args)
        else:
            skip_execution = True

    if skip_execution:
        cmd += ['-dry', 1]

    cmd += ['-simID', job.ID,
            '-meshname', os.path.join(args.mesh_path, final_mesh_name),
            '-meshformat', args.meshformat]

    cmd += setup_time_variables(args)

    cmd += setup_output(args)

    cmd += setup_material(args)

    cmd += setup_bc(args)

    if args.type_of_simulation == 'unloading':
        cmd += setup_unloading(args)

    if args.type_of_simulation == 'contraction':
        cmd += setup_stimuli(args)  # define stimuli

    cmd += setup_active_stress(args)  # active stress setting

    cmd += set_solver_options(args)

    cmd += set_cv_sys(args)

    job.carp(cmd)

# =============================================================================
#    FUNCTIONS
# =============================================================================


def full_list_of_parameters(args):
    auxiliary_parameters = {'meshformat': 1, # Mesh in binary format
                            'atria_tags': [3, 4],
                            'num_mechanic_bs': 3,  # Number of mechanical boundary springs. Veins
                            'num_mechanic_ed': 1,  # Number of mechanical data sets. Pericardium, if contraction.
                            'num_mechanic_nbc': 5,  # Number of mechanical Neumann boundary conditions.
                                                    # Veins + cavities
                            'numSurfVols': 2,  # Number of surface enclosed volumes to keep track of. Ventricles.
                            'passive_tags': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                            'ventricular_tags': [1, 2, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
                            }
    unloading_parameters = {'loadStepping': 50.,  # Apply load in steps to avoid instabilities: (0) off, (1) auto,
                            # (<0) manual choice
                            'timedt': 10.,  # Time between temporal output (ms)
                            'unload_conv': 0,  # 0 - volume-based, 1 - point-based.
                            'unload_err': 1,  # Relative error. 0 for absolute error.
                            'unload_maxit': 10, # They are usually around 5-7
                            'unload_stagtol': 10.,
                            'unload_tol': 1e-3 # Microliter of difference
                            }
    guccione_parameters = {'bf_guccione': 8.,
                           'b_fs_guccione': 4.,
                           'b_t_guccione': 3.,
                           }
    neohookean_parameters = {'scaling_aorta': 26.66,
                             'scaling_extra_tissue': 1000.,
                             'scaling_PA': 3.7,
                             }
    tanh_parameters = {'lambda_0': 0.7,  # Sarcomere length ratio (a_7)
                       'ld': 6.,  # Degree of length dependence (a_6)
                       'ldOn': 1,  # Turn on/off length-dependence
                       'ld_up': 0.,  # Length dependence of upstroke time (a_4)
                       'tau_c0': 50.,  # Time constant contraction rate (t_r)
                       'tau_r': 50.,  # Time constant relaxation rate (t_d)
                       't_emd': 20.,  # Electromechanical delay, ms. Rodero 2021
                       'veldep': 0,  # Velocity dependence
                       'VmThresh': -60.,  # Threshold Vm for deriving LAT
                       }
    windkessel_parameters = {'aortic_valve_Rfwd': 0.,  # Forward aortic resistance. 0 to avoid stenosis.
                             'lv_wk3_C': 5.16,  # Windkessel capacitor. 10.1007/s10237-013-0523-y
                             'lv_wk3_R2': 0.63,  # Parallel Windkessel resistor. 10.1007/s10237-013-0523-y
                             'mitral_valve_Rbwd': 1000.,  # Pick backward flow resistance. High to avoid regurgitation.
                             'mitral_valve_Rfwd': 0.05,  # Pick forward flow resistance. Ad hoc.
                             'pulmonic_valve_Rfwd': 0.,
                             'rv_wk3_C': 4.1*5.16,
                             'rv_wk3_R2': 0.16*0.63,
                             'tricuspid_valve_Rbwd': 1000.,
                             'tricuspid_valve_Rfwd': 0.05,
                             }
    numerics_parameters = {'mechDT': 1.,  # Time-step for mechanics (ms)
                           'krylov_maxit_mech': 1e3,
                           'krylov_norm_mech': 0,
                           'krylov_tol_mech': 1e-8,
                           'mass_lumping': 1,
                           'mech_activate_inertia': 1,  # Activate inertia term in mechanics simulations
                           'mech_lambda_upd': 2,
                           'mech_rho_inf': 0,
                           'mech_stiffness_damping': 0.2,
                           'mech_mass_damping': 0.2,
                           'newton_adaptive_tol_mech': 1,
                           'newton_forcing_term': 3,
                           'newton_line_search': 0,
                           'newton_maxit_mech': 20,
                           'newton_tol_mech': 1e-6,
                           'newton_tol_cvsys': 1e-3,
                           'newton_atol_mech': 1e-6,
                           'pstrat': 2,  # KDtree-based partitioning strategy
                           'pstrat_i': 2,  # KDtree-based partitioning strategy for intracellular
                           'spacedt': 10.,  # Time between spatial output (ms)
                           'tend': 1000.,  # Length of simulation (ms)
                           }

    output_parameters = {'gridout_i': 0,  # Set bits: 0=none. Prevents the output of large files.
                         'gzip_data': 1,  # Turn on/off gzipping of grid data output
                         'mech_output': 1,  # Mech output: 1 = igb vector output (set gzip_data for zipped output)
                         'strain_value': 0,  # No strain value
                         'stress_value': 0,  # Compute no stress
                         'vtk_output_mode': 0,  # No output vtk

                         }

    all_parameters = dict(itertools.chain(auxiliary_parameters.items(), unloading_parameters.items(),
                                          guccione_parameters.items(), neohookean_parameters.items(),
                                          tanh_parameters.items(), windkessel_parameters.items(),
                                          numerics_parameters.items(), output_parameters.items()))

    # Keep these parameters unless introduced by user
    for parameter in all_parameters:
        if not hasattr(args, parameter) or getattr(args, parameter) is None:
            setattr(args, parameter, all_parameters[parameter])

    return args


def use_unloaded_mesh(args):
    """
    Function to move the reference.pts from the unloading simulation to the simulation folder and clear the non-binary
    files.
    """

    os.system(MESHTOOL_EXE_PATH + " convert -ifmt=carp_bin -ofmt=carp_txt -imsh=" +
              os.path.join(args.mesh_path, args.mesh_name + "_ED") + " -omsh=" +
              os.path.join(args.mesh_path, args.mesh_name + "_unloaded")
              )

    os.system("cp " + os.path.join(args.sim_name[:-11] + "unloading", "reference.pts ") +
              os.path.join(args.mesh_path, args.mesh_name + "_unloaded.pts"))

    os.system(MESHTOOL_EXE_PATH + " convert -ifmt=carp_txt -ofmt=carp_bin -imsh=" +
              os.path.join(args.mesh_path, args.mesh_name + "_unloaded") + " -omsh=" +
              os.path.join(args.mesh_path, args.mesh_name + "_unloaded")
              )

    os.system("rm " + os.path.join(args.mesh_path, "*.pts"))
    os.system("rm " + os.path.join(args.mesh_path, "*.elem"))
    os.system("rm " + os.path.join(args.mesh_path, "*.lon"))


def setup_time_variables(args):
    """
    Assignment of time-dependent parameters.
    """

    time_opts = ['-timedt', args.timedt,
                 '-mechDT', args.mechDT,
                 '-spacedt', args.spacedt,
                 '-tend', args.tend,
                 '-loadStepping', args.loadStepping]

    return time_opts


def setup_output(args):
    """
    Assignment of output-related parameters.
    """

    vis_opts = ['-gridout_i', args.gridout_i,
                '-mech_output', args.mech_output,
                '-vtk_output_mode', args.vtk_output_mode,
                '-strain_value', args.strain_value,
                '-stress_value', args.stress_value,
                '-gzip_data', args.gzip_data
                ]

    return vis_opts


def setup_material(args):
    """
    Assignment of parameters related to the material laws.
    """

    mech_opts = []

    # set bulk modulus kappa depending on finite element
    kappa = 1000 if args.mech_element == 'P1-P0' else 1e100

    ventricles = model.mechanics.GuccioneMaterial(args.ventricular_tags, 'Ventricles', kappa=kappa,
                                                  a=args.scaling_Guccione, b_f=args.bf_guccione,
                                                  b_fs=args.b_fs_guccione, b_t=args.b_t_guccione)
    atria = model.mechanics.NeoHookeanMaterial([3, 4], 'Atria', kappa=kappa, c=args.scaling_neohookean)
    valves = model.mechanics.NeoHookeanMaterial([7, 8, 9, 10], 'Valve planes', kappa=kappa, c=args.scaling_extra_tissue)
    inlets = model.mechanics.NeoHookeanMaterial([11, 12, 13, 14, 15, 16, 17], 'Inlet planes', kappa=kappa, c=1000.0)
    aorta = model.mechanics.NeoHookeanMaterial([5], 'Aorta', kappa=kappa, c=args.scaling_aorta)
    pa = model.mechanics.NeoHookeanMaterial([6], 'Pulmonary_Artery', kappa=kappa, c=args.scaling_PA)
    veins = model.mechanics.NeoHookeanMaterial([18, 19, 20, 21, 22, 23, 24], 'BC Veins', kappa=kappa,
                                               c=args.scaling_neohookean)

    mech_opts += model.optionlist([ventricles, atria, valves, inlets, aorta, pa, veins])

    mech_opts += ['-mech_vol_split_aniso', 1]

    return mech_opts


def setup_bc(args):
    """
    Assignment of the parameters related to the boundary and initial conditions.
    """

    if args.type_of_simulation == 'contraction' or args.type_of_simulation == 'unloading':
        args.num_mechanic_nbc += 1
        args.num_mechanic_bs += 1

    nbc = ['-num_mechanic_nbc', args.num_mechanic_nbc,
           '-num_mechanic_bs', args.num_mechanic_bs,
           '-mechanic_bs[0].value', args.spring_BC,  # LSPV
           '-mechanic_bs[1].value', args.spring_BC,  # RSPV
           '-mechanic_bs[2].value', args.spring_BC,  # SVC
           '-mechanic_nbc[0].name', 'LSPV',
           '-mechanic_nbc[0].surf_file', os.path.join(args.mesh_path, 'LSPV'),
           '-mechanic_nbc[0].spring_idx', 0,
           '-mechanic_nbc[1].name', 'RSPV',
           '-mechanic_nbc[1].surf_file', os.path.join(args.mesh_path, 'RSPV'),
           '-mechanic_nbc[1].spring_idx', 1,
           '-mechanic_nbc[2].name', 'SVC',
           '-mechanic_nbc[2].surf_file', os.path.join(args.mesh_path, 'SVC'),
           '-mechanic_nbc[2].spring_idx', 2,
           '-mechanic_nbc[3].name', 'lvendo_closed',
           '-mechanic_nbc[3].surf_file', os.path.join(args.mesh_path, 'lvendo_closed'),
           '-mechanic_nbc[3].pressure', str(float(args.LV_EDP)*0.133322),  # kPa
           '-mechanic_nbc[4].name', 'rvendo_closed',
           '-mechanic_nbc[4].surf_file', os.path.join(args.mesh_path, 'rvendo_closed'),
           '-mechanic_nbc[4].pressure', str(float(args.RV_EDP)*0.133322)  # kPa
           ]

    if args.type_of_simulation == 'contraction' or args.type_of_simulation == 'unloading':
        pericardium_file = os.path.join(args.mesh_path, 'pericardium_penalty')
        nbc += ['-num_mechanic_ed', args.num_mechanic_ed,
                '-mechanic_ed[0].ncomp', 1,
                '-mechanic_ed[0].file', pericardium_file,
                '-mechanic_bs[3].edidx', 0,
                '-mechanic_bs[3].value', args.spring_BC,
                '-mechanic_nbc[5].name', 'Pericardium',
                '-mechanic_nbc[5].surf_file', os.path.join(args.mesh_path, 'biv.epi'),
                '-mechanic_nbc[5].spring_idx', 3,
                '-mechanic_nbc[5].nspring_idx', 0,
                '-mechanic_nbc[5].nspring_config', 1]

    return nbc


def setup_unloading(args):
    """
    Assignment of parameters relevant only for the unloading step.
    """
    unload_opts = ['-experiment',   5,
                   '-loadStepping', args.loadStepping,
                   '-unload_conv',  args.unload_conv,
                   '-unload_tol',   args.unload_tol,
                   '-unload_err',   args.unload_err,
                   '-unload_maxit', args.unload_maxit,
                   '-unload_stagtol', args.unload_stagtol]
    return unload_opts


def setup_stimuli(args):
    """
    Assignment of parameters related to the activation time file.
    """

    act_seq_file = os.path.join(args.AT_path, args.AT_name + '.dat')
    # general stimulus options
    stm_opts = ['-num_stim', 1,
                '-stimulus[0].stimtype', 8,  # Prescribed takeoff
                '-stimulus[0].data_file', act_seq_file,
                '-diffusionOn', 0,  # No depolarization
                '-mech_deform_elec', 0  # Mechanics doesn't affect electric grid
                ]

    return stm_opts


def setup_active_stress(args):
    """
    Assignment of parameters related to the active stress model.
    """

    if args.type_of_simulation == 'unloading':  # All passive
        opts = ['-num_imp_regions', 1,
                '-num_stim', 0,
                '-imp_region[0].name', 'passive',
                '-imp_region[0].im', 'PASSIVE',
                '-imp_region[0].num_IDs', len(args.ventricular_tags) + len(args.atria_tags) + len(args.passive_tags)]

        for ventricular_tag in range(len(args.ventricular_tags) + len(args.atria_tags) + len(args.passive_tags)):
            opts += ['-imp_region[0].ID['+str(ventricular_tag)+']', ventricular_tag+1]
    else:
        opts = ['-mech_use_actStress', 1,
                '-num_imp_regions', 3,
                '-imp_region[0].name', 'ventricles',
                '-imp_region[0].im', 'TT2',
                '-imp_region[0].num_IDs', len(args.ventricular_tags)]

        for i, tag in enumerate(args.ventricular_tags):
            opts += ['-imp_region[0].ID[' + str(i) + ']',  tag]

        opts += ['-imp_region[1].name', 'atria',
                 '-imp_region[1].im', 'COURTEMANCHE',
                 '-imp_region[1].num_IDs', len(args.atria_tags)]

        for i, tag in enumerate(args.atria_tags):
            opts += ['-imp_region[1].ID[' + str(i) + ']',  tag]

        opts += ['-imp_region[2].name', 'others',
                 '-imp_region[2].im', 'PASSIVE',
                 '-imp_region[2].num_IDs', len(args.passive_tags)]

        for i, tag in enumerate(args.passive_tags):
            opts += ['-imp_region[2].ID[' + str(i) + ']',  tag]

        tanh_pars = 't_emd={},' \
                    'Tpeak={},' \
                    'tau_c0={},' \
                    'tau_r={},' \
                    't_dur={},' \
                    'lambda_0={},' \
                    'ld={},' \
                    'ld_up={},' \
                    'ldOn={},' \
                    'VmThresh={}'.format(args.t_emd,
                                         args.peak_isometric_tension,
                                         args.tau_c0,
                                         args.tau_r,
                                         args.transient_dur,
                                         args.lambda_0,
                                         args.ld,
                                         args.ld_up,
                                         args.ldOn,
                                         args.VmThresh)

        opts += ['-imp_region[0].plugins', 'TanhStress',
                 '-imp_region[0].plug_param', tanh_pars,
                 '-veldep', args.veldep]
    return opts


def set_solver_options(args):
    """
    Assignment of the parameters related to the mechanics solver.
    """

    # Track tissue volume
    mech_opts = ['-volumeTracking', 1,  # Keep track of myocardial volume changes due to mechanical deformation
                 '-numElemVols', 1,  # Number of mesh volumes to keep track of.
                 '-elemVols[0].name', 'tissue',
                 '-elemVols[0].grid', model.mechanics.grid_id(),
                 '-pstrat', args.pstrat,
                 '-pstrat_i', args.pstrat_i,
                 '-krylov_tol_mech', args.krylov_tol_mech,
                 '-krylov_norm_mech', args.krylov_norm_mech,
                 '-krylov_maxit_mech', args.krylov_maxit_mech,
                 '-newton_atol_mech', args.newton_atol_mech,
                 '-newton_tol_mech', args.newton_tol_mech,
                 '-newton_adaptive_tol_mech', args.newton_adaptive_tol_mech,
                 '-newton_forcing_term', args.newton_forcing_term,
                 '-newton_tol_cvsys', args.newton_tol_cvsys,
                 '-newton_line_search', args.newton_line_search,
                 '-newton_maxit_mech', args.newton_maxit_mech,
                 '-mech_activate_inertia', args.mech_activate_inertia,
                 '-mass_lumping', args.mass_lumping,
                 '-mech_rho_inf', args.mech_rho_inf,
                 '-mech_stiffness_damping', args.mech_stiffness_damping,
                 '-mech_mass_damping', args.mech_mass_damping,
                 '-mech_lambda_upd', args.mech_lambda_upd
                 ]

    return mech_opts


def set_cv_sys(args):
    """
    Assignment of the parameters related to the circulation system model.
    """

    mech_grid = model.mechanics.grid_id()
    vols = ['-numSurfVols', args.numSurfVols,
            '-surfVols[0].name', 'lvendo_closed',
            '-surfVols[0].surf_file', os.path.join(args.mesh_path, 'lvendo_closed'),
            '-surfVols[0].grid', mech_grid,
            '-surfVols[1].name', 'rvendo_closed',
            '-surfVols[1].surf_file', os.path.join(args.mesh_path, 'rvendo_closed'),
            '-surfVols[1].grid', mech_grid,
            ]

    if args.type_of_simulation == 'contraction':
        vols += ['-CV_coupling', 0,
                 '-CV_FE_coupling', 1,
                 '-CVS_mode', 0,
                 '-num_cavities', 2,
                 '-cavities[0].cav_type', 0,
                 '-cavities[0].cavP', 3,
                 # '-cavities[0].tube', 2,
                 '-cavities[0].cavVol', 0,
                 '-cavities[0].p0_cav', args.LV_EDP,  # mmHg
                 '-cavities[0].p0_in', args.LV_EDP,  # mmHg
                 '-cavities[0].p0_out', args.Ao_EDP,
                 # '-cavities[0].valve', 0,
                 '-cavities[0].state', -1,
                 '-lv_wk3.name', 'Aorta',
                 '-lv_wk3.R1', args.AV_resistance,
                 '-lv_wk3.R2', args.lv_wk3_R2,
                 '-lv_wk3.C', args. lv_wk3_C,
                 '-mitral_valve.Rfwd', args.mitral_valve_Rfwd,
                 '-mitral_valve.Rbwd', args.mitral_valve_Rbwd,
                 '-aortic_valve.Rfwd', args.aortic_valve_Rfwd,
                 '-cavities[1].cav_type', 1,
                 '-cavities[1].cavP', 4,
                 # '-cavities[1].tube', 2,
                 '-cavities[1].cavVol', 1,
                 '-cavities[1].p0_cav', args.RV_EDP,  # mmHg
                 '-cavities[1].p0_in', args.RV_EDP,  # mmHg
                 '-cavities[1].p0_out', args.PA_EDP,
                 # '-cavities[1].valve', 0,
                 '-cavities[1].state', -1,
                 '-rv_wk3.name', 'PA',
                 '-rv_wk3.R1', args.PV_resistance,
                 '-rv_wk3.R2', args.rv_wk3_R2,
                 '-rv_wk3.C', args.rv_wk3_C,
                 '-tricuspid_valve.Rfwd', args.tricuspid_valve_Rfwd,
                 '-tricuspid_valve.Rbwd', args.tricuspid_valve_Rbwd,
                 '-pulmonic_valve.Rfwd', args.pulmonic_valve_Rfwd
                 ]
    return vols


if __name__ == '__main__':
    # teardown decorator function and assign it to constant function RUN
    RUN = tools.carpexample(parser_commands, job_id)(run)
    RUN()
