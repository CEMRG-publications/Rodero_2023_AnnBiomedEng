#!/usr/bin/env python
"""
Electromechanics benchmark for 4 chamber heart
"""

# --- import packages ---------------------------------------------------------
from __future__ import print_function
import os

from carputils import settings
from carputils import tools
from carputils import model

import itertools

# --- command line options ----------------------------------------------------
def parser_commands():

    parser = tools.standard_parser()
    parser.add_argument('--type_of_simulation',
                        options=['unloading', 'contraction'],
                        help='Type of simulation to run.')
    parser.add_argument('--sim_name',
                        type=str,
                        default='test',
                        help='Name of the simulation. Recommendation is wave_batch_typeofsimulation.')
    parser.add_argument('--mesh_name',
                        type=str,
                        default='',
                        help='Name of the mesh as concatenation of all the modes.')
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

    mesh_dir = ''  #TODO

    cmd = tools.carp_cmd()

    cmd += ['-simID', job.ID,
            '-meshname', args.mesh_name]

    # --- configuration variables ---------------------------------------------
    cmd += setup_time_variables(args)

    # --- boundary settings ---------------------------------------------------
    cmd += setup_bc(args, mesh_dir)

    # --- cardiovascular system settings --------------------------------------
    cmd += set_cv_sys(args, mesh_dir)

    # --- mechanical settings -------------------------------------------------
    # define materials for different regions
    cmd += setup_material(args)

    # set solver properties
    #TODO: Continue here
    cmd += set_mechanic_options(args)

    # --- electrical settings ------------------------------------------------
    #if args.experiment == 'wk3':
        #cmd += setup_electrical_parameters()     # setup resolution dependent electrical parameters
    cmd += setup_stimuli(args)     # define stimuli
    cmd += setup_active(args)     # active stress setting

    # visualization
    cmd += setup_visualization(args.visualize, args.postprocess)

    if args.experiment == 'unloading':
        cmd += setup_unloading(args)

    # Alpha method
    if args.alpha_method:
        cmd += setup_alpha_method()

    # Run simulation
    job.carp(cmd)

# =============================================================================
#    FUNCTIONS
# =============================================================================


def full_list_of_parameters(args):
    auxiliary_parameters = {'num_mechanic_bs': 3,  # Number of mechanical boundary springs. Veins
                            'num_mechanic_ed': 1,  # Number of mechanical data sets. Pericardium, if contraction.
                            'num_mechanic_nbc': 5,  # Number of mechanical Neumann boundary conditions.
                                                    # Veins + cavities
                            'numSurfVols': 2,  # Number of surface enclosed volumes to keep track of. Ventricles. #TODO not generate the atrial surfaces
    }
    unloading_parameters = {}
    guccione_parameters = {'bf_guccione': 8., #TODO: review these numbers
                           'b_fs_guccione': 4.,
                           'b_t_guccione': 3.,
                           }
    neohookean_parameters = {'scaling_aorta': 26.66, #TODO: review these numbers
                             'scaling_extra_tissue': 1000.,
                             'scaling_PA': 3.7,
                             }
    tanh_parameters = {}
    windkessel_parameters = {'aortic_valve_Rfwd': 0., # Forward aortic resistance. 0 to avoid stenosis.
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
    numerics_parameters = {'loadStepping': 1.,  # Apply load in steps to avoid instabilities: (0) off, (1) auto,
                                                # (<0) manual choice'timedt': 10.,  # Time between temporal output (ms)
                           'mechDT': 1.,  # Time-step for mechanics (ms)
                           'spacedt': 10.,  # Time between spatial output (ms)
                           'tend': 1000.,  # Length of simulation (ms)
                           }

    all_parameters = dict(itertools.chain(auxiliary_parameters.items(), unloading_parameters.items(),
                                          guccione_parameters.items(), neohookean_parameters.items(),
                                          tanh_parameters.items(), windkessel_parameters.items(),
                                          numerics_parameters.items()))

    # Keep these parameters unless introduced by user
    for parameter in all_parameters:
        if not hasattr(args, parameter) or getattr(args, parameter) is None:
            setattr(args, parameter, all_parameters[parameter])

    return args


def setup_time_variables(args):

    time_opts = []

    time_opts += ['-timedt', args.timedt,
                  '-mechDT', args.mechDT,
                  '-spacedt', args.spacedt,
                  '-tend', args.tend,
                  '-loadStepping', args.loadStepping]

    return time_opts


def setup_bc(args, mesh_dir):

    if args.type_of_experiment == 'contraction':
        args.num_mechanic_nbc += 1
        args.num_mechanic_bs += 1

    nbc = ['-num_mechanic_nbc', args.num_mechanic_nbc,
           '-num_mechanic_bs', args.num_mechanic_bs,
           '-mechanic_bs[0].value', args.spring_BC,  # LSPV
           '-mechanic_bs[1].value', args.spring_BC,  # RSPV
           '-mechanic_bs[2].value', args.spring_BC,  # SVC
           '-mechanic_nbc[0].name', 'LSPV',
           '-mechanic_nbc[0].surf_file', os.path.join(mesh_dir, 'LSPV'),  #TODO: Remove the creation of the other BCs.
           '-mechanic_nbc[0].spring_idx', 0,
           '-mechanic_nbc[1].name', 'RSPV',
           '-mechanic_nbc[1].surf_file', os.path.join(mesh_dir, 'RSPV'),
           '-mechanic_nbc[1].spring_idx', 1,
           '-mechanic_nbc[2].name', 'SVC',
           '-mechanic_nbc[2].surf_file', os.path.join(mesh_dir, 'SVC'),
           '-mechanic_nbc[2].spring_idx', 2,
           '-mechanic_nbc[3].name', 'lvendo_closed',
           '-mechanic_nbc[3].surf_file', os.path.join(mesh_dir, 'lvendo_closed'),
           '-mechanic_nbc[3].pressure', str(float(args.LV_EDP)*0.133322),  # kPa
           '-mechanic_nbc[4].name', 'rvendo_closed',
           '-mechanic_nbc[4].surf_file', os.path.join(mesh_dir, 'rvendo_closed'),
           '-mechanic_nbc[4].pressure', str(float(args.RV_EDP)*0.133322)  # kPa
           ]

    if args.type_of_simulation == 'contraction':
        pericardium_file = os.path.join(mesh_dir, 'pericardium_penalty')
        nbc += ['-num_mechanic_ed', args.num_mechanic_ed,
                '-mechanic_ed[0].ncomp', 1,
                '-mechanic_ed[0].file', pericardium_file,
                '-mechanic_bs[3].edidx', 0,
                '-mechanic_bs[3].value', args.spring_BC,
                '-mechanic_nbc[5].name', 'Pericardium',
                '-mechanic_nbc[5].surf_file', os.path.join(mesh_dir, 'biv.epi'),
                '-mechanic_nbc[5].spring_idx', 3,
                '-mechanic_nbc[5].nspring_idx', 0,
                '-mechanic_nbc[5].nspring_config', 1]

    return nbc


def set_cv_sys(args,mesh_dir):

    mech_grid = model.mechanics.grid_id()
    vols = ['-numSurfVols', args.numSurfVols,
            '-surfVols[0].name', 'lvendo_closed',
            '-surfVols[0].surf_file', os.path.join(mesh_dir, 'lvendo_closed'),
            '-surfVols[0].grid', mech_grid,
            '-surfVols[1].name', 'rvendo_closed',
            '-surfVols[1].surf_file', os.path.join(mesh_dir, 'rvendo_closed'),
            '-surfVols[1].grid', mech_grid,
            ]

    if args.type_of_simulation == 'contraction':
        vols += ['-CV_coupling', 0,
                 '-CV_FE_coupling', 1,
                 '-CVS_mode', 0,
                 '-num_cavities', 2,
                 '-cavities[0].cav_type', 0,
                 '-cavities[0].cavP', 3,
                 '-cavities[0].tube', 2,
                 '-cavities[0].cavVol', 0,
                 '-cavities[0].p0_cav', args.LV_EDP,  # mmHg
                 '-cavities[0].p0_in', args.LV_EDP,  # mmHg
                 '-cavities[0].p0_out', args.Ao_EDP,
                 '-cavities[0].valve', 0,
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
                 '-cavities[1].tube', 2,
                 '-cavities[1].cavVol', 1,
                 '-cavities[1].p0_cav', args.RV_EDP,  # mmHg
                 '-cavities[1].p0_in', args.RV_EDP,  # mmHg
                 '-cavities[1].p0_out', args.PA_EDP,
                 '-cavities[1].valve', 0,
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


def setup_material(args):

    mech_opts = []

    # set bulk modulus kappa depending on finite element
    kappa = 1000 if args.mech_element == 'P1-P0' else 1e100

    ventricular_tags = [] #TODO: add the dictionary of the FEC % to tags here

    ventricles = model.mechanics.GuccioneMaterial(ventricular_tags, 'Ventricles', kappa=kappa, a=args.scaling_Guccione,
                                                  b_f=args.bf_guccione, b_fs=args.b_fs_guccione, b_t=args.b_t_guccione)
    atria = model.mechanics.NeoHookeanMaterial([3, 4], 'Atria', kappa=kappa, c=args.scaling_neohookean)
    valves = model.mechanics.NeoHookeanMaterial([7, 8, 9, 10], 'Valve planes', kappa=kappa, c=args.scaling_extra_tissue)
    inlets = model.mechanics.NeoHookeanMaterial([11, 12, 13, 14, 15, 16, 17], 'Inlet planes', kappa=kappa, c=1000.0)
    aorta = model.mechanics.NeoHookeanMaterial([5], 'Aorta', kappa=kappa, c=args.scaling_aorta)
    pa = model.mechanics.NeoHookeanMaterial([6], 'Pulmonary_Artery', kappa=kappa, c=args.scaling_PA)
    veins = model.mechanics.NeoHookeanMaterial([18, 19, 20, 21, 22, 23, 24], 'BC Veins', kappa=kappa,
                                               c=args.scaling_extra_tissue)

    mech_opts += model.optionlist([ventricles, atria, valves, inlets, aorta, pa, veins])

    mech_opts += ['-mech_vol_split_aniso', 1]

    return mech_opts


def setup_stimuli(args):
    """ Set stimulus
    """
    act_seq_path = join(settings.config.MESH_DIR, 'h4ckcl/actSeq')
    act_seq_file = join(act_seq_path, '{}_{}.dat'.format(args.case,args.act_seq_name))
    # general stimulus options
    stm_opts = ['-num_stim', 1,
                '-stimulus[0].stimtype', 8,
                '-stimulus[0].data_file', act_seq_file,
		'-diffusionOn', 0]

    return stm_opts

# ----------------------------------------------------------------------------
def setup_electrical_parameters():
    """ setup electrical parameters"""

    ep_opts = ['-bidomain', 0,
               '-diffusionOn', 0]

    return ep_opts

# ----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
def setup_solver_mech(args):
    """ adapt solver parameters """

    # for pt the switch is set automatically
    sopts = ['-pstrat', 1,
             '-pstrat_i', 1,
             '-krylov_tol_mech', 1e-10,
             '-krylov_norm_mech', 0,
             '-krylov_maxit_mech', 5000]

    sopts += ['-newton_atol_mech', 1e-6,
              '-newton_tol_mech', 1e-8,
              '-newton_adaptive_tol_mech', 2,
              '-newton_tol_cvsys', 1e-6,
              '-newton_line_search', 0,
              '-newton_maxit_mech', 20]

    solve_opts = ''

    if solve_opts != '':
        sopts += ['+F', solve_opts]

    return sopts

# -----------------------------------------------------------------------------
def set_mechanic_options(args):
    """ set mechanical options """

    # Track tissue volume
    mech_opts = ['-volumeTracking', 1,
                 '-numElemVols', 1,
                 '-elemVols[0].name', 'tissue',
                 '-elemVols[0].grid', model.mechanics.grid_id()]

    mech_opts += setup_solver_mech(args)

    return mech_opts

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def setup_active(args):
    """ setup active stress setting """

    if args.postprocess:
        return []
    if args.experiment == 'unloading': # All passive
        opts = ['-num_imp_regions', 1,
                '-num_stim', 0,
                '-imp_region[0].name','passive',
                '-imp_region[0].im','PASSIVE',
                '-imp_region[0].num_IDs',26,
                '-imp_region[0].ID[0]',1,
                '-imp_region[0].ID[1]',2,
                '-imp_region[0].ID[2]',3,
                '-imp_region[0].ID[3]',4,
                '-imp_region[0].ID[4]',5,
                '-imp_region[0].ID[5]',6,
                '-imp_region[0].ID[6]',7,
                '-imp_region[0].ID[7]',8,
                '-imp_region[0].ID[8]',9,
                '-imp_region[0].ID[9]',10,
                '-imp_region[0].ID[10]',11,
                '-imp_region[0].ID[11]',12,
                '-imp_region[0].ID[12]',13,
                '-imp_region[0].ID[13]',14,
                '-imp_region[0].ID[14]',15,
                '-imp_region[0].ID[15]',16,
                '-imp_region[0].ID[16]',17,
                '-imp_region[0].ID[17]',18,
                '-imp_region[0].ID[18]',19,
                '-imp_region[0].ID[19]',20,
                '-imp_region[0].ID[20]',21,
                '-imp_region[0].ID[21]',22,
                '-imp_region[0].ID[22]',23,
                '-imp_region[0].ID[23]',24,
                '-imp_region[0].ID[24]',25,
                '-imp_region[0].ID[25]',26]
    else:
        opts = ['-num_imp_regions', 3,
                '-imp_region[0].name', 'ventricles',
                '-imp_region[0].im', 'TT2',
                '-imp_region[0].num_IDs', 4,
                '-imp_region[0].ID[0]', 1,
                '-imp_region[0].ID[1]', 2,
                '-imp_region[0].ID[2]', 25,
                '-imp_region[0].ID[3]', 26,
                '-imp_region[1].name', 'atria',
                '-imp_region[1].im', 'COURTEMANCHE',
                '-imp_region[1].num_IDs', 2,
                '-imp_region[1].ID[0]', 3,
                '-imp_region[1].ID[1]', 4,
                '-imp_region[2].name', 'others',
                '-imp_region[2].im', 'PASSIVE',
                '-imp_region[2].num_IDs', 20,
                '-imp_region[2].ID[0]', 5,
                '-imp_region[2].ID[1]', 6,
                '-imp_region[2].ID[2]', 7,
                '-imp_region[2].ID[3]', 8,
                '-imp_region[2].ID[4]', 9,
                '-imp_region[2].ID[5]', 10,
                '-imp_region[2].ID[6]', 11,
                '-imp_region[2].ID[7]', 12,
                '-imp_region[2].ID[8]', 13,
                '-imp_region[2].ID[9]', 14,
                '-imp_region[2].ID[10]', 15,
                '-imp_region[2].ID[11]', 16,
                '-imp_region[2].ID[12]', 17,
                '-imp_region[2].ID[13]', 18,
                '-imp_region[2].ID[14]', 19,
                '-imp_region[2].ID[15]', 20,
                '-imp_region[2].ID[16]', 21,
                '-imp_region[2].ID[17]', 22,
                '-imp_region[2].ID[18]', 23,
                '-imp_region[2].ID[19]', 24]

        tanh_pars = set_tanh_stress_pars(args.t_peak, args.Tanh_time_relax, args.Tanh_time_contract, args.t_dur)
        opts += ['-imp_region[0].plugins', 'TanhStress',
                 '-imp_region[0].plug_param', tanh_pars]
    return opts

# -----------------------------------------------------------------------------
def setup_em_coupling():
    """
    Setup electromechanical coupling
    """
    # setup weak coupling
    coupling = ['-mech_use_actStress', 1,
                '-mech_lambda_upd', 1,
                '-mech_deform_elec', 0]

    # add velocity dependence fudge factor
    veldep = 0  # fundge factor to attenuate force-velocity dependence in [0,1]
    coupling += ['-veldep', veldep]

    return coupling

# =============================================================================
#    Active stress models
# =============================================================================

# -----------------------------------------------------------------------------
def set_tanh_stress_pars(arg_t_peak, arg_tau_r, arg_tau_c0, arg_t_dur):
    """
    Tanh stress parameters
    Active stress model as used in Andrew's thesis and
    Niederer et al 2011 Cardiovascular Research 89
    """
    # current settings   #   default values (see TanhStress.c in LIMPET)
    # ------------------ # ----------------------------------------------------
    t_emd = 20           #Marina, but check Electro-mechanical delay, https://doi.org/10.1093/oxfordjournals.eurheartj.a060210
    t_peak = arg_t_peak  # 100.0 kPa peak isometric tension
    tau_c0 = arg_tau_c0      #  40.0 ms  time constant contraction (t_r)
    tau_r = arg_tau_r        # 110.0 ms  time constant relaxation (t_d)
    t_dur = arg_t_dur       # 550.0 ms  duration of transient (t_max)
    lambda_0 = 0.7       #   0.7 -   sacomere length ratio (a_7)
    ld_deg = 6.0         #   5.0 -   degree of length dependence (a_6)
    ld_up = 500.0        # 500.0 ms  length dependence of upstroke time (a_4)
    ld_on = 0            #   0   -   turn on/off length dependence
    vm_thresh = -60.0    # -60.0 mV  threshold Vm for deriving LAT

    tpl = 't_emd={},Tpeak={},tau_c0={},tau_r={},t_dur={},lambda_0={},' + \
          'ld={},ld_up={},ldOn={},VmThresh={}'
    return tpl.format(t_emd, t_peak, tau_c0, tau_r, t_dur, lambda_0, ld_deg,
                      ld_up, ld_on, vm_thresh)

# -----------------------------------------------------------------------------
def setup_visualization(visualize, postprocess):
    """
    Visualization settings
    """
    # prevent output of large files
    vis_opts = ['-gridout_i', 0]

    if visualize:
        if postprocess:
            stress_val = 8  # principal stresses, elementwise
            strain_val = 4+8  # principal strains, elementwise
            vis_opts += ['-mech_output', 1+2,         # igb and vtk
                         '-vtk_output_mode', 3,       # vtu with zlib compr.
                         '-strain_value', strain_val,
                         '-stress_value', stress_val]
        else:
            vis_opts += ['-mech_output', 1+2,
                         '-vtk_output_mode', 3,       # vtu with zlib compr.
                         '-gzip_data', 0,
                         '-strain_value', 0,
                         '-stress_value', 0]
    else:
        vis_opts += ['-mech_output', 1,
                     '-strain_value', 0,
                     '-stress_value', 0]
    return vis_opts

# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
def setup_unloading(args):
    unload_opts = ['-experiment',   5,
                   '-loadStepping', args.loadStepping, 
                   '-unload_conv',  0, # 0  Volume based, 1 point based
                   '-unload_tol',   1e-3,
                   '-unload_err',   1,
                   '-unload_maxit', 10,
                   '-unload_stagtol',10.0]
    return unload_opts

# -----------------------------------------------------------------------------
def setup_alpha_method():
    alpha_opts = ['-mech_activate_inertia', 1,
                  '-mass_lumping', 1,
                  '-mech_rho_inf', 0.0,
                  '-mech_stiffness_damping', 0.1,
                  '-mech_mass_damping', 0.1]
    return alpha_opts

# --- MAIN -------------------------------------------------------------------
if __name__ == '__main__':
    # teardown decorator function and assign it to constant function RUN
    RUN = tools.carpexample(parser_commands, job_id)(run)
    RUN()
