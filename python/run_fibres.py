#!/usr/bin/env python2

"""

To run the experiment:

case_number=00; for fibre_exp in 'apba' 'epi' 'endoLV' 'endoRV'; do /home/crg17/Desktop/scripts/4chmodel/Python/run_fibers.py --experiment $fibre_exp --current_case $case_number --np 20; done; alert
"""

import os
import carputils
from carputils import tools

def parser():
    parser = carputils.tools.standard_parser()
    parser.add_argument('--experiment',
                        default='epi',
                        choices=['apba','epi','endoLV','endoRV'],
                        help='pick experiment type')
    parser.add_argument('--current_case',
                        help='pick the number of the case you are working on')
    return parser
   

def jobID(args):
    """
    Generate name of top level output directory.
    """
    subtest = args.experiment
    meshdir  = "/data/fitting/Full_Heart_Mesh_" + args.current_case + "/biv"
    simdir = os.path.join(meshdir,"fibres")

    if not os.path.exists(simdir):
      os.makedirs(simdir)

    return  os.path.join(simdir,subtest)

@tools.carpexample(parser, jobID)
def run(args, job):

    meshdir  = "/data/fitting/Full_Heart_Mesh_" + args.current_case + "/biv"

    # Add all the non-general arguments
    cmd  = carputils.tools.carp_cmd()
    cmd += ['-simID', job.ID]

    # Generate mesh
    meshname = '{}/biv'.format(meshdir)

    cmd += ['-meshname', meshname]


    # stimuli
    numStims, stims = setupStimuli(args.experiment, meshdir)

    # Bidomain simulation 

    cmd += ['-bidomain',1]

    cmd += ['-experiment',2]

    stimOpts = [ '-num_stim', numStims ] + stims
    cmd += stimOpts

    cmd += ['-num_gregions', 1,
            '-gregion[0].num_IDs', 1, 
            '-gregion[0].ID[0]', 1,
            '-gregion[0].g_il', 1,
            '-gregion[0].g_it', 1,
            '-gregion[0].g_in', 1,
            '-gregion[0].g_el', 1,
            '-gregion[0].g_et', 1,
            '-gregion[0].g_en', 1]

    # Run main CARP simulation
    # ------------------------
    job.carp(cmd)



# ============================================================================
#    EP FUNCTIONS
# ============================================================================

# --- set stimulus -----------------------------------------------------------
def setupStimuli(experiment,meshdir):

    # Generate electrical trigger options
    numStims = 2

    if experiment=='apba':
      filename_ground = 'LV_apex_epi'
      filename_pot = 'MVTV_base.surf'
    elif experiment=='epi':
      filename_ground = 'biv_endo.surf'
      filename_pot = 'biv.epi.surf'
    elif experiment=='endoLV':
      filename_ground = 'biv_noLVendo.surf'
      filename_pot = 'biv.lvendo.surf'
    elif experiment=='endoRV':
      filename_ground = 'biv_noRVendo.surf'
      filename_pot = 'biv.rvendo.surf'
    else:
      raise Exception('Unsupported experiment type')

    # define stimulus

    stims = [ '-stimulus[0].stimtype', 3,
              '-stimulus[0].vtx_file',meshdir + '/' + filename_ground,
              '-stimulus[1].stimtype', 2,
              '-stimulus[1].vtx_file',meshdir + '/' + filename_pot,
              '-stimulus[1].start',    0.,
              '-stimulus[1].strength', 1.,
              '-stimulus[1].duration', 1.]

    return numStims, stims

if __name__ == '__main__':
    run()
