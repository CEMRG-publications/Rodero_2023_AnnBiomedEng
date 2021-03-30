import os

def run_laplacian(heart, experiment = None):

    if experiment != None:
        experiment_vec = [experiment]
    else:
        experiment_vec = ["apba","epi","endoLV","endoRV"]

    scripts_folder = "/home/crg17/Desktop/KCL_projects/fitting/python"

    for exp in experiment_vec:
        os.system(os.path.join(scripts_folder,"run_fibres.py") + \
            " --experiment " + exp + " --current_case " + str(heart) + \
            " --np 20 --overwrite-behaviour overwrite")

def rb_bayer(heart, alpha_epi = -60, alpha_endo = 80, beta_epi = 25, beta_endo = -65):

    biv_dir = "/data/fitting/Full_Heart_Mesh_" + str(heart) + "/biv"
    fibre_dir = biv_dir + "/fibres"
    outname = "rb" + "_" + str(alpha_epi) + "_" + str(alpha_endo) + "_" + str(beta_epi) + "_" + str(beta_endo)

    for exp in ["apba","epi","endoLV","endoRV"]:

        os.system("igbextract " + os.path.join(fibre_dir,exp,"phie.igb") + \
                  " -o ascii -O " + os.path.join(fibre_dir,exp,"phie.dat"))
        os.system("sed -i s/'\\s'/'\\n'/g " + os.path.join(fibre_dir,exp,"phie.dat"))

    os.system("GlRuleFibers -m " + os.path.join(biv_dir,"biv") + \
                            " --type biv"  + \
                            " -a " + os.path.join(fibre_dir,"apba","phie.dat") + \
                            " -e " + os.path.join(fibre_dir,"epi","phie.dat") + \
                            " -l " + os.path.join(fibre_dir,"endoLV","phie.dat") + \
                            " -r " + os.path.join(fibre_dir,"endoRV","phie.dat") + \
                            " --alpha_endo " + str(alpha_endo) + \
                            " --alpha_epi " + str(alpha_epi) + \
                            " --beta_endo " + str(beta_endo) + \
                            " --beta_epi " + str(beta_epi) + \
                            " -o " + os.path.join(fibre_dir,outname + ".lon")
                            )

