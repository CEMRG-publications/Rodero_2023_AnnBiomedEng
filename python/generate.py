from os import sep
import skopt
import os
import numpy as np
from joblib import Parallel, delayed
import tqdm
import pathlib
import shutil

import prepare_mesh
import fibres
import UVC
import run_EP
import files_manipulations
from Historia.shared.design_utils import read_labels

SEED = 2

def input_EP_param(n_samples = None):

    param_names = ["mode_" + str(mode) for mode in range(1,19)]

    param_ranges = [
        (-64.624573989, 64.624573989),
        (-49.3066597463, 49.3066597463),
        (-41.573252807, 41.573252807),
        (-30.4268528225, 30.4268528225),
        (-25.7218779298, 25.7218779298),
        (-22.304645526, 22.304645526),
        (-21.5206199592, 21.5206199592),
        (-17.351659107, 17.351659107),
        (-17.0448821118, 17.0448821118),
        (-15.2509822608, 15.2509822608),
        (-14.4696049643, 14.4696049643),
        (-13.8879578509, 13.8879578509),
        (-12.6375573967, 12.6375573967),
        (-11.2271563098, 11.2271563098),
        (-11.0001216799, 11.0001216799),
        (-10.0484889853, 10.0484889853),
        (-9.2438333607, 9.2438333607),
        (-8.7797037167, 8.7797037167)
    ]

    param_names += [
        "alpha_endo",
        "alpha_epi",
        "lastFECtag",
        "CV_l",
        "k_fibre",
        "k_FEC"
    ]

    param_ranges += [
        (40.,90.), # degrees
        (-90., -40.), # degrees
        (26, 38),
        (0.01, 2.), # m/s
        (0.01, 1.), 
        (1., 10.)
    ]

    if(n_samples is None):
        n_samples = 10*len(param_ranges)

    space = skopt.space.Space(param_ranges)

    sobol = skopt.sampler.Sobol(min_skip = SEED, max_skip = SEED)
    x = sobol.generate(space.dimensions, n_samples, random_state = SEED)

    f = open(os.path.join("/data", "fitting", "EP_param_labels.txt"), "w")
    [f.write('%s\n' % key) for key in param_names]
    f.close()

    f = open(os.path.join("/data", "fitting", "EP_param_values.txt"), "w")
    [f.write('%s\n' % ' '.join(map(str, lhs_array))) for lhs_array in x]
    f.close()

def synthetic_mesh(mesh_ID):

    with open("/data/fitting/EP_param_labels.txt") as f:
        param_names = f.read().splitlines()

    with open("/data/fitting/EP_param_values.txt") as f:
        param_values = f.read().splitlines()

    modes_idx = np.where([x[0:4] == "mode" for x in param_names])[0]

    line2extract = param_values[mesh_ID].split(' ')

    modes_weights = [float(w) for w in np.take(line2extract, modes_idx)]

def EP_pipeline(mesh_ID):

    with open("/data/fitting/EP_param_labels.txt") as f:
        param_names = f.read().splitlines()

    with open("/data/fitting/EP_param_values.txt") as f:
        param_values = f.read().splitlines()

    alpha_epi_idx = int(np.where([x == "alpha_epi" for x in param_names])[0])
    alpha_endo_idx = int(np.where([x == "alpha_endo" for x in param_names])[0])
    lastFECtag_idx = int(np.where([x == "lastFECtag" for x in param_names])[0])
    CV_l_idx = int(np.where([x == "CV_l" for x in param_names])[0])
    k_fibre_idx = int(np.where([x == "k_fibre" for x in param_names])[0])
    k_FEC_idx = int(np.where([x == "k_FEC" for x in param_names])[0])

    line2extract = param_values[mesh_ID].split(' ')

    alpha_epi = float(line2extract[alpha_epi_idx])
    alpha_endo = float(line2extract[alpha_endo_idx])
    lastFECtag = int(line2extract[lastFECtag_idx])
    CV_l = float(line2extract[CV_l_idx])
    k_fibre = float(line2extract[k_fibre_idx])
    k_FEC = float(line2extract[k_FEC_idx])


    # prepare_mesh.extract_LDRB_biv(mesh_ID)
    # prepare_mesh.extract_MVTV_base(mesh_ID)
    # fibres.run_laplacian(mesh_ID)
    # fibres.full_pipeline(mesh_ID, alpha_epi, alpha_endo)
    # UVC.create(1, "MVTV")
    # UVC.bottom_third(1, "MVTV")
    # UVC.create_FEC(1, "MVTV")
    run_EP.carp2init(mesh_ID, lastFECtag, CV_l, k_fibre, k_FEC)
    run_EP.launch_init(mesh_ID, alpha_endo, alpha_epi)

def template_mesh_setup():
    prepare_mesh.extract_LDRB_biv("Template")
    prepare_mesh.extract_MVTV_base("Template")
    UVC.create("Full_Heart_Mesh_Template", "MVTV")
    UVC.bottom_third("Template", "MVTV")
    UVC.create_FEC("Template", "MVTV")
    fibres.run_laplacian("Template")

def EP_funct_param(n_samples = None, waveno = 0, subfolder = "."):

    path_lab = os.path.join("/data","fitting")
    path_match = os.path.join("/data","fitting","match")
    path_gpes = os.path.join(path_lab, subfolder, "wave" + str(waveno))

    pathlib.Path(path_gpes).mkdir(parents = True, exist_ok = True)
    
    param_names = read_labels(os.path.join(path_lab, "EP_funct_labels.txt"))


    param_ranges_lower = np.loadtxt(os.path.join(path_match, "input_range_lower.dat"), dtype=float)
    param_ranges_upper = np.loadtxt(os.path.join(path_match, "input_range_upper.dat"), dtype=float)

    param_ranges = [(param_ranges_lower[i],param_ranges_upper[i]) for i in range(len(param_ranges_lower))]

    if(n_samples is None):
        n_samples = 10*len(param_ranges)

    space = skopt.space.Space(param_ranges)

    sobol = skopt.sampler.Sobol(min_skip = SEED, max_skip = SEED)
    x = sobol.generate(space.dimensions, n_samples, random_state = SEED)

    # grid = skopt.sampler.Grid(border="include", use_full_layout=False)
    # x = grid.generate(space.dimensions, n_samples, random_state = SEED)

    f = open(os.path.join("/data", "fitting", "EP_funct_labels.txt"), "w")
    [f.write('%s\n' % key) for key in param_names]
    f.close()

    f = open(os.path.join(path_gpes, "X.dat"), "w")
    [f.write('%s\n' % ' '.join(map(str,[format(i, '.2f') for i in lhs_array]))) for lhs_array in x]
    f.close()

def template_EP(param_line):

    with open("/data/fitting/EP_funct_labels.txt") as f:
        param_names = f.read().splitlines()

    with open("/data/fitting/EP_funct_values.txt") as f:
        param_values = f.read().splitlines()

    alpha_idx = int(np.where([x == "alpha" for x in param_names])[0])
    lastFECtag_idx = int(np.where([x == "lastFECtag" for x in param_names])[0])
    CV_l_idx = int(np.where([x == "CV_l" for x in param_names])[0])
    k_fibre_idx = int(np.where([x == "k_fibre" for x in param_names])[0])
    k_FEC_idx = int(np.where([x == "k_FEC" for x in param_names])[0])

    line2extract = param_values[param_line].split(' ')

    alpha_endo = float(line2extract[alpha_idx])
    alpha_epi = -alpha_endo
    lastFECtag = int(line2extract[lastFECtag_idx])
    CV_l = float(line2extract[CV_l_idx])
    k_fibre = float(line2extract[k_fibre_idx])
    k_FEC = float(line2extract[k_FEC_idx])

    fibres.full_pipeline("Template", alpha_epi, alpha_endo)
    run_EP.carp2init("Template", lastFECtag, CV_l, k_fibre, k_FEC, suffix = param_line)
    run_EP.launch_init("Template", alpha_endo, alpha_epi, suffix = param_line)

def template_EP_parallel(line_from = 0, line_to = 10, waveno = 0, subfolder = "."):

    path_lab = os.path.join("/data","fitting")
    path_gpes = os.path.join(path_lab, subfolder, "wave" + str(waveno))
    path_EP = os.path.join("/data","fitting","Full_Heart_Mesh_Template","biv",
                            "EP_simulations")

    pathlib.Path(path_EP).mkdir(parents = True, exist_ok = True)

    def template_EP_explicit_arguments(alpha_endo_vec, lastFECtag_vec, CV_l_vec,
                                       k_fibre_vec, k_FEC_vec, param_line,
                                       path_EP,FEC_height_vec):

        simulation_file_name = os.path.join(path_EP,
                                str(format(alpha_endo_vec[param_line],'.2f')) +\
                                str(format(FEC_height_vec[param_line],'.2f')) +\
                                str(format(CV_l_vec[param_line],'.2f')) +\
                                str(format(k_fibre_vec[param_line],'.2f')) +\
                                str(format(k_FEC_vec[param_line],'.2f')) +\
                                ".dat"
                                )
        
        if not os.path.isfile(simulation_file_name):

            fibres.full_pipeline("Template",
                                alpha_epi = -alpha_endo_vec[param_line],
                                alpha_endo = alpha_endo_vec[param_line]
                                )

            run_EP.carp2init("Template",
                            lastFECtag_vec[param_line],
                            CV_l_vec[param_line],
                            k_fibre_vec[param_line],
                            k_FEC_vec[param_line],
                            simulation_file_name = simulation_file_name,
                            path_EP = path_EP
                            )

            run_EP.launch_init("Template", 
                            alpha_endo = alpha_endo_vec[param_line], 
                            alpha_epi = -alpha_endo_vec[param_line], 
                            simulation_file_name = simulation_file_name,
                            path_EP = path_EP
                            )

    with open(os.path.join(path_lab,"EP_funct_labels.txt")) as f:
        param_names = f.read().splitlines()

    with open(os.path.join(path_gpes,"X.dat")) as f:
        param_values = f.read().splitlines()

    line_to = min(line_to, len(param_values)-1)

    alpha_idx = int(np.where([x == "alpha" for x in param_names])[0])
    FEC_height_idx = int(np.where([x == "FEC_height" for x in param_names])[0])
    CV_l_idx = int(np.where([x == "CV_l" for x in param_names])[0])
    k_fibre_idx = int(np.where([x == "k_fibre" for x in param_names])[0])
    k_FEC_idx = int(np.where([x == "k_FEC" for x in param_names])[0])

    alpha_endo_vec = [float(line.split(' ')[alpha_idx]) for line in param_values]
    FEC_height_vec = [float(line.split(' ')[FEC_height_idx]) for line in param_values]
    CV_l_vec = [float(line.split(' ')[CV_l_idx]) for line in param_values]
    k_fibre_vec = [float(line.split(' ')[k_fibre_idx]) for line in param_values]
    k_FEC_vec = [float(line.split(' ')[k_FEC_idx]) for line in param_values]

    def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return array[idx]
        
    FEC_height_to_lastFECtag = {33:25,
                                35:26,
                                40:27,
                                45:28,
                                50:29,
                                55:30,
                                60:31,
                                65:32,
                                70:33,
                                75:34,
                                80:35,
                                85:36,
                                90:37,
                                95:38,
                                100:39}

    lastFECtag_vec = []
    for height in FEC_height_vec:
        height_key = find_nearest(list(FEC_height_to_lastFECtag.keys()),round(float(height)))
        lastFECtag = FEC_height_to_lastFECtag[height_key]
        lastFECtag_vec = np.append(lastFECtag_vec, lastFECtag)

    for param_line in tqdm.tqdm(range(line_from,line_to+1)):

        template_EP_explicit_arguments(alpha_endo_vec,
                                        lastFECtag_vec, 
                                        CV_l_vec, 
                                        k_fibre_vec, 
                                        k_FEC_vec, 
                                        param_line,
                                        path_EP,
                                        FEC_height_vec)
    return line_to + 1

def EP_output(waveno = 0, subfolder = "."):

    EP_dir = os.path.join("/data","fitting","Full_Heart_Mesh_Template","biv", 
                          "EP_simulations")
    labels_dir = os.path.join("/data","fitting")
    path2UVC = os.path.join("/data","fitting","Full_Heart_Mesh_Template","biv",
                            "UVC_MVTV","UVC")

    outpath = os.path.join("/data", "fitting",subfolder, "wave" + str(waveno))

    output_names = ["TAT","TATLV"]

    f = open(os.path.join(labels_dir,"EP_output_labels.txt"), "w")
    [f.write("%s\n" % phenotype) for phenotype in output_names]
    f.close()

    lvendo_vtx = files_manipulations.vtx.read(os.path.join("/data","fitting",
                                                      "Full_Heart_Mesh_Template",
                                                      "biv","biv.lvendo.surf.vtx"),
                                        "biv")

    UVC_Z = np.genfromtxt(os.path.join(path2UVC, "COORDS_Z.dat"),dtype = float)

    Z_90 = np.where(UVC_Z < 0.9)[0]

    Z_90_endo_mask = np.in1d(lvendo_vtx.indices, Z_90)
    Z_90_endo = lvendo_vtx.indices[Z_90_endo_mask]

    with open(os.path.join(outpath,"X.dat")) as f:
        param_values = f.read().splitlines()
    
    X_names = [line.replace(" ","") for line in param_values]

    TAT = [float("nan")]*len(X_names)
    TATLV = [float("nan")]*len(X_names)

    for i in tqdm.tqdm(range(len(X_names))):
        with open(os.path.join(EP_dir, X_names[i] + ".dat")) as g:
            AT_vec = g.read().splitlines()
        g.close()
        # AT_vec_float = []
        # for x in AT_vec:
        #     if i > 28:
        #         print(x)
        #         print(X_names[i])
        #     AT_vec_float.append(float(x))
        AT_vec_float = [float(x) for x in AT_vec]
        TAT[i] = max(AT_vec_float)
        TATLV[i] = max(np.array(AT_vec_float)[Z_90_endo.astype(int)])

    output_numbers = [TAT, TATLV]

    for i,varname in enumerate(output_names):
        np.savetxt(os.path.join(outpath, varname + ".dat"),
                    output_numbers[i],
                    fmt="%.2f")

def anatomical_output(heart_name, return_ISWT = True, return_WT = True,
                      return_EDD = True, return_LVmass = True,
                      return_LVendovol = True, close_LV = True,
                      return_LVOTdiam = True, return_RVOTdiam = True,
                      close_RV = True, close_LA = True, return_LAendovol = True,
                      close_RA = True, return_RAendovol = True,
                      return_RVlongdiam = True, return_RVbasaldiam = True,
                      return_RVendovol = True):
    
    path2fourch = os.path.join("/data","fitting",heart_name)
    biv_path = os.path.join(path2fourch,"biv")
    output_list = {}

    if close_LV and (return_LVendovol or return_LVOTdiam):
        prepare_mesh.close_LV_endo(heart_name)

    if return_LVOTdiam:
        av_la_pts = files_manipulations.pts.read(os.path.join(biv_path,"..","av_ao.pts"))
        av_la_surf = files_manipulations.surf.read(os.path.join(biv_path,"..","av_ao.elem"),heart_name)

        area_array = files_manipulations.area_or_vol_surface(pts_file = av_la_pts,
                        surf_file = av_la_surf, with_vol = False,
                        with_area = True)
        LVOT_area = sum(area_array)*1e-6 # In mm2

        LVOTdiam = 2*np.sqrt(LVOT_area/np.pi)

        output_list["LVOT diameter, mm"] = round(LVOTdiam,2)

    if close_RV and (return_RVOTdiam or return_RVlongdiam or return_RVbasaldiam or return_RVendovol):
        prepare_mesh.close_RV_endo(heart_name)

    if return_RVOTdiam:
        pv_pa_pts = files_manipulations.pts.read(os.path.join(biv_path,"..","pv_pa.pts"))
        pv_pa_surf = files_manipulations.surf.read(os.path.join(biv_path,"..","pv_pa.elem"),heart_name)

        area_array = files_manipulations.area_or_vol_surface(pts_file = pv_pa_pts,
                        surf_file = pv_pa_surf, with_vol = False,
                        with_area = True)
        RVOT_area = sum(area_array)*1e-6 # In mm2

        RVOTdiam = 2*np.sqrt(RVOT_area/np.pi)

        output_list["RVOT diameter, mm"] = round(RVOTdiam,2)        

    if return_RVendovol or return_RVlongdiam:
        rvendo_closed = files_manipulations.pts.read(os.path.join(path2fourch,"rvendo_closed.pts"))

        if return_RVendovol:
            rvendo_closed_surf = files_manipulations.surf.read(os.path.join(path2fourch,"rvendo_closed.elem"), heart_name)
            vol_array = files_manipulations.area_or_vol_surface(pts_file = rvendo_closed,
                        surf_file = rvendo_closed_surf, with_vol = True,
                        with_area = False)
            RVendovol = sum(vol_array)*1e-12 # In mL

            output_list["RV volume, mL"] = np.abs(round(RVendovol,2))


    if return_RVlongdiam or return_RVbasaldiam:
        tvrv_pts = files_manipulations.pts.read(os.path.join(path2fourch,"tv_rv.pts"))

        tvrv_surf = files_manipulations.surf.read(os.path.join(biv_path,"..","tv_rv.elem"),heart_name)

        area_array = files_manipulations.area_or_vol_surface(pts_file = tvrv_pts,
                        surf_file = tvrv_surf, with_vol = False,
                        with_area = True)
        tvrv_area = sum(area_array)*1e-6 # In mm2

        RVbasaldiam = 2*np.sqrt(tvrv_area/np.pi)

        if return_RVbasaldiam:
            output_list["RV basal diameter, mm"] = round(RVbasaldiam,2)
        
        if return_RVlongdiam:

            num_pts =  tvrv_pts.size

            sum_x = np.sum(tvrv_pts.p1)
            sum_y = np.sum(tvrv_pts.p2)
            sum_z = np.sum(tvrv_pts.p3)

            centroid = np.array([sum_x/num_pts, sum_y/num_pts, sum_z/num_pts])

            dist_vec = np.zeros(rvendo_closed.size)

            for i in range(len(dist_vec)):
                new_point = np.array([rvendo_closed.p1[i],rvendo_closed.p2[i],rvendo_closed.p3[i]])
                dist_vec[i] = np.linalg.norm(centroid - new_point)

            RVlongdiam_centroid = max(dist_vec)*1e-3

            RVlongdiam = np.sqrt((RVbasaldiam/2.)**2 + RVlongdiam_centroid**2)

            output_list["RV long. diameter, mm"] = round(RVlongdiam,2)

    if close_LA and return_LAendovol:
        prepare_mesh.close_LA_endo(heart_name)
    
    if return_LAendovol:
        laendo_closed_pts = files_manipulations.pts.read(os.path.join(biv_path,"..","laendo_closed.pts"))
        laendo_closed_surf = files_manipulations.surf.read(os.path.join(biv_path,"..","laendo_closed.elem"),heart_name)

        vol_array = files_manipulations.area_or_vol_surface(pts_file = laendo_closed_pts,
                        surf_file = laendo_closed_surf, with_vol = True,
                        with_area = False)
        LAendovol = sum(vol_array)*1e-12 # In mL

        output_list["LA volume, mL"] = round(LAendovol,2)

    if close_RA and return_RAendovol:
        prepare_mesh.close_RA_endo(heart_name)
    
    if return_RAendovol:
        raendo_closed_pts = files_manipulations.pts.read(os.path.join(biv_path,"..","raendo_closed.pts"))
        raendo_closed_surf = files_manipulations.surf.read(os.path.join(biv_path,"..","raendo_closed.elem"),heart_name)

        vol_array = files_manipulations.area_or_vol_surface(pts_file = raendo_closed_pts,
                        surf_file = raendo_closed_surf, with_vol = True,
                        with_area = False)
        RAendovol = sum(vol_array)*1e-12 # In mL

        output_list["RA volume, mL"] = round(RAendovol,2)

    if return_LVendovol:
        lvendo_closed_pts = files_manipulations.pts.read(os.path.join(biv_path,"..","lvendo_closed.pts"))
        lvendo_closed_surf = files_manipulations.surf.read(os.path.join(biv_path,"..","lvendo_closed.elem"),heart_name)

        temp_vol = files_manipulations.area_or_vol_surface(pts_file = lvendo_closed_pts,
                        surf_file = lvendo_closed_surf, with_vol = True,
                        with_area = False)

        # d12 = np.array([np.nan,np.nan,np.nan])
        # d13 = np.array([np.nan,np.nan,np.nan])
        # temp_vol = np.array([])
        # for i in range(lvendo_closed_surf.size):
        #     d13[0] = lvendo_closed_pts.p1[lvendo_closed_surf.i1[i]] - \
        #             lvendo_closed_pts.p1[lvendo_closed_surf.i3[i]]
        #     d13[1] = lvendo_closed_pts.p2[lvendo_closed_surf.i1[i]] - \
        #             lvendo_closed_pts.p2[lvendo_closed_surf.i3[i]]
        #     d13[2] = lvendo_closed_pts.p3[lvendo_closed_surf.i1[i]] - \
        #             lvendo_closed_pts.p3[lvendo_closed_surf.i3[i]]

        #     d12[0] = lvendo_closed_pts.p1[lvendo_closed_surf.i1[i]] - \
        #             lvendo_closed_pts.p1[lvendo_closed_surf.i2[i]]
        #     d12[1] = lvendo_closed_pts.p2[lvendo_closed_surf.i1[i]] - \
        #             lvendo_closed_pts.p2[lvendo_closed_surf.i2[i]]
        #     d12[2] = lvendo_closed_pts.p3[lvendo_closed_surf.i1[i]] - \
        #             lvendo_closed_pts.p3[lvendo_closed_surf.i2[i]]
            
        #     cr = np.cross(d13,d12)
        #     crNorm = np.linalg.norm(cr)

        #     area_tr = 0.5*crNorm
        #     zMean = (lvendo_closed_pts.p3[lvendo_closed_surf.i1[i]] + \
        #             lvendo_closed_pts.p3[lvendo_closed_surf.i2[i]] + \
        #             lvendo_closed_pts.p3[lvendo_closed_surf.i3[i]])/3.

        #     nz = cr[2]/crNorm

        #     temp_vol = np.append(temp_vol,area_tr*zMean*nz)

        LV_chamber_vol = np.abs(sum(temp_vol)*1e-12) # In mL
        
        output_list["LV end-diastolic volume, mL"] = round(LV_chamber_vol,2)
    
    if return_ISWT or return_WT or return_EDD or return_LVmass:
        path2UVC = os.path.join(biv_path, "UVC_MVTV", "UVC")
        biv_pts = files_manipulations.pts.read(os.path.join(biv_path,"biv.pts"))
        UVC_Z_elem = np.genfromtxt(os.path.join(path2UVC, "COORDS_Z_elem_scaled.dat"),dtype = float)
        UVC_Z = np.genfromtxt(os.path.join(path2UVC, "COORDS_Z.dat"),dtype = float)

        if return_LVmass:
            biv_elem = files_manipulations.elem.read(os.path.join(biv_path,"biv.elem"))
            UVC_V_elem = np.genfromtxt(os.path.join(path2UVC, "COORDS_V_elem.dat"),dtype = float)

            def six_vol_element_cm3(i):
                result =  np.linalg.det(np.array([np.array([1e-4*biv_pts.p1[biv_elem.i1[i]], 1e-4*biv_pts.p2[biv_elem.i1[i]], 1e-4*biv_pts.p3[biv_elem.i1[i]], 1.], dtype = float),
                                    np.array([1e-4*biv_pts.p1[biv_elem.i2[i]], 1e-4*biv_pts.p2[biv_elem.i2[i]], 1e-4*biv_pts.p3[biv_elem.i2[i]], 1.], dtype = float),
                                    np.array([1e-4*biv_pts.p1[biv_elem.i3[i]], 1e-4*biv_pts.p2[biv_elem.i3[i]], 1e-4*biv_pts.p3[biv_elem.i3[i]], 1.], dtype = float),
                                    np.array([1e-4*biv_pts.p1[biv_elem.i4[i]], 1e-4*biv_pts.p2[biv_elem.i4[i]], 1e-4*biv_pts.p3[biv_elem.i4[i]], 1.], dtype = float)
                                    ], dtype = float)
                                    )
                return np.abs(result)
            
            LV_mass_idx = np.intersect1d(np.where(UVC_Z_elem < 0.9), np.where(UVC_V_elem < 0))

            six_vol_LV = []
            six_vol_LV.append(Parallel(n_jobs=20)(delayed(six_vol_element_cm3)(i) for i in range(len(LV_mass_idx))))

            LV_mass = 1.05*sum(six_vol_LV[0])/6.

            output_list["LV mass, g"] = round(LV_mass,2)
        if return_ISWT or return_WT or return_EDD:
            septum_idx = files_manipulations.vtx.read(os.path.join(biv_path,"biv.rvsept.surf.vtx"),"biv")
            lvendo_idx = files_manipulations.vtx.read(os.path.join(biv_path,"biv.lvendo.surf.vtx"),"biv")
            UVC_PHI = np.genfromtxt(os.path.join(path2UVC, "COORDS_PHI.dat"),dtype = float)
            septum_Z = UVC_Z[septum_idx.indices]
            lvendo_Z = UVC_Z[lvendo_idx.indices]
            septum_band_idx = septum_idx.indices[np.intersect1d(np.where(septum_Z > 0.6),
                                                        np.where(septum_Z < 0.9)
                                                        )
                                                ]
            lvendo_band_idx = lvendo_idx.indices[np.intersect1d(np.where(lvendo_Z > 0.6),
                                                        np.where(lvendo_Z < 0.9)
                                                        )
                                        ]
            septum_band_PHI = UVC_PHI[septum_band_idx]
            lvendo_band_PHI = UVC_PHI[lvendo_band_idx]

            midpoint_PHI = (max(septum_band_PHI) + min(septum_band_PHI))/2.
            bandwith_PHI = max(septum_band_PHI) - min(septum_band_PHI)
            min_PHI = midpoint_PHI-bandwith_PHI/6.
            max_PHI = midpoint_PHI+bandwith_PHI/6.

        if return_ISWT or return_EDD:
            lvendo_septum_ROI_idx = lvendo_band_idx[np.intersect1d(np.where(lvendo_band_PHI > min_PHI),
                                                    np.where(lvendo_band_PHI < max_PHI)
                                                    )
                                    ]
            lvendo_septum_ROI_pts = biv_pts.extract(files_manipulations.vtx(lvendo_septum_ROI_idx,"biv"))

            if return_ISWT:
                septum_ROI_idx = septum_band_idx[np.intersect1d(np.where(septum_band_PHI > min_PHI),
                                                        np.where(septum_band_PHI < max_PHI)
                                                        )
                                        ]
                septum_ROI_pts = biv_pts.extract(files_manipulations.vtx(septum_ROI_idx,"biv"))

                def dist_from_septum_to_lvendo_septum(i):
                    return lvendo_septum_ROI_pts.min_dist(np.array([septum_ROI_pts.p1[i], septum_ROI_pts.p2[i], septum_ROI_pts.p3[i]]))
                
                septum_ROI_thickness = []
                septum_ROI_thickness.append(Parallel(n_jobs=20)(delayed(dist_from_septum_to_lvendo_septum)(i) for i in range(septum_ROI_pts.size)))

                # Itraventricular septal wall thickness, in mm
                ISWT = 1e-3*np.median(septum_ROI_thickness)

                output_list["Interventricular septal wall thickness, mm"] = round(ISWT,2)
        
        if return_EDD or return_WT:
            # Wall thickness as the opposite side of the septum
            if min_PHI <= 0:
                wall_min_PHI = min_PHI + np.pi
            else:
                wall_min_PHI = min_PHI - np.pi
            
            
            if max_PHI <= 0:
                wall_max_PHI = max_PHI + np.pi
            else:
                wall_max_PHI = max_PHI - np.pi

            lvendo_wall_ROI_idx = lvendo_band_idx[np.intersect1d(np.where(lvendo_band_PHI > wall_min_PHI),
                                                    np.where(lvendo_band_PHI < wall_max_PHI)
                                                    )
                                    ]
            lvendo_wall_ROI_pts = biv_pts.extract(files_manipulations.vtx(lvendo_wall_ROI_idx,"biv"))

            if return_WT:
                bivepi_idx = files_manipulations.vtx.read(os.path.join(biv_path,"biv.epi.surf.vtx"),
                                                    "biv")
                bivepi_Z = UVC_Z[bivepi_idx.indices]
                bivepi_band_idx = bivepi_idx.indices[np.intersect1d(np.where(bivepi_Z > 0.6),
                                                        np.where(bivepi_Z < 0.9)
                                                        )
                                        ]
                bivepi_band_PHI = UVC_PHI[bivepi_band_idx]
                lvepi_ROI_idx = bivepi_band_idx[np.intersect1d(np.where(bivepi_band_PHI > wall_min_PHI),
                                                        np.where(bivepi_band_PHI < wall_max_PHI)
                                                        )
                                        ]
                lvepi_ROI_pts = biv_pts.extract(files_manipulations.vtx(lvepi_ROI_idx,"biv"))

                def dist_from_lvepi_to_lvendo_wall(i):
                    return lvendo_wall_ROI_pts.min_dist(np.array([lvepi_ROI_pts.p1[i], lvepi_ROI_pts.p2[i], lvepi_ROI_pts.p3[i]]))

                wall_ROI_thickness = []
                wall_ROI_thickness.append(Parallel(n_jobs=20)(delayed(dist_from_lvepi_to_lvendo_wall)(i) for i in range(lvepi_ROI_pts.size)))

                # Wall thickness, in mm
                WT = 1e-3*np.median(wall_ROI_thickness)

                output_list["Posterior wall thickness, mm"] = round(WT,2)

            if return_EDD:
                def dist_from_lvendo_septum_to_lvendo_wall(i):
                    return lvendo_wall_ROI_pts.min_dist(np.array([lvendo_septum_ROI_pts.p1[i], lvendo_septum_ROI_pts.p2[i], lvendo_septum_ROI_pts.p3[i]]))

                ED_dimension_ROI = []            
                ED_dimension_ROI.append(Parallel(n_jobs=20)(delayed(dist_from_lvendo_septum_to_lvendo_wall)(i) for i in range(lvendo_septum_ROI_pts.size)))

                # End-diastolic dimension, mm
                EDD = 1e-3*np.median(ED_dimension_ROI)

                output_list["Diastolic LV internal dimension, mm"] = round(EDD,2)

    return output_list

def mechanics_archer2_setup(fourch_name):
    
    path2fourch = os.path.join("/data","fitting",fourch_name)
    path2biv = os.path.join(path2fourch,"biv")

    prepare_mesh.extract_peri_base(fourch_name)
    UVC.create(fourch_name, "peri")
    
    # We take the maximum UVC_Z between the original UVC and the peri UVC

    UVC_Z_MVTV_elem = np.genfromtxt(os.path.join(path2biv, "UVC_MVTV", "UVC", "COORDS_Z_elem.dat"),dtype = float)
    UVC_Z_peri_elem = np.genfromtxt(os.path.join(path2biv, "UVC_peri", "UVC", "COORDS_Z_elem.dat"),dtype = float)

    UVC_Z_max = np.maximum(UVC_Z_MVTV_elem, UVC_Z_peri_elem)

    # The polinomial for the pericardium. Max penalty at the apex, nothing from where UVC >= 0.82

    penalty_biv = 1.5266*(0.82 - UVC_Z_max)**3 - 0.37*(0.82 - UVC_Z_max)**2 + 0.4964*(0.82 - UVC_Z_max)
    penalty_biv[UVC_Z_max > 0.82] = 0.0

    # All this is on the biv, we need to map it to the whole heart.

    fourch_elem = files_manipulations.elem.read(os.path.join(path2fourch, fourch_name + ".elem"))
    tags_vec = fourch_elem.tags

    penalty_fourch = np.zeros(fourch_elem.size)

def filter_output(waveno = 0, subfolder = ".", skip = False):
    
    path_match = os.path.join("/data","fitting","match")
    path_lab = os.path.join("/data","fitting")
    path_gpes = os.path.join(path_lab, subfolder, "wave" + str(waveno))

    exp_mean = np.loadtxt(os.path.join(path_match, "exp_mean.txt"), dtype=float)
    exp_std = np.loadtxt(os.path.join(path_match, "exp_std.txt"), dtype=float)
    ylabels = read_labels(os.path.join(path_lab, "EP_output_labels.txt"))

    if skip:
        shutil.copy(os.path.join(path_gpes, "X.dat"),os.path.join(path_gpes, "X_feasible.dat"))
    else:

        X = np.loadtxt(os.path.join(path_gpes, "X.dat"), dtype=float)
        
        idx_feasible = np.array([], dtype = int)

        for i, output_name in enumerate(ylabels):

            unfeas_lower = max(0, exp_mean[i] - 5 * exp_std[i])
            unfeas_upper = exp_mean[i] + 5 * exp_std[i]

            y = np.loadtxt(os.path.join(path_gpes, output_name + ".dat"),dtype=float)
            idx_no_lower = np.where(y > unfeas_lower)[0]
            idx_no_upper = np.where(y < unfeas_upper)[0]
            if  i == 0:
                idx_feasible = np.intersect1d(idx_no_lower, idx_no_upper)
            else:
                idx_feasible = np.intersect1d(idx_feasible,np.intersect1d(idx_no_lower, idx_no_upper))

        idx_feasible = np.unique(idx_feasible)

        np.savetxt(os.path.join(path_gpes, "X_feasible.dat"), X[idx_feasible],
                    fmt="%.2f")

    for output_name in ylabels:
        if skip:
            shutil.copy(os.path.join(path_gpes, output_name + ".dat"),
                        os.path.join(path_gpes, output_name + "_feasible.dat"))
        else:
            y = np.loadtxt(os.path.join(path_gpes, output_name + ".dat"),dtype=float)

            np.savetxt(os.path.join(path_gpes, output_name + "_feasible.dat"),
                        y[idx_feasible], fmt="%.2f")
    if not skip:
        print("A total number of " + str(len(idx_feasible)) + " points are feasible")
