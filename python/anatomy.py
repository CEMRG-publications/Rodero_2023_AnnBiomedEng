import csv
from joblib import Parallel, delayed
import numpy as np
import os
import pathlib
import prepare_mesh
import skopt
import time
import tqdm

import CardiacMeshConstruction.pipeline as PCApip
import fibres
import UVC
import run_EP
import files_manipulations

SEED = 2
# ----------------------------------------------------------------
# Make the code reproducible
# random.seed(SEED)
np.random.seed(SEED)
# torch.manual_seed(SEED)


def input(n_samples = None, waveno = 0, subfolder = "."):
    """Function to generate the input parameter space using sobol' semi random
    sequences.

    Args:
        n_samples (int, optional): Number of points to generate. Defaults to 
        None.
        waveno (int, optional):  Wave number, specifies the folder name. 
        Defaults to 0.
        subfolder (str, optional): Subfolder name of /data/fitting to work on. 
        Defaults to ".".
    """

    path_lab = os.path.join("/data","fitting")
    path_match = os.path.join("/data","fitting","match")
    path_gpes = os.path.join(path_lab, subfolder, "wave" + str(waveno))

    pathlib.Path(path_gpes).mkdir(parents = True, exist_ok = True)
    
    param_ranges_lower_anatomy = np.loadtxt(os.path.join(path_match, "anatomy_input_range_lower.dat"), dtype=float)
    param_ranges_upper_anatomy = np.loadtxt(os.path.join(path_match, "anatomy_input_range_upper.dat"), dtype=float)

    param_ranges_lower_EP = np.loadtxt(os.path.join(path_match, "EP_input_range_lower.dat"), dtype=float)
    param_ranges_upper_EP = np.loadtxt(os.path.join(path_match, "EP_input_range_upper.dat"), dtype=float)

    param_ranges_lower = np.append(param_ranges_lower_anatomy, param_ranges_lower_EP)
    param_ranges_upper = np.append(param_ranges_upper_anatomy, param_ranges_upper_EP)

    param_ranges = [(param_ranges_lower[i],param_ranges_upper[i]) for i in range(len(param_ranges_lower))]

    if(n_samples is None):
        n_samples = 10*len(param_ranges)

    space = skopt.space.Space(param_ranges)

    sobol = skopt.sampler.Sobol(min_skip = SEED, max_skip = SEED)
    x = sobol.generate(space.dimensions, int(n_samples), random_state = SEED)

    x_EP = []
    x_anatomy = []
    for row in x:
        x_anatomy.append(row[:len(param_ranges_lower_anatomy)])
        x_EP.append(row[len(param_ranges_lower_anatomy):(len(param_ranges_lower_anatomy)+len(param_ranges_lower_EP))])

    f = open(os.path.join(path_gpes, "X.dat"), "w")
    [f.write('%s\n' % ' '.join(map(str,[format(i, '.2f') for i in lhs_array]))) for lhs_array in x]
    f.close()

    f = open(os.path.join(path_gpes, "X_EP.dat"), "w")
    [f.write('%s\n' % ' '.join(map(str,[format(i, '.2f') for i in lhs_array]))) for lhs_array in x_EP]
    f.close()

    with open(os.path.join(path_gpes,"X_anatomy.csv"), mode='w') as f:
        f_writer = csv.writer(f, delimiter=',')

        f_writer.writerow(["Mode" + str(i) for i in range(1,19)])

        for current_line in range(len(x_anatomy)):
            output = np.zeros(18)
            output[0:9] = x_anatomy[current_line]
            f_writer.writerow(["{0:.2f}".format(round(i,2)) for i in output])

    f.close()
def build_meshes(waveno = 0, subfolder = ".", force_construction = False):
    """Function to generate meshes using deformetrica given the modes values.

    Args:
        waveno (int, optional): Wave number, defines the folder. Defaults to 0.
        subfolder (str, optional): Subfolder where to work on. Defaults to ".".
        force_construction (bool, optional): If True, it generates the meshes,
        even if they already exit. Defaults to False.

    Returns:
        had_to_run_new (bool): Boolean variable to know if a new mesh was 
        created.
    """
    
    path_lab = os.path.join("/data","fitting")
    path_gpes = os.path.join(path_lab, subfolder, "wave" + str(waveno))
    temp_outpath = os.path.join(path_lab,"temp_meshes")
    had_to_run_new = False

    with open(os.path.join(path_gpes,"X_anatomy.csv")) as f:
        anatomy_values = f.read().splitlines()

    pathlib.Path(temp_outpath).mkdir(parents = True, exist_ok = True)
    
    for i in tqdm.tqdm(range(len(anatomy_values)-1)):
        temp_base_name = "wave" + str(waveno) + "_" + str(i)
        final_base_name = "heart_" + anatomy_values[i+1].replace(",","")[:-36]
        mesh_path = os.path.join(path_lab, final_base_name)

        if (not os.path.isfile(os.path.join(mesh_path,final_base_name + ".elem")) and not os.path.isfile(os.path.join(mesh_path,final_base_name + "_default.elem"))) or force_construction:
            had_to_run_new = True
            if not os.path.isfile(os.path.join(temp_outpath,"wave" + str(waveno) + "_" + str(i) + ".vtk")):
                csv_file_name = os.path.join(temp_outpath,"wave" + str(waveno) + "_" + str(i) + ".csv")
                np.savetxt(csv_file_name,np.array([anatomy_values[0],anatomy_values[i+1]]),fmt='%s',delimiter='\n')
                
                print(final_base_name)
                print(time.strftime("%H:%M:%S", time.localtime()))

                os.chdir(os.path.join("/home","crg17","Desktop","KCL_projects","fitting","python","CardiacMeshConstruction_outside"))
                os.system("python3.6 ./pipeline.py " + csv_file_name + " " + temp_outpath + " " + str(waveno) + "_" + str(i))

            pathlib.Path(mesh_path).mkdir(parents = True, exist_ok = True)

            os.system("cp " + os.path.join(temp_outpath,temp_base_name + ".vtk") +\
                    " " + os.path.join(mesh_path, final_base_name + ".vtk"))
            prepare_mesh.vtk_mm2carp_um(fourch_name = final_base_name)
    
    os.system("rm -rf " + temp_outpath)
    return had_to_run_new
def EP_setup(waveno = 0, subfolder = "."):
    """Function to prepare the mesh ready for an EP simulation.

    Args:
        waveno (int, optional): Wave number, defines the folder. Defaults to 0.
        subfolder (str, optional): Subfolder name of /data/fitting to work on. 
        Defaults to ".".

    Returns:
        had_to_run_new (bool): If a new simulation was run, returns to True.
        Otherwise is False.
    """

    path_lab = os.path.join("/data","fitting")
    path_gpes = os.path.join(path_lab, subfolder, "wave" + str(waveno))
    had_to_run_new = False
    with open(os.path.join(path_gpes,"X_anatomy.csv")) as f:
        anatomy_values = f.read().splitlines()
    
    for i in tqdm.tqdm(range(len(anatomy_values)-1)):
        final_base_name = "heart_" + anatomy_values[i+1].replace(",","")[:-36]
        mesh_path = os.path.join(path_lab, final_base_name)

        if not os.path.isfile(os.path.join(mesh_path,"biv","biv_noRVendo.surf.vtx")):
            prepare_mesh.extract_LDRB_biv(final_base_name)
        if not os.path.isfile(os.path.join(mesh_path,"biv","MVTV_base.surf.vtx")):
            prepare_mesh.extract_MVTV_base(final_base_name)
        if (not os.path.isfile(os.path.join(mesh_path,"biv","UVC_MVTV", "UVC","COORDS_V_elem_scaled.dat"))) or\
           (not os.path.isfile(os.path.join(mesh_path,"biv","UVC_MVTV", "UVC","COORDS_PHI_elem_scaled.dat"))) or\
           (not os.path.isfile(os.path.join(mesh_path,"biv","UVC_MVTV", "UVC","COORDS_Z_elem_scaled.dat"))) or\
           (not os.path.isfile(os.path.join(mesh_path,"biv","UVC_MVTV", "UVC","COORDS_RHO_elem_scaled.dat"))):
            UVC.create(final_base_name, "MVTV")
        if not os.path.isfile(os.path.join(mesh_path,"biv","EP","bottom_third.vtx")):
            UVC.bottom_third(final_base_name, "MVTV")
        if not os.path.isfile(os.path.join(mesh_path,"biv","biv_FEC.elem")):
            UVC.create_FEC(final_base_name, "MVTV")
        if not os.path.isfile(os.path.join(mesh_path,"biv","fibres","endoRV","phie.igb")):
            had_to_run_new = True
            fibres.run_laplacian(final_base_name)
    
    return had_to_run_new
def EP_simulations(waveno = 0, subfolder = ".", map_fibres_to_fourch = False):
    """Function to prepare the mesh and run the EP simulation. It works in a 
    sequential way to improve debugging.

    Args:
        waveno (int, optional): Wave number, defines the folder name. Defaults 
        to 0.
        subfolder (str, optional): [description]. Folder name in /data/fitting to work  on. 
        Defaults to ".".
        map_fibres_to_fourch (bool, optional): If True, maps the fibres from the
        biventricular mesh to the four chamber mesh. Defaults to False.

    Returns:
        had_to_run_new (bool): If a new simulation was run, returns to True.
        Otherwise is False.
    """

    path_lab = os.path.join("/data","fitting")
    path_gpes = os.path.join(path_lab, subfolder, "wave" + str(waveno))

    had_to_run_new = False

    with open(os.path.join(path_lab,"EP_funct_labels.txt")) as f:
        param_names = f.read().splitlines()

    with open(os.path.join(path_gpes,"X_EP.dat")) as f:
        param_values = f.read().splitlines()
    
    with open(os.path.join(path_gpes,"X_anatomy.csv")) as f:
        anatomy_values = f.read().splitlines()
    
    alpha_idx = int(np.where([x == "alpha" for x in param_names])[0])
    FEC_height_idx = int(np.where([x == "FEC_height" for x in param_names])[0])
    CV_l_idx = int(np.where([x == "CV_l" for x in param_names])[0])
    k_fibre_idx = int(np.where([x == "k_fibre" for x in param_names])[0])
    k_FEC_idx = int(np.where([x == "k_FEC" for x in param_names])[0])

    def find_nearest(array, value):
        """Function to find the closest value in an array to a given value.

        Args:
            array (array): Array to look into to find the closest value.
            value (same as array): Value to find the closest number in array.

        Returns:
            Closest value to "value" in "array".
        """
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

    for line_num in tqdm.tqdm(range(len(param_values))):
        FEC_height = round(float(param_values[line_num].split(' ')[FEC_height_idx]),2)
        height_key = find_nearest(list(FEC_height_to_lastFECtag.keys()),round(float(FEC_height)))

        alpha = round(float(param_values[line_num].split(' ')[alpha_idx]),2)
        lastFECtag = round(float(FEC_height_to_lastFECtag[height_key]),2)
        CV_l = round(float(param_values[line_num].split(' ')[CV_l_idx]),2) 
        k_fibre = round(float(param_values[line_num].split(' ')[k_fibre_idx]),2)
        k_FEC = round(float(param_values[line_num].split(' ')[k_FEC_idx]),2)

        fourch_name = "heart_" + anatomy_values[line_num+1].replace(",","")[:-36]
        
        path_EP = os.path.join("/data","fitting",fourch_name,"biv",
                            "EP_simulations")
        
        pathlib.Path(path_EP).mkdir(parents = True, exist_ok = True)

        
        simulation_file_name = os.path.join(path_EP,
                                '{0:.2f}'.format(alpha) +\
                                '{0:.2f}'.format(FEC_height) +\
                                '{0:.2f}'.format(CV_l) +\
                                '{0:.2f}'.format(k_fibre) +\
                                '{0:.2f}'.format(k_FEC) +\
                                ".dat"
                                )

        if not os.path.isfile(os.path.join("/data","fitting",fourch_name,"biv",
                              "fibres", "rb_-" + str('{0:.2f}'.format(alpha)) + "_" + str('{0:.2f}'.format(alpha)) + ".elem")):
            had_to_run_new = True

            fibres.full_pipeline(fourch_name = fourch_name,
                                alpha_epi = -alpha,
                                alpha_endo = alpha,
                                map_fibres_to_fourch = map_fibres_to_fourch
                                )
        if not os.path.isfile(os.path.join(path_EP,simulation_file_name)):
            had_to_run_new = True

            run_EP.carp2init(fourch_name = fourch_name,
                            lastFECtag = lastFECtag,
                            CV_l = CV_l,
                            k_fibre = k_fibre,
                            k_FEC = k_FEC,
                            simulation_file_name = simulation_file_name,
                            path_EP = path_EP
                            )

            run_EP.launch_init(fourch_name = fourch_name, 
                            alpha_endo = alpha, 
                            alpha_epi = -alpha, 
                            simulation_file_name = simulation_file_name,
                            path_EP = path_EP
                            )
        
    return had_to_run_new
def output(heart_name, return_ISWT = True, return_WT = True,
                      return_EDD = True, return_LVmass = True,
                      return_LVendovol = True, close_LV = True,
                      return_LVOTdiam = True, return_RVOTdiam = True,
                      close_RV = True, close_LA = True, return_LAendovol = True,
                      close_RA = True, return_RAendovol = True,
                      return_RVlongdiam = True, return_RVbasaldiam = True,
                      return_RVendovol = True, return_TAT = True,
                      return_TATLVendo = True, simulation_name = None):
    """Function that computes the output of the mesh generation and EP 
    simulation.

    Args:
        heart_name (str): Name of the four chamber mesh file.
        return_ISWT (bool, optional): If True, returns the intra-septal wall
        thickness. Defaults to True.
        return_WT (bool, optional): If True, returns the lateral wall thickness.
        Defaults to True.
        return_EDD (bool, optional): If True, returns the end-diastolic 
        diameter. Defaults to True.
        return_LVmass (bool, optional): If True, returns the mass of the LV. 
        Defaults to True.
        return_LVendovol (bool, optional): If True, returns the LV endocardial
        volume. Defaults to True.
        close_LV (bool, optional): If True, generates the closed LV endocardium
        surface. Defaults to True.
        return_LVOTdiam (bool, optional): If True, returns the diameter of the
        LV outflow tract. Defaults to True.
        return_RVOTdiam (bool, optional): If True, returns the diameter of the
        RV outflow tract. Defaults to True.
        close_RV (bool, optional):  If True, generates the closed RV endocardium
        surface. Defaults to True.
        close_LA (bool, optional):  If True, generates the closed LA endocardium
        surface. Defaults to True.
        return_LAendovol (bool, optional): If True, returns the LA endocardial
        volume. Defaults to True.
        close_RA (bool, optional): f True, generates the closed RA endocardium
        surface. Defaults to True.
        return_RAendovol (bool, optional): If True, returns the RA endocardial
        volume. Defaults to True.
        return_RVlongdiam (bool, optional): If True, returns the longitudinal
        diameter of the RV. Defaults to True.
        return_RVbasaldiam (bool, optional): If True, returns the basal
        diameter of the RV. Defaults to True.
        return_RVendovol (bool, optional): If True, returns the RV endocardial
        volume. Defaults to True.
        return_TAT (bool, optional): If True, returns the total activation time
        of the ventricles. Defaults to True.
        return_TATLVendo (bool, optional): If True, returns the total activation
        time of the LV endocardium. Defaults to True.
        simulation_name (str, optional): Name of the simulation file. Defaults 
        to None.

    Returns:
        dictionary: Dictionary with the specified output.
    """
    
    path2fourch = os.path.join("/data","fitting",heart_name)
    biv_path = os.path.join(path2fourch,"biv")
    output_list = {}

    if close_LV and (return_LVendovol or return_LVOTdiam):
        prepare_mesh.close_LV_endo(heart_name)

    if return_LVOTdiam:
        av_la_pts = files_manipulations.pts.read(os.path.join(biv_path,"..","av_ao.pts"))
        av_la_surf = files_manipulations.surf.read(os.path.join(biv_path,"..","av_ao.elem"),heart_name)

        print(os.path.join(biv_path,"..","av_ao.pts"))

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

    if close_RA and (return_RAendovol or return_RVlongdiam or return_RVbasaldiam):
        prepare_mesh.close_RA_endo(heart_name)
    
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

            if wall_max_PHI*wall_min_PHI > 0:
                lvendo_wall_ROI_idx = lvendo_band_idx[np.intersect1d(np.where(lvendo_band_PHI > wall_min_PHI)[0],
                                                        np.where(lvendo_band_PHI < wall_max_PHI)[0]
                                                        )
                                        ]
            else:

                upper_dir = np.where(lvendo_band_PHI > wall_min_PHI)
                lower_dir = np.where(lvendo_band_PHI < wall_max_PHI)

                #Otherwise is an array of arrays of arrays
                all_indices_flat = [subitem for sublist in [lower_dir,upper_dir] for item in sublist for subitem in item]


                lvendo_wall_ROI_idx = lvendo_band_idx[np.unique(all_indices_flat)]


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

                if wall_max_PHI*wall_min_PHI > 0:
                    lvepi_ROI_idx = bivepi_band_idx[np.intersect1d(np.where(bivepi_band_PHI > wall_min_PHI),
                                                            np.where(bivepi_band_PHI < wall_max_PHI)
                                                            )
                                            ]
                else:
                    upper_dir = np.where(bivepi_band_PHI > wall_min_PHI)
                    lower_dir = np.where(bivepi_band_PHI < wall_max_PHI)

                    #Otherwise is an array of arrays of arrays
                    all_indices_flat = [subitem for sublist in [lower_dir,upper_dir] for item in sublist for subitem in item]

                    lvepi_ROI_idx = bivepi_band_idx[np.unique(all_indices_flat)]


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

    if return_TAT or return_TATLVendo:
        with open(os.path.join(biv_path,"EP_simulations",simulation_name + ".dat")) as g:
            AT_vec = g.read().splitlines()
        g.close()

        AT_vec_float = np.array([float(x) for x in AT_vec])

        if return_TAT:
            filtered_TAT = AT_vec_float[np.where(AT_vec_float < 300)]
            output_list["TAT, ms"] = round(max(filtered_TAT),2)
        
        if return_TATLVendo:

            lvendo_vtx = files_manipulations.vtx.read(os.path.join(biv_path,
                                                      "biv.lvendo.surf.vtx"),
                                                      "biv")

            UVC_Z = np.genfromtxt(os.path.join(path2UVC, "COORDS_Z.dat"),dtype = float)

            Z_90 = np.where(UVC_Z < 0.9)[0]

            Z_90_endo_mask = np.in1d(lvendo_vtx.indices, Z_90)
            Z_90_endo = lvendo_vtx.indices[Z_90_endo_mask]
            
            AT_vec_endo = np.array(AT_vec_float)[Z_90_endo.astype(int)]
            filtered_TATLVendo = AT_vec_endo[np.where(AT_vec_endo < 300)]
            output_list["TAT LV endo, ms"] = round(max(filtered_TATLVendo),2)

    print(output_list)
    return output_list
def write_output_casewise(waveno = 0, subfolder = "."):
    """Function to write the output of a given wave. The outputs are specified
    in the written output_labels file.

    Args:
        waveno (int, optional): Wave number. Defaults to 0.
        subfolder (str, optional): Subfolder name. Defaults to ".".
    """
    outpath = os.path.join("/data", "fitting",subfolder, "wave" + str(waveno))
    labels_dir = os.path.join("/data","fitting",subfolder)


    with open(os.path.join(outpath,"X_anatomy.csv")) as f:
        anatomy_values = f.read().splitlines()
    with open(os.path.join(outpath,"X_EP.dat")) as f:
        param_values = f.read().splitlines()


    output_names = ["LVV","RVV","LAV","RAV",
                    "LVOTdiam", "RVOTdiam",
                    "LVmass", "LVWT", "LVEDD", "SeptumWT",
                    "RVlongdiam", "RVbasaldiam",
                    "TAT","TATLVendo"]
    output_units = ["mL", "mL", "mL", "mL",
                    "mm", "mm",
                    "g", "mm", "mm", "mm",
                    "mm", "mm",
                    "ms", "ms"]


    f = open(os.path.join(labels_dir,"output_labels.txt"), "w")
    [f.write("%s\n" % phenotype) for phenotype in output_names]
    f.close()

    f = open(os.path.join(labels_dir,"output_units.txt"), "w")
    [f.write("%s\n" % phenotype) for phenotype in output_units]
    f.close()

    simulation_names = [line.replace(" ","") for line in param_values]
    mesh_names = ["heart_" + anatomy_values[i+1].replace(",","")[:-36] for i in range(len(anatomy_values)-1)]

    for i in tqdm.tqdm(range(len(mesh_names))):
        print("Computing output...")
        EP_dir = os.path.join("/data","fitting",mesh_names[i],"biv", 
                          "EP_simulations")
        if os.path.isfile(os.path.join(EP_dir,output_names[-1] + ".dat")):
            continue 
        if os.path.isfile(os.path.join(EP_dir, simulation_names[i] + ".dat")):
            
            flag_close_LV = True
            flag_close_RV = True
            flag_close_LA = True
            flag_close_RA = True

            if(os.path.isfile(os.path.join(mesh_names[i], "lvendo_closed.elem"))):
                flag_close_LV = False
            if(os.path.isfile(os.path.join(mesh_names[i], "laendo_closed.elem"))):
                flag_close_LA = False
            if(os.path.isfile(os.path.join(mesh_names[i], "rvendo_closed.elem"))):
                flag_close_RV = False
            if(os.path.isfile(os.path.join(mesh_names[i], "raendo_closed.elem"))):
                flag_close_RA = False
            
            output_list = output(heart_name = mesh_names[i],
                                simulation_name = simulation_names[i],
                                return_ISWT = True, return_WT = True,
                                return_EDD = True, return_LVmass = True,
                                return_LVendovol = True, return_LVOTdiam = True,
                                return_RVOTdiam = True, return_LAendovol = True,
                                return_RAendovol = True,
                                return_RVlongdiam = True,
                                return_RVbasaldiam = True, 
                                return_RVendovol = True, return_TAT = True,
                                return_TATLVendo = True,
                                close_LV = flag_close_LV,
                                close_RV = flag_close_RV,
                                close_LA = flag_close_LA,
                                close_RA = flag_close_RA, 
                       )

            LVV = output_list["LV end-diastolic volume, mL"]
            RVV = output_list["RV volume, mL"]
            LAV = output_list["LA volume, mL"]
            RAV = output_list["RA volume, mL"]
            LVOTdiam = output_list["LVOT diameter, mm"]
            RVOTdiam = output_list["RVOT diameter, mm"]
            LVmass = output_list["LV mass, g"]
            LVWT = output_list["Posterior wall thickness, mm"]
            LVEDD = output_list["Diastolic LV internal dimension, mm"]
            SeptumWT = output_list["Interventricular septal wall thickness, mm"]
            RVlongdiam = output_list["RV long. diameter, mm"]
            RVbasaldiam = output_list["RV basal diameter, mm"]
            TAT = output_list["TAT, ms"]
            TATLVendo = output_list["TAT LV endo, ms"]

        else:
            EP_setup(waveno = waveno, subfolder = subfolder)
            EP_simulations(waveno = waveno, subfolder = subfolder)
            i = i - 1
        
        output_numbers = [LVV, RVV, LAV, RAV,
                        LVOTdiam, RVOTdiam,
                        LVmass, LVWT, LVEDD, SeptumWT,
                        RVlongdiam, RVbasaldiam,
                        TAT,TATLVendo]

        for var_i, varname in enumerate(output_names):
            np.savetxt(os.path.join(EP_dir, varname + ".dat"),
                        [output_numbers[var_i]],
                        fmt="%s")
def collect_output(waveno = 0, subfolder = "."):
    """Function to merge the output of all the simulations in a single file.

    Args:
        waveno (int, optional): Wave number. Defaults to 0.
        subfolder (str, optional): Output subfolder. Defaults to ".".
    """
    outpath = os.path.join("/data", "fitting",subfolder, "wave" + str(waveno))

    with open(os.path.join(outpath,"X_anatomy.csv")) as f:
        anatomy_values = f.read().splitlines()
    with open(os.path.join(outpath,"X_EP.dat")) as f:
        param_values = f.read().splitlines()


    output_names = ["LVV","RVV","LAV","RAV",
                    "LVOTdiam", "RVOTdiam",
                    "LVmass", "LVWT", "LVEDD", "SeptumWT",
                    "RVlongdiam", "RVbasaldiam",
                    "TAT","TATLVendo"]
    
    mesh_names = ["heart_" + anatomy_values[i+1].replace(",","")[:-36] for i in range(len(anatomy_values)-1)]

    LVV = []
    RVV = []
    LAV = []
    RAV = []
    LVOTdiam = []
    RVOTdiam = []
    LVmass = []
    LVWT = []
    LVEDD = []
    SeptumWT = []
    RVlongdiam = []
    RVbasaldiam = []
    TAT = []
    TATLVendo =[]

    output_numbers = [LVV, RVV, LAV, RAV,
                        LVOTdiam, RVOTdiam,
                        LVmass, LVWT, LVEDD, SeptumWT,
                        RVlongdiam, RVbasaldiam,
                        TAT,TATLVendo]

    print("Gathering output...")
    for i in tqdm.tqdm(range(len(mesh_names))):
        EP_dir = os.path.join("/data","fitting",mesh_names[i],"biv", 
                          "EP_simulations")
        for i,outname in enumerate(output_names):
            output_number = np.loadtxt(os.path.join(EP_dir, outname + ".dat"), dtype=float)
            output_numbers[i].append(output_number)

    for i,varname in enumerate(output_names):
        np.savetxt(os.path.join(outpath, varname + ".dat"),
                    output_numbers[i],
                    fmt="%.2f")
def preprocess_input(waveno = 0, subfolder = "."):
    """Function to split the input from a .dat file to a .csv file. This is
    needed for deformetrica.

    Args:
        waveno (int, optional): Wave number, specifies the folder name.
        Defaults to 0.
        subfolder (str, optional): Subfolder name of /data/fitting to work on. 
        Defaults to ".".
    """

    path_gpes = os.path.join("/data","fitting", subfolder, "wave" + str(waveno))

    with open(os.path.join(path_gpes,"X.dat")) as f:
        anatomy_and_EP_values = f.read().splitlines()

    x_anatomy = []
    x_EP = []
    
    for full_line in anatomy_and_EP_values:
        line = full_line.split(' ')
        x_anatomy.append(line[0:9])
        x_EP.append(line[9:14])

    f = open(os.path.join(path_gpes, "X_EP.dat"), "w")
    f.writelines(' '.join(row) + '\n' for row in x_EP)
    f.close()

    with open(os.path.join(path_gpes,"X_anatomy.csv"), mode='w') as f:
        f_writer = csv.writer(f, delimiter=',')

        f_writer.writerow(["Mode" + str(i) for i in range(1,19)])

        for current_line in range(len(x_anatomy)):
            output = np.zeros(18)
            output[0:9] = x_anatomy[current_line]
            f_writer.writerow(["{0:.2f}".format(round(i,2)) for i in output])

    f.close()