import numpy as np
import os
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib
import pathlib
import time
import torchmetrics
import tqdm

import seaborn as sns

import fibres
import files_manipulations
import fitting_hm
from global_variables_config import *
from gpytGPE.gpe import GPEmul
from gpytGPE.utils.metrics import IndependentStandardError as ISE
import postprocessing
import prepare_mesh
import run_EP
import UVC

from Historia.shared.design_utils import read_labels



np.random.seed(SEED)

def build_meshes(subfolder="CT_anatomy", force_construction=False):
    path_lab = os.path.join(PROJECT_PATH, subfolder)
    temp_outpath = os.path.join(path_lab, "temp_meshes")

    with open(os.path.join(path_lab, "X_anatomy.csv")) as f:
        anatomy_values = f.read().splitlines()

    pathlib.Path(temp_outpath).mkdir(parents=True, exist_ok=True)

    for i in tqdm.tqdm(range(len(anatomy_values) - 1)):
        temp_base_name = "CT_case_" + str(i)

        separated_modes = anatomy_values[i + 1].split(',')
        specific_modes = separated_modes[0:9]
        final_base_name = "heart_" + ''.join(specific_modes)
        mesh_path = os.path.join(path_lab, final_base_name)

        if (not os.path.isfile(os.path.join(mesh_path, final_base_name + ".elem")) and
            not os.path.isfile(os.path.join(mesh_path, final_base_name + "_default.elem"))) or force_construction:
            if not os.path.isfile(os.path.join(temp_outpath, "CT_case_" + str(i) + ".vtk")):
                csv_file_name = os.path.join(temp_outpath, "CT_case_" + str(i) + ".csv")
                np.savetxt(csv_file_name, np.array([anatomy_values[0], anatomy_values[i + 1]]), fmt='%s',
                           delimiter='\n')

                print(final_base_name)
                print(time.strftime("%H:%M:%S", time.localtime()))

                os.chdir(os.path.join("/home", "crg17", "Desktop", "KCL_projects", "fitting", "python",
                                      "CardiacMeshConstruction_outside"))
                os.system(
                    "python3.6 ./pipeline.py " + csv_file_name + " " + temp_outpath + " " + "CT_case_" + str(i))

            pathlib.Path(mesh_path).mkdir(parents=True, exist_ok=True)

            os.system("cp " + os.path.join(temp_outpath, "wave" + temp_base_name + ".vtk") +
                      " " + os.path.join(mesh_path, final_base_name + ".vtk"))
            prepare_mesh.vtk_mm2carp_um(fourch_name=final_base_name, subfolder=subfolder)

    os.system("rm -rf " + temp_outpath)


def ep_setup(subfolder="CT_anatomy"):
    path_lab = os.path.join(PROJECT_PATH, subfolder)

    with open(os.path.join(path_lab, ANATOMY_CSV)) as f:
        anatomy_values = f.read().splitlines()

    for i in tqdm.tqdm(range(len(anatomy_values) - 1)):

        separated_modes = anatomy_values[i + 1].split(',')
        specific_modes = separated_modes[0:9]
        final_base_name = "heart_" + ''.join(specific_modes)
        mesh_path = os.path.join(path_lab, final_base_name)

        if not os.path.isfile(os.path.join(mesh_path, "biv", "biv_noRVendo.surf.vtx")):
            prepare_mesh.extract_ldrb_biv(final_base_name, subfolder=subfolder)
        if not os.path.isfile(os.path.join(mesh_path, "biv", "mvtv_base.surf.vtx")):
            prepare_mesh.extract_mvtv_base(fourch_name=final_base_name, subfolder=subfolder)
        if (not os.path.isfile(os.path.join(mesh_path, "biv", "UVC_mvtv", "UVC", "COORDS_V_elem_scaled.dat"))) or \
                (not os.path.isfile(os.path.join(mesh_path, "biv", "UVC_mvtv", "UVC", "COORDS_PHI_elem_scaled.dat"))) or \
                (not os.path.isfile(os.path.join(mesh_path, "biv", "UVC_mvtv", "UVC", "COORDS_Z_elem_scaled.dat"))) or \
                (not os.path.isfile(os.path.join(mesh_path, "biv", "UVC_mvtv", "UVC", "COORDS_RHO_elem_scaled.dat"))):
            UVC.create(fourch_name=final_base_name, base="mvtv", subfolder=subfolder)
        if not os.path.isfile(os.path.join(mesh_path, "biv", "EP", "bottom_third.vtx")):
            UVC.bottom_third(fourch_name=final_base_name, UVC_base="mvtv", subfolder=subfolder)
        if not os.path.isfile(os.path.join(mesh_path, "biv", "biv_fec.elem")):
            UVC.create_fec(fourch_name=final_base_name, uvc_base="mvtv", subfolder=subfolder)
        if not os.path.isfile(os.path.join(mesh_path, "biv", "fibres", "endoRV", "phie.igb")):
            fibres.run_laplacian(fourch_name=subfolder + "/" + final_base_name)


def ep_simulations(subfolder="CT_anatomy"):
    path_lab = os.path.join(PROJECT_PATH, subfolder)

    with open(os.path.join(path_lab, ANATOMY_CSV)) as f:
        anatomy_values = f.read().splitlines()

    alpha = 80.00
    fec_height = 70.00
    cv = 0.80
    k_fibre = 0.29
    k_fec = 7.00
    lastfectag = '33'

    for i in tqdm.tqdm(range(len(anatomy_values) - 1)):

        separated_modes = anatomy_values[i + 1].split(',')
        specific_modes = separated_modes[0:9]
        fourch_name = "heart_" + ''.join(specific_modes)

        path_ep = os.path.join(path_lab, fourch_name, "biv", "EP_simulations")

        pathlib.Path(path_ep).mkdir(parents=True, exist_ok=True)

        simulation_file_name = os.path.join(path_ep,
                                            '{0:.2f}'.format(alpha) + \
                                            '{0:.2f}'.format(fec_height) + \
                                            '{0:.2f}'.format(cv) + \
                                            '{0:.2f}'.format(k_fibre) + \
                                            '{0:.2f}'.format(k_fec) + \
                                            ".dat"
                                            )

        if not os.path.isfile(os.path.join(path_lab, fourch_name, "biv", "fibres",
                                           "rb_-" + str('{0:.2f}'.format(alpha)) + "_" + str('{0:.2f}'.format(alpha)) +
                                           ".elem")):
            fibres.full_pipeline(fourch_name=fourch_name,
                                 subfolder=subfolder,
                                 alpha_epi='{0:.2f}'.format(-alpha),
                                 alpha_endo='{0:.2f}'.format(alpha),
                                 map_to_fourch=False
                                 )
        if not os.path.isfile(os.path.join(path_ep, simulation_file_name)):

            run_EP.carp2init(fourch_name=subfolder + "/" + fourch_name,
                             lastfectag=lastfectag,
                             CV_l=cv,
                             k_fibre=k_fibre,
                             k_fec=k_fec,
                             simulation_file_name=simulation_file_name,
                             path_ep=path_ep
                             )

            run_EP.launch_init(fourch_name=subfolder + "/" + fourch_name,
                               alpha_endo=alpha,
                               alpha_epi=-alpha,
                               simulation_file_name=simulation_file_name,
                               path_ep=path_ep,
                               map_to_fourch=False
                               )


def output(heart_name, return_ISWT=True, return_WT=True,
           return_EDD=True, return_LVmass=True,
           return_LVendovol=True, close_LV=True,
           return_LVOTdiam=True, return_RVOTdiam=True,
           close_RV=True, close_LA=True, return_LAendovol=True,
           close_RA=True, return_RAendovol=True,
           return_RVlongdiam=True, return_RVbasaldiam=True,
           return_RVendovol=True, return_TAT=True,
           return_TATLVendo=True, simulation_name=None, subfolder="CT_anatomy"):
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

    path2fourch = os.path.join(PROJECT_PATH, subfolder, heart_name)
    biv_path = os.path.join(path2fourch, "biv")
    output_list = {}

    if close_LV and (return_LVendovol or return_LVOTdiam):
        prepare_mesh.close_lv_endo(heart_name, subfolder = subfolder)

    if return_LVOTdiam:
        av_la_pts = files_manipulations.pts.read(os.path.join(biv_path, "..", "av_mapped.pts"))
        av_la_surf = files_manipulations.surf.read(os.path.join(biv_path, "..", "av_ao.surf"), heart_name)

        print(os.path.join(biv_path, "..", "av_ao.pts"))

        area_array = files_manipulations.area_or_vol_surface(pts_file=av_la_pts,
                                                             surf_file=av_la_surf, with_vol=False,
                                                             with_area=True)
        LVOT_area = sum(area_array) * 1e-6  # In mm2

        LVOTdiam = 2 * np.sqrt(LVOT_area / np.pi)

        output_list["LVOT diameter, mm"] = round(LVOTdiam, 2)

    if close_RV and (return_RVOTdiam or return_RVlongdiam or return_RVbasaldiam or return_RVendovol):
        prepare_mesh.close_rv_endo(heart_name, subfolder = subfolder)

    if return_RVOTdiam:
        pv_pa_pts = files_manipulations.pts.read(os.path.join(biv_path, "..", "pv_mapped.pts"))
        pv_pa_surf = files_manipulations.surf.read(os.path.join(biv_path, "..", "pv_pa.surf"), heart_name)

        area_array = files_manipulations.area_or_vol_surface(pts_file=pv_pa_pts,
                                                             surf_file=pv_pa_surf, with_vol=False,
                                                             with_area=True)
        RVOT_area = sum(area_array) * 1e-6  # In mm2

        RVOTdiam = 2 * np.sqrt(RVOT_area / np.pi)

        output_list["RVOT diameter, mm"] = round(RVOTdiam, 2)

    if return_RVendovol or return_RVlongdiam:
        rvendo_closed = files_manipulations.pts.read(os.path.join(path2fourch, heart_name + ".pts"))

        if return_RVendovol:
            rvendo_closed_surf = files_manipulations.surf.read(os.path.join(path2fourch, "rvendo_closed.surf"),
                                                               heart_name)
            vol_array = files_manipulations.area_or_vol_surface(pts_file=rvendo_closed,
                                                                surf_file=rvendo_closed_surf, with_vol=True,
                                                                with_area=False)
            RVendovol = sum(vol_array) * 1e-12  # In mL

            output_list["RV volume, mL"] = np.abs(round(RVendovol, 2))

    if close_RA and (return_RAendovol or return_RVlongdiam or return_RVbasaldiam):
        prepare_mesh.close_RA_endo(heart_name, subfolder=subfolder)

    if return_RVlongdiam or return_RVbasaldiam:
        tvrv_pts = files_manipulations.pts.read(os.path.join(path2fourch, "tv_mapped.pts"))

        tvrv_surf = files_manipulations.surf.read(os.path.join(biv_path, "..", "tv_ra.surf"), heart_name)

        area_array = files_manipulations.area_or_vol_surface(pts_file=tvrv_pts,
                                                             surf_file=tvrv_surf, with_vol=False,
                                                             with_area=True)
        tvrv_area = sum(area_array) * 1e-6  # In mm2

        RVbasaldiam = 2 * np.sqrt(tvrv_area / np.pi)

        # if return_RVbasaldiam:
        #     output_list["RV basal diameter, mm"] = round(RVbasaldiam, 2)

        if return_RVlongdiam:

            num_pts = tvrv_pts.size

            sum_x = np.sum(tvrv_pts.p1)
            sum_y = np.sum(tvrv_pts.p2)
            sum_z = np.sum(tvrv_pts.p3)

            centroid = np.array([sum_x / num_pts, sum_y / num_pts, sum_z / num_pts])

            dist_vec = np.zeros(rvendo_closed.size)

            for i in range(len(dist_vec)):
                new_point = np.array([rvendo_closed.p1[i], rvendo_closed.p2[i], rvendo_closed.p3[i]])
                dist_vec[i] = np.linalg.norm(centroid - new_point)

            RVlongdiam_centroid = max(dist_vec) * 1e-3

            RVlongdiam = np.sqrt((RVbasaldiam / 2.) ** 2 + RVlongdiam_centroid ** 2)

            output_list["RV long. diameter, mm"] = round(RVlongdiam, 2)

    if close_LA and return_LAendovol:
        prepare_mesh.close_LA_endo(heart_name, subfolder=subfolder)

    if return_LAendovol:
        laendo_closed_pts = files_manipulations.pts.read(os.path.join(biv_path, "..", "laendo_closed.pts"))
        laendo_closed_surf = files_manipulations.surf.read(os.path.join(biv_path, "..", "laendo_closed.elem"),
                                                           heart_name)

        vol_array = files_manipulations.area_or_vol_surface(pts_file=laendo_closed_pts,
                                                            surf_file=laendo_closed_surf, with_vol=True,
                                                            with_area=False)
        LAendovol = sum(vol_array) * 1e-12  # In mL

        output_list["LA volume, mL"] = round(LAendovol, 2)

    if return_RAendovol:
        raendo_closed_pts = files_manipulations.pts.read(os.path.join(biv_path, "..", "raendo_closed.pts"))
        raendo_closed_surf = files_manipulations.surf.read(os.path.join(biv_path, "..", "raendo_closed.elem"),
                                                           heart_name)

        vol_array = files_manipulations.area_or_vol_surface(pts_file=raendo_closed_pts,
                                                            surf_file=raendo_closed_surf, with_vol=True,
                                                            with_area=False)
        RAendovol = sum(vol_array) * 1e-12  # In mL

        output_list["RA volume, mL"] = round(RAendovol, 2)

    if return_LVendovol:
        lvendo_closed_pts = files_manipulations.pts.read(os.path.join(biv_path, "..", heart_name + ".pts"))
        lvendo_closed_surf = files_manipulations.surf.read(os.path.join(biv_path, "..", "lvendo_closed.surf"),
                                                           heart_name)

        temp_vol = files_manipulations.area_or_vol_surface(pts_file=lvendo_closed_pts,
                                                           surf_file=lvendo_closed_surf, with_vol=True,
                                                           with_area=False)

        LV_chamber_vol = np.abs(sum(temp_vol) * 1e-12)  # In mL

        output_list["LV end-diastolic volume, mL"] = round(LV_chamber_vol, 2)

    if return_ISWT or return_WT or return_EDD or return_LVmass:
        path2UVC = os.path.join(biv_path, "UVC_mvtv", "UVC")
        biv_pts = files_manipulations.pts.read(os.path.join(biv_path, "biv.pts"))
        UVC_Z_elem = np.genfromtxt(os.path.join(path2UVC, "COORDS_Z_elem_scaled.dat"), dtype=float)
        UVC_Z = np.genfromtxt(os.path.join(path2UVC, "COORDS_Z.dat"), dtype=float)

        if return_LVmass:
            biv_elem = files_manipulations.elem.read(os.path.join(biv_path, "biv.elem"))
            UVC_V_elem = np.genfromtxt(os.path.join(path2UVC, "COORDS_V_elem.dat"), dtype=float)

            def six_vol_element_cm3(i):
                result = np.linalg.det(np.array([np.array(
                    [1e-4 * biv_pts.p1[biv_elem.i1[i]], 1e-4 * biv_pts.p2[biv_elem.i1[i]],
                     1e-4 * biv_pts.p3[biv_elem.i1[i]], 1.], dtype=float),
                                                 np.array([1e-4 * biv_pts.p1[biv_elem.i2[i]],
                                                           1e-4 * biv_pts.p2[biv_elem.i2[i]],
                                                           1e-4 * biv_pts.p3[biv_elem.i2[i]], 1.], dtype=float),
                                                 np.array([1e-4 * biv_pts.p1[biv_elem.i3[i]],
                                                           1e-4 * biv_pts.p2[biv_elem.i3[i]],
                                                           1e-4 * biv_pts.p3[biv_elem.i3[i]], 1.], dtype=float),
                                                 np.array([1e-4 * biv_pts.p1[biv_elem.i4[i]],
                                                           1e-4 * biv_pts.p2[biv_elem.i4[i]],
                                                           1e-4 * biv_pts.p3[biv_elem.i4[i]], 1.], dtype=float)
                                                 ], dtype=float)
                                       )
                return np.abs(result)

            LV_mass_idx = np.intersect1d(np.where(UVC_Z_elem < 0.9), np.where(UVC_V_elem < 0))

            six_vol_LV = []
            six_vol_LV.append(Parallel(n_jobs=20)(delayed(six_vol_element_cm3)(i) for i in range(len(LV_mass_idx))))

            LV_mass = 1.05 * sum(six_vol_LV[0]) / 6.

            output_list["LV mass, g"] = round(LV_mass, 2)
        if return_ISWT or return_WT or return_EDD:
            septum_idx = files_manipulations.vtx.read(os.path.join(biv_path, "biv.rvsept.surf.vtx"), "biv")
            lvendo_idx = files_manipulations.vtx.read(os.path.join(biv_path, "biv.lvendo.surf.vtx"), "biv")
            UVC_PHI = np.genfromtxt(os.path.join(path2UVC, "COORDS_PHI.dat"), dtype=float)
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

            midpoint_PHI = (max(septum_band_PHI) + min(septum_band_PHI)) / 2.
            bandwith_PHI = max(septum_band_PHI) - min(septum_band_PHI)
            min_PHI = midpoint_PHI - bandwith_PHI / 6.
            max_PHI = midpoint_PHI + bandwith_PHI / 6.

        if return_ISWT or return_EDD:
            lvendo_septum_ROI_idx = lvendo_band_idx[np.intersect1d(np.where(lvendo_band_PHI > min_PHI),
                                                                   np.where(lvendo_band_PHI < max_PHI)
                                                                   )
            ]
            lvendo_septum_ROI_pts = biv_pts.extract(files_manipulations.vtx(lvendo_septum_ROI_idx, "biv"))

            if return_ISWT:
                septum_ROI_idx = septum_band_idx[np.intersect1d(np.where(septum_band_PHI > min_PHI),
                                                                np.where(septum_band_PHI < max_PHI)
                                                                )
                ]
                septum_ROI_pts = biv_pts.extract(files_manipulations.vtx(septum_ROI_idx, "biv"))

                def dist_from_septum_to_lvendo_septum(i):
                    return lvendo_septum_ROI_pts.min_dist(
                        np.array([septum_ROI_pts.p1[i], septum_ROI_pts.p2[i], septum_ROI_pts.p3[i]]))

                septum_ROI_thickness = []
                septum_ROI_thickness.append(Parallel(n_jobs=20)(
                    delayed(dist_from_septum_to_lvendo_septum)(i) for i in range(septum_ROI_pts.size)))

                # Itraventricular septal wall thickness, in mm
                ISWT = 1e-3 * np.median(septum_ROI_thickness)

                output_list["Interventricular septal wall thickness, mm"] = round(ISWT, 2)

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

            if wall_max_PHI * wall_min_PHI > 0:
                lvendo_wall_ROI_idx = lvendo_band_idx[np.intersect1d(np.where(lvendo_band_PHI > wall_min_PHI)[0],
                                                                     np.where(lvendo_band_PHI < wall_max_PHI)[0]
                                                                     )
                ]
            else:

                upper_dir = np.where(lvendo_band_PHI > wall_min_PHI)
                lower_dir = np.where(lvendo_band_PHI < wall_max_PHI)

                # Otherwise is an array of arrays of arrays
                all_indices_flat = [subitem for sublist in [lower_dir, upper_dir] for item in sublist for subitem in
                                    item]

                lvendo_wall_ROI_idx = lvendo_band_idx[np.unique(all_indices_flat)]

            lvendo_wall_ROI_pts = biv_pts.extract(files_manipulations.vtx(lvendo_wall_ROI_idx, "biv"))

            if return_WT:
                bivepi_idx = files_manipulations.vtx.read(os.path.join(biv_path, "biv.epi.surf.vtx"),
                                                          "biv")
                bivepi_Z = UVC_Z[bivepi_idx.indices]
                bivepi_band_idx = bivepi_idx.indices[np.intersect1d(np.where(bivepi_Z > 0.6),
                                                                    np.where(bivepi_Z < 0.9)
                                                                    )
                ]
                bivepi_band_PHI = UVC_PHI[bivepi_band_idx]

                if wall_max_PHI * wall_min_PHI > 0:
                    lvepi_ROI_idx = bivepi_band_idx[np.intersect1d(np.where(bivepi_band_PHI > wall_min_PHI),
                                                                   np.where(bivepi_band_PHI < wall_max_PHI)
                                                                   )
                    ]
                else:
                    upper_dir = np.where(bivepi_band_PHI > wall_min_PHI)
                    lower_dir = np.where(bivepi_band_PHI < wall_max_PHI)

                    # Otherwise is an array of arrays of arrays
                    all_indices_flat = [subitem for sublist in [lower_dir, upper_dir] for item in sublist for subitem in
                                        item]

                    lvepi_ROI_idx = bivepi_band_idx[np.unique(all_indices_flat)]

                lvepi_ROI_pts = biv_pts.extract(files_manipulations.vtx(lvepi_ROI_idx, "biv"))

                def dist_from_lvepi_to_lvendo_wall(i):
                    return lvendo_wall_ROI_pts.min_dist(
                        np.array([lvepi_ROI_pts.p1[i], lvepi_ROI_pts.p2[i], lvepi_ROI_pts.p3[i]]))

                wall_ROI_thickness = []
                wall_ROI_thickness.append(
                    Parallel(n_jobs=20)(delayed(dist_from_lvepi_to_lvendo_wall)(i) for i in range(lvepi_ROI_pts.size)))

                # Wall thickness, in mm
                WT = 1e-3 * np.median(wall_ROI_thickness)

                output_list["Posterior wall thickness, mm"] = round(WT, 2)

            if return_EDD:
                def dist_from_lvendo_septum_to_lvendo_wall(i):
                    return lvendo_wall_ROI_pts.min_dist(np.array(
                        [lvendo_septum_ROI_pts.p1[i], lvendo_septum_ROI_pts.p2[i], lvendo_septum_ROI_pts.p3[i]]))

                ED_dimension_ROI = []

                ED_dimension_ROI.append(Parallel(n_jobs=20)(
                    delayed(dist_from_lvendo_septum_to_lvendo_wall)(i) for i in range(lvendo_septum_ROI_pts.size)))

                # End-diastolic dimension, mm
                EDD = 1e-3 * np.median(ED_dimension_ROI)

                output_list["Diastolic LV internal dimension, mm"] = round(EDD, 2)

    if return_TAT or return_TATLVendo:
        with open(os.path.join(biv_path, "EP_simulations", simulation_name + ".dat")) as g:
            AT_vec = g.read().splitlines()
        g.close()

        AT_vec_float = np.array([float(x) for x in AT_vec])

        if return_TAT:
            filtered_TAT = AT_vec_float[np.where(AT_vec_float < 300)]
            output_list["TAT, ms"] = round(max(filtered_TAT), 2)

        if return_TATLVendo:
            lvendo_vtx = files_manipulations.vtx.read(os.path.join(biv_path,
                                                                   "biv.lvendo.surf.vtx"),
                                                      "biv")

            UVC_Z = np.genfromtxt(os.path.join(path2UVC, "COORDS_Z.dat"), dtype=float)

            Z_90 = np.where(UVC_Z < 0.9)[0]

            Z_90_endo_mask = np.in1d(lvendo_vtx.indices, Z_90)
            Z_90_endo = lvendo_vtx.indices[Z_90_endo_mask]

            AT_vec_endo = np.array(AT_vec_float)[Z_90_endo.astype(int)]
            filtered_TATLVendo = AT_vec_endo[np.where(AT_vec_endo < 300)]
            output_list["TAT LV endo, ms"] = round(max(filtered_TATLVendo), 2)

    print(output_list)
    return output_list


def write_output_casewise(subfolder="CT_anatomy"):
    """Function to write the output of a given wave. The outputs are specified
    in the written output_labels file.

    Args:
        waveno (int, optional): Wave number. Defaults to 0.
        subfolder (str, optional): Subfolder name. Defaults to ".".
    """
    outpath = os.path.join(PROJECT_PATH, subfolder)
    labels_dir = outpath

    with open(os.path.join(outpath, "X_anatomy.csv")) as f:
        anatomy_values = f.read().splitlines()

    output_names = ["LVV", "RVV", "LAV", "RAV",
                    "LVOTdiam", "RVOTdiam",
                    "LVmass", "LVWT", "LVEDD", "SeptumWT",
                    "RVlongdiam",
                    "TAT", "TATLVendo"]
    output_units = ["mL", "mL", "mL", "mL",
                    "mm", "mm",
                    "g", "mm", "mm", "mm",
                    "mm",
                    "ms", "ms"]

    f = open(os.path.join(labels_dir, "output_labels.txt"), "w")
    [f.write("%s\n" % phenotype) for phenotype in output_names]
    f.close()

    f = open(os.path.join(labels_dir, "output_units.txt"), "w")
    [f.write("%s\n" % phenotype) for phenotype in output_units]
    f.close()

    simulation_names = ['80.0070.000.800.297.00' for line in anatomy_values]
    mesh_names = ["heart_" + anatomy_values[i + 1].replace(",", "")[:-36] for i in range(len(anatomy_values) - 1)]

    for i in tqdm.tqdm(range(len(mesh_names))):
        print("Computing output...")
        EP_dir = os.path.join("/data", "fitting", "CT_anatomy", mesh_names[i], "biv",
                              "EP_simulations")
        if os.path.isfile(os.path.join(EP_dir, output_names[-1] + ".dat")):
            continue
        if os.path.isfile(os.path.join(EP_dir, simulation_names[i] + ".dat")):

            flag_close_LV = True
            flag_close_RV = True
            flag_close_LA = True
            flag_close_RA = True

            if (os.path.isfile(os.path.join(mesh_names[i], "lvendo_closed.elem"))):
                flag_close_LV = False
            if (os.path.isfile(os.path.join(mesh_names[i], "laendo_closed.elem"))):
                flag_close_LA = False
            if (os.path.isfile(os.path.join(mesh_names[i], "rvendo_closed.elem"))):
                flag_close_RV = False
            if (os.path.isfile(os.path.join(mesh_names[i], "raendo_closed.elem"))):
                flag_close_RA = False

            output_list = output(heart_name=mesh_names[i],
                                 simulation_name=simulation_names[i],
                                 return_ISWT=True, return_WT=True,
                                 return_EDD=True, return_LVmass=True,
                                 return_LVendovol=True, return_LVOTdiam=True,
                                 return_RVOTdiam=True, return_LAendovol=True,
                                 return_RAendovol=True,
                                 return_RVlongdiam=True,
                                 return_RVbasaldiam=True,
                                 return_RVendovol=True, return_TAT=True,
                                 return_TATLVendo=True,
                                 close_LV=flag_close_LV,
                                 close_RV=flag_close_RV,
                                 close_LA=flag_close_LA,
                                 close_RA=flag_close_RA,
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
            TAT = output_list["TAT, ms"]
            TATLVendo = output_list["TAT LV endo, ms"]

        output_numbers = [LVV, RVV, LAV, RAV,
                          LVOTdiam, RVOTdiam,
                          LVmass, LVWT, LVEDD, SeptumWT,
                          RVlongdiam,
                          TAT, TATLVendo]

        for var_i, varname in enumerate(output_names):
            np.savetxt(os.path.join(EP_dir, varname + ".dat"),
                       [output_numbers[var_i]],
                       fmt="%s")


def collect_output(subfolder="CT_anatomy"):
    """Function to merge the output of all the simulations in a single file.

    Args:
        waveno (int, optional): Wave number. Defaults to 0.
        subfolder (str, optional): Output subfolder. Defaults to ".".
    """
    outpath = os.path.join("/data", "fitting", subfolder)

    with open(os.path.join(outpath, "X_anatomy.csv")) as f:
        anatomy_values = f.read().splitlines()

    output_names = ["LVV", "RVV", "LAV", "RAV",
                    "LVOTdiam", "RVOTdiam",
                    "LVmass", "LVWT", "LVEDD", "SeptumWT",
                    "RVlongdiam",
                    "TAT", "TATLVendo"]

    mesh_names = ["heart_" + anatomy_values[i + 1].replace(",", "")[:-36] for i in range(len(anatomy_values) - 1)]

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
    TAT = []
    TATLVendo = []

    output_numbers = [LVV, RVV, LAV, RAV,
                      LVOTdiam, RVOTdiam,
                      LVmass, LVWT, LVEDD, SeptumWT,
                      RVlongdiam,
                      TAT, TATLVendo]

    print("Gathering output...")
    for i in tqdm.tqdm(range(len(mesh_names))):
        EP_dir = os.path.join("/data", "fitting", "CT_anatomy", mesh_names[i], "biv",
                              "EP_simulations")
        for i, outname in enumerate(output_names):
            output_number = np.loadtxt(os.path.join(EP_dir, outname + ".dat"), dtype=float)
            output_numbers[i].append(output_number)

    for i, varname in enumerate(output_names):
        np.savetxt(os.path.join(outpath, varname + ".dat"),
                   output_numbers[i],
                   fmt="%.2f")


def extend_emulators(train=False, anatomy_waveno=2):

    anatomy_values = np.loadtxt(open(os.path.join(PROJECT_PATH,"CT_anatomy","X_anatomy.csv")),
                                delimiter=',',skiprows=1)

    CT_y_train = np.loadtxt(os.path.join(PROJECT_PATH,"CT_anatomy","LVV.dat"), dtype=float)
    CT_x_train = np.hstack((anatomy_values[0:19,0:9],np.tile([80,70,0.8,0.29,7],(19,1))))

    original_x_train, original_y_train, original_emul = fitting_hm.run_GPE(waveno=anatomy_waveno, train=False, active_feature=["LVV"],
                                                               n_training_pts=420, training_set_memory=2,
                                                               subfolder="anatomy_max_range", only_feasible=False)

    for i in range(-1,len(CT_y_train)):
        extended_x_train = np.vstack((original_x_train, CT_x_train[0:(i+1),:]))
        extended_y_train = np.append(original_y_train,CT_y_train[0:(i+1)])

        if i == -1:
            extended_emul = original_emul
        else:
            if train:
                extended_emul = GPEmul(extended_x_train, extended_y_train)
                extended_emul.train(X_val = None, y_val = None, max_epochs=100, n_restarts=5,
                                    savepath=os.path.join(PROJECT_PATH,"CT_anatomy") + "/")
                extended_emul.save("LVV_extended_CT" + str(i) + "_waveno" + str(anatomy_waveno) + ".gpe")
            else:
                extended_emul = GPEmul.load(X_train = extended_x_train, y_train = extended_y_train,
                                   loadpath=os.path.join(PROJECT_PATH,"CT_anatomy") + "/",
                                   filename="LVV_extended_CT" + str(i) + "_waveno" + str(anatomy_waveno) + ".gpe")

        y_pred_mean, y_pred_std = extended_emul.predict(CT_x_train)
        average_pred_mean, average_pred_std = extended_emul.predict([[0,0,0,0,0,0,0,0,0,80,70,0.8,0.29,7]])

        ci = 2  # ~95% confidance interval

        inf_bound = []
        sup_bound = []

        height = 9.36111
        width = 5.91667
        fig, axes = plt.subplots(1, 1, figsize=(2 * width, 2 * height / 4))

        # l = np.argsort(y_pred_mean)  # for the sake of a better visualisation
        l = range(len(y_pred_mean))

        inf_bound.append((y_pred_mean - ci * y_pred_std).min())
        sup_bound.append((y_pred_mean + ci * y_pred_std).max())

        axes.scatter(
            np.arange(1, len(l) + 1),
            CT_y_train[l],
            facecolors="none",
            edgecolors="C0",
            label="simulated",
        )
        axes.scatter(
            np.arange(1, len(l) + 1),
            y_pred_mean[l],
            facecolors="C0",
            s=16,
            label="emulated",
        )
        axes.errorbar(
            np.arange(1, len(l) + 1),
            y_pred_mean[l],
            yerr=ci * y_pred_std[l],
            c="C0",
            ls="none",
            lw=0.5,
            label=f"uncertainty ({ci} SD)",
        )

        xlabels = np.array(["#" + str(i) for i in range(1,20)])
        axes.set_xticks(range(1,20))
        axes.set_xticklabels(xlabels[l])
        axes.set_ylabel("mL", fontsize=12)
        axes.set_xlabel("CT subject")
        if i == -1:
            axes.set_title(
                "Original enlarged LVV GPE using | Predicted average volume: " + '{0:.2f}'.format(average_pred_mean[0]) +
                u"\u00B1" + '{0:.2f}'.format(average_pred_std[0]) + " mL",
                fontsize=12,
            )
        else:
            axes.set_title(
                "Extended enlarged LVV GPE (from wave " + str(anatomy_waveno) + ") using  " + str(i+1) +
                " meshes from the CT cohort | Predicted average volume: " + '{0:.2f}'.format(average_pred_mean[0]) +
                u"\u00B1" + '{0:.2f}'.format(average_pred_std[0]) + " mL",
                fontsize=12,
            )
        axes.legend(loc="upper left")

        # axes.set_ylim([np.min(inf_bound), np.max(sup_bound)])
        axes.set_ylim(50,180)

        fig.tight_layout()
        plt.savefig(
            os.path.join(PROJECT_PATH, "CT_anatomy", "figures", "max_range_LVV_waveno" + str(anatomy_waveno) + "_" +
                         f'{(i+1):02}' + "_CT_points.png"),
            bbox_inches="tight", dpi=300
        )
        plt.close()


def find_ct_outside_range():

    anatomy_values = np.loadtxt(open(os.path.join(PROJECT_PATH, "CT_anatomy", "X_anatomy.csv")),
                                delimiter=',', skiprows=1)

    std_list = np.std(anatomy_values[:20,:],0)[0:9]
    total_cases = []
    for i in range(20):
        list_outside = []
        for m in range(9):
            if anatomy_values[i,m] > 2*std_list[m]:
                list_outside.append(m+1)
        if len(list_outside) > 0:
            print("Case " + str(i+1) + " is outside the training range due to mode(s) ", end="")
            print(list_outside)
            total_cases.append(i+1)
        else:
            print("Case " + str(i+1) + " is completely inside the range")

    print("Summary: " + str(round(100*len(total_cases)/19,2)) + "% of the cases are outside of bounds")


def validation_emulator_vs_ct(subfolder="anatomy_max_range", waveno=2, active_features=["LVV","RVV","LAV","RAV",
                                                                                        "LVOTdiam","RVOTdiam","LVmass",
                                                                                        "LVWT","LVEDD","SeptumWT",
                                                                                        "RVlongdiam","TAT","TATLVendo"]):
    anatomy_values = np.loadtxt(open(os.path.join(PROJECT_PATH, "CT_anatomy", "X_anatomy.csv")),
                                delimiter=',', skiprows=1)
    ylabels = np.genfromtxt(os.path.join(PROJECT_PATH, subfolder, "output_units.txt"), dtype=str)

    for feature_i in range(len(active_features)):
        CT_y_train = np.loadtxt(os.path.join(PROJECT_PATH, "CT_anatomy", active_features[feature_i] + ".dat"), dtype=float)
        CT_x_train = np.hstack((anatomy_values[0:19, 0:9], np.tile([80, 70, 0.8, 0.29, 7], (19, 1))))

        original_x_train, original_y_train, original_emul = fitting_hm.run_GPE(waveno=waveno, train=False,
                                                                               active_feature=[active_features[feature_i]],
                                                                               n_training_pts=420, training_set_memory=2,
                                                                               subfolder=subfolder,
                                                                               only_feasible=False)
        extended_emul = original_emul

        y_pred_mean, y_pred_std = extended_emul.predict(CT_x_train)

        ci = 2  # ~95% confidance interval

        inf_bound = []
        sup_bound = []

        height = 9.36111
        width = 5.91667
        fig, axes = plt.subplots(1, 1, figsize=(2 * width, 2 * height / 4))

        # l = np.argsort(y_pred_mean)  # for the sake of a better visualisation
        l = range(len(y_pred_mean))

        inf_bound.append((y_pred_mean - ci * y_pred_std).min())
        sup_bound.append((y_pred_mean + ci * y_pred_std).max())

        axes.scatter(
            np.arange(1, len(l) + 1),
            CT_y_train[l],
            facecolors="none",
            edgecolors="C0",
            label="simulated",
        )
        axes.scatter(
            np.arange(1, len(l) + 1),
            y_pred_mean[l],
            facecolors="C0",
            s=16,
            label="emulated",
        )
        axes.errorbar(
            np.arange(1, len(l) + 1),
            y_pred_mean[l],
            yerr=ci * y_pred_std[l],
            c="C0",
            ls="none",
            lw=0.5,
            label=f"uncertainty ({ci} SD)",
        )

        xlabels = np.array(["#" + str(i) for i in range(1, 20)])

        axes.set_xticks(range(1, 20))
        axes.set_xticklabels(xlabels[l])
        axes.set_ylabel(ylabels[feature_i], fontsize=12)
        axes.set_xlabel("CT subject")

        axes.set_title(active_features[feature_i] + " GPE against CT simulations" ,fontsize=12)

        axes.legend(loc="upper left")

        axes.set_ylim([np.min(inf_bound), np.max(sup_bound)])

        fig.tight_layout()
        plt.savefig(
            os.path.join(PROJECT_PATH, "CT_anatomy", "figures", subfolder + "_" + active_features[feature_i] + " _waveno" +
                         str(waveno) + "_CT_points.png"),
            bbox_inches="tight", dpi=300
        )
        plt.close()


def report_scores_emulation(subfolder="anatomy_max_range", waveno=2, active_features=["LVV", "RVV", "LAV", "RAV",
                                                                                      "LVOTdiam", "RVOTdiam", "LVEDD",
                                                                                      "RVlongdiam", "LVWT", "SeptumWT",
                                                                                      "TAT", "TATLVendo", "LVmass"]):

    anatomy_values = np.loadtxt(open(os.path.join(PROJECT_PATH, "CT_anatomy", "X_anatomy.csv")),
                                delimiter=',', skiprows=1)

    R2Score_vec = []
    iseScore_vec = []
    impl_mean_vec = []
    impl_std_vec = []

    for feature_i in range(len(active_features)):
        CT_y_train = np.loadtxt(os.path.join(PROJECT_PATH, "CT_anatomy", active_features[feature_i] + ".dat"), dtype=float)
        CT_x_train = np.hstack((anatomy_values[0:19, 0:9], np.tile([80, 70, 0.8, 0.29, 7], (19, 1))))

        _, _, emul = fitting_hm.run_GPE(waveno=waveno, train=False, active_feature=[active_features[feature_i]],
                                        n_training_pts=420, training_set_memory=2, subfolder=subfolder, only_feasible=False)

        y_pred_mean, y_pred_std = emul.predict(CT_x_train)

        R2Score = torchmetrics.R2Score()(emul.tensorize(y_pred_mean), emul.tensorize(CT_y_train))

        iseScore = ISE(
            emul.tensorize(CT_y_train),
            emul.tensorize(y_pred_mean),
            emul.tensorize(y_pred_std),
        )

        impl = postprocessing.impl_measure_per_output(emul_mean = y_pred_mean, lit_mean = CT_y_train,
                                                      emul_var = y_pred_std**2, lit_var = y_pred_std*0)

        R2Score_vec.append(R2Score)
        iseScore_vec.append(iseScore)
        impl_mean_vec.append(np.mean(impl))
        impl_std_vec.append(np.std(impl))

    return R2Score_vec, iseScore_vec, impl_mean_vec, impl_std_vec

def plot_CT_output_vs_literature():
    matplotlib.rcParams.update({'font.size': 22})

    active_features=read_labels(os.path.join("/data","fitting","anatomy_max_range","output_labels.txt"))
    CT_output_matrix = np.zeros((19, len(active_features)), dtype=float)

    for feature_i in range(len(active_features)):
        CT_output = np.loadtxt(os.path.join(PROJECT_PATH, "CT_anatomy", active_features[feature_i] + ".dat"), dtype=float)
        CT_output_matrix[:,feature_i] = CT_output

    exp_means = np.loadtxt(os.path.join("/data", "fitting", "match", "exp_mean_anatomy_EP.txt"), dtype=float)
    exp_stds = np.loadtxt(os.path.join("/data", "fitting", "match", "exp_std_anatomy_EP.txt"), dtype=float)

    for i in range(CT_output_matrix.shape[0]):
        if i > 8:
            plt.plot(CT_output_matrix[i,:].T, marker=r'$' + str(i+1) + '$', linestyle = 'None', markersize=10)
        else:
            plt.plot(CT_output_matrix[i, :].T, marker=r'$0' + str(i + 1) + '$', linestyle='None', markersize=10)
    # plt.legend(['1','2'])
    plt.xticks(range(CT_output_matrix.shape[1]), active_features,rotation = 20)

    y_lim_max=225.
    plt.ylim((0, y_lim_max))

    min_rectangles = (exp_means - 2*exp_stds)/y_lim_max
    max_rectangles = (exp_means + 2 * exp_stds) / y_lim_max

    for i in range(len(min_rectangles)):
        plt.axvspan(xmin=i-0.25, xmax=i+0.25, ymin = min_rectangles[i], ymax=max_rectangles[i], alpha=0.25, color='green')

    plt.title('CT cohort biomarkers compared with literature ranges')

    plt.show()