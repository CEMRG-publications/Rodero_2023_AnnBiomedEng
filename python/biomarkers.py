from joblib import Parallel, delayed
import numpy as np
import os
import tqdm

from global_variables_config import *

import files_manipulations
import prepare_mesh

def lvv(anatomy_values, i):

    # heart_name = "heart_" + anatomy_values[i + 1].replace(",", "")[:-36]
    if not os.path.isfile(os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv", "EP_simulations", "LVV.dat")):
        if not os.path.isfile(os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "av_ao.surf")):
            prepare_mesh.close_lv_endo(fourch_name="heart_" + anatomy_values[i + 1].replace(",", "")[:-36], subfolder="meshes")
        np.savetxt(os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv", "EP_simulations", "LVV.dat"),
                            [round(np.abs(sum(files_manipulations.area_or_vol_surface(
                                pts_file=files_manipulations.pts.read(os.path.join(
                                    PROJECT_PATH, "meshes",
                                    "heart_" + anatomy_values[i + 1].replace(",", "")[:-36],
                                    "heart_" + anatomy_values[i + 1].replace(",", "")[:-36] + ".pts")
                                ),surf_file=files_manipulations.surf.read(
                                    os.path.join(PROJECT_PATH, "meshes",
                                                 "heart_" + anatomy_values[i + 1].replace(",", "")[:-36],
                                                 "lvendo_closed.surf"),mesh_from="heart_" + anatomy_values[i + 1].replace(",", "")[:-36]),
                                with_vol=True,with_area=False)) * 1e-12), 2)],fmt="%s")
        # lvendo_closed_pts = files_manipulations.pts.read(os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "heart_" + anatomy_values[i + 1].replace(",", "")[:-36] + ".pts"))
        # lvendo_closed_surf = files_manipulations.surf.read(os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "lvendo_closed.surf"),
        #                                                "heart_" + anatomy_values[i + 1].replace(",", "")[:-36])
        #
        # temp_vol = files_manipulations.area_or_vol_surface(pts_file=lvendo_closed_pts,
        #                                                    surf_file=lvendo_closed_surf, with_vol=True,
        #                                                    with_area=False)
        # LV_chamber_vol = round(np.abs(sum(temp_vol) * 1e-12),2)  # In mL
        #
        # np.savetxt(os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "heart_" + anatomy_values[i + 1].replace(",", "")[:-36]), "LVV.dat"),
        #                     [output_numbers[var_i]],
        #                     fmt="%s")

def rvv(anatomy_values, i):
    # heart_name = "heart_" + anatomy_values[i + 1].replace(",", "")[:-36]
    if not os.path.isfile(os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv", "EP_simulations", "RVV.dat")):
        if not os.path.isfile(os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "pv_pa.surf")):
            prepare_mesh.close_rv_endo(fourch_name="heart_" + anatomy_values[i + 1].replace(",", "")[:-36], subfolder="meshes")

        np.savetxt(os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv",
                                "EP_simulations", "RVV.dat"),
                   [round(np.abs(sum(files_manipulations.area_or_vol_surface(
                       pts_file=files_manipulations.pts.read(os.path.join(
                           PROJECT_PATH, "meshes",
                           "heart_" + anatomy_values[i + 1].replace(",", "")[:-36],
                           "heart_" + anatomy_values[i + 1].replace(",", "")[:-36] + ".pts")
                       ), surf_file=files_manipulations.surf.read(
                           os.path.join(PROJECT_PATH, "meshes",
                                        "heart_" + anatomy_values[i + 1].replace(",", "")[:-36],
                                        "rvendo_closed.surf"),
                           mesh_from="heart_" + anatomy_values[i + 1].replace(",", "")[:-36]),
                       with_vol=True, with_area=False)) * 1e-12), 2)], fmt="%s")
        # rvendo_closed = files_manipulations.pts.read(os.path.join(path2fourch, heart_name + ".pts"))
        #
        #     rvendo_closed_surf = files_manipulations.surf.read(os.path.join(path2fourch, "rvendo_closed.surf"),
        #                                                        heart_name)
        #     vol_array = files_manipulations.area_or_vol_surface(pts_file=rvendo_closed,
        #                                                         surf_file=rvendo_closed_surf, with_vol=True,
        #                                                         with_area=False)
        #     RVendovol = sum(vol_array) * 1e-12  # In mL
        #
        #     output_list["RV volume, mL"] = np.abs(round(RVendovol, 2))
        # np.savetxt(os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36],
        #                         "heart_" + anatomy_values[i + 1].replace(",", "")[:-36]), "RVV.dat"),
        #                     [output_numbers[var_i]],
        #                     fmt="%s")

def lav(anatomy_values, i):
    # heart_name = "heart_" + anatomy_values[i + 1].replace(",", "")[:-36]
    if not os.path.isfile(os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv", "EP_simulations", "LAV.dat")):
        if not os.path.isfile(os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "mv_lv.pts")):
            prepare_mesh.close_LA_endo(fourch_name="heart_" + anatomy_values[i + 1].replace(",", "")[:-36], subfolder="meshes")

        np.savetxt(os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv",
                                "EP_simulations", "LAV.dat"),
                   [round(np.abs(sum(files_manipulations.area_or_vol_surface(
                       pts_file=files_manipulations.pts.read(os.path.join(
                           PROJECT_PATH, "meshes",
                           "heart_" + anatomy_values[i + 1].replace(",", "")[:-36],
                           "laendo_closed.pts")
                       ), surf_file=files_manipulations.surf.read(
                           os.path.join(PROJECT_PATH, "meshes",
                                        "heart_" + anatomy_values[i + 1].replace(",", "")[:-36],
                                        "laendo_closed.elem"),
                           mesh_from="heart_" + anatomy_values[i + 1].replace(",", "")[:-36]),
                       with_vol=True, with_area=False)) * 1e-12), 2)], fmt="%s")

def rav(anatomy_values, i):
    # heart_name = "heart_" + anatomy_values[i + 1].replace(",", "")[:-36]
    if not os.path.isfile(os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv", "EP_simulations", "RAV.dat")):
        if not os.path.isfile(os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "tv_rv.pts")):
            prepare_mesh.close_RA_endo(fourch_name="heart_" + anatomy_values[i + 1].replace(",", "")[:-36], subfolder="meshes")

        np.savetxt(os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv",
                                "EP_simulations", "RAV.dat"),
                   [round(np.abs(sum(files_manipulations.area_or_vol_surface(
                       pts_file=files_manipulations.pts.read(os.path.join(
                           PROJECT_PATH, "meshes",
                           "heart_" + anatomy_values[i + 1].replace(",", "")[:-36],
                           "raendo_closed.pts")
                       ), surf_file=files_manipulations.surf.read(
                           os.path.join(PROJECT_PATH, "meshes",
                                        "heart_" + anatomy_values[i + 1].replace(",", "")[:-36],
                                        "raendo_closed.elem"),
                           mesh_from="heart_" + anatomy_values[i + 1].replace(",", "")[:-36]),
                       with_vol=True, with_area=False)) * 1e-12), 2)], fmt="%s")

def lvotdiam(anatomy_values, i):
    if not os.path.isfile(
            os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv",
                         "EP_simulations", "LVOTdiam.dat")):
        if not os.path.isfile(
                os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36],
                             "av_ao.surf")):
            prepare_mesh.close_lv_endo(fourch_name="heart_" + anatomy_values[i + 1].replace(",", "")[:-36],
                                       subfolder="meshes")
        np.savetxt(os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv",
                                "EP_simulations", "LVOTdiam.dat"),
                   [round(2 * np.sqrt((sum(files_manipulations.area_or_vol_surface(
                                pts_file=files_manipulations.pts.read(os.path.join(
                                    PROJECT_PATH, "meshes",
                                    "heart_" + anatomy_values[i + 1].replace(",", "")[:-36],
                                    "av_mapped.pts")
                                ),surf_file=files_manipulations.surf.read(
                                    os.path.join(PROJECT_PATH, "meshes",
                                                 "heart_" + anatomy_values[i + 1].replace(",", "")[:-36],
                                                 "av_ao.surf"),mesh_from="heart_" + anatomy_values[i + 1].replace(",", "")[:-36]),
                                with_vol=False,with_area=True)) * 1e-6) / np.pi), 2)], fmt="%s")

def rvotdiam(anatomy_values, i):
    if not os.path.isfile(
            os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv",
                         "EP_simulations", "RVOTdiam.dat")):
        if not os.path.isfile(
                os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36],
                             "pv_pa.surf")):
            prepare_mesh.close_rv_endo(fourch_name="heart_" + anatomy_values[i + 1].replace(",", "")[:-36],
                                       subfolder="meshes")
        np.savetxt(os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv",
                                "EP_simulations", "RVOTdiam.dat"),
                   [round(2 * np.sqrt((sum(files_manipulations.area_or_vol_surface(
                                pts_file=files_manipulations.pts.read(os.path.join(
                                    PROJECT_PATH, "meshes",
                                    "heart_" + anatomy_values[i + 1].replace(",", "")[:-36],
                                    "pv_mapped.pts")
                                ),surf_file=files_manipulations.surf.read(
                                    os.path.join(PROJECT_PATH, "meshes",
                                                 "heart_" + anatomy_values[i + 1].replace(",", "")[:-36],
                                                 "pv_pa.surf"),mesh_from="heart_" + anatomy_values[i + 1].replace(",", "")[:-36]),
                                with_vol=False,with_area=True)) * 1e-6) / np.pi), 2)], fmt="%s")

def tat(anatomy_values, param_values, i):
    if not os.path.isfile(
            os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv",
                         "EP_simulations", "TAT.dat")):

        np.savetxt(os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv",
                                "EP_simulations", "TAT.dat"),[round(max(np.array([float(x) for x in open(os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv", "EP_simulations", '{0:.2f}'.format(round(float(param_values[i].split(' ')[0]), 2)) + \
                                           '{0:.2f}'.format(round(float(param_values[i].split(' ')[1]), 2)) + \
                                           '{0:.2f}'.format(round(float(param_values[i].split(' ')[2]), 2)) + \
                                           '{0:.2f}'.format(round(float(param_values[i].split(' ')[3]), 2)) + \
                                           '{0:.2f}'.format(round(float(param_values[i].split(' ')[4]), 2)) + \
                                           ".dat")).read().splitlines()])[np.where(np.array([float(x) for x in open(os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv", "EP_simulations", '{0:.2f}'.format(round(float(param_values[i].split(' ')[0]), 2)) + \
                                           '{0:.2f}'.format(round(float(param_values[i].split(' ')[1]), 2)) + \
                                           '{0:.2f}'.format(round(float(param_values[i].split(' ')[2]), 2)) + \
                                           '{0:.2f}'.format(round(float(param_values[i].split(' ')[3]), 2)) + \
                                           '{0:.2f}'.format(round(float(param_values[i].split(' ')[4]), 2)) + \
                                           ".dat")).read().splitlines()]) < 300)]), 2)],fmt="%s")

def tatlvendo(anatomy_values, param_values, i):
    # simulation_name = '{0:.2f}'.format(round(float(param_values[i].split(' ')[0]), 2)) + \
    #                   '{0:.2f}'.format(round(float(param_values[i].split(' ')[1]), 2)) + \
    #                   '{0:.2f}'.format(round(float(param_values[i].split(' ')[2]), 2)) + \
    #                   '{0:.2f}'.format(round(float(param_values[i].split(' ')[3]), 2)) + \
    #                   '{0:.2f}'.format(round(float(param_values[i].split(' ')[4]), 2))
    # biv_path = os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv")
    # path2UVC = os.path.join(biv_path, "UVC_mvtv", "UVC")
    if not os.path.isfile(
            os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv",
                         "EP_simulations", "TATLVendo.dat")):
        # with open(os.path.join(biv_path, "EP_simulations", simulation_name + ".dat")) as g:
        #     AT_vec = g.read().splitlines()
        # g.close()
        # AT_vec = open(os.path.join(biv_path, "EP_simulations", simulation_name + ".dat")).read().splitlines()
        # AT_vec_float = np.array([float(x) for x in AT_vec])

        # lvendo_vtx = files_manipulations.vtx.read(os.path.join(biv_path,"biv.lvendo.surf.vtx"),"biv")

        # UVC_Z = np.genfromtxt(os.path.join(path2UVC, "COORDS_Z.dat"), dtype=float)

        # Z_90 = np.where(UVC_Z < 0.9)[0]

        # Z_90_endo_mask = np.in1d(lvendo_vtx.indices, Z_90)
        # Z_90_endo = lvendo_vtx.indices[Z_90_endo_mask]

        # AT_vec_endo = np.array(AT_vec_float)[Z_90_endo.astype(int)]
        np.savetxt(os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv",
                                "EP_simulations", "TATLVendo.dat"),
                   [round(max(np.array(np.array([float(x) for x in open(os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv", "EP_simulations",
                                                                                     '{0:.2f}'.format(round(float(
                                                                                         param_values[i].split(' ')[0]),
                                                                                                            2)) + \
                                                                                     '{0:.2f}'.format(round(float(
                                                                                         param_values[i].split(' ')[1]),
                                                                                                            2)) + \
                                                                                     '{0:.2f}'.format(round(float(
                                                                                         param_values[i].split(' ')[2]),
                                                                                                            2)) + \
                                                                                     '{0:.2f}'.format(round(float(
                                                                                         param_values[i].split(' ')[3]),
                                                                                                            2)) + \
                                                                                     '{0:.2f}'.format(round(float(
                                                                                         param_values[i].split(' ')[4]),
                                                                                                            2)) + ".dat")).read().splitlines()]))[files_manipulations.vtx.read(os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv","biv.lvendo.surf.vtx"),"biv").indices[np.in1d(files_manipulations.vtx.read(os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv","biv.lvendo.surf.vtx"),"biv").indices, np.where(np.genfromtxt(os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv", "UVC_mvtv", "UVC", "COORDS_Z.dat"), dtype=float) < 0.9)[0])].astype(int)][np.where(np.array(np.array([float(x) for x in open(os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv", "EP_simulations", '{0:.2f}'.format(round(float(param_values[i].split(' ')[0]), 2)) + \
                      '{0:.2f}'.format(round(float(param_values[i].split(' ')[1]), 2)) + \
                      '{0:.2f}'.format(round(float(param_values[i].split(' ')[2]), 2)) + \
                      '{0:.2f}'.format(round(float(param_values[i].split(' ')[3]), 2)) + \
                      '{0:.2f}'.format(round(float(param_values[i].split(' ')[4]), 2)) + ".dat")).read().splitlines()]))[files_manipulations.vtx.read(os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv","biv.lvendo.surf.vtx"),"biv").indices[np.in1d(files_manipulations.vtx.read(os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv","biv.lvendo.surf.vtx"),"biv").indices, np.where(np.genfromtxt(os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv", "UVC_mvtv", "UVC", "COORDS_Z.dat"), dtype=float) < 0.9)[0])].astype(int)] < 300)]), 2)], fmt="%s")
    # simulation_name = '{0:.2f}'.format(round(float(param_values[i].split(' ')[0]), 2)) + \
    #                                        '{0:.2f}'.format(round(float(param_values[i].split(' ')[1]), 2)) + \
    #                                        '{0:.2f}'.format(round(float(param_values[i].split(' ')[2]), 2)) + \
    #                                        '{0:.2f}'.format(round(float(param_values[i].split(' ')[3]), 2)) + \
    #                                        '{0:.2f}'.format(round(float(param_values[i].split(' ')[4]), 2))
    # biv_path = os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv")
    # path2UVC = os.path.join(biv_path, "UVC_mvtv", "UVC")
    # if not os.path.isfile(
    #         os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv",
    #                      "EP_simulations", "TATLVendo.dat")):
    #
    #     with open(os.path.join(biv_path, "EP_simulations", simulation_name + ".dat")) as g:
    #         AT_vec = g.read().splitlines()
    #     g.close()
    #
    #     AT_vec_float = np.array([float(x) for x in AT_vec])
    #
    #     lvendo_vtx = files_manipulations.vtx.read(os.path.join(biv_path,
    #                                                            "biv.lvendo.surf.vtx"),
    #                                               "biv")
    #
    #     UVC_Z = np.genfromtxt(os.path.join(path2UVC, "COORDS_Z.dat"), dtype=float)
    #
    #     Z_90 = np.where(UVC_Z < 0.9)[0]
    #
    #     Z_90_endo_mask = np.in1d(lvendo_vtx.indices, Z_90)
    #     Z_90_endo = lvendo_vtx.indices[Z_90_endo_mask]
    #
    #     AT_vec_endo = np.array(AT_vec_float)[Z_90_endo.astype(int)]
    #     filtered_TATLVendo = AT_vec_endo[np.where(AT_vec_endo < 300)]
    #     np.savetxt(os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv",
    #                             "EP_simulations", "TATLVendo.dat"),[round(max(filtered_TATLVendo), 2)],fmt="%s")

def non_parallelizable(anatomy_values, i):
    return_LVmass = False
    return_ISWT = False
    return_WT = False
    return_EDD = False
    return_RVlongdiam = False
    if not os.path.isfile(
            os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv",
                         "EP_simulations", "LVmass.dat")):
        return_LVmass = True
    if not os.path.isfile(
            os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv",
                         "EP_simulations", "LVWT.dat")):
        return_WT = True
    if not os.path.isfile(
            os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv",
                         "EP_simulations", "LVEDD.dat")):
        return_EDD = True
    if not os.path.isfile(
            os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv",
                         "EP_simulations", "SeptumWT.dat")):
        return_ISWT = True
    if not os.path.isfile(
            os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv",
                         "EP_simulations", "RVlongdiam.dat")):
        return_RVlongdiam = True
    path2fourch = os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36])
    biv_path = os.path.join(path2fourch, "biv")
    if return_RVlongdiam:
        if not os.path.isfile(os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "pv_pa.surf")):
            prepare_mesh.close_rv_endo(fourch_name="heart_" + anatomy_values[i + 1].replace(",", "")[:-36], subfolder="meshes")
        if not os.path.isfile(os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "tv_rv.pts")):
            prepare_mesh.close_RA_endo(fourch_name="heart_" + anatomy_values[i + 1].replace(",", "")[:-36], subfolder="meshes")

        rvendo_closed = files_manipulations.pts.read(os.path.join(path2fourch, heart_name + ".pts"))

        tvrv_pts = files_manipulations.pts.read(os.path.join(path2fourch, "tv_mapped.pts"))

        tvrv_surf = files_manipulations.surf.read(os.path.join(biv_path, "..", "tv_ra.surf"), heart_name)

        area_array = files_manipulations.area_or_vol_surface(pts_file=tvrv_pts,
                                                             surf_file=tvrv_surf, with_vol=False,
                                                             with_area=True)
        tvrv_area = sum(area_array) * 1e-6  # In mm2

        RVbasaldiam = 2 * np.sqrt(tvrv_area / np.pi)

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

        np.savetxt(os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv",
                                "EP_simulations", "RVlongdiam.dat"),
                   [round(RVlongdiam, 2)], fmt="%s")

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

            np.savetxt(
                os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv",
                             "EP_simulations", "LVmass.dat"),
                [round(LV_mass, 2)], fmt="%s")

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

                np.savetxt(
                    os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv",
                                 "EP_simulations", "SeptumWT.dat"),
                    [round(ISWT, 2)], fmt="%s")


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

                np.savetxt(
                    os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv",
                                 "EP_simulations", "LVWT.dat"),
                    [round(WT, 2)], fmt="%s")


            if return_EDD:
                def dist_from_lvendo_septum_to_lvendo_wall(i):
                    return lvendo_wall_ROI_pts.min_dist(np.array(
                        [lvendo_septum_ROI_pts.p1[i], lvendo_septum_ROI_pts.p2[i], lvendo_septum_ROI_pts.p3[i]]))

                ED_dimension_ROI = []

                ED_dimension_ROI.append(Parallel(n_jobs=20)(
                    delayed(dist_from_lvendo_septum_to_lvendo_wall)(i) for i in range(lvendo_septum_ROI_pts.size)))

                # End-diastolic dimension, mm
                EDD = 1e-3 * np.median(ED_dimension_ROI)

                np.savetxt(
                    os.path.join(PROJECT_PATH, "meshes", "heart_" + anatomy_values[i + 1].replace(",", "")[:-36], "biv",
                                 "EP_simulations", "LVEDD.dat"),
                    [round(EDD, 2)], fmt="%s")

def extract(subfolder="initial_sweep", anatomy_csv_file="input_anatomy_training.csv", ep_dat_file="input_ep_training.dat"):

    with open(os.path.join(PROJECT_PATH, subfolder, anatomy_csv_file)) as f:
        anatomy_values = f.read().splitlines()

    with open(os.path.join(PROJECT_PATH, subfolder, ep_dat_file)) as f:
        param_values = f.read().splitlines()

    Parallel(n_jobs=20)(delayed(lvv)(anatomy_values=anatomy_values, i=i) for i in range(20))
    Parallel(n_jobs=20)(delayed(rvv)(anatomy_values=anatomy_values, i=i) for i in range(20))
    Parallel(n_jobs=20)(delayed(lav)(anatomy_values=anatomy_values, i=i) for i in range(20))
    Parallel(n_jobs=20)(delayed(rav)(anatomy_values=anatomy_values, i=i) for i in range(20))
    Parallel(n_jobs=20)(delayed(lvotdiam)(anatomy_values=anatomy_values, i=i) for i in range(20))
    Parallel(n_jobs=20)(delayed(rvotdiam)(anatomy_values=anatomy_values, i=i) for i in range(20))
    Parallel(n_jobs=20)(delayed(tat)(anatomy_values=anatomy_values, param_values=param_values, i=i) for i in range(20))
    Parallel(n_jobs=20)(delayed(tatlvendo)(anatomy_values=anatomy_values, param_values=param_values, i=i) for i in range(20))

    for i in tqdm.tqdm(range(20)):
        # In parallel we'd have to read the elem file more often so it's actually slower.
        non_parallelizable(anatomy_values=anatomy_values, i=i)

