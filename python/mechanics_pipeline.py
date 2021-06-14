
def penalty_map(fourch_name):
    """The chambers closed are in independent scripts in prepare_mesh. 
    - Here we make the penalty map. One value for each element in the whole mesh
    - Move the ATs.
    - For each vein, we need a surf and neubc.
    - For each endo, we make surf and neubc.
    - Surf and neubc of epicardium (where to apply the PM).
    - Binary mesh.

    Args:
        fourch_name ([type]): [description]
    """
    
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

    np.savetxt(os.path.join(path2biv, "pericardium_penalty.dat"),
                        penalty_biv, fmt="%.2f")
    
    os.system("meshtool insert data -msh=" + os.path.join(path2fourch,fourch_name) +\
              " -submsh=" + os.path.join(path2biv,"biv") +\
              " -submsh_data=" + os.path.join(path2biv,"pericardium_penalty.dat") +\
              " -odat=" + path2fourch +\
              " -mode=1"
            )
