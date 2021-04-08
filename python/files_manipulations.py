import numpy as np
import csv

class vtx:

    def __init__(self, indices, mesh_from):
        """Init function for the vtx class.

        Args:
            indices (numpy array of integers): Array of indices.
            mesh_from (str): Mesh where the vtx comes from, for debugging
            purposes.
        """
        self.indices = indices
        self.size = len(indices)
        self.mesh_from = mesh_from

    def __str__(self):
        """Function to improve the printing of a vtx object.

        Returns:
            str: Message with number of vertices and the mesh where it comes
            from.
        """
        return "Vtx variable of {} vertices extracted from {}".format(self.size,self.mesh_from)

    @classmethod
    def read(cls, pathname, mesh_from):
        """Function to read a vtx file and convert it to a vtx object.

        Args:
            pathname (str): Full path (including filename and extension).
            mesh_from (str): Mesh where the vtx comes from, for debugging
            purposes.
        Returns:
            vtx: vtx object from the file.
        """
        vtx_var = np.genfromtxt(pathname, dtype = int, skip_header = 2)

        return cls(vtx_var, mesh_from)

    def write(self,pathname):
        """Function to write a vtx object to a file.

        Args:
            pathname (str): Full path (including filename and extension).
        """
        array = np.array(self.indices)
        header = [self.size, "intra"]
        str_to_write = np.append(header, array)

        with open(pathname, 'w') as f:
            for item in str_to_write:
                f.write("%s\n" % item)

class surf:
    
    def __init__(self, i1, i2, i3, mesh_from, tags = None):
        """Init function for the surf class.

        Args:
            i1 (numpy array of integers): Array with the first index for each 
            element.
            i2 (numpy array of integers): Array with the second index for each 
            element.
            i3 (numpy array of integers): Array with the third index for each 
            element.
            mesh_from (str): Mesh where the vtx comes from, for debugging
            purposes.
            tags (numpy array of integers, optional): In case of being a
            surfmesh, the list of tag for each element. If it is a surf, takes
            the value None. Defaults to None.
        """
        self.i1 = i1.astype(int)
        self.i2 = i2.astype(int)
        self.i3 = i3.astype(int)
        if(tags is None):
            self.tags = tags
        else:
            self.tags = tags.astype(int)

        self.size = i1.shape[0]
        self.mesh_from = mesh_from

    @classmethod
    def read(cls, pathname, mesh_from):
        """Function to read a .surf, .elem or .surfmesh file and convert it to a
        surf object.

        Args:
            pathname (str): Full path (including filename and extension).
            mesh_from (str): Mesh where the vtx comes from, for debugging
            purposes.
        Returns:
            surf: surf object extracted from the file.
        """

        surfmesh_str = pathname.split(".")[-1]
        if(surfmesh_str == "surfmesh"):
            num_cols = (1,2,3,4)
            is_surfmesh = True
        else:
            num_cols = (1,2,3)
            is_surfmesh = False
    
        surffile = np.genfromtxt(pathname, delimiter = ' ',
                                    dtype = int, skip_header = True,
                                    usecols= num_cols
                                    )

        i1 = surffile[:,0]
        i2 = surffile[:,1]
        i3 = surffile[:,2]

        if(is_surfmesh):
            tags = surffile[:,3]
        else:
            tags = None
        
        return cls(i1, i2, i3, mesh_from, tags)

    @classmethod
    def merge(cls, surf1, surf2):
        """Function to merge two surf objects into one. Might have duplicates.

        Args:
            surf1 (surf): First surf object to merge.
            surf2 (surf): Second surf object to merge.

        Raises:
            NameError: If the meshes they come from are not the same.
            NameError: If one have tags and the other not.

        Returns:
            surf: Surf object of the two surfs merged.
        """
        if(surf1.mesh_from != surf2.mesh_from):
            raise NameError("Surfaces come from different meshes.")
        if((surf1.tags is None) != (surf2.tags is None)):
            raise NameError("Both meshes need to be surf or surfmeshes, but not mixed.")

        i1 = np.append(surf1.i1, surf2.i1)
        i2 = np.append(surf1.i2, surf2.i2)
        i3 = np.append(surf1.i3, surf2.i3)
        mesh_from = surf1.mesh_from

        if(surf1.tags is not None):
            tags = np.append(surf1.tags, surf2.tags)
        else:
            tags = None


        return cls(i1, i2, i3, mesh_from, tags)

    @classmethod
    def tosurf(cls, surfmesh_var):
        """Function to convert surfmesh (or .surf.elem) to surf object.

        Args:
            surfmesh_var (surf): Surf object (with tags).

        Returns:
            surf: Surf object (without tags).
        """
        
        return cls(surfmesh_var.i1, surfmesh_var.i2, surfmesh_var.i3, surfmesh_var.mesh_from)
    
    @classmethod
    def tosurfmesh(cls, surf_var):
        """Function to convert a surf object to surfmesh (or .surf.elem) 
        adding zeroes to the tags.

        Args:
            surf_var (surf): Surf object (without tags).

        Returns:
            surf: Surf object (with tags).
        """

        tags = np.zeros(surf_var.size)
        return cls(surf_var.i1, surf_var.i2, surf_var.i3, surf_var.mesh_from, tags)

    def tovtx(self):
        """Function to extract the vtx from a surf object, removing duplicates.

        Returns:
            vtx: vtx object with the indices from the surf.
        """
        vtxvec =  np.unique(np.concatenate((self.i1, self.i2, self.i3),
                                            axis = 0))
        
        return vtx(vtxvec, self.mesh_from)

    def write(self,pathname):
        """Function to write a surf object to a file.

        Args:
            pathname (str): Full path (including filename and extension).
        """

        header = np.array([str(self.size)])
        elemtype = np.repeat("Tr",self.size)
        data = [elemtype, self.i1, self.i2, self.i3]

        if(self.tags is not None):
            data = [elemtype, self.i1, self.i2, self.i3, self.tags]

        np.savetxt(pathname, header, fmt='%s')

        with open(pathname, "ab") as f:
            np.savetxt(f, np.transpose(data), fmt = "%s")

    @classmethod
    def extract(cls, self, vtx_var):
        if(self.tags is not None):
            sub_tags = self.tags[vtx_var]
        else:
            sub_tags = self.tags

        sub_i1 = self.i1[vtx_var]
        sub_i2 = self.i2[vtx_var]
        sub_i3 = self.i3[vtx_var]

        return cls(sub_i1, sub_i2, sub_i3, self.mesh_from, sub_tags)

class pts:
    
    def __init__(self, p1, p2, p3, name):
        """Function to initialise a pts object.

        Args:
            p1 (numpy array): Array of the first coordinate for all the points.
            p2 (numpy array): Array of the second coordinate for all the points.
            p3 (numpy array): Array of the third coordinate for all the points.
            name (str): Name of the pts (without extension) for debugging
            purposes.
        """
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.size = p1.shape[0]
        self.name = name

    @classmethod
    def read(cls, pathname):
        """Function to generate a pts object from a file.

        Args:
            pathname (str): Full path including filename and extension.

        Returns:
            pts: pts object extracted from the file.
        """
    
        ptsfile = np.genfromtxt(pathname, delimiter = ' ',
                                    dtype = float, skip_header = 1
                                    )

        p1 = ptsfile[:,0]
        p2 = ptsfile[:,1]
        p3 = ptsfile[:,2]

        full_name = pathname.split("/")[-1]
        name_noext_vec = full_name.split(".")
        name_noext = '.'.join(name_noext_vec[:-1])

        return cls(p1, p2, p3, name_noext)

    def extract(self,vtx_array):
        """Function to extract the pts of a submesh based on a vtx.

        Args:
            vtx_array (vtx): vtx of the submesh

        Raises:
            NameError: If the vtx does not come from this pts.

        Returns:
            pts: pts object of the submesh.
        """

        if(vtx_array.mesh_from != self.name):
            raise NameError("The vtx does not come from that mesh. You should use {}.pts".format(vtx_array.mesh_from))
        
        new_p1 = self.p1[vtx_array.indices]
        new_p2 = self.p2[vtx_array.indices]
        new_p3 = self.p3[vtx_array.indices]

        return pts(new_p1, new_p2, new_p3, self.name)

    def write(self,pathname):
        """Function to write a pts object to a file.

        Args:
            pathname (str): Full path (including filename and extension).
        """

        header = np.array([str(self.size)])
        data = np.array([self.p1, self.p2, self.p3])

        np.savetxt(pathname, header, fmt='%s')

        with open(pathname, "ab") as datafile_id:
            np.savetxt(datafile_id, np.transpose(data), fmt = "%s")

class lon:
    
    def __init__(self, f1, f2, f3, s1 = None, s2 = None, s3 = None):
        """Function to initialize a lon object (from fibres).

        Args:
            f1 (numpy array): First coordinate of the fibre directions for all
            the elements.
            f2 (numpy array): Second coordinate of the fibre directions for all
            the elements.
            f3 (numpy array): Third coordinate of the fibre directions for all
            the elements.
            s1 (numpy array): First coordinate of the sheet directions for all
            the elements.
            s2 (numpy array): Second coordinate of the sheet directions for all
            the elements.
            s3 (numpy array): Third coordinate of the sheet directions for all
            the elements.
        """
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3

        self.s1 = s1
        self.s2 = s2
        self.s3 = s3

    @classmethod
    def read(cls, pathname):
        """Function to read a lon file and convert it in a lon object.

        Args:
            pathname (str): Full path including filename and extension.

        Returns:
            lon: lon object from the file.
        """

        print("Reading lon file...")

        lonfile = np.genfromtxt(pathname, delimiter = ' ',
                                    dtype = float, skip_header = 1
                                    )
        print("File read.")
        f1 = lonfile[:,0]
        f2 = lonfile[:,1]
        f3 = lonfile[:,2]

        if(lonfile.shape[1] > 3):
            s3 = lonfile[:,-1]
            s2 = lonfile[:,-2]
            s1 = lonfile[:,-3]
        else:
            s1 = None
            s2 = None
            s3 = None

        return cls(f1, f2, f3, s1, s2, s3)

    def orthogonalise(self):
        """Function to orthogonalise the sheet directions based on the fibre
        direction. The assumption is if the z coordinate is too small (seems a
        bug from CARP).
        """
        count = 0

        for i in range(len(self.f1)):
            f = np.array([self.f1[i], self.f2[i], self.f3[i]])
            s = np.array([self.s1[i], self.s2[i], self.s3[i]])
            if(np.dot(f,s) > 1e-5):
                s_corrected = orthogonalise(f,s)
                self.s1[i] = s_corrected[0]
                self.s2[i] = s_corrected[1]
                self.s3[i] = s_corrected[2]
                count = count + 1
        print("Corrected sheet direction for " + str(count) + " elements.")

    def write(self,pathname):
        """Function to write the lon object in a file.

        Args:
            pathname (str): Full path including filename and extension.
        """

        header = np.array([str(self.f1.size)])

        if(self.s1 is None):
            header = np.array(["1"])
            data = np.array([self.f1, self.f2, self.f3])
        else:
            header = np.array(["2"])
            data = np.array([self.f1, self.f2, self.f3, self.s1, self.s2, self.s3])

        np.savetxt(pathname, header, fmt='%s')

        with open(pathname, "ab") as datafile_id:
            np.savetxt(datafile_id, np.transpose(data), fmt = "%s")

    @classmethod
    def normalise(cls, self):
        matrix_lon = np.transpose(np.array([self.f1, self.f2, self.f3]))
        row_norms = np.linalg.norm(matrix_lon, axis = 1)
        new_matrix = matrix_lon / row_norms[:, np.newaxis]

        return cls(new_matrix[:,0], new_matrix[:,1], new_matrix[:,2])

def orthogonalise(f, s):
    """Function to rotate a vector to make it orthogonal to the other.

    Args:
        f (numpy array): Vector base.
        s (numpy array): Vector to move to make it orthogonal to f.

    Returns:
        numpy array: Corrected vector.
    """

    c = numpy.cross(f,s)
    d = np.dot(f,s)

    norm_c = np.linalg.norm(c)
    axis = (1/norm_c)*c

    theta = np.pi - np.arccos(d)

    R00 = axis[0]**2 + np.cos(theta) + (1-axis[0]**2)
    R01 = (1-np.cos(theta))*axis[0]*axis[1] - axis[2]*np.sin(theta)
    R02 = (1-np.cos(theta))*axis[0]*axis[2] + axis[1]*np.sin(theta)

    R10 = (1-np.cos(theta))*axis[0]*axis[1] + axis[2]*np.sin(theta)
    R11 = axis[1]**2 + np.cos(theta) + (1-axis[1]**2)
    R12 = (1-np.cos(theta))*axis[1]*axis[2] - axis[0]*np.sin(theta)

    R20 = (1-np.cos(theta))*axis[0]*axis[2] - axis[1]*np.sin(theta)
    R21 = (1-np.cos(theta))*axis[1]*axis[2] + axis[0]*np.sin(theta)
    R22 = axis[2]**2 + np.cos(theta) + (1-axis[2]**2)

    R = np.array([[R00, R01, R02], [R10, R11, R12], [R20, R21, R22]])

    return R.dot(s)

class elem:
    
    def __init__(self, i1, i2, i3, i4, tags):
        """Init function for the elem class.

        Args:
            i1 (numpy array of integers): Array with the first index for each 
            element.
            i2 (numpy array of integers): Array with the second index for each 
            element.
            i3 (numpy array of integers): Array with the third index for each 
            element.
            i4 (numpy array of integers): Array with the fourth index for each 
            element.
            tags (numpy array of integers): List of tags for each element.
        """
        self.i1 = i1.astype(int)
        self.i2 = i2.astype(int)
        self.i3 = i3.astype(int)
        self.i4 = i4.astype(int)
        self.tags = tags.astype(int)

        self.size = i1.shape[0]

    @classmethod
    def read(cls, pathname):
        """Function to read a .elem and convert it to an elem object.

        Args:
            pathname (str): Full path (including filename and extension).

        Returns:
            elem: elem object extracted from the file.
        """
        print("Reading elem file...")
        elemfile = np.genfromtxt(pathname, delimiter = ' ',
                                    dtype = int, skip_header = True,
                                    usecols= (1,2,3,4,5)
                                    )
        print("File read.")

        return cls(elemfile[:,0], elemfile[:,1], elemfile[:,2],
                   elemfile[:,3], elemfile[:,4])


    def write(self,pathname):
        """Function to write a elem object to a file.

        Args:
            pathname (str): Full path (including filename and extension).
        """

        header = np.array([str(self.size)])
        elemtype = np.repeat("Tt",self.size)

        data = [elemtype, self.i1, self.i2, self.i3, self.i4, self.tags]

        np.savetxt(pathname, header, fmt='%s')

        with open(pathname, "ab") as f:
            np.savetxt(f, np.transpose(data), fmt = "%s")
