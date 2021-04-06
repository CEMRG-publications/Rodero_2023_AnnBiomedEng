import numpy as np
import csv

class vtx:

    def __init__(self, indices, mesh_from):
        self.indices = indices
        self.size = len(indices)
        self.mesh_from = mesh_from

    def __str__(self):
        return "Vtx variable of {} vertices extracted from {}".format(self.size,self.mesh_from)

    @classmethod
    def read(cls, pathname, mesh_from):
        vtx_var = np.genfromtxt(pathname, dtype = int, skip_header = 2)

        return cls(vtx_var, mesh_from)

    def write(self,pathname):
        array = np.array(self.indices)
        header = [self.size, "intra"]
        str_to_write = np.append(header, array)

        with open(pathname, 'w') as f:
            for item in str_to_write:
                f.write("%s\n" % item)
class surf:
    
    def __init__(self, i1, i2, i3, mesh_from, tags = None):
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
        
        return cls(surfmesh_var.i1, surfmesh_var.i2, surfmesh_var.i3, surfmesh_var.mesh_from)
    
    @classmethod
    def tosurfmesh(cls, surfmesh_var):

        tags = np.zeros(surfmesh_var.size)
        return cls(surfmesh_var.i1, surfmesh_var.i2, surfmesh_var.i3, surfmesh_var.mesh_from, tags)

    def tovtx(self):
        vtxvec =  np.unique(np.concatenate((self.i1, self.i2, self.i3),
                                            axis = 0))
        
        return vtx(vtxvec, self.mesh_from)

    def write(self,pathname):

        header = np.array([str(self.size)])
        elemtype = np.repeat("Tr",self.size)
        data = [elemtype, self.i1, self.i2, self.i3]

        if(self.tags is not None):
            data = [elemtype, self.i1, self.i2, self.i3, self.tags]

        np.savetxt(pathname, header, fmt='%s')

        with open(pathname, "ab") as f:
            np.savetxt(f, np.transpose(data), fmt = "%s")

class pts:
    
    def __init__(self, p1, p2, p3, name):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.size = p1.shape[0]
        self.name = name

    @classmethod
    def read(cls, pathname):
    
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

        if(vtx_array.mesh_from != self.name):
            raise NameError("The vtx does not come from that mesh. You should use {}.pts".format(vtx_array.mesh_from))
        
        new_p1 = self.p1[vtx_array.indices]
        new_p2 = self.p2[vtx_array.indices]
        new_p3 = self.p3[vtx_array.indices]

        return pts(new_p1, new_p2, new_p3, self.name)

    def write(self,pathname):

        header = np.array([str(self.size)])
        data = np.array([self.p1, self.p2, self.p3])

        np.savetxt(pathname, header, fmt='%s')

        with open(pathname, "ab") as datafile_id:
            np.savetxt(datafile_id, np.transpose(data), fmt = "%s")


class lon:
    
    def __init__(self, f1, f2, f3, s1, s2, s3):
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3

        self.s1 = s1
        self.s2 = s2
        self.s3 = s3

    @classmethod
    def read(cls, pathname):

        lonfile = np.genfromtxt(pathname, delimiter = ' ',
                                    dtype = float, skip_header = 1
                                    )

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

        header = np.array([str(self.f1.size)])

        if(self.s1 is None):
            data = np.array([self.f1, self.f2, self.f3])
        else:
            data = np.array([self.f1, self.f2, self.f3, self.s1, self.s2, self.s3])

        np.savetxt(pathname, header, fmt='%s')

        with open(pathname, "ab") as datafile_id:
            np.savetxt(datafile_id, np.transpose(data), fmt = "%s")

def orthogonalise(f, s):

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
