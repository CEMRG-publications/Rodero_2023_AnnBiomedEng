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
