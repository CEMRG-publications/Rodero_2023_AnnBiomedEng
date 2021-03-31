import numpy as np

def write_vtx(pathname, array):

    array = np.array(array)
    header = [array.size, "intra"]
    str_to_write = np.append(header, array)

    with open(pathname, 'w') as f:
        for item in str_to_write:
            f.write("%s\n" % item)

def read_vtx(pathname):

    vtx = np.genfromtxt(pathname, dtype = int, skip_header = 2)

    return vtx

class surf:
    
    def __init__(self, p1, p2, p3, tags):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.tags = tags
        self.size = p1.shape[0]

    @classmethod
    def read(cls, pathname):
    
        surffile = np.genfromtxt(pathname, delimiter = ' ',
                                    dtype = int, skip_header = True,
                                    usecols= (1,2,3,4)
                                    )

        p1 = surffile[:,0]
        p2 = surffile[:,1]
        p3 = surffile[:,2]
        tags = surffile[:,3]

        return cls(p1, p2, p3, tags)

    def tovtx(self):
        vtxvec =  np.unique(np.concatenate((self.p1, self.p2, self.p3),
                                            axis = 0))
        
        return vtxvec

class pts:
    
    def __init__(self, p1, p2, p3):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.size = p1.shape[0]

    @classmethod
    def read(cls, pathname):
    
        ptsfile = np.genfromtxt(pathname, delimiter = ' ',
                                    dtype = float, skip_header = 1
                                    )

        p1 = ptsfile[:,0]
        p2 = ptsfile[:,1]
        p3 = ptsfile[:,2]

        return cls(p1, p2, p3)

    def extract(self,vtx_array):
        
        new_p1 = self.p1[vtx_array]
        new_p2 = self.p2[vtx_array]
        new_p3 = self.p3[vtx_array]

        return pts(new_p1, new_p2, new_p3)

    def write(self,pathname):

        header = np.array([str(self.size)])
        data = np.array([self.p1, self.p2, self.p3])

        np.savetxt(pathname, header, fmt='%s')

        with open(pathname, "ab") as datafile_id:
            datafile_id.write(b"\n")
            np.savetxt(datafile_id, np.transpose(data), fmt = "%s")