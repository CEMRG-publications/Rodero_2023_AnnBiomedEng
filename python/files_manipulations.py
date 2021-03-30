import numpy as np

def write_vtx(pathname, array):

    header = [len(array), "intra"]
    str_to_write = np.append(header, array)

    with open(pathname, 'w') as f:
        for item in str_to_write:
            f.write("%s\n" % item)

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
