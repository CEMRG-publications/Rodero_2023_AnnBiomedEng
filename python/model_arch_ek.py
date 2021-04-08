#!/usr/bin/python2


EXAMPLE_DESCRIPTIVE_NAME = 'Model Architecture: Laplacian'
EXAMPLE_AUTHOR = 'Karli Gillette <karli.gillette@medunigraz.at>,' \
                 'Toni Prassl <anton.prassl@medunigraz.at>'

import os
import sys
import time
import math
import scipy
import pandas
import subprocess
from io import StringIO
from datetime import date

import numpy as np
import matplotlib

from glob import glob
from scipy import spatial
from carputils.carpio import meshtool as mt
from carputils import tools, settings, model, mesh

# provide Python2 and Python3 compatibility
isPy2 = True
if sys.version_info.major > 2:
    isPy2 = False
    raw_input = input
    xrange = range

def parser():
    parser = tools.standard_parser()

    runopts= parser.add_argument_group(':::::::::: execution options :::::::::')
    runopts.add_argument('--uvc',
                        action='store_true',
                        help='Construct a UVC coordinate system on mesh.')
    runopts.add_argument('--fibers',
                        action='store_true',
                        help='Generate rule based fibers')
    runopts.add_argument('--hetero',
                        action='store_true',
                        help='Reassign element tags for 9 chamber heterogenities.')
    runopts.add_argument('--vtk',
                        action='store_true',
                        help='Generate VTK files (meshes and fibers) for visualisation in Paraview.')

    runopts.add_argument('--rerun_fibers',
                        action='store_true',
                        help='Re-run fibers computation, if directories exist.')
    runopts.add_argument('--rerun_uvc',
                        action='store_true',
                        help='Re-run uvc computation, if directories exist.')
    runopts.add_argument('--rerun_laplace',
                        action='store_true',
                        help='Re-run laplace computation, if directories exist.')
    runopts.add_argument('--rerun_hetero',
                        action='store_true',
                        help='Re-run hetero computation, if directories exist.')
    runopts.add_argument('--rerun_meshes',
                        action='store_true',
                        help='Re-run mesh construction, if directories exist.')


    fiberopts = parser.add_argument_group(':::::::::: fiber creation :::::::::')
    fiberopts.add_argument('--fibre_rule',
                           type=float, nargs=4, action='append',
                           default=[],
                           help='specify fibre rule angles in order alpha_endo, '
                                'alpha_epi, beta_endo, beta_epi')
    fiberopts.add_argument('--fibre_rule_alpha',
                           type=float, nargs=2, action='append',
                           default=[],
                           help='specify fibre rule angles in order alpha_endo, '
                                'alpha_epi with default beta (-65, 25)')

    meshopts = parser.add_argument_group(':::::::::: mesh options :::::::::')
    meshopts.add_argument('--basename',
                          help='Basename of heart mesh with blood volume to process. '
                               'Please provide associated tagging (--tags) and mesh type (--mode).'
                               'If no basename provided, an ellipsoidal mesh is generated.')
    meshopts.add_argument('--tags',
                          default='etags.sh',
                          help='Bash file providing tags for all anatomies in mesh. '
                               'Please provide one tag per line with T_LV=*, T_RV=*, etc.')
    meshopts.add_argument('--mode',
                          default='lv',
                          choices=['biv','lv','h4c'],
                          help='Identity the type of mesh provided.')
    meshopts.add_argument('--ellipse_res',
                          default=1,
                          type=float,
                          help='Resolution of the elliptic mesh')
    meshopts.add_argument('--ellipse_rad',
                          default=25,
                          type=float,
                          help='Radius of the elliptic mesh')

    hetopts = parser.add_argument_group(':::::::::: incorporating heterogeneity :::::::::')
    hetopts.add_argument('--tags_apba',
                         type=int, nargs=3, action='append',
                         default=[100,200,300],
                         help='Element tag for apico-basal heart regions')
    hetopts.add_argument('--tags_wall',
                         type=int, nargs=3, action='append',
                         default=[25,50,75],
                         help='Element tag for endocardial wall elements (default: 25)')
    hetopts.add_argument('--epi_endo_cutoffs',
                          type=float,
                          default=[25.,30.],
                          help='Thickness of epicardial wall layer (default: 25%%) and endocardial wall layer (default: 30%%)')
    hetopts.add_argument('--apex_base_cutoffs',
                          type=float,
                          default=[45.,75.],
                          help='Size of apex based on Laplace solution (default: below 45%%) '
                               'and size of heart base based on Laplace solution (default: above 75%%)')
    return parser

def jobID(args):
        """
        Generate name of top level output directory.
        """
        today = date.today()
        args.ID='{}_model_arch_{}_{}_{}'.format(today.isoformat(), args.flv, args.np, args.mode)

        return args.ID


class dspace(object):

    def __init__(self, **kwargs):
        self._attrdict = {}
        self.extend(kwargs)

    def add(self, attrname, attrvalue):
        if not attrname[0].isalpha():
            raise SyntaxError('attribute names have to begin with an alphbetic character (`{}`)!'.format(attrname))
        self._attrdict[attrname] = attrvalue

    def extend(self, attrdict):
        if isPy2: it = attrdict.iteritems
        else:     it = attrdict.items
        for attrname, attrvalue in it():
            self.add(attrname, attrvalue)

    def iterattr(self):
        if isPy2: it = self._attrdict.iteritems
        else:     it = self._attrdict.items
        return it()

    def __getattr__(self, attrname):
        return self._attrdict[attrname]

    #
    def __getitem__(self, attrname):
        return self._attrdict[attrname]

    def __setitem__(self, attrname, attrvalue):
        self.add(attrname, attrvalue)

    def __len__(self):
        return len(self._attrdict)

    def __str__(self):
        return str(self._attrdict)

    def pretty_string(self, level=0, indent=3):
        pstr, istr = '', ' ' * indent * level

        if isPy2:   it = self._attrdict.iteritems
        else:       it = self._attrdict.items
        for attrname, attrvalue in it():
            if isinstance(attrvalue, dspace):
                pstr += istr + '{}:\n'.format(attrname)
                pstr += attrvalue.pretty_string(level + 1, indent)
            else:
                pstr += istr + '{}: {}\n'.format(attrname, attrvalue)
        return pstr

    __repr__ = __str__

    def __iadd__(self, other):
        if isinstance(other, dict):
            self.extend(other)
            return self
        elif isinstance(other, dspace):
            self.extend(other._attrdict)
            return self
        else:
            raise TypeError('can only add dictionaries of dictspaces to a dictspace!')

class simplemath:

    def __init__(self):
        pass

    @staticmethod
    def PCA(data, dims_rescaled_data=3):
        """
        returns: data transformed in 2 dims/columns + regenerated original data
        pass in: data as 2D/3D NumPy array
        """
        m, n = data.shape
        # mean center the data
        #data -= data.mean(axis=0)
        cdata = data.mean(axis=0)
        # calculate the covariance matrix
        R = np.cov(data-cdata, rowvar=False)
        # calculate eigenvectors & eigenvalues of the covariance matrix
        # use 'eigh' rather than 'eig' since R is symmetric,
        # the performance gain is substantial
        evals, evecs = scipy.linalg.eigh(R)
        # sort eigenvalue in decreasing order
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:,idx]
        # sort eigenvectors according to same index
        evals = evals[idx]
        # select the first n eigenvectors (n is desired dimension
        # of rescaled data array, or dims_rescaled_data)
        evecs = evecs[:, :dims_rescaled_data]
        # carry out the transformation on the data using eigenvectors
        # and return the re-scaled data, eigenvalues, and eigenvectors
        #return
        # , evals, evecs
        return (evals, evecs)

    # input
    #   array:   1D/2D numpy array of integers
    #   indices: 1D numpy array of integers
    @staticmethod
    def ismember1d(arr,inds):
        arr       = arr.squeeze()
        inds      = inds.squeeze()

        arr_bool  = np.in1d(arr, inds)
        arr_bool  = np.reshape(arr_bool, arr.shape)

        memberIDs = np.asarray(arr_bool.nonzero())
        if isinstance(memberIDs, int):
            return arr[memberIDs], memberIDs
        else:
            return arr[memberIDs.squeeze()], memberIDs.squeeze()

    @staticmethod
    def ismember2d(arr,inds):

        arr       = arr.squeeze()
        inds      = inds.squeeze()

        arr_bool  = np.in1d(arr, inds)
        arr_bool  = np.reshape(arr_bool, arr.shape)

        memberIDs = np.count_nonzero(arr_bool, axis=1)
        if isinstance(memberIDs, int):
            return arr_bool.squeeze(), memberIDs
        else:
            return arr_bool.squeeze(), memberIDs.squeeze()

    ## angle between two vectors
    #def angle(a,b):
    #    c  = np.dot(a,b)
    #    c /= np.linalg.norm(a)
    #    c /= np.linalg.norm(b)#
    #
    #    angle   = np.arccos(np.clip(c, -1, 1))
    #    angle_d = angle * np.pi/180
    #    return angle, angle_d

    # vector length
    @staticmethod
    def length(v):
      return np.sqrt(v.dot(v))

    @staticmethod
    def dotproduct(v1, v2):
      return sum((a*b) for a, b in zip(v1, v2))

    # normalize vector
    @staticmethod
    def normalize(vec):
        vec /= np.linalg.norm(vec)
        return vec

    @staticmethod
    def angle(v1, v2):
        angle   = math.acos(simplemath.dotproduct(v1, v2) / (simplemath.length(v1) * simplemath.length(v2)))
        angle_d = angle * np.pi/180
        return angle, angle_d

    # edge length
    @staticmethod
    def edgelen(a, b, pts) :
        x = np.subtract(pts[a,:], pts[b,:])
        return simplemath.length(x)

    # distance between list of nodes and a single node

    @staticmethod
    def dists(node, nodes):
        dists_ = scipy.spatial.distance.cdist([node], nodes)
        return dists_.squeeze()

    # normalize distance along a path
    @staticmethod
    def normalize_distance(path, pts):
        cdist = 0
        ndist = [cdist]
        p0    = pts[path[0]]

        for i in range(1,len(path)):
            p1     = pts[path[i]]
            cdist += simplemath.length(p1-p0)
            ndist.append( cdist )
            p0     = p1

        total_len = ndist[-1]
        ndist    /= total_len
        return ndist

    # Node to elements connectivity
    @staticmethod
    def node2elem(elemList, numNodes=None ):
        # Note: the nodes list MAY not have the length of the pointsfile if
        #       numNodes was not passed to the function

        numElems = elemList.shape[0]
        if numNodes is None:
            numNodes = max(elemList.flatten())

        n2e = [[] for i in range(numNodes)] # list of elements each node is in

        for i in xrange(numElems) :
            elem = elemList[i,:]

            # list of attached elements for each node
            for p in elem:
                n2e[p].append(i)

        return n2e

    # Note: this algorith is available as speeedy meshtool standalone
    @staticmethod
    def same_side_of_plane(tri, pRef, pTest):

        pRef  = pRef.squeeze()
        pTest = pTest.squeeze()

        if not len(pRef) == 3:
          print('Invalid call of SameSideOfPlane -> pRef')

        if not len(pTest) ==3:
          print('Invalid call of SameSideOfPlane -> pTest')

        p01        = tri[:,1] - tri[:,0]
        p02        = tri[:,2] - tri[:,0]

        inw_normal = np.cross(p01, p02)
        value      = np.dot(inw_normal, (pRef-tri[:,0]).reshape([3,1]))
        inw_normal = inw_normal * np.sign(value)

        if   np.dot(inw_normal, pTest-tri[:,0]) < 0:     # point on different sides
            return False
        elif np.dot(inw_normal, pTest-tri[:,0]) > 0:     # point on same side
            return True
        else:
            return True                                     # point on plane

    @staticmethod
    def distance_point_plane(xyzA, xyzB, xyzC, xyzP, planeNormal=None):

        # nodes A, B and C are forming the plane
        # nVec is the plane normal
        # node P is the distant (test) node

        # considering two versions of input
        # version 1: - single node of plane (xyzA)
        #            - plane normal
        #            - distant node to be investigated (xyzP)
        #
        # version 2: - three plane nodes (xyzA, xyzB, xyzC)
        #            - distant node to be investigated (xyzP)

        if planeNormal is None:
            planeNormal = np.cross(xyzB-xyzA, xyzC-xyzA)

        planeNormal = simplemath.normalize(planeNormal)

        distance    = np.dot(planeNormal, xyzP) - np.dot(planeNormal, xyzA)
        distance    = np.fabs(distance)

        # Check if the normal vector shows into the direction of P
        if np.dot(planeNormal, xyzP-xyzA) < 0:
          planeNormal = -planeNormal

        return distance, planeNormal

    # starting with a seed
    @staticmethod
    def align_neighbors(vtx_list, pts, seed=[]):

        if not seed:
            seed = vtx_list[0]

        aligned_vtx_list = [seed]
        search_dists     = [ ]      # keep track of distances between aligned nodes
        vtx_list         = np.setdiff1d(vtx_list, seed) # remove seed from search list

        while len(vtx_list):

            seed = simplemath.find_nearest_neighbor(vtx_list, seed, pts)

            # update lists
            search_dists.append(simplemath.edgelen(seed, aligned_vtx_list[-1], pts))

            if len(search_dists) > 5:
                # searching backwards is a no-go
                mean_search_dists = np.mean(search_dists[:-1])
                if search_dists[-1] > 5*mean_search_dists:
                    # drop this element
                    search_dists = search_dists[:-1]
                    vtx_list     = np.setdiff1d(vtx_list, seed)
                    # use previous seed again
                    seed = aligned_vtx_list[-1]
                    continue

            vtx_list = np.setdiff1d(vtx_list, seed)
            aligned_vtx_list.append(seed)

        return aligned_vtx_list

    @staticmethod
    def convert_to_rgb(minval, maxval, val, colors):

        EPSILON = sys.float_info.epsilon

        fi = float(val-minval) / float(maxval-minval) * (len(colors)-1)
        i = int(fi)
        f = fi - i
        if f < EPSILON:
            return colors[i],val
        else:
            (r1, g1, b1), (r2, g2, b2) = colors[i], colors[i+1]
            return int(r1 + f*(r2-r1)), int(g1 + f*(g2-g1)), int(b1 + f*(b2-b1)), val

    # search for node in array a which is closest to any node in array b
    @staticmethod
    def find_nearest_neighbor(arr_a, arr_b, pts):

        if len(arr_a) == 0:
            return -1

        # allocate list of distances
        dists = []

        # provide inds of b as points list
        pts_b = pts[arr_b,:]

        for a in arr_a:
            pt_a   = pts[a,:]
            dist_a = np.subtract(pts_b, pt_a)

            if len(dist_a) == 3:
                dist_a = np.linalg.norm(dist_a)
                dist_a = dist_a.reshape([1,1])
            else:
                dist_a = np.linalg.norm(dist_a, axis=1)

            argmin = np.argmin(dist_a)
            dists.append(dist_a[argmin])

        return arr_a[np.argmin(dists)]

class base:

    def __init__(self):
        pass

    @staticmethod
    def guess_ext_and_check(basename, name, specified=None,exts=['.vtx','.surf.vtx']):

        n_flag = False

        if exts is None:
            n_flag = True
            exts=['.elem']

        exts = ['.elem' if v is None else v for v in exts]

        for ext in exts:
            if specified is not None:
                if not os.path.exists(specified + ext):
                    msg = 'Specified file {}.{} does not exist'.format(specified,ext)
                    path=None
                return specified

            # Attempt to guess name
            fname = '{}.{}{}'.format(basename, name, ext)
            # print fname

            if os.path.isfile(fname):
                if n_flag:
                    return fname.strip('.elem')
                else:
                    return fname
        return None

    @staticmethod
    def makevtxfcn(job,args,lap_lvphi,rvjsurf_lv,rvjsurf_rv):

        base.debugString(args,'Generating vtxfcn file: '+rvjsurf_rv)

        vtx_rvjrv=rvjsurf_rv.replace('.surf.vtx','.vtxfcn.surf.vtx')

        dat_lvphi = pandas.read_csv(lap_lvphi, delimiter='\n', header=None).values.squeeze()
        df_surf_lv  = pandas.read_csv(rvjsurf_lv, skiprows=2, delimiter=' ', header=None)
        df_surf_rv  = pandas.read_csv(rvjsurf_rv, skiprows=2, delimiter=' ', header=None)

        surf_vtx_lv = df_surf_lv.values.astype(np.int).flatten()
        surf_vtx_rv = df_surf_rv.values.astype(np.int).flatten()

        ind=surf_vtx_lv.T

        np.savetxt(vtx_rvjrv,np.c_[surf_vtx_rv,dat_lvphi[ind]],fmt='%u %f')
        job.bash(['sed', '-i', '1iintra', vtx_rvjrv], None)
        job.bash(['sed', '-i', '1i'+ str(len(surf_vtx_rv)), vtx_rvjrv], None)

        return vtx_rvjrv.replace('.surf.vtx','.surf')

    @staticmethod
    def replace_all(lst, dic):

        newlst=[]

        for i in range(0,len(lst)):
            curr=lst[i].split('=')
            tmp=[curr[1]]

            for c2sub in [':',',','-','(',')']:
                for k in range(0,len(tmp)):
                    tmp[k]=base.intersperse(tmp[k].split(c2sub),c2sub)
                tmp = [y for x in tmp for y in x]

            for k in range(0,len(tmp)):
                for i, j in dic.iterattr():
                    if tmp[k] == i:
                        tmp[k]= str(j)

            curr[1]=''.join(tmp)
            newlst.append(curr)



        return dict(newlst)

    @staticmethod
    def intersperse(lst, item):
        result = [item] * (len(lst) * 2 - 1)
        result[0::2] = lst
        return result

    @staticmethod
    def debugString(args,msg,mode=1):

        try:
            r,c=os.popen('stty size','r').read().split()
        except:
            c=80

        if not args.silent:
            if mode == 0:
                print('='*int(c))
                print(msg)
                print('='*int(c))
                print(' ')
            elif mode ==1:
                c_shift=int(int(c)/2)-int(len(msg)/2)
                print('#'+'-'*(int(c)-1))
                print('#'+' '*c_shift+msg)
                print('#'+'-'*(int(c)-1))
                print(' ')
            elif mode == 2:
                print(msg)
        return

    @staticmethod
    def self_path():
        path = os.path.dirname(sys.argv[0])
        if not path:
            path = '.'
        return path

    @staticmethod
    def get_elem_tags(job, meshname):

        m_tags   = [] # myocardial tags
        b_tags   = [] # bath tags
        READNEXT = False

        strstream = StringIO.StringIO()
        job.meshtool(['query', 'tags', '-msh={}'.format(meshname)], stdout=strstream)
        output = strstream.getvalue().split('\n')
        strstream.close()

        # assuming user.name present
        for line in output:
            if 'Myocardium' in line or 'Bath' in line:
                READNEXT = True
                continue

            if READNEXT:
                READNEXT = False
                if not len(m_tags):
                    # read first myocardial tags
                    m_tags = line.rstrip().split(',')
                    m_tags = np.asarray(m_tags, dtype=np.int)
                    continue
                if not len(b_tags):
                    b_tags = line.rstrip()
                    if b_tags:
                        b_tags = np.asarray(b_tags.split(','), dtype=np.int)
                    else:
                        b_tags = np.array([],dtype=np.int)
                    continue

        return m_tags, b_tags

class model_arch(dspace):
    """
    Creation of the simulation infrastructure.
    """
    def __init__(self,args,job):
        super(model_arch, self).__init__()

        self.args = self.check_args(args)
        self.check_dependencies()

        self.tags_dict(self.args)
        self.msh_dict(self.args)
        self.sols_dict(self.args)

        print(self.pretty_string())

        gen_meshes(job,self.args,self)
        gen_sols(job,self.args,self)

        if self.args.fibers: gen_fibers(job, self.args,self)  # generate fibers
        if self.args.uvc: gen_uvc(job, self.args,self)  # generate UVC coordinates
        if self.args.hetero: gen_hetero(job, self.args,self)  # generate heterogeneity

    @staticmethod
    def check_args(args):
        """
        Check that all arguments are correct and files exist.
        """
        if args.basename:

            if args.tags is None:
                raise IOError('Please specify a tags file.')

            if not os.path.isfile(args.basename+'.elem'):
                raise IOError('Could not locate the basemesh elem file: '+args.basename+'.elem')
            if not os.path.isfile(args.basename+'.pts'):
                raise IOError('Could not locate the basemesh pts file: '+args.basename+'.pts')

            if not os.path.isfile(args.tags):
                raise IOError('Could not locate tag information: '+args.tags)

            assert args.mode in ['biv','lv','h4c']

            if not 'um' or 'mm' in args.basename:
                raise IOError('Please specify resolution in basename. Automatic query not implemented.')

            if base.guess_ext_and_check(args.basename, 'base') is None and args.mode != 'h4c':
                raise IOError('The base of the mesh must exist in the basename directory for the biv and lv modes.')

            if base.guess_ext_and_check(args.basename,'apex') is None and args.mode == 'lv':
                raise IOError('The apex of the mesh must exist in basename directory for lv mode.')

            if base.guess_ext_and_check(args.basename,'rvsept_point') is None and args.mode == 'lv':
                raise IOError('The rvsept_point of the mesh must exist in basename directory for lv mode.')

        else:

            assert args.mode in ['lv']

            print('Assuming no tags file....Using default tags')

        # Argument modification
        if args.mode !='lv':
            args.mmode='biv'
        else:
            args.mmode='lv'

        return args

    @staticmethod
    def check_dependencies():

        if subprocess.check_output(['whereis','GlElemCenters'],stdin=subprocess.PIPE).split(':')[1] == '':
            raise ImportWarning('GlElemCenters is not found in executable path. Hint: run make tools from carp directory')

        if subprocess.check_output(['whereis','samesideofplane'],stdin=subprocess.PIPE).split(':')[1] == '':
            raise ImportWarning('samesideofplane not found in executable path. Hint: please make meshtool and '
                                'then make standalones in the FEMLIB directory of carp.')

    def tags_dict(self,args):

        if args.tags:

            f=open(args.tags,'r')
            tmp=[line.split(' ')[0].strip('\n') for line in f.readlines() if line.startswith('T_') and line.split(' ')[0].split('=')[1] is not '']
            f.close()

            list_tags=dspace(**dict([x.split('=') for x in tmp]))

            #Generating submeshes
            ext_op=['lv={},{}'.format( list_tags.T_LV, list_tags.T_LVBP),
                    'rv={},{}'.format( list_tags.T_RV, list_tags.T_RVBP),
                    'biv={},{}'.format(list_tags.T_RV, list_tags.T_LV),
                    'bivBP={},{},{},{}'.format(list_tags.T_RV, list_tags.T_LV,list_tags.T_LVBP,list_tags.T_RVBP)]

            sub_op={}
            if args.mode == 'h4c':
                ext_op+=['biv.rvj={}:{}'.format(list_tags.T_LV,list_tags.T_RV),
                         'biv.rvsept={}:{}'.format(list_tags.T_RVBP,list_tags.T_LV),
                         'biv.rvendo_nosept={}:{}'.format(list_tags.T_RVBP,list_tags.T_RV),
                         'biv.epi={},{},{},{}-{},{},{},{},{},{},{},{}'.format(list_tags.T_RV,list_tags.T_LV,list_tags.T_RVBP,list_tags.T_LVBP,
                                                                              list_tags.T_RA,list_tags.T_LA,list_tags.T_RABP,list_tags.T_LABP,
                                                                              list_tags.T_PULMARTERY,list_tags.T_PULMARTERYBP,
                                                                              list_tags.T_AORTA,list_tags.T_AORTABP),
                         'biv.rvendo={}-{},{}'.format(list_tags.T_RVBP,list_tags.T_RABP,list_tags.T_PULMARTERYBP),
                         'biv.lvendo={}-{},{}'.format(list_tags.T_LVBP,list_tags.T_LABP,list_tags.T_AORTABP),
                         'biv.base={},{}:{},{},{}'.format(list_tags.T_RV,list_tags.T_LV,
                                                          list_tags.T_AORTA,list_tags.T_RA,list_tags.T_LA)]

            if args.mode =='biv':
                ext_op+=['biv.rvj={}:{}'.format(list_tags.T_LV,list_tags.T_RV),
                         'biv.rvsept={}:{}'.format(list_tags.T_RVBP,list_tags.T_LV),
                         'biv.rvendo_nosept={}:{}'.format(list_tags.T_RVBP,list_tags.T_RV),
                         'biv.epi={},{}-{},{}'.format(list_tags.T_LV,list_tags.T_RV,list_tags.T_LVBP,list_tags.T_RVBP),
                         'biv.rvendo={}:{},{}'.format(list_tags.T_RVBP,list_tags.T_LV,list_tags.T_RV),
                         'biv.lvendo={}:{}'.format(list_tags.T_LVBP,list_tags.T_LV)]
                sub_op['epi']='epi-base'

            if args.mode =='lv':
                ext_op+=['lv.epi={},{}-{}'.format(list_tags.T_LV,list_tags.T_LVBP,list_tags.T_LVBP),
                         'lv.lvendo={}:{}'.format(list_tags.T_LVBP,list_tags.T_LV),
                         'lv={}'.format(list_tags.T_LV)]
                sub_op['epi']='epi-base'


            self.add('tags',dspace(**{'list':list_tags}))

            list_num=base.replace_all(ext_op,list_tags)
            self['tags'].add('ext',dspace(**list_num))
            self['tags'].add('sub',dspace(**sub_op))

        else:
            list_tags=dspace(**dict({'T_LV':'60','T_RV':'80'}))
            self.add('tags',dspace(**{'list':list_tags}))
            pass


    def msh_dict(self,args):

        biv={'surflist':dspace(**dict.fromkeys(['base','lvendo','rvendo','rvsept','rvendo_nosept','epi','rvj'])),
             'biv':dspace(**dict.fromkeys(['base','lvendo','uvcapex','rvendo','rvsept','rvendo_nosept','rvj','rvja','rvjp','epi','apex'])),
             'lv':dspace(**dict.fromkeys(['lvendo','rvsept','rvja','rvjp','epi','uvcapex','apex','base','rvj'])),
             'rv':dspace(**dict.fromkeys(['rvendo_nosept','rvja','rvjp','epi']))}

        lv={'surflist':dspace(**dict.fromkeys(['base','lvendo','epi','apex','rvsept_point'])),
            'lv': dspace(**dict.fromkeys(['lvendo','uvcapex','base','epi','apex','rvsept_point']))}

        # uvc={'lv'  : dspace(**dict.fromkeys(['lvseptmid','lvpmid','lvamid','lvwallmid']))}

        if args.mmode=='lv':
            self.add('msh',dspace(**lv))
        else:
            self.add('msh',dspace(**biv))
        # if args.uvc: self.extend(uvc)

    def sols_dict(self,args):

        uvc_biv = {'apba'  : dspace(basename='biv',bc=['uvcapex', 'base'],stim=[0,1]),
                   'rv_t'  : dspace(basename='rv', bc=['rvendo_nosept', 'epi'], stim=[0,1]),
                   'lv_t'  : dspace(basename='lv', bc=['lvendo', 'rvsept','rvj','epi'], stim=[0,1,1,1]),
                   'rv_phi': dspace(basename='rv', bc=['rvja','rvjp'],stim=['vtx_fcn','vtx_fcn']),
                   'lv_phi_a': dspace(basename='lv', bc=['lvseptmid','lvwallmid'], stim=[0,  1]),
                   'lv_phi_p': dspace(basename='lv', bc=['lvseptmid','lvwallmid'], stim=[0,  1])}

        uvc_lv = {'apba'  : dspace(basename='lv', bc=['uvcapex', 'base'], stim=[0,1]),
                  'lv_t'  : dspace(basename='lv', bc=['lvendo','epi'], stim=[0,1]),
                  'lv_phi_a': dspace(basename='lv', bc=['lvseptmid','lvwallmid'], stim=[0,  1]),
                   'lv_phi_p': dspace(basename='lv', bc=['lvseptmid','lvwallmid'], stim=[0,  1])}

        fibers_biv = {'apba'    : dspace(basename='biv', bc=['uvcapex', 'base'], stim=[0,1]),
                      'endo_epi': dspace(basename='biv', bc=['lvendo','rvendo','epi'],stim=[0,0,1]),
                      'rv'      : dspace(basename='biv', bc=['lvendo','rvendo','epi'], stim=[0,1,0]),
                      'lv'      : dspace(basename='biv', bc=['lvendo','rvendo','epi'], stim=[1,0,0])
                      }

        fibers_lv =  {'apba': dspace(basename='lv', bc=['uvcapex', 'base'], stim=[0,1]),
                      'lv_t': dspace(basename='lv', bc=['lvendo','epi'], stim=[0,1])
                     }

        hetero_biv=  {'apba': dspace(basename='biv',bc=['uvcapex', 'base'], stim=[0,1]),
                      'rv_t': dspace(basename='rv', bc=['rvendo_nosept', 'epi'], stim=[0,1]),
                      'lv_t': dspace(basename='lv', bc=['lvendo', 'rvsept','rvj','epi'], stim=[0,1,1,1])
                      }

        hetero_lv = {'apba': dspace(basename='lv', bc=['uvcapex', 'base'], stim=[0,1]),
                     'lv_t': dspace(basename='lv', bc=['lvendo','epi'], stim=[0,1])
                    }

        self.add('uvc',dspace(**dict.fromkeys([])))

        sols={}

        if args.fibers:
            sols.update(locals()['fibers_'+args.mmode])

        if args.uvc:
            sols.update(**locals()['uvc_'+args.mmode])

        if args.hetero:
            sols.update(**locals()['hetero_'+args.mmode])

        self.add('sols',dspace(**sols))

class gen_uvc(dspace):

    def __init__(self,job,args,MA_info):
        super(gen_uvc, self).__init__()
        self+=MA_info
        self.job=job
        self.args=args

        t0 = time.time() # start process timer
        base.debugString(args,'GENERATION OF UVC COORDINATES',mode=0)

        curr_dir = os.path.join(args.ID,'UVC')

        [self.uvc.add('{}file'.format(x),os.path.join(curr_dir,'COORDS_{}.pts').format(x)) for x in ['RHO','PHI','Z','V']]

        self.uvc.add('uvcpnts',os.path.join(curr_dir,'COMBINED_COORDS_Z_RHO_PHI_V.pts'))
        # self.uvc.add('cartpnts',args.basename+'.pts')

        curr_dir = os.path.join(args.ID,'UVC')
        if not os.path.exists(curr_dir) or args.rerun_uvc:
            job.mkdir(curr_dir,parents=True)

            if args.mmode == 'biv':

                #TRANSMURAL COORDINATE
                base.debugString(args,'Writing transmural coordinate (rho)')
                self.RHOdata = self.maplvrvdata2biv(self.msh.biv.mesh,
                                                self.msh.lv.mesh,
                                                self.msh.rv.mesh,
                                                self.sols.lv_t,
                                                self.sols.rv_t,
                                                ofile=self.uvc.RHOfile)

                #PHI COORDINATE
                base.debugString(args,'Writing rotational coordinate (phi)')

                self.PHIdata = self.maplvrvdata2biv(self.msh.biv.mesh,
                                                self.msh.lv.mesh,
                                                self.msh.rv.mesh,
                                                self.sols.lv_phi,
                                                self.sols.rv_phi,
                                                ofile=self.uvc.PHIfile)


                #APBP COORDINATE
                base.debugString(args,'Writing apico-basal coordinate (z)')
                self.Zdata=self._write_apba()

                #VENTRICULAR COORDINATE
                base.debugString(args,'Writing ventricular coordinate (v)')
                self.Vdata=self.maplvrvdata2biv(self.msh.biv.mesh,
                                                self.msh.lv.mesh,
                                                self.msh.rv.mesh,
                                                -1,
                                                1,
                                                ofile=self.uvc.Vfile)


            else:
                 #TRANSMURAL COORDINATE
                base.debugString(args,'Writing transmural coordinate (rho)')
                self.RHOdata = self._write_coord(self.sols.lv_t,self.uvc.RHOfile)

                #PHI COORDINATE
                base.debugString(args,'Writing rotational coordinate (phi)')
                self.PHIdata = self._write_coord(self.sols.lv_phi,self.uvc.PHIfile)


                #APBP COORDINATE
                base.debugString(args,'Writing apico-basal coordinate (z)')
                self.Zdata=self._write_apba()

                #VENTRICULAR COORDINATE
                base.debugString(args,'Writing ventricular coordinate (v)')
                self.Vdata=self._write_coord(-1,self.uvc.Vfile)


            if not os.path.isfile(self.uvc.uvcpnts): self._write_comb()

        else:

            base.debugString(args, 'UVC coordinates already exist. '
                              'Please use rerun_uvc to compute again.')

        base.debugString(self.args, 'GEN UVC: Done in {:.0f}s'.format(time.time()-t0))

    def _write_comb(self):

        base.debugString(self.args,'Writing combined coordinate file')

        nlArr = np.chararray((len(self.Vdata),1)); nlArr[:] = '\n'
        spArr = np.chararray((len(self.Vdata),1)); spArr[:] = ' '

        with open((self.uvc.uvcpnts),'w') as fp:
            arr = np.column_stack((self.Zdata.astype(str),   spArr,
                                   self.RHOdata.astype(str), spArr,
                                   self.PHIdata.astype(str), spArr,
                                   self.Vdata.astype(str),   nlArr))
            Cdata = ''.join(arr.flatten())
            fp.write(Cdata)

        self.job.cp(self.uvc.uvcpnts,self.uvc.uvcpnts.replace('.pts','.dat'))
        self.job.bash(['sed', '-i', '1i{}'.format(len(self.Vdata)), self.uvc.uvcpnts], None)

        return

    def _write_coord(self,lv_sol,filename):

        if 'COORDS_V' in filename:
            Vdata = np.ones(len(self.PHIdata),dtype=np.int)
            Vdata *= lv_sol
            np.savetxt(filename,Vdata)
            self.job.cp(filename,filename.replace('.pts','.dat'))

        else:
            self.job.cp(lv_sol, filename)
            self.job.cp(filename,filename.replace('.pts','.dat'))

        with open(filename.replace('.pts','.dat'),'r') as fp:
            Cdata = fp.read().split('\n')
            Cdata = [float(x) for x in Cdata if x is not '']
            Cdata = np.array(Cdata, dtype=np.float)

        # add correct pts header
        print(filename)
        self.job.bash(['sed', '-i', '1i{}'.format(len(Cdata)), filename], None)

        return Cdata


    def _write_apba(self):

        self.job.cp(self.sols.apba,self.uvc.Zfile)
        self.job.cp(self.uvc.Zfile, self.uvc.Zfile.replace('.pts','.dat'))

        with open(self.uvc.Zfile.replace('.pts','.dat'),'r') as fp:
                Zdata = fp.read().split('\n')
                Zdata = [float(x) for x in Zdata if x is not '']
                Zdata = np.array(Zdata, dtype=np.float)

        # add correct pts header
        self.job.bash(['sed', '-i', '1i{}'.format(len(Zdata)), self.uvc.Zfile], None)

        return Zdata

    def maplvrvdata2biv(self,bivmesh,lvmesh,rvmesh,lvlap,rvlap,ofile=None,scale=[1,1]):

        """
        Maps the lv and rv laplacians back onto the bivmesh. The lv and rv submeshes
        must be extracted from the same biv mesh so a *.nod file is present.

        parameters:
        bivmesh     (input) path to base biv mesh
        lvmesh      (input) path to base lv mesh
        rvmesh      (input) path to base rv mesh
        lvlap       (input) path to lv laplacian phie.dat file
        lvlap       (input) path to rv laplacian phie.dat file
        ofile       (optional) name for outfile for the combined coordinates.
        scale       (optional) [lv, rv] scale the lv and rv laplacian by a scalar
        """

        lvnod = np.fromfile(lvmesh+'.nod', dtype=int)
        rvnod = np.fromfile(rvmesh+'.nod', dtype=int)

        if isinstance(lvlap,int) or isinstance(lvlap,float):
            lvlapdat=np.ones(len(lvnod))*lvlap
        else:
            lvlapdat= pandas.read_csv(lvlap, dtype=np.float, header=None).values.squeeze()

        if isinstance(rvlap,int) or isinstance(rvlap,float):
            rvlapdat=np.ones(len(lvnod))*rvlap
        else:
            rvlapdat= pandas.read_csv(rvlap, dtype=np.float, header=None).values.squeeze()

        nbivpts = -1
        with open(bivmesh+'.pts', 'r') as f:
            nbivpts = int(f.readline())

        lapbiv = None
        if nbivpts > 0:
            lapbiv = np.ones(nbivpts)*-1.0
            for i, nod in enumerate(rvnod):
                lapbiv[nod] = rvlapdat[i]*scale[1]

            for i, nod in enumerate(lvnod):
                lapbiv[nod] = lvlapdat[i]*scale[0]

        if ofile is not None:
            df = pandas.DataFrame(lapbiv)
            df.to_csv(ofile, header=None, sep=' ', index=False)
            df.to_csv(ofile.replace('.pts','.dat'), header=None, sep=' ', index=False)
            self.job.bash(['sed', '-i', '1i{}'.format(len(lapbiv)), ofile], None)


        return lapbiv

        # merge anterior and posterior lv_phi's

class gen_hetero(dspace):

    def __init__(self,job,args,MA_info):
        super(gen_hetero, self).__init__()
        self+=MA_info
        self.args=args
        self.job=job
        self.outdir=os.path.join(job.ID, 'hetero')

        t0 = time.time() # start process timer
        base.debugString(args,'ESTABLISHING HETEROGENEITY',mode=0)

        if args.rerun_hetero or not os.path.exists(self.outdir):
            job.mkdir(self.outdir, parents=True)

            self.init_hetero()
        else:
            base.debugString(args, 'Heterogeneity has already been formulated. '
                              'Please use rerun_hetero flag.')

        base.debugString(args, 'GEN HETEROGENEITY: Done in {:.0f}s'.format(time.time()-t0))

    def init_hetero(self):

        tagregs  = {'apex':  self.args.tags_apba[2],
                    'mid':   self.args.tags_apba[1],
                    'base':  self.args.tags_apba[0],
                    'endo':  self.args.tags_wall[0],
                    'mcell': self.args.tags_wall[1],
                    'epi':   self.args.tags_wall[2]}

        wallopts = {'epi':   self.args.epi_endo_cutoffs[0],
                    'rendo': self.args.epi_endo_cutoffs[1],
                    'lendo': self.args.epi_endo_cutoffs[1],
                    'apex':  self.args.apex_base_cutoffs[0],
                    'base':  self.args.apex_base_cutoffs[1]}

        tags           = self.tags.list
        # only these elements receive heterogeneity information
        tags_whitelist = [tags['T_LV'], tags['T_RV']]

        # apply apico-basal laplace solution to define new set of element tags
        basemesh   = self.msh[self.args.mmode]['mesh']
        n_apbatags = self.set_apba_elemtags(self.args,self.job,basemesh, self.sols.apba,
                                       [wallopts['apex']/100., wallopts['base']/100.],        # wall options:
                                       [tagregs['apex'], tagregs['mid'], tagregs['base']],  # tag regions
                                        tags_whitelist, self.outdir)


        # map transmural solution for lv
        # Note: assuming that every element in the lv submodel gets a new tag
        basemesh = self.msh['lv']['mesh']
        n_lvtags = self.set_tm_elemtags(self.args,self.job,basemesh, self.sols.lv_t,
                                   [wallopts['lendo']/100., 1-wallopts['epi']/100.],       # wall options
                                   [tagregs['endo'], tagregs['mcell'], tagregs['epi']],    # tag regions
                                   [], self.outdir)


        if self.args.mmode =='biv':
            # map transmural solutions for the rv
            # Note: assuming that every element in the rv submodel gets a new tag
            basemesh = self.msh['rv']['mesh']
            n_rvtags = self.set_tm_elemtags(self.args,self.job, basemesh, self.sols.rv_t,
                                       [wallopts['rendo']/100., 1-wallopts['epi']/100.],       # wall options:
                                       [tagregs['endo'], tagregs['mcell'], tagregs['epi']],    # tag regions
                                       [], self.outdir)


        # finally assemble aquired information
        # TODO: to do this efficiently, we need to know about the meshtool eidx/nod content
        o_etags_file    = os.path.join(self.outdir, os.path.basename(self.msh[self.args.mmode]['mesh']) + ".original.tags.dat")
        #o_lv_etags_file = os.path.join(outdir, os.path.basename(self.msh['lv']['mesh']) + ".original.tags.dat")
        #o_rv_etags_file = os.path.join(outdir, os.path.basename(self.msh['rv']['mesh']) + ".original.tags.dat")

        o_biv_etags = pandas.read_csv(o_etags_file,    delimiter=' ', header=None).values.squeeze()
        #o_lv_etags  = pandas.read_csv(o_lv_etags_file, delimiter=' ', header=None).values.squeeze()
        #o_rv_etags  = pandas.read_csv(o_rv_etags_file, delimiter=' ', header=None).values.squeeze()a


        if self.args.mmode =='biv':
            _,lv_inds = simplemath.ismember1d(o_biv_etags, np.asarray(tags['T_LV'], dtype=np.int))
            n_apbatags[lv_inds] += n_lvtags
            _,rv_inds = simplemath.ismember1d(o_biv_etags, np.asarray(tags['T_RV'], dtype=np.int))
            n_apbatags[rv_inds] += n_rvtags
        else:
            _,lv_inds = simplemath.ismember1d(o_biv_etags, np.asarray([0,1,2,3]))
            n_apbatags[lv_inds] += n_lvtags

        n_etags_file = os.path.join(self.outdir, os.path.basename(self.msh[self.args.mmode]['mesh']) + ".tags.dat")
        df = pandas.DataFrame(n_apbatags, dtype=np.int)
        df.to_csv(n_etags_file, header=None, sep=' ', index=False)

        # replace element tags in original file
        basemesh = os.path.join(self.outdir, os.path.basename(self.msh[self.args.mmode]['mesh']))
        self.replaceElemTags(self.job, basemesh, n_etags_file)

        return

    @staticmethod
    def replaceElemTags(job, basemesh, ntagsfile):
        # TODO: check if tagsfile matches length of element file
        tmpFile  = '_tmp.elem'

        with open(basemesh+'.elem') as fid:
            numElems = int(fid.readline().rstrip())

        # make a copy of the 'old' element tags
        otagsfile = basemesh + '.tags.dat'
        if not os.path.exists(otagsfile):
            cmd  = 'tail -n +2 ' + basemesh+'.elem | rev | cut -d " " -f-1 '
            cmd += '| rev > ' + otagsfile
            os.system(cmd)

        # replace 'old' element tags with the new ones
        cmd  = 'tail -n +2 ' + basemesh+'.elem | rev | cut -d " " -f2- '
        cmd += '| rev > ' + tmpFile
        os.system(cmd)

        print(ntagsfile)

        # assemble new element file including new tag information
        cmd  = 'paste -d" " ' + tmpFile + ' ' + ntagsfile + ' > ' + basemesh+'.elem'
        os.system(cmd)

        # insert correct element header information
        job.bash(['sed', '-i', '1i'+str(numElems), basemesh+".elem"], None)

        # cleanup
        os.remove(tmpFile)
        return

    # store element labels to a given tagsfile
    @staticmethod
    def extractElemTags(args, meshname, tagsfile=None):

        if tagsfile is None:
            tagsfile = meshname + '.tags.dat'

        if not os.path.exists(tagsfile):
            cmd  = 'tail -n +2 ' + meshname+'.elem | rev | cut -d " " -f-1 '
            cmd += '| rev > ' + tagsfile
            os.system(cmd)
        else:
            base.debugString(args, '{} already exists. Skipping creation...'.format(tagsfile))

        return tagsfile

    @staticmethod
    def set_apba_elemtags(args, job, meshname, lapfile, wallopts, tagregs, tags_whitelist, outdir):

        # data conversion from string to np integer array
        tags_whitelist = np.asarray(tags_whitelist, dtype=np.int)


        # files to store original and newly created tags in
        o_etags_file = meshname + ".original.tags.dat"
        n_etags_file = os.path.join(outdir, os.path.basename(meshname) + ".tags.dat")


        if not os.path.exists(o_etags_file):
            base.debugString(args, 'Creating {}'.format(o_etags_file))
            gen_hetero.extractElemTags(args,meshname, tagsfile=o_etags_file)
        else:
            base.debugString(args, '{} exists - skipping creation'.format(o_etags_file))

        # copy over original tags to output directory, if necessary
        if not os.path.exists(os.path.join(outdir, os.path.basename(meshname) + ".original.tags.dat")):
            job.cp(o_etags_file, outdir)


        # interpolate phie from vertices to element centers
        odatfile = lapfile
        odatfile = odatfile.replace('.dat', '.ectr.dat')
        cmd      = ['interpolate', 'node2elem',
                    '-omsh=' + meshname,
                    '-idat=' + lapfile,
                    '-odat=' + odatfile]
        job.meshtool(cmd,None)


        # read original element tags
        o_etags = pandas.read_csv(o_etags_file, delimiter=' ', header=None).values.squeeze()


        # read laplace solutions on element centers
        lapdata = pandas.read_csv(odatfile, delimiter=' ', header=None).values.squeeze()


        nelems  = len(o_etags)
        n_etags = np.ones(nelems) * tagregs[1]


        # re-label apical region
        inds          = lapdata < wallopts[0]
        inds          = np.where(inds == True)
        inds          = np.asarray(inds).squeeze()
        n_etags[inds] = tagregs[0]


        # re-label basal region
        inds          = lapdata > wallopts[1]
        inds          = np.where(inds == True)
        inds          = np.asarray(inds).squeeze()
        n_etags[inds] = tagregs[2]


        # element tags specified in 'tags_whitelist' shall receive heterogeneity
        # any other tag must be preserved
        nochange_tags = np.setdiff1d(o_etags, tags_whitelist)
        tf, inds      = simplemath.ismember1d(o_etags, nochange_tags)
        n_etags[inds] = o_etags[inds]


        # ---Export new element tags-----------------------------------------------
        base.debugString(args, 'Writing new tags to {}'.format(n_etags_file))
        df = pandas.DataFrame(n_etags).astype(np.int)
        df.to_csv(n_etags_file, header=None, sep=' ', index=False)


        # TODO: this could be done at some point using meshtool insert data as well

        # Export new version of element file
        n_elemfile = os.path.join(outdir, os.path.basename(meshname) + ".elem")

        base.debugString(args, 'Writing new elements file to {}'.format(n_elemfile))

        # Create temporary file
        tmpfile = os.path.join(outdir, 'tmp.elem')

        # Remove tags from original element file
        cmd = 'tail -n +2 ' + meshname + '.elem | ' # start with second line
        cmd += 'sed "s/\s*$//g" | '                 # remove trailing blanks
        cmd += 'expand | '                          # replace tabs with spaces
        cmd += 'rev | cut -d" " -f2- | rev '        # skip tag column
        cmd += '> ' + tmpfile                       # pipe into new file
        msg = os.system(cmd)

        # Merging (old) element information with new tags
        cmd = 'paste ' + tmpfile + ' ' + n_etags_file + ' > ' + n_elemfile
        msg = os.system(cmd)

        # Finally, adding correct header to element file.
        job.bash(['sed', '-i', '1i' + str(nelems), n_elemfile], None)

        # And remove all tabs
        cmd = "sed -i 's/\t/ /g' " + n_elemfile
        msg = os.system(cmd)

        # Cleanup temporary files
        os.remove(tmpfile)

        return n_etags

    @staticmethod
    def set_tm_elemtags(args, job, meshname, lapfile, wallopts,
                        tagregs, tags_whitelist, outdir):

        # data conversion from string to np integer array
        tags_whitelist = np.asarray(tags_whitelist, dtype=np.int)


        # files to store original and newly created tags in
        o_etags_file = meshname + ".original.tags.dat"
        n_etags_file = os.path.join(outdir, os.path.basename(meshname) + ".tags.dat")

        if not os.path.exists(o_etags_file):
            base.debugString(args, 'Creating {}'.format(o_etags_file))
            gen_hetero.extractElemTags(args, job, meshname, tagsfile=o_etags_file)
        else:
            base.debugString(args, '{} exists - skipping creation'.format(o_etags_file))

        # copy over original tags to output directory, if necessary
        if not os.path.exists(os.path.join(outdir, os.path.basename(meshname) + ".original.tags.dat")):
            job.cp(o_etags_file, outdir)


        # interpolate phie from vertices to element centers
        odatfile = lapfile
        odatfile = odatfile.replace('.dat', '.ectr.dat')
        cmd      = ['interpolate', 'node2elem',
                    '-omsh=' + meshname,
                    '-idat=' + lapfile,
                    '-odat=' + odatfile]
        job.meshtool(cmd,None)


        # read original element tags
        o_etags = pandas.read_csv(o_etags_file, delimiter=' ', header=None).values.squeeze()


        # read laplace solutions on element centers
        lapdata = pandas.read_csv(odatfile, delimiter=' ', header=None).values.squeeze()


        nelems  = len(o_etags)
        n_etags = np.ones(nelems, dtype=np.int) * tagregs[1]


        # re-label epicardial region
        inds          = lapdata > wallopts[1]
        inds          = np.where(inds == True)
        inds          = np.asarray(inds).squeeze()
        n_etags[inds] = tagregs[2]


        # re-label endocardial region
        inds          = lapdata < wallopts[0]
        inds          = np.where(inds == True)
        inds          = np.asarray(inds).squeeze()
        n_etags[inds] = tagregs[0]


        ## element tags specified in 'tags_whitelist' shall receive heterogeneity
        ## any other tag must be preserved
        #nochange_tags          = np.setdiff1d(o_etags, tags_whitelist)
        #n_etags[nochange_tags] = o_etags[nochange_tags]


        # ---Export new element tags------------------------------------------------
        base.debugString(args, 'Writing new tags to {}'.format(n_etags_file))
        df = pandas.DataFrame(n_etags).astype(np.int)
        df.to_csv(n_etags_file, header=None, sep=' ', index=False)

        return n_etags

class gen_fibers(dspace):

    def __init__(self,job,args,MA_info):
        super(gen_fibers, self).__init__()
        self+=MA_info
        self.args=args
        self.job=job

        t0 = time.time() # start process timer
        base.debugString(self.args,'FIBER GENERATION',mode=0)
        fibre_rules=self.sort_fibers(self.args)
        if not os.path.exists(os.path.join(job.ID,'fibers')) or self.args.rerun_fibers:
            job.mkdir(os.path.join(job.ID,'fibers'),parents=True)

            lonfiles = []
            for rule in fibre_rules:
                if args.mode == 'lv':
                    fargs = [self.job, self.msh.lv.mesh, self.sols.lv_t, self.sols.apba, rule, os.path.join(self.job.ID,'fibers')]
                else:
                    fargs = [self.job, self.msh.biv.mesh, self.sols.endo_epi, self.sols.apba, rule, os.path.join(self.job.ID,'fibers')]
                    fargs += ['biv', self.sols.lv, self.sols.rv]


                tpl = 'fibres_{}_{}_{}_{}.lon'
                lonfile = os.path.join(os.path.join(job.ID,'fibers'), tpl.format(*rule))

                if not os.path.isfile(lonfile):
                    lonfiles.append(self.generate_fibres(*fargs))

                self.gen_vtk_fiber_files(lonfiles)

        else:
            base.debugString(args, 'Fibers have already been created. '
                                       'Please use rerun_fibers flag to run again.')

        base.debugString(args, 'GEN FIBERS: Done in {:.0f}s'.format(time.time()-t0))

    def gen_vtk_fiber_files(self,lonfiles):

        vtkfile = os.path.join(self.job.ID, 'fibers','fibers_biv')

        cmd = [settings.execs.GLVTKCONVERT,
               '-m', self.msh.biv.mesh,
               '-n', self.sols.endo_epi,
               '-n', self.sols.apba]

        if self.args.mode == 'biv':
            cmd += ['-n', self.sols.lv,
                    '-n', self.sols.rv]

        for fname in lonfiles:
            cmd += ['-l', fname]

        cmd += ['-o', vtkfile]

        self.job.bash(cmd, 'Generating VTK File with Fibres')

        return

    @staticmethod
    def generate_fibres(job, meshname, laplace_epi, laplace_apba, fibre_rule,
                        outdir, mode='lv', laplace_lv=None, laplace_rv=None,
                        nonmyo=[]):

        assert mode in ['lv', 'biv']
        if mode == 'biv':
            assert laplace_lv is not None and laplace_rv is not None

        tpl = 'fibres_{}_{}_{}_{}.lon'
        lonfile = os.path.join(outdir, tpl.format(*fibre_rule))

        cmd = [settings.execs.GLRULEFIBERS,
               '-m', meshname,
               '--type', mode,
               '-a', laplace_apba,
               '-e', laplace_epi]

        if mode == 'biv':
            cmd += ['-l', laplace_lv,
                    '-r', laplace_rv]

        for tag in nonmyo:
            cmd += ['-n', str(tag)]

        cmd += ['--alpha_endo', fibre_rule[0],
                '--alpha_epi',  fibre_rule[1],
                '--beta_endo',  fibre_rule[2],
                '--beta_epi',   fibre_rule[3],
                '-o', lonfile]

        job.bash(cmd, 'Generate Rule-Based Fibres')

        return lonfile

    @staticmethod
    def sort_fibers(args):

        fibre_rules = []
        for rule in args.fibre_rule:
            fibre_rules.append(rule)
        for rule in args.fibre_rule_alpha:
            fibre_rules.append(rule + [-65, 25])
        if len(fibre_rules) == 0:
            fibre_rules = [[40, -50, -65, 25]]

        return fibre_rules

class gen_meshes(dspace):

    def __init__(self,job,args,MA_info):
        super(gen_meshes, self).__init__()
        self+=MA_info
        self.job=job
        self.args=args

        """
        Generate meshes, submeshes, and surfaces as determined by simulation architecture.
        """

        t0       = time.time() # start process timer
        base.debugString(args,'MESH GENERATION', mode=0)

        #Copy over the basemesh and generate all associated surfaces
        if args.basename:
            self.gen_model()
        else:
            self.gen_ellip_wBP()

        base.debugString(args,'GEN MESHES: Done in {:.0f}s={:.2f}s'.format(time.time()-t0,(time.time()-t0)/60))

    def gen_model(self):

        curr_dir = os.path.join(self.args.ID,'model','base')
        # check_dir = os.path.join(self.args.ID,'model','base','check')
        base.debugString(self.args,'Setting up base mesh in working directory: '+curr_dir)
        bname    = os.path.basename(self.args.basename)
        basemesh = os.path.join(curr_dir,bname)

        #Copy over the basemesh:
        if self.args.rerun_meshes or not os.path.exists(curr_dir):
            self.job.mkdir(curr_dir, parents=True)
            #Copy over the base file
            try:
                [self.job.cp(self.args.basename+x,curr_dir) for x in ['.elem','.pts','.lon']]

                gen_hetero.extractElemTags(self.args,basemesh, tagsfile=basemesh+'.original.tags.dat')
            except:
                raise IOError('Could not copy over basename files')
        else:
            base.debugString(self.args, 'Base mesh already copied over and generated. To overwrite, '
                              'please use rerun_meshes flag. Checking surfaces...',mode=2)

        # self.job.mkdir(check_dir,parents=True)

        #Generate the relevant surfaces:
        for surf,_ in self.msh.surflist.iterattr():
            if base.guess_ext_and_check(basemesh, surf) is None:
                if (surf == 'base' and self.args.mode != 'h4c') or base.guess_ext_and_check(self.args.basename, surf) is not None:
                    filebase='.'.join((self.args.basename,surf))
                    [self.job.cp(filebase+x,curr_dir) for x in ['.surf','.surf.vtx','.vtx'] if os.path.exists(filebase+x)]
                elif self.args.rerun_meshes or base.guess_ext_and_check(self.args.basename, surf) is None:
                    srf2mk = os.path.join(curr_dir, '.'.join((bname, surf)))
                    mt.extract_surface(self.job,self.args.basename, srf2mk,op=self.tags.ext['.'.join((self.args.mmode,surf))])

            else:
                base.debugString(self.args, surf + ' already copied over and generated. To overwrite, '
                                           'please use rerun_meshes flag. Checking surfaces...',mode=2)

        #Subtract relevant surfaces if needed
        for key, values in self.tags.sub.iterattr():
            srf2mk   = os.path.join(curr_dir,'.'.join((bname,key)))
            srf2rm   = os.path.join(curr_dir,'.'.join((bname,values.split('-')[1])))
            srf2rmfm = os.path.join(curr_dir,'.'.join((bname,values.split('-')[0])))
            self.sub(self.job,self.args,srf2mk,srf2rm,srf2rmfm)

        #Split the RVJ
        if self.args.mmode == 'biv':
            test = [base.guess_ext_and_check(basemesh,x) for x in ['rvja','rvjp','rvj.apex']]
            if None in test or self.args.rerun_meshes: self.splitrvj(basemesh)
            else: base.debugString(self.args,'RVJ already split...',mode=2)

        #Generate submeshes and map over all surfaces
        for attr,keys in self.msh.iterattr():
            if attr == 'surflist':
                continue
            curr_dir=os.path.join(self.args.ID,'model',attr)
            # check_dir = os.path.join(self.args.ID,'model',attr,'check')

            msh2mk = os.path.join(curr_dir,'.'.join((bname,attr)))
            base.debugString(self.args,'Constructing {} in working directory: {}'.format(attr, curr_dir))

            #Generate the mesh
            if not os.path.exists(curr_dir):
                mt.extract_mesh(self.job,basemesh,msh2mk,self.tags.ext[attr],silent=self.args.silent)
                mt.extract_surface(self.job,msh2mk,msh2mk,self.tags.ext[attr],silent=self.args.silent)
                gen_hetero.extractElemTags(self.args, msh2mk, tagsfile=msh2mk+'.original.tags.dat')

                # query_unreachable(self.job,self.args,msh2mk,odir=check_dir)

                if self.args.vtk:
                    mt.convert(self.job, msh2mk,'carp_txt',msh2mk,'carp_bin', silent=self.args.silent)
            else:
                base.debugString(self.args, 'Mesh already exists. To overwrite, '
                                  'please use rerun_meshes flag. Checking surfaces...',mode=2)

            for surf,_ in keys.iterattr():
                if self.args.rerun_meshes or base.guess_ext_and_check(msh2mk,surf) is None:
                    curr_surf='.'.join((bname,attr,surf))
                    mt.map(self.job,msh2mk,'.'.join((basemesh,surf))+'.*',curr_dir,mapName=curr_surf+'.')

                self.msh[attr][surf] = base.guess_ext_and_check(msh2mk,surf) #set the environment

            self.msh[attr]['mesh'] = msh2mk #set the environment -> must be after surface setting

            #Set so that the lv and rv meshes are based from the biv mesh
            if attr == 'biv':
                bname    = os.path.basename(self.msh.biv.mesh)
                basemesh = self.msh.biv.mesh

        test = [base.guess_ext_and_check(self.msh.lv.mesh,x) for x in ['lvpmid','lvamid','lvseptmid','lvwallmid','lvawall','lvpwall']]
        if None in test or self.args.rerun_meshes:
            self.msh.lv.extend(self.splitlv())
        else:
            [self.msh.lv.add(i,test[x]) for x, i in enumerate(['lvpmid','lvamid','lvseptmid','lvwallmid','lvawall','lvpwall'])]

        if base.guess_ext_and_check(self.msh.lv.mesh,'uvcapex') is None: self.findLVapex()

        self.msh.lv.add('uvcapex',self.msh.lv.mesh+'.uvcapex.vtx')
        self.msh.biv.add('uvcapex',self.msh.biv.mesh+'.uvcapex.vtx')

        #Convert to binary for faster laplacian loading -> must be after lv and rvj split code
        for attr,keys in self.msh.iterattr():
            if attr == 'surflist':
                continue
            if not os.path.isfile(self.msh[attr]['mesh']+'.belem'):
                mt.convert(self.job, self.msh[attr]['mesh'],'carp_txt',self.msh[attr]['mesh'],'carp_bin', silent=self.args.silent)

        return self

    def gen_ellip_wBP(self):

        curr_dir = os.path.join(self.args.ID,'model','base')
        meshname = os.path.join(curr_dir, 'LV_ellipsoid_res{}_rad{}'.format(str(self.args.ellipse_res),str(self.args.ellipse_rad)).replace('.','p'))
        self.msh.lv.add('mesh',meshname)

        print('No basename provided. Generating an ellipsoidal mesh....')
        base.debugString(self.args,'Setting up ellipsoidal mesh in working directory: '+curr_dir)

        if self.args.rerun_meshes or not os.path.exists(curr_dir):
            self.job.mkdir(curr_dir,parents=True)

            tagLV=self.tags.list['T_LV']

            geom = mesh.Ellipsoid.with_resolution(self.args.ellipse_rad,self.args.ellipse_res,tetrahedrise=True)
            geom.generate_carp(meshname, ['top', 'inside', 'outside'])
            geom.generate_carp_apex_vtx(meshname)

            [self.job.mv(meshname+'_top.'+x,meshname+'.base.'+x) for x in ['surf','vtx','neubc']]
            [self.job.mv(meshname+'_inside.'+x,meshname+'.lvendo.'+x) for x in ['surf','vtx','neubc']]
            [self.job.mv(meshname+'_outside.'+x,meshname+'.epi.'+x) for x in ['surf','vtx','neubc']]
            [self.job.mv(x,x.replace('.vtx','.surf.vtx')) for x in glob(meshname+'*.vtx')]
            self.job.mv(meshname+'_apex.surf.vtx',meshname+'.apex.vtx')

            mt.extract_surface(self.job,meshname,meshname)

            with open(meshname+'.elem') as fid:
                numElems = int(fid.readline().rstrip())

            tmpfile=os.path.dirname(meshname)+'/tmp.elem'

            # Remove tags from original element file
            cmd = 'tail -n +2 ' + meshname + '.elem | ' # start with second line
            cmd += 'sed "s/\s*$//g" | '                 # remove trailing blanks
            cmd += 'expand | '                          # replace tabs with spaces
            cmd += 'rev | cut -d" " -f2- | rev '        # skip tag column
            cmd += '> ' + tmpfile                    # pipe into new file
            msg = os.system(cmd)

            cmd = 'sed -i "s/$/ '+tagLV+'/" '+tmpfile
            msg=os.system(cmd)

            with open(meshname+'.elem') as fid:
                numElems = int(fid.readline().rstrip())
            self.job.bash(['sed', '-i', '1i'+ str(numElems), tmpfile], None)

            self.job.mv(tmpfile,meshname+'.elem')

            with open(meshname+'.rvsept_point.vtx','w') as fid:
                fid.write('1'+'\n'+'extra'+'\n'+'30')

            if self.args.uvc:
                self.msh.lv.extend(self.splitlv(outdir=os.path.join(self.job.ID,'model','base')))

        else:
            base.debugString(self.args, 'Assuming mesh already exists. To overwrite,'
                                       'please use rerun_meshes flag.', mode=2)

        test = [base.guess_ext_and_check(self.msh.lv.mesh,x) for x in ['lvpmid','lvamid','lvseptmid','lvwallmid']]
        if len(test):
            [self.msh.lv.add(i,test[x]) for x, i in enumerate(['lvpmid','lvamid','lvseptmid','lvwallmid'])]

        for attr,_ in self.msh.lv.iterattr():
            if attr == 'mesh': continue
            self.msh.lv[attr]=base.guess_ext_and_check(meshname,attr)

        if base.guess_ext_and_check(self.msh.lv.mesh,'uvcapex') is None: self.findLVapex()
        self.msh.lv.add('uvcapex',self.msh.lv.mesh+'.uvcapex.vtx')

        return self

    def findLVapex(self):

        base.debugString(self.args,'Finding the Apex of the LV')

        # read points file
        df_pts = pandas.read_csv(self.msh.lv.mesh + '.pts', skiprows=1, delimiter=' ', header=None)
        pts    = df_pts.values.squeeze()

        # query for average edge lengths
        _,meanEL,_ = gen_meshes.queryEdges(self.msh.lv.mesh)


        # Default constants do not work, people use meshes where the
        # underlying unit is millimeters :-(
        ZERO_PHI_RADIUS_APEX = meanEL

         # read some more boundaries
        lv_septmid_vtx = pandas.read_csv(self.msh.lv.lvseptmid, delimiter=' ', skiprows=2, header=None).values.squeeze() # septal nodes of cavity
        lv_wallmid_vtx = pandas.read_csv(self.msh.lv.lvwallmid, delimiter=' ', skiprows=2, header=None).values.squeeze() # para-septal nodes of cavity
        lv_endo_vtx    = pandas.read_csv(self.msh.lv.lvendo,    delimiter=' ', skiprows=2, header=None).values.squeeze() # endocardial nodes of cavity
        lv_base_vtx    = pandas.read_csv(self.msh.lv.base,      delimiter=' ', skiprows=2, header=None).values.squeeze() # endocardial nodes of cavity

        # determine singularity at the apex
        singular_vtx_lv   = np.intersect1d(lv_septmid_vtx, lv_wallmid_vtx)

        # Extend transmural apical line of nodes to a cylindric region with
        # ZERO_PHI_APEX_RADIUS microns. Set rotational coordinates to be zero.
        tmp_vtx = np.copy(singular_vtx_lv)
        for vtx in tmp_vtx:
            dists        = simplemath.dists(pts[vtx,:], pts)
            inds         = np.where(dists < ZERO_PHI_RADIUS_APEX)
            singular_vtx_lv = np.append(singular_vtx_lv, inds[0].flatten())
            singular_vtx_lv = np.unique(singular_vtx_lv)

        print('Number of apex points found in LV: '+str(len(singular_vtx_lv)))

        del tmp_vtx, inds, dists

        df = pandas.DataFrame(singular_vtx_lv)
        df.to_csv(self.msh.lv.mesh+'.uvcapex.vtx', header=None, sep=' ', index=False)
        self.job.bash(['sed', '-i', '1iextra',                 self.msh.lv.mesh+'.uvcapex.vtx'], None)
        self.job.bash(['sed', '-i', '1i'+ str(len(singular_vtx_lv)), self.msh.lv.mesh+'.uvcapex.vtx'], None)

        if self.args.mmode != 'lv':

            cmd='correspondance -msh1='+self.msh.biv.mesh+' -msh2='+self.msh.lv.mesh+' -ifmt=carp_txt'
            os.system(cmd)
            print('Done....')

            corr_file=os.path.join(os.path.dirname(self.msh.lv.mesh),'biv_lv_corr.txt')
            self.job.mv('corr.txt',corr_file)

            singular_vtx_biv=[]
            f=open(corr_file,'r')

            for idx,line in enumerate(f.readlines()):

                if idx==0: pass
                else:
                    if int(line.split(':')[1]) in singular_vtx_lv:
                        singular_vtx_biv.append(int(line.split(':')[0]))
            f.close()

            print('Number of apex points found in BiV: '+str(len(singular_vtx_biv)))

            df = pandas.DataFrame(singular_vtx_biv)
            df.to_csv(self.msh.biv.mesh+'.uvcapex.vtx', header=None, sep=' ', index=False)
            self.job.bash(['sed', '-i', '1iextra',                 self.msh.biv.mesh+'.uvcapex.vtx'], None)
            self.job.bash(['sed', '-i', '1i'+ str(len(singular_vtx_biv)), self.msh.biv.mesh+'.uvcapex.vtx'], None)


    def splitrvj(self,basemesh):

        """
        Split the rvj surface file into the posterior and anterior portions
        based on PCA analysis on the lvendo.

        parameters:
        basemesh     (input) file name with path you want to save the output surface to
        verbose      (optional) show the PCA visualization. Default is false.

        Note:
            The basemesh must have associated lvendo, base, and rvj surfaces (*.surf.vtx)

        """
        base.debugString(self.args,'Splitting RV-LV Junction')

        rvjfile    = base.guess_ext_and_check(basemesh,'rvj')
        lvendofile = base.guess_ext_and_check(basemesh,'lvendo')
        basefile   = base.guess_ext_and_check(basemesh,'base')

        eigvals = np.empty([3,3])
        eigvecs = np.empty([3,1])

        # import mesh vertices
        df_pts   = pandas.read_csv(basemesh+'.pts', skiprows=1, delimiter=' ', header=None)
        pts      = df_pts.values

        # load lvendo vertices
        df_surf  = pandas.read_csv(lvendofile, skiprows=2, delimiter=' ', header=None)
        surf_vtx = df_surf.values.astype(np.int).flatten()
        # load base vertices
        df_base  = pandas.read_csv(basefile, skiprows=2, delimiter=' ', header=None)
        base_vtx = df_base.values.astype(np.int)
        # load rvj vertices
        df_rvj  = pandas.read_csv(rvjfile, skiprows=2, delimiter=' ', header=None)
        rvj_vtx = df_rvj.values.astype(np.int)
        # load rvj surfaces
        if os.path.isfile(rvjfile.replace('.vtx','')):
            df_rvj_surf = pandas.read_csv(rvjfile.replace('.vtx',''), skiprows=1, delimiter=' ', header=None)
            rvj_surf    = df_rvj_surf.values[:,1:].astype(np.int).squeeze()
        else:
            df_rvj_surf = None
        # load rvj neubc files
        if os.path.isfile(rvjfile.replace('.surf.vtx','.neubc')):
            df_rvj_neubc = pandas.read_csv(rvjfile.replace('.surf.vtx','.neubc'), skiprows=1, delimiter=' ', header=None)
            rvj_neubc    = df_rvj_neubc.values[:,1:].astype(np.int).squeeze()
        else:
            df_rvj_neubc = None

        ptcloud = pts[surf_vtx,:]
        eigvals, eigvecs = simplemath.PCA(ptcloud, 3) # eigenvecs delivered in columns!

        # base center of left cavity
        cm_cavity = np.mean(pts[surf_vtx,:], axis=0)

        # determine if the first pca-vector is pointing in apico-basal direction
        cav_basering_vtx_val,cav_basering_vtx_ind = simplemath.ismember1d(base_vtx, surf_vtx)
        cm_cavbasering   = np.mean(pts[cav_basering_vtx_val,:].squeeze(), axis=0)

        # ---change direction of first eigenvector if necessary--------------------
        if np.dot(cm_cavbasering-cm_cavity, eigvecs[:,0]) < 0.:
            eigvecs[:,0] *= -1


        # ---test plot area--------------------------------------------------------
        verbose=False
        if verbose:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax  = fig.gca(projection='3d')

            ax.plot(ptcloud[::10,0], ptcloud[::10,1], ptcloud[::10,2], 'ro', alpha=0.3, markersize=2)
            ax.set_aspect('equal')
            #ax.plot(pts[lvbasering_vtx,0], pts[lvbasering_vtx,1], pts[lvbasering_vtx,2], 'bo', alpha=0.3, markersize=2)
            ax.set_aspect('equal')

            i=0
            colors = ['red', 'green', 'blue']
            for evec in eigvecs.T:
                ax.plot([cm_cavbasering[0], cm_cavbasering[0]+evec[0]*10],
                        [cm_cavbasering[1], cm_cavbasering[1]+evec[1]*10],
                        [cm_cavbasering[2], cm_cavbasering[2]+evec[2]*10], color=colors[i], lw=3)
                i+=1

            ax.set_aspect('equal')
            plt.show()
        # -------------------------------------------------------------------------

        # ---determine septal direction (seen from left cavity)
        # lvsept_vec x apba_vec -> ant_post_vec
        cm_rvj      = np.mean(pts[rvj_vtx,:].squeeze(), axis=0)
        lvsept_vec  = cm_rvj - cm_cavbasering
        antpost_vec = np.cross(lvsept_vec, eigvecs[:,0])

        # determine lowest apical node on epicardium
        plane_normal = eigvecs[:,0]
        plane_point  = cm_cavbasering
        max_dist     = 0.

        for vtx in rvj_vtx:
            rvj_pt = pts[vtx,:]
            dist = np.dot(plane_normal, np.subtract(rvj_pt,plane_point).T)

            if abs(dist) > max_dist:
                max_dist     = abs(dist)
                rvj_apex_vtx = vtx


        # separate between anterior and posterior RVJ using a same side of plane algorithm
        # storing the three plane nodes in the columns!
        rvj_posterior_vtx = []
        rvj_anterior_vtx  = []
        tri      = np.zeros([3,3], np.float)
        tri[:,0] = pts[rvj_apex_vtx]
        tri[:,1] = pts[rvj_apex_vtx] + lvsept_vec
        tri[:,2] = pts[rvj_apex_vtx] + eigvecs[:,0]
        pRef     = pts[rvj_apex_vtx] + antpost_vec

        for vtx in rvj_vtx:
            pTest = pts[vtx,:].reshape([3,1])
            tf    = simplemath.same_side_of_plane(tri, pRef, pTest)
            if tf:
                rvj_posterior_vtx.append(vtx)
            else:
                rvj_anterior_vtx.append(vtx)
        del pTest, pRef, tri

        rvj_posterior_vtx = np.asarray(rvj_posterior_vtx, dtype=np.int)
        rvj_anterior_vtx  = np.asarray(rvj_anterior_vtx,  dtype=np.int)


        # ---determine associated anterior surf/neubc files------------------------
        if rvj_surf is not None:

            mbIDs, mbCounts = simplemath.ismember2d(rvj_surf, rvj_anterior_vtx)
            del mbIDs

            # remove non-anterior surface patches
            rmIDs = np.where(mbCounts < 2)
            df_tmp = df_rvj_surf.drop(df_rvj_surf.index[rmIDs], inplace=False)
            del mbCounts

            # export new surface file, re-insert proper header information
            df_tmp.to_csv(basemesh+'.rvja.surf', header=None, sep=' ', index=False)
            self.job.bash(['sed', '-i', '1i' + str(df_tmp.shape[0]), basemesh+'.rvja.surf'], None)
            del df_tmp


        # will not do a neubc reduction without a related surface file !!!
        if df_rvj_neubc is not None and rvj_surf is not None:

            # remove non-anterior neubc patches
            df_tmp = df_rvj_neubc.drop(df_rvj_neubc.index[rmIDs], inplace=False)

            # export new neubc file, re-insert proper header information
            df_tmp.to_csv(basemesh+'.rvja.neubc', header=None, sep=' ', index=False)
            self.job.bash(['sed', '-i', '1i' + str(df_tmp.shape[0]), basemesh+'.rvja.neubc'], None)
            del df_tmp
        # -------------------------------------------------------------------------


        # ---determine associated anterior surf/neubc files------------------------
        if rvj_surf is not None:
            mbIDs, mbCounts = simplemath.ismember2d(rvj_surf, rvj_posterior_vtx)
            del mbIDs

            # remove non-posterior surface patches
            rmIDs = np.where(mbCounts < 2)
            df_tmp = df_rvj_surf.drop(df_rvj_surf.index[rmIDs], inplace=False)
            del mbCounts

            # export new surface file, re-insert proper header information
            df_tmp.to_csv(basemesh + '.rvjp.surf', header=None, sep=' ', index=False)
            self.job.bash(['sed', '-i', '1i' + str(df_tmp.shape[0]), basemesh + '.rvjp.surf'], None)
            del df_tmp

            # will not do a neubc reduction without a related surface file !!!
        if df_rvj_neubc is not None and rvj_surf is not None:
            # remove non-posterior neubc patches
            df_tmp = df_rvj_neubc.drop(df_rvj_neubc.index[rmIDs], inplace=False)

            # export new neubc file, re-insert proper header information
            df_tmp.to_csv(basemesh + '.rvjp.neubc', header=None, sep=' ', index=False)
            self.job.bash(['sed', '-i', '1i' + str(df_tmp.shape[0]), basemesh + '.rvjp.neubc'], None)
            del df_tmp
            # -------------------------------------------------------------------------


        # export to dedicated anterior and posterior surfaces/vertex/neubc files
        # TODO: not yet complete

        # export anterior rvj vertices
        np.savetxt(basemesh + '.rvja.surf.vtx', rvj_anterior_vtx, fmt='%u', delimiter='\n')
        self.job.bash(['sed', '-i', '1iextra',                        basemesh+".rvja.surf.vtx"], None)
        self.job.bash(['sed', '-i', '1i'+ str(len(rvj_anterior_vtx)), basemesh+".rvja.surf.vtx"], None)

        # export posterior rvj vertices
        np.savetxt(basemesh + '.rvjp.surf.vtx', rvj_posterior_vtx, fmt='%u', delimiter='\n')
        self.job.bash(['sed', '-i', '1iextra',                         basemesh+".rvjp.surf.vtx"], None)
        self.job.bash(['sed', '-i', '1i'+ str(len(rvj_posterior_vtx)), basemesh+".rvjp.surf.vtx"], None)

        # export rvj epicardial apex
        np.savetxt(basemesh + '.rvj.apex.vtx', rvj_apex_vtx, fmt='%u', delimiter='\n')
        self.job.bash(['sed', '-i', '1iextra', basemesh+".rvj.apex.vtx"], None)
        self.job.bash(['sed', '-i', '1i1',     basemesh+".rvj.apex.vtx"], None)


        # ---meshalyzer test data file to check correctness of rvja and rvjb vertices---------
        testdata = np.ones([len(pts),1],dtype=np.int)*-1
        testdata[rvj_anterior_vtx ] = 0
        testdata[rvj_posterior_vtx] = 1
        np.savetxt(basemesh + '.rvj.dat', testdata, fmt='%u', delimiter='\n')
        # -------------------------------------------------------------------------

        return basemesh+'.rvja.surf.vtx', basemesh+'.rvjp.surf.vtx'

    def splitlv(self,outdir=None,verbose=False):

        """
        Extract the septal, posterior, anterior, and mid wall slices within the lvendo based
        on PCA analysis of the lvendo.

        parameters:
        basemesh     (input) file name with path you want to save the output surface to
        verbose      (optional) show the PCA visualization. Default is false.

        Note:
            The basemesh must have associated lvendo, base, apex, and rvj surfaces (*.surf.vtx)
        """

        base.debugString(self.args,'SPLITTING LV')

        basemesh   = self.msh[self.args.mmode].mesh
        bivbase    = os.path.basename(basemesh)

        if outdir is None:
            outdir     = os.path.join(self.job.ID,'model','lv')

        if self.args.mmode != 'lv':
            meshfile   = os.path.join(outdir,bivbase+'.lv')

        else:
            meshfile   = os.path.join(outdir,bivbase)


        # # map needed files
        # files  = basemesh + '.lvendo.surf,'
        # files += basemesh + '.lvendo.surf.vtx,'
        # files += basemesh + '.apex.vtx,'
        # files += basemesh + '.base.surf,'
        # files += basemesh + '.base.surf.vtx,'
        # if self.args.mmode == 'biv':
        #     files += basemesh + '.rvj.surf,'
        #     files += basemesh + '.rvj.surf.vtx'


        # ---SETUP APICO-BASAL, ANTERO-POSTERO AND LVRV VECTORS--------------------
        # import mesh vertices
        df_pts   = pandas.read_csv(meshfile+'.pts', skiprows=1, delimiter=' ', header=None)
        pts      = df_pts.values
        # load lvendo vertices
        df_surf  = pandas.read_csv(meshfile+'.lvendo.surf.vtx', skiprows=2, delimiter=' ', header=None)
        surf_vtx = df_surf.values.astype(np.int).flatten()
        # load apical vertex (assume that it is sitting on the lv)
        df_apex     = pandas.read_csv(meshfile+'.rvj.apex.vtx', skiprows=2, delimiter=' ', header=None)
        lv_apex_vtx = df_apex.values.astype(np.int)
        # load base vertices
        df_base  = pandas.read_csv(meshfile+'.base.surf.vtx', skiprows=2, delimiter=' ', header=None)
        base_vtx = df_base.values.astype(np.int)


        # base center of left cavity
        cav_basering_vtx,_ = simplemath.ismember1d(base_vtx, surf_vtx)
        cm_cavbasering     = np.mean(pts[cav_basering_vtx,:].squeeze(), axis=0)

        # determine the apico-basal direction
        apba_vec = np.subtract(cm_cavbasering, pts[lv_apex_vtx[0], :].squeeze())
        apba_vec = simplemath.normalize(apba_vec)

        # center of left cavity
        cm_cavity = np.mean(pts[surf_vtx,:], axis=0)

        # ---change direction of first eigenvector if necessary--------------------
        if np.dot(np.subtract(cm_cavbasering,cm_cavity), apba_vec) < 0.:
            #eigvecs[:,0] = eigvecs[:,0] * (-1)
            apba_vec     = apba_vec     * (-1)
        # -------------------------------------------------------------------------


        # determine lowest apical node on left endocardium
        plane_normal = apba_vec
        plane_point  = cm_cavbasering
        max_dist     = 0.

        for vtx in surf_vtx:
            lv_pt = pts[vtx,:]
            dist  = np.dot(plane_normal, np.subtract(lv_pt,plane_point))

            if abs(dist) > max_dist:
                max_dist    = abs(dist)
                lv_apex_vtx = vtx

        # reset apico-basal direction
        apba_vec = np.subtract(cm_cavbasering, pts[lv_apex_vtx,:])
        apba_vec = simplemath.normalize(apba_vec)

        if self.args.mmode =='biv':
            #load rvj vertices
            df_rvj   = pandas.read_csv(meshfile+'.rvj.surf.vtx', skiprows=2, delimiter=' ', header=None)
            rvj_vtx  = df_rvj.values.astype(np.int)

            # determine the lowest node on the rv-junction
            plane_normal = apba_vec
            plane_point  = cm_cavbasering
            max_dist     = 0.

            for vtx in rvj_vtx:
                rvj_pt = pts[vtx,:]
                dist = np.dot(plane_normal, np.subtract(rvj_pt,plane_point).T)

                if abs(dist) > max_dist:
                    max_dist     = abs(dist)
                    rvj_apex_vtx = vtx
        else:
            with open(meshfile+'.rvsept_point.vtx','r') as fid:
                rvj_apex_vtx = int(fid.readlines()[2])


        # ---determine septal direction (seen from left cavity)
        # lvrv_vec x apba_vec -> ant_post_vec
        lvrv_vec    = np.subtract(pts[rvj_apex_vtx,:], pts[lv_apex_vtx,:])
        antpost_vec = np.cross(lvrv_vec, apba_vec).squeeze()

        # orthogonalize vectors
        lvrv_vec    = np.cross(apba_vec, antpost_vec).squeeze()

        # ---test plot area--------------------------------------------------------
        if verbose:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax  = fig.gca(projection='3d')

            ax.plot(pts[surf_vtx[::10],0],
                    pts[surf_vtx[::10],1],
                    pts[surf_vtx[::10],2], 'o', color='yellow', alpha=0.3, markersize=2)

            evec = apba_vec
            ax.plot([cm_cavbasering[0], cm_cavbasering[0]+evec[0]*10000],
                    [cm_cavbasering[1], cm_cavbasering[1]+evec[1]*10000],
                    [cm_cavbasering[2], cm_cavbasering[2]+evec[2]*10000], color='red', lw=3)

            ax.plot([cm_cavbasering[0], cm_cavbasering[0]+antpost_vec[0]],
                    [cm_cavbasering[1], cm_cavbasering[1]+antpost_vec[1]],
                    [cm_cavbasering[2], cm_cavbasering[2]+antpost_vec[2]], color='green', lw=3)

            ax.plot([cm_cavbasering[0], cm_cavbasering[0]+lvrv_vec[0]],
                    [cm_cavbasering[1], cm_cavbasering[1]+lvrv_vec[1]],
                    [cm_cavbasering[2], cm_cavbasering[2]+lvrv_vec[2]], color='blue', lw=3)
            ax.set_aspect('equal')
            plt.show()
        # -------------------------------------------------------------------------

        # compute element centers of lv mesh
        ectrfile = meshfile+'.ectrs.pts'
        if not os.path.isfile(ectrfile):
            cmd = [settings.execs.GLELEMCENTERS, '-m', meshfile, '-o', ectrfile]

            self.job.bash(cmd, None)

        # call external same-side-of-plane algorithm
        # nodes file to be processed
        datfile  = meshfile + '.quarters.dat'

        # first run: antero/postero separation
        # 3 nodes defining the plane
        xyzA = cm_cavbasering
        xyzB = cm_cavbasering+apba_vec
        xyzC = cm_cavbasering+lvrv_vec
        pRef = cm_cavbasering-antpost_vec

        # remove any existing data file
        if os.path.exists(datfile):
            os.remove(datfile)

        cmd  = 'samesideofplane ' + ectrfile + ' ' + datfile + ' 2 '
        cmd += np.array2string(xyzA).replace('[','').replace(']',' ')
        cmd += np.array2string(xyzB).replace('[','').replace(']',' ')
        cmd += np.array2string(xyzC).replace('[','').replace(']',' ')
        cmd += np.array2string(pRef).replace('[','').replace(']',' ')

        os.system(cmd)

        if verbose:
            # ---export auxilliar grid file for inspection with meshalyzer---------
            fid = open(meshfile+'saggital.pts_t','w')
            fid.write("1\n")
            fid.write("4\n")
            fid.write("{} {} {}\n".format(xyzA[0], xyzA[1], xyzA[2]))
            fid.write("{} {} {}\n".format(xyzB[0], xyzB[1], xyzB[2]))
            fid.write("{} {} {}\n".format(xyzC[0], xyzC[1], xyzC[2]))
            fid.write("{} {} {}  ".format(pRef[0], pRef[1], pRef[2]))
            fid.close()
            # ---------------------------------------------------------------------


        # ---test plot area--------------------------------------------------------
        if verbose:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax  = fig.gca(projection='3d')

            ax.plot(pts[surf_vtx[::10],0],
                    pts[surf_vtx[::10],1],
                    pts[surf_vtx[::10],2], 'o', color='yellow', alpha=0.3, markersize=2)

            ax.scatter(pts[surf_vtx[::10],0],
                       pts[surf_vtx[::10],1],
                       pts[surf_vtx[::10],2], c='yellow', alpha=0.3, s=2)
            ax.scatter(xyzA[0], xyzA[1], xyzA[2], c='red',   s=50)
            ax.scatter(xyzB[0], xyzB[1], xyzB[2], c='green', s=50)
            ax.scatter(xyzC[0], xyzC[1], xyzC[2], c='blue',  s=50)
            ax.scatter(pRef[0], pRef[1], pRef[2], c='cyan',  s=50)

            ax.set_aspect('equal')
            plt.show()
        # -------------------------------------------------------------------------


        # second run: septal/paraseptal separation
        # 3 nodes defining the plane
        xyzA = cm_cavbasering
        xyzB = cm_cavbasering+apba_vec
        xyzC = cm_cavbasering+antpost_vec
        pRef = cm_cavbasering+lvrv_vec

        cmd = 'samesideofplane ' + ectrfile + ' ' + datfile + ' 1 '
        cmd += np.array2string(xyzA).replace('[','').replace(']',' ')
        cmd += np.array2string(xyzB).replace('[','').replace(']',' ')
        cmd += np.array2string(xyzC).replace('[','').replace(']',' ')
        cmd += np.array2string(pRef).replace('[','').replace(']',' ')

        os.system(cmd)

        if verbose:
            # ---export auxilliar grid file for inspection with meshalyzer---------
            fid = open(meshfile+'frontal.pts_t','w')
            fid.write("4\n")
            fid.write("3\n")
            fid.write("{} {} {}\n".format(xyzA[0], xyzA[1], xyzA[2]))
            fid.write("{} {} {}\n".format(xyzB[0], xyzB[1], xyzB[2]))
            fid.write("{} {} {}\n".format(xyzC[0], xyzC[1], xyzC[2]))
            fid.write("{} {} {}  ".format(pRef[0], pRef[1], pRef[2]))
            fid.close()
            # ---------------------------------------------------------------------


        # replace existing element tags with the ones in the datafile
        gen_hetero.replaceElemTags(self.job, meshfile, datfile)

        # ---create some more vertex files needed for later processing-------------
        T_POSTERIORWALL = 0     # knowledge is based on samesideofplane setup
        T_ANTERIORWALL  = 2     # knowledge is based on samesideofplane setup

        etags      = pandas.read_csv(datfile, delimiter=' ', header=None).values.squeeze()
        pwall_eIDs = np.where(etags == T_POSTERIORWALL)[0]
        awall_eIDs = np.where(etags == T_ANTERIORWALL) [0]

        df_mesh   = pandas.read_csv(meshfile+'.elem', skiprows=1, delimiter=' ', header=None)
        pwall_vtx = df_mesh.values[pwall_eIDs,1:-1].astype(np.int).squeeze().flatten()
        awall_vtx = df_mesh.values[awall_eIDs,1:-1].astype(np.int).squeeze().flatten()
        del df_mesh, pwall_eIDs, awall_eIDs
        pwall_vtx = np.unique(pwall_vtx)
        awall_vtx = np.unique(awall_vtx)


        # exclusive export of posterior wall indices
        df = pandas.DataFrame(pwall_vtx)
        df.to_csv(meshfile+'.lvpwall.vtx', header=None, sep=' ', index=False)
        self.job.bash(['sed', '-i', '1iextra',                 meshfile+".lvpwall.vtx"], None)
        self.job.bash(['sed', '-i', '1i'+ str(len(pwall_vtx)), meshfile+".lvpwall.vtx"], None)
        del df

        # export of anterior wall indices
        df = pandas.DataFrame(awall_vtx)
        df.to_csv(meshfile+'.lvawall.vtx', header=None, sep=' ', index=False)
        self.job.bash(['sed', '-i', '1iextra',                 meshfile+".lvawall.vtx"], None)
        self.job.bash(['sed', '-i', '1i'+ str(len(awall_vtx)), meshfile+".lvawall.vtx"], None)
        del df
        # -------------------------------------------------------------------------

        # determine interfaces
        surfs = [meshfile + '.lvseptmid',
                 meshfile + '.lvpmid',
                 meshfile + '.lvamid',
                 meshfile + '.lvwallmid']

        ops   = '1:3;1:0;3:2;2:0'
        mt.extract_surface(self.job,meshfile, ','.join((surfs)), op=ops, ifmt='carp_txt')

        # remove generated neubc files
        for item in surfs:
            if os.path.exists(item + '.neubc'):
                os.remove(item + '.neubc')

        new_dict=dict({'lvpmid':   meshfile + '.lvpmid.surf.vtx',
                       'lvamid':   meshfile + '.lvamid.surf.vtx',
                       'lvseptmid':meshfile + '.lvseptmid.surf.vtx',
                       'lvwallmid':meshfile + '.lvwallmid.surf.vtx',
                       'apex':meshfile+'.apex.vtx'})

        return new_dict

    @staticmethod
    def find_geodesic_dists(seed_vtx, pts, surf, n2e, actTm):

        #find the distances from a point to all other points

        seed            = [ seed_vtx ]
        actTm[seed_vtx] = 0.
        cp_n2e          = list(n2e)

        while len(seed):
            seed = gen_meshes.propagate(seed, pts, surf, cp_n2e, actTm)
        return

    @staticmethod
    def propagate(seeds, pts, surf, n2e, actTm):
        # propagate the wavefront
        new_seeds = set()
        for s in seeds :
            for elem in n2e[s] :
                for p in surf[elem] :
                    if p != s :
                        actTm[p] = min(actTm[p], actTm[s]+simplemath.edgelen(p,s,pts))
                        if p not in seeds :
                            new_seeds.add(p)
            n2e[s] = []
        return new_seeds

    @staticmethod
    def queryEdges(basemesh):

        """ QUERYEDGES
        query min/mean/max sizes of with given mesh (obtained from 'meshtool query edges')

        parameters:
        basemesh     (input) path to basename of the mesh

        returns:
            tuple of minimum, mean and maximum existing edge length

        """
        min_el  = -1
        mean_el = -1
        max_el  = -1

        queryFile = basemesh + '.mtquery'

        if not os.path.exists(queryFile):

            # compile meshtool command
            cmd = 'meshtool query edges -msh={} > {}'.format(basemesh, queryFile)
            # Note: no need to silence command
            os.system(cmd)

        with open(queryFile,'r') as fid:
            lines = fid.readlines()

            rows  = -1
            for line in lines:
                rows += 1

                if "Edge lengths" in line:
                    strOI   = lines[rows+2].strip().rstrip(')').split(',')
                    print(strOI)

                    mean_el = float(strOI[1].split(':')[1])
                    min_el  = float(strOI[2].split(':')[1])
                    max_el  = float(strOI[3].split(':')[1])
                    break

        return min_el, mean_el, max_el

    @staticmethod
    def query_unreachable(job,args,msh,odir=None):

        base.debugString(args,'Checking Connectivity: ' + msh)

        if odir is None: odir = os.path.dirname(msh)

        job.mkdir(odir,parents=True)

        srf2chk=os.path.join(odir,os.path.basename(msh))
        thirdline=subprocess.check_output(['sed','-n','3p',msh+'.elem'],stdin=subprocess.PIPE)
        ind=str(thirdline).split(' ')[1]

        cmd=['meshtool','extract','unreachable','-msh='+msh,'-idx='+ind,'-submsh='+srf2chk]
        print(' '.join(cmd))
        subprocess.call(cmd)

        firstline=subprocess.check_output(['sed','-n','1p',srf2chk+'.unreachable.elem'],stdin=subprocess.PIPE)
        n_elems=int(str(firstline).split(' ')[0])

        print('Number of unreachable elements: '+str(n_elems))

        if n_elems != 0:

            print('Surface extraction yielded an island!')
            print('Press any Key to continue.')
            raw_input()

    @staticmethod
    def sub(job,args,srf2mk,srf2rm,srf2rmfm):

        # TODO: make a separate surf2vtx function
        # TODO: shrink neubc files based on the srf2rmdIDs

        """
        Substract a surface from another surface.

        parameters:
        srf2mk      (input) file name with path you want to save the output surface to
        srf2rm      (input) path to file you want removed
        srf2rmfm    (input) surface you want to remove from

        Note: Surface files srf2rm and sr2rmfm must have existing *.surf and *.surf.vtx files.

        """

        base.debugString(args,'Substracting '+srf2rm+' from '+srf2rmfm)

        # Note: this code is currently just implemented for triangular surfaces
        #       it will fail for quads
        #       df .. dataframe

        # subtract one surface from the other
        df      = pandas.read_csv(srf2rmfm+'.surf', skiprows=1, delimiter=' ', header=None)
        goodsrf = df.values[:,1:].astype(np.int).squeeze()

        df_badverts = pandas.read_csv(srf2rm+'.surf.vtx', skiprows=2, header=None)
        badverts    = df_badverts.values.astype(np.int)

        # actual ismember comes now
        goodsrf_bool = np.in1d(goodsrf, badverts)
        goodsrf_bool = np.reshape(goodsrf_bool, np.shape(goodsrf))
        srf2rmIDs    = np.count_nonzero(goodsrf_bool, axis=1)
        srf2rmIDs    = np.where(srf2rmIDs == 3)
        del goodsrf, goodsrf_bool, df_badverts, badverts

        # remove rows
        df.drop(df.index[srf2rmIDs], inplace=True)

        # export new surface file, re-insert proper header information
        df.to_csv(srf2mk+'.surf', header=None, sep=' ', index=False)
        job.bash(['sed', '-i', '1i' + str(df.shape[0]), srf2mk+'.surf'], None)

        # export updated vertex file and re-insert proper header information
        goodverts = df.values[:,1:].astype(np.int).flatten()
        goodverts = np.unique(goodverts)

        cmd  = 'tail -n+2 ' + srf2mk+'.surf | '
        cmd += "cut  -f2- -d' '             | "
        cmd += "tr   ' ' '\n'               | "
        cmd += 'sort -u                     > '
        cmd += srf2mk + ".surf.vtx"
        msg  = os.system(cmd)

        job.bash(['sed', '-i', '1iextra',                 srf2mk+".surf.vtx"], None)
        job.bash(['sed', '-i', '1i'+ str(len(goodverts)), srf2mk+".surf.vtx"], None)
        return

class gen_sols(dspace):

    def __init__(self,job,args,MA_info):
        super(gen_sols, self).__init__()
        self+=MA_info
        self.job=job
        self.args=args

        self.sols_dir=os.path.join(job.ID,'sols','{}')
        print(self.pretty_string())

        t0 = time.time() # start process timer

        base.debugString(args,'EIKONAL GENERATION',mode=0)

        self.init_dir= self.sols_dir.format('init')


        if not os.path.exists(self.sols_dir.format('')):
            self.gen_init_files()
            self.ek_batch()
        else: print('Assuming init files already generated')

        for key,value in sorted(self.sols.iterattr()):
            if key != 'rv_phi':

                self.sols[key] = self.sols_dir.format('ek_'+key+'.dat')
                if not os.path.isfile(self.sols[key]):

                    if 'lv_phi' in key:
                        curr_dir =self.sols_dir.format('ek_'+key)
                        bname = self.msh[value['basename']]['mesh']


                        bc0=[]
                        bc1=[]
                        m_tags=[[0,1],[2,3]][(0,1)['_a' in key]]
                        b_tags=[[0,1],[2,3]][(1,0)['_a' in key]]

                        for entry,val in zip(value['bc'],value['stim']):
            #
                            if val == 0:
                                bc0.append(self.msh[value['basename']][entry].replace('.vtx',''))
                            elif val == 1:
                                bc1.append(self.msh[value['basename']][entry].replace('.vtx',''))
                            else:
                                raise IOError('wrong value specified for stim. Please check stimdict.')

                        distance  = self.computeEikonalDistanceCARP(bname,curr_dir,bc0,bc1,m_tags,b_tags)
                        if 'lv_phi_p' in key: distance *= -1

                        self.job.bash(['rm', '-r', curr_dir+'0'])
                        self.job.bash(['rm', '-r', curr_dir+'1'])

                        # output normalized distances
                        with open(self.sols[key], 'w') as fp:
                            # convert numpy array to string and add carriage return
                            d = '\n'.join(distance.astype(str))
                            fp.write(d)
                            del d

                    else:
                        self.computeEikonalDistance(self.sols_dir.format('ek_'+key))

                else:
                    print('Already generated!')

        self.sols['lv_phi']=self.sols_dir.format('ek_lv_phi.dat')
        if not os.path.isfile(self.sols.lv_phi):
            base.debugString(args,'Generating joint LV_phi file in: '+self.sols.lv_phi)
            self.merge_lvphi_anterior_posterior(self.sols.lv_phi)
        else:
            print('LV PHI solution already exists')

        print('DONE')
        print(time.time()-t0)

        base.debugString(args,'Generating Laplacian Solution: '+'lap_rv_phi',mode=0)
        curr_dir =self.sols_dir.format('lap_rv_phi')
        if not os.path.exists(curr_dir):
            job.mkdir(curr_dir, parents=True)

            bc=[]

            bc.append(base.makevtxfcn(job,args,self.sols.lv_phi,self.msh.lv.rvja,self.msh.rv.rvja))
            bc.append(base.makevtxfcn(job,args,self.sols.lv_phi,self.msh.lv.rvjp,self.msh.rv.rvjp))

            self.sols['rv_phi']=self.solve_laplace(self.msh.rv.mesh,bc,self.sols.rv_phi.stim,odir=curr_dir)

        else:
            self.sols['rv_phi']=self.sols_dir.format('lap_rv_phi/phie.dat')

        #
        #
        # raw_input()
        #
        # for key,value in sorted(self.sols.iterattr()):
        #
        #     if key == 'rv_phi':
        #
        #         base.debugString(args,'Generating Laplacian Solution: '+'lap_'+key,mode=0)
        #
        #         curr_dir =self.sols_dir.format('lap_'+key)
        #         sol_file=os.path.join(curr_dir, 'phie.dat')
        #
        #         if not os.path.exists(curr_dir):
        #                 job.mkdir(curr_dir, parents=True)
        #
        #         if os.path.exists(sol_file):
        #             self.sols[key] = sol_file
        #             base.debugString(args, 'Already generated: '+sol_file+'\n',mode=2)
        #
        #         else:
        #
        #             bc=[]
        #             bname=self.msh[value['basename']]['mesh']
        #
        #             bc.append(base.makevtxfcn(job,args,self.sols.lv_phi,self.msh.lv.rvja,self.msh.rv.rvja))
        #             bc.append(base.makevtxfcn(job,args,self.sols.lv_phi,self.msh.lv.rvjp,self.msh.rv.rvjp))
        #
        #             self.sols[key]=self.solve_laplace(bname,bc,value['stim'],odir=curr_dir)
        #
        #     else:
        #
        #         base.debugString(args,'Generating Eikonal Solution: '+'ek_'+key,mode=0)
        #
        #         curr_dir =sols_dir.format('ek_'+key)
        #         sol_file      = os.path.join(curr_dir, key + '.dat')
        #
        #         if os.path.exists(sol_file):
        #             self.sols[key] = sol_file
        #             base.debugString(args, 'Already generated: '+sol_file+'\n',mode=2)
        #
        #         else:
        #
        #             if not os.path.exists(curr_dir):
        #                 job.mkdir(curr_dir, parents=True)
        #
        #             bc0    = []
        #             bc1    = []
        #             bname = self.msh[value['basename']]['mesh']
        #
        #             #define myo and bath tags
        #             if 'lv_phi' in key:
        #                 m_tags=[[0,1],[2,3]][(0,1)['_a' in key]]
        #                 b_tags=[[0,1],[2,3]][(1,0)['_a' in key]]
        #
        #                 print('m_tags= '+str(m_tags))
        #
        #             else:
        #                 m_tags, b_tags = base.get_elem_tags(job, bname)
        #                 b_tags = np.append(b_tags, -1000)
        #
        #             #Setup the boundary conditions!
        #             for entry,val in zip(value['bc'],value['stim']):
        #
        #                 if val == 0:
        #                     bc0.append(self.msh[value['basename']][entry].replace('.vtx',''))
        #                 elif val == 1:
        #                     bc1.append(self.msh[value['basename']][entry].replace('.vtx',''))
        #                 else:
        #                     raise IOError('wrong value specified for stim. Please check stimdict.')
        #
        #
        #             distance  = self.computeEikonalDistance(bname,curr_dir,bc0,bc1,m_tags,b_tags)
        #             if 'lv_phi_p' in key: distance *= -1
        #
        #             # clean up and dangerous thing to do !!!
        #             job.bash(['rm', '-r', curr_dir+'0'])
        #             job.bash(['rm', '-r', curr_dir+'1'])
        #
        #             # output normalized distances
        #             self.sols[key] = sol_file
        #             with open(sol_file, 'w') as fp:
        #                 # convert numpy array to string and add carriage return
        #                 d = '\n'.join(distance.astype(str))
        #                 fp.write(d)
        #                 del d
        #
        #         # finally merge anterior and posterior lv_phi files
        #         if key == 'lv_phi_p':
        #             self.sols['lv_phi']=sols_dir.format('ek_lv_phi/vm.act.seq.dat')
        #
        #             if not os.path.exists(self.sols.lv_phi):
        #                 job.mkdir(os.path.dirname(self.sols.lv_phi),parents=True)
        #                 base.debugString(args,'Generating joint LV_phi file: '+self.sols.lv_phi)
        #                 self.merge_lvphi_anterior_posterior(self.sols.lv_phi)
        #             else:
        #                 print('LV PHI solution already exists')

        base.debugString(args,'Done in {:.0f}s'.format(time.time()-t0))
        base.debugString(args,self.pretty_string(),mode=0)

        base.debugString(args,'GEN SOLUTIONS: Done in {:.0f}s'.format(time.time()-t0))

    def ek_batch(self):

        base.debugString(self.args,'EK BATCH SUBMISSION',mode=0)

        if self.biv_batch:
            base.debugString(self.args,'Submitting EK Batch for BIV',mode=1)
            biv_cmd=['ekbatch',self.msh.biv.mesh,','.join((self.biv_batch))]
            subprocess.check_call(biv_cmd)

            [self.job.mv(x+'.dat',self.sols_dir.format('')) for x in self.biv_batch]

        if self.rv_batch:
            base.debugString(self.args,'Submitting EK Batch for RV',mode=1)
            rv_cmd=['ekbatch',self.msh.rv.mesh,','.join((self.rv_batch))]
            subprocess.check_call(rv_cmd)

            [self.job.mv(x+'.dat',self.sols_dir.format('')) for x in self.rv_batch]

        print(self.lv_batch)
        if self.lv_batch:
            base.debugString(self.args,'Submitting EK Batch for LV',mode=1)
            lv_cmd=['ekbatch',self.msh.lv.mesh,','.join((self.lv_batch))]
            print(' '.join((lv_cmd)))
            subprocess.check_call(lv_cmd)

            [self.job.mv(x+'.dat',self.sols_dir.format('')) for x in self.lv_batch]

    def gen_init_files(self):

        self.init_dir=self.sols_dir.format('init')
        self.job.mkdir(self.init_dir,parents=True)

        biv_batch=[]
        lv_batch=[]
        rv_batch=[]

        self.lv_batch_tags=[]

        for key,value in sorted(self.sols.iterattr()):

            if key != 'rv_phi':

                # if 'lv_phi' in key:
                #     self.lv_batch_tags.append(['0,1','2,3'][(0,1)['_a' in key]])
                #         # b_tags=[[0,1],[2,3]][(1,0)['_a' in key]]
                #
                #
                # else:

                if 'lv_phi' in key:
                    pass
                else:

                    msh_base=self.msh[value['basename']]['mesh'][-3::].strip('.')

                    ifile0=os.path.join(self.init_dir,'ek_{}0.init'.format(key))
                    ifile1=os.path.join(self.init_dir,'ek_{}1.init'.format(key))
                    base.debugString(self.args,'Generating Eikonal Init Files: '+'ek_'+key,mode=1)

                    if not os.path.isfile(ifile0) or not os.path.isfile(ifile1):

                        bc0=[]
                        bc1=[]

                        for entry,val in zip(value['bc'],value['stim']):
                            if val == 0:
                                bc0.append(self.msh[value['basename']][entry])
                            elif val == 1:
                                bc1.append(self.msh[value['basename']][entry])
                            else:
                                raise IOError('wrong value specified for stim. Please check stimdict.')

                        base.debugString(self.args,'Generating Eikonal Init File: '+ifile0,mode=1)
                        gen_sols.ek_init_file(ifile0,bc0)

                        base.debugString(self.args,'Generating Eikonal Init File: '+ifile1,mode=1)
                        gen_sols.ek_init_file(ifile1,bc1)

                    else:
                        print('Init files already exist:\n')
                        print(ifile0)
                        print(ifile1)

                    locals()[msh_base+'_batch'].append(ifile0.replace('.init',''))
                    locals()[msh_base+'_batch'].append(ifile1.replace('.init',''))

        self.biv_batch=biv_batch
        self.lv_batch=lv_batch
        self.rv_batch=rv_batch

    @staticmethod
    def ek_init_file(init_file,bc):

        print('Boundary Conditions: \n'+' \n'.join((bc)))
        bc_nodes=[]
        [bc_nodes.append(np.loadtxt(entry,dtype=int,skiprows=2)) for entry in bc]
        vtx_list = [item for sublist in bc_nodes for item in sublist]

        # print('USING A CONDUCTION VELOCITY OF '+str(vel_f))
        f=open(init_file,'w')

        #Conduction velocity in myo:
        f.write('vf: 1.0 vs: 1.0 vn: 1.0 vPS: 1.0\n')
        f.write('retro_delay: 0.0 antero_delay: 0.0\n')

        f.write(str(len(vtx_list))+' 0\n')

        for vtx in vtx_list:
            f.write(str(vtx)+' 0\n')

        f.close()

        print('Wrote file '+init_file)

    def merge_lvphi_anterior_posterior(self, ofile):

        df_elem = pandas.read_csv(self.msh.lv.mesh + '.elem', skiprows=1, delimiter=' ', header=None)
        elem    = df_elem.values[:, 1:-1].squeeze().astype(np.int)
        etags   = df_elem.values[:,   -1].squeeze().astype(np.int)

        ant_elems  = np.take(elem, np.where(etags > 1), axis=0)[0]  # all anterior elements
        lv_ant_vtx = np.unique(ant_elems.flatten())
        del df_elem, elem, etags

        # read laplace solutions
        lv_phi_a = pandas.read_csv(self.sols.lv_phi_a, delimiter='\n', header=None)
        lv_phi_a = np.asarray(lv_phi_a.values, dtype=np.float).squeeze()
        lv_phi_p = pandas.read_csv(self.sols.lv_phi_p, delimiter='\n', header=None)
        lv_phi_p = np.asarray(lv_phi_p.values, dtype=np.float).squeeze()

        # assemble global lv_phi values
        lv_phi             = np.copy(lv_phi_p)
        lv_phi[lv_ant_vtx] = lv_phi_a[lv_ant_vtx]

        lv_phi=(((lv_phi-np.min(lv_phi))/np.max((lv_phi-np.min(lv_phi))))*2*np.pi)-np.pi

        self.sols.add('lv_phi', ofile)

        # save to file
        with open(ofile, 'w') as fp:
            # write as column vector
            data = '\n'.join(lv_phi.astype(str))
            fp.write(data)

        return self

    # # computing the lvphie map for the anterior and posterior laplace range
    # def map_lvphi(self,mshbase, phi, cm_vtx_list, pts):
    #
    #     # determine, if we are handling the anterior or posterior half of the LV
    #     if   max(phi) > 3:
    #         mode = 'anterior'
    #     elif min(phi) < -3:
    #         mode  = 'posterior'
    #     else:
    #         mode = ''
    #
    #     # sort the vertices based on their laplace values
    #     indices = np.argsort(phi[cm_vtx_list])
    #
    #     cm_vtx_list = cm_vtx_list[indices]
    #     cm_values   = np.take(phi, cm_vtx_list)
    #
    #
    #     # ---DEBUGGING OUTPUT------------------------------------------------------
    #     # selected anterior nodes used for normalization stored as auxilliary file
    #     geoFile = os.path.join(os.path.dirname(mshbase), mode + '.geo.pts_t')
    #     df = pandas.DataFrame(pts[cm_vtx_list,:])
    #     df.to_csv(geoFile, header=None, sep=' ', index=False)
    #
    #     self.job.bash(['sed', '-i', '1i' + str(len(cm_vtx_list)), geoFile], None)
    #     self.job.bash(['sed', '-i', '1i1',                        geoFile], None)
    #     del df, geoFile
    #
    #     geoFile = os.path.join(os.path.dirname(mshbase), mode + '.geo.dat_t')
    #     df = pandas.DataFrame(cm_values)
    #     df.to_csv(geoFile, header=None, sep=' ', index=False)
    #
    #     self.job.bash(['sed', '-i', '1i' + str(len(cm_values)), geoFile], None)
    #     self.job.bash(['sed', '-i', '1i1',                      geoFile], None)
    #     del df, geoFile
    #     # ---END DEBUGGING OUTPUT--------------------------------------------------
    #
    #
    #     # normalize distances
    #     ndist_cm  = simplemath.normalize_distance(cm_vtx_list, pts)
    #
    #
    #     # rescale the map to +/-pi depending on anterior or posterior
    #     #
    #     # NOTE: make sure that the interpolation function is increasing in x (-> cm_values).
    #     #       Otherwise, the output is nonsense!
    #     if mode is 'anterior':
    #         ndist_cm = ndist_cm * (+ np.pi)
    #         phi      = np.interp(phi, cm_values, ndist_cm, left=0, right=np.pi)
    #     elif mode is 'posterior':
    #         ndist_cm = ndist_cm * (- np.pi)
    #         phi      = np.interp(phi, cm_values, ndist_cm[::-1], left = -np.pi, right = 0)
    #     else:
    #         ndist_cm = []
    #
    #
    #     # make sure it's the global index list
    #     return phi

    def solve_laplace(self, msh, bc_stim_files, bc_stim_str, odir,
                      bc_tags=[], no_bc_tags=[], rerun_laplace=True):

        """
        Initiates a laplacian simulation.
        """

        # Get basic command line, including solver options
        cmd = tools.carp_cmd()

        if isinstance(bc_stim_files, str):
            bc_stim_files = [bc_stim_files]

        # Set up conductivity within regions a laplace solution is computed in
        cond  = ['-num_gregions', 1]
        cond += ['-gregion[0].name',    'LAPLACE',
                 '-gregion[0].g_il',    1,
                 '-gregion[0].g_it',    1,
                 '-gregion[0].g_in',    1,
                 '-gregion[0].g_el',    1,
                 '-gregion[0].g_et',    1,
                 '-gregion[0].g_en',    1,
                 '-gregion[0].num_IDs', len(bc_tags)]
        for i, t in enumerate(bc_tags):
            cond += ['-gregion[0].ID[{}]'.format(i), t]

        # if len(no_bc_tags) > 0:
        #     cond += ['-num_gregions', 2]
        #     cond += ['-gregion[1].name',    'IGNORE_REGION',
        #              '-gregion[1].g_il',    0.00000001,
        #              '-gregion[1].g_it',    0.00000001,
        #              '-gregion[1].g_in',    0.00000001,
        #              '-gregion[1].g_el',    0.00000001,
        #              '-gregion[1].g_et',    0.00000001,
        #              '-gregion[1].g_en',    0.00000001,
        #              '-gregion[1].num_IDs', len(no_bc_tags)]
        #     for i, t in enumerate(no_bc_tags):
        #         cond += ['-gregion[1].ID[{}]'.format(i), t]

        # Define the geometry of the stimulus at one end of the block
        stimuli = ['-num_stim', len(bc_stim_files)]

        istim = 0

        for vtx,bc_str in zip(bc_stim_files,bc_stim_str):

            if bc_str == 0:

                stimuli += ['-stimulus[{}].vtx_file'.format(istim), vtx,
                            '-stimulus[{}].stimtype'.format(istim), 3]
                istim   += 1

            elif bc_str=='vtx_fcn':
                stimuli += ['-stimulus[{}].vtx_fcn'.format(istim), 1,
                            '-stimulus[{}].vtx_file'.format(istim), vtx,
                            '-stimulus[{}].stimtype'.format(istim), 2,
                            '-stimulus[{}].start'.format(   istim), 0,
                            '-stimulus[{}].strength'.format(istim), 1,
                            '-stimulus[{}].duration'.format(istim), 1]
                istim += 1

            else:

                stimuli += ['-stimulus[{}].vtx_file'.format(istim), vtx,
                            '-stimulus[{}].stimtype'.format(istim), 2,
                            '-stimulus[{}].start'.format(   istim), 0,
                            '-stimulus[{}].duration'.format(istim), 1,
                            '-stimulus[{}].strength'.format(istim), bc_str]
                istim += 1

        cmd += ['-simID',      odir,
                '-meshname',   msh,
                '-gzip_data',  0,
                '-experiment', 2, # Laplacian solve
                '-bidomain',   1] # This option must be set or the code segfaults.
                                  # The laplacian solve takes place on the extracellular grid.
        cmd += stimuli
        cmd += cond

        # Run simulation
        if os.path.exists(odir) and not rerun_laplace:
            base.debugString(self.args, '... {} already exists. Skipping generation!').format(odir)
        else:
            self.job.carp(cmd, odir)


        # Create .dat file
        phie = os.path.join(odir, 'phie.igb')
        dat  = os.path.join(odir, 'phie.dat')
        cmd  = [settings.execs.IGBEXTRACT, phie, '-o', 'ascii_1pL', '-O', dat]

        self.job.bash(cmd, None)

        # Return filename of solution
        return dat

    def computeEikonalDistance(self,simID):

        base.debugString(self.args,'Normalizing solutions for '+simID,mode=1)

        simID0 = '{}0.dat'.format(simID)
        simID1 = '{}1.dat'.format(simID)
        solID  = '{}.dat'.format(simID)

        actseq0 = np.loadtxt(os.path.join(simID0), dtype=float)
        actseq1 = np.loadtxt(os.path.join(simID1), dtype=float)
        distance = np.divide(actseq0, actseq0+actseq1, out=np.zeros_like(actseq0), where=(actseq0+actseq1)>0.0)
        distance = (distance-distance.min())/(distance.max()-distance.min())

        self.job.bash(['rm', '-r', simID0])
        self.job.bash(['rm', '-r', simID1])

        with open(solID, 'w') as fp:
            # convert numpy array to string and add carriage return
            d = '\n'.join(distance.astype(str))
            fp.write(d)
            del d

        return distance

    def computeEikonalDistanceCARP(self,basename, simID, vtxfname0, vtxfname1, atags, ptags):

          simID0 = '{}0'.format(simID)
          simID1 = '{}1'.format(simID)
          self.run_eikonal(basename, simID0, vtxfname0, active_tags=atags, passive_tags=ptags)
          self.run_eikonal(basename, simID1, vtxfname1, active_tags=atags, passive_tags=ptags)
          actseq0 = np.loadtxt(os.path.join(simID0, 'vm_act_seq.dat'), dtype=float) - 2.0
          actseq1 = np.loadtxt(os.path.join(simID1, 'vm_act_seq.dat'), dtype=float) - 2.0
          distance = np.divide(actseq0, actseq0+actseq1, out=np.zeros_like(actseq0), where=(actseq0+actseq1)>0.0)
          distance = (distance-distance.min())/(distance.max()-distance.min())

          return distance

    def run_eikonal(self,basename, simID, source,active_tags=[1,6], passive_tags=[41,46]):

        cmd = tools.carp_cmd()

        active_tags = list(set(active_tags))
        passive_tags = list(set(passive_tags))

        gregargs = {'g_il': 0.174, 'g_it': 0.019, 'g_in': 0.019, 'g_el': 0.625, 'g_et': 0.236, 'g_en': 0.236}
        gregion_active = model.ConductivityRegion(active_tags, 'active', **gregargs)
        gregargs = {'g_il': 0.0001, 'g_it': 0.0001, 'g_in': 0.0001, 'g_el': 0.0001, 'g_et': 0.0001, 'g_en': 0.0001}
        gregion_passive = model.ConductivityRegion(passive_tags, 'passive', **gregargs)
        ek_active = model.EkRegion(0, 'active', 1.0, 1.0, 1.0)
        ek_passive = model.EkRegion.passive(1, 'passive')

        cmd += ['-meshname', basename,
              '-simID', simID,
              '-experiment', 6,
              '-localize_pts', 1,
              '-pstrat', 1,
              '-pstrat_i', 1,
              '-meshformat', 0]

        cmd += model.optionlist([gregion_active, gregion_passive, ek_active, ek_passive])

        cmd +=['-num_stim',             len(source)]
        for i in range(0,len(source)):
            if 'vtxfcn' in source[i]:
                cmd += ['-stimulus[{}].vtx_fcn'.format(i), 1,
                  '-stimulus[{}].strength'.format(i), 1]
            else:
                cmd+= ['-stimulus[{}].strength'.format(i), 150.0]

                cmd += ['-stimulus[{}].stimtype'.format(i),   0,
                    '-stimulus[{}].vtx_file'.format(i), source[i],
                    '-stimulus[{}].start'.format(i),      0,
                    '-stimulus[{}].duration'.format(i),   2.0]

        self.job.carp(cmd)

@tools.carpexample(parser,jobID)
def main(args, job):
    t0=time.time()
    base.debugString(args,'MODEL ARCHITECTURE VERSION: ORIGINAL EIKONAL ROOT',mode=0)
    model_arch(args,job)
    MAt=time.time()-t0
    base.debugString(args,'MODEL ARCHITECTURE GENERATION DONE IN: {:.0f}m={:.0f}s'.format(MAt/60,MAt))

if __name__ == '__main__':
    main()