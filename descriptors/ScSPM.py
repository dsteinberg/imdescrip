""" A modified implementation of Yang's ScSPM image descriptor. 

"""

import math
import numpy as np
from sklearn.linear_model import orthogonal_mp_gram as omp
from sklearn.decomposition import MiniBatchDictionaryLearning as MBdiclearn 
#from sklearn.cluster import KMeans
#from sklearn.decomposition import DictionaryLearning as diclearn
#from skimage import color
from utils import patch as pch, siftwrap as sw
from descriptor import Descriptor

class ScSPM (Descriptor):
    """ A modified sparse coding spatial pyramid matching image descriptor.

    """


    def __init__ (self, maxdim=320, psize=16, pstride=8, active=10, dsize=512,
                    levels=(1,2,4), compress_dim=None):
        """
        """

        self.maxdim = maxdim
        self.psize = psize
        self.pstride = pstride 
        self.active = active
        self.levels = levels
        self.dsize = dsize
        self.compress_dim = compress_dim
        self.dic = None       # Sparse code dictionary (D)
        self.dic_gram = None  # Sparse code dictionary gram matrix (D * D.T)
        
        if self.compress_dim is not None:
            D = np.sum(np.array(levels)**2) * self.dsize
            self.rmat = np.random.randn(D, self.compress_dim)
            self.rmat = self.rmat / np.sqrt((self.rmat**2).sum(axis=0))


    def extract (self, impath):
        """
        """

        if self.dic is None:
            raise ValueError('No dictionary has been learned!')

        # Get and resize image 
        img = pch.imread_resize(impath, self.maxdim) 

        # Extract SIFT patches
        patches, cx, cy = sw.DSIFT_patches(img, self.psize, self.pstride)

        # Get OMP codes
        scpatch = np.transpose(omp(self.dic_gram, np.dot(self.dic, patches.T), 
                                self.active))

        ## Pyramid pooling and normalisation
        fea = pch.pyramid_pooling(scpatch, cx, cy, img.shape, self.levels)
        fea = fea / math.sqrt((fea**2).sum() + 1e-10)

        if self.compress_dim is not None:
            return np.dot(fea, self.rmat)
        else:
            return fea
        

    def learn_dictionary (self, images, npatches, niter=5000):
        """
        """

        # Get SIFT training patches 
        patches = sw.training_patches(images, npatches, self.psize, self.maxdim)
          
        # Learn dictionary
        print('Learning dictionary...')
        sdic = MBdiclearn(n_atoms=self.dsize, verbose=True, n_iter=niter,
                            transform_n_nonzero_coefs=self.active)
        sdic.fit(patches)
        print('done')

        # Normalise and save the dictionary
        self.dic = pch.norm_patches(sdic.components_)
        self.dic_gram = np.dot(self.dic, self.dic.T)
        
        return

