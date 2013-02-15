""" A modified implementation of Yang's ScSPM image descriptor. 

"""

import numpy as np
from sys import stdout
from sklearn import decomposition as dec
from sklearn.linear_model import orthogonal_mp_gram as omp
from skimage import color
from scipy.linalg import orth
from utils import imgpatches as imp
from descriptor import Descriptor

class ScSPM (Descriptor):
    """ A modified sparse coding spatial pyramid matching image descriptor.

    """


    def __init__ (self, maxdim=320, psize=16, pstride=8, active=10,
                    levels=(1,2,4), l1_dsize=128, l2_dsize=512, colour=False, 
                    compress_dim=None):
        """
        """

        self.maxdim = maxdim
        self.psize = psize
        self.pstride = pstride 
        self.active = active
        self.levels = levels
        self.l1_dsize = l1_dsize
        self.l2_dsize = l2_dsize
        self.col = colour
        self.compress_dim = compress_dim
        self.dic_l1 = None       # ICA mixing matrix (A)
        self.dic_l2 = None       # Sparse code dictionary (D)
        self.dic_l1_umix = None  # ICA unmixing matrix (W)
        self.dic_l2_gram = None  # Sparse code dictionary gram matrix (D * D.T)
        
        if self.compress_dim is not None:
            D = np.sum(np.array(levels)**2) * self.l2_dsize
            self.comp_mat = np.random.randn(D, self.compress_dim)


    def extract (self, image):
        """
        """

        if (self.dic_l1 is None) or (self.dic_l2 is None):
            raise ValueError('No dictionaries have been learned!')

        # Get and resize image, convert to gray if necessary
        if self.col == True:
            img = imp.imread_resize(image, self.maxdim) 
        else:
            img = color.rgb2gray(imp.imread_resize(image, self.maxdim))

        # Extract patches
        patches, cx, cy = imp.grid_patches(img, self.psize, self.pstride)

        # Centre and contrast normalise patches
        patches = imp.norm_patches(imp.centre_patches(patches))
       
        # Get ICA codes/responses and re-normalise
        l1_patch = imp.norm_patches(np.dot(patches, self.dic_l1_umix.T))
        print l1_patch.shape

        # Get OMP codes
        l2_patch = np.transpose(omp(self.dic_l2_gram, np.dot(self.dic_l2, 
                                    l1_patch.T), self.active))

        # Pyramid pooling and normalisation
        fea = imp.norm_patches(imp.pyramid_pooling(l2_patch, cx, cy, img.shape, 
                                self.levels))

        if self.compress_dim is not None:
            return np.dot(fea, self.comp_mat)
        else:
            return fea
        

    def learn_dictionaries (self, images, npatches, l2_iter=5000, 
                            visualise=False):
        """
        """

        # Get training patches 
        patches = imp.training_patches(images, npatches, self.psize, 
                                        self.maxdim, self.col)

        # Centre and contrast normalise patches
        patches = imp.norm_patches(imp.centre_patches(patches))
           
        # Learn ICA dictionary (whiten/dewhiten integrated into ICA)
        stdout.write('Learning ICA dictionary...')
        stdout.flush()
        ica = dec.FastICA(self.l1_dsize)
        ica.fit(patches)
        print('done')

        self.dic_l1 = ica.get_mixing_matrix()
        self.dic_l1_umix = ica.unmixing_matrix_ 

        if visualise == True:
            imp.disp_patches(self.dic_l1.T, self.col)

        # ICA Encode training patches (and whiten)
        stdout.write('Encoding and whitening patches...')
        stdout.flush()
        pca = dec.PCA(whiten=True)
        l1_patch = pca.fit_transform(ica.transform(patches))
        print('done')

        # Learn OMP dictionary
        print('Learning sparse code dictionary...')
        sdic = dec.MiniBatchDictionaryLearning(self.l2_dsize, verbose=True,
                transform_n_nonzero_coefs=self.active, n_iter=l2_iter)
        sdic.fit(l1_patch)
        print('done')

        self.dic_l2 = imp.norm_patches(pca.inverse_transform(sdic.components_))
        self.dic_l2_gram = np.dot(self.dic_l2, self.dic_l2.T)
        
        if visualise == True:
            imp.disp_patches(np.dot(self.dic_l2, self.dic_l1.T), self.col)
        
        return

