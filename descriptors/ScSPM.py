
import numpy as np
from sys import stdout
from sklearn import decomposition as dec
from skimage import color 
from utils import imgpatches as imp
from descriptor import Descriptor

class ScSPM (Descriptor):
    """
    """

    def __init__ (self, maxdim=320, psize=16, pstride=8, active=10, 
                    colour=False, compress_dim=None):
        """
        """

        self.maxdim = maxdim
        self.psize = psize
        self.pstride = pstride 
        self.active = active
        self.col = colour
        self.compress_dim = compress_dim
        self.dic_l1 = None      # ICA unmixing matrix (W)
        self.dic_l1_mix = None  # ICA mixing matrix (A)
        self.dic_l2 = None      # Sparse code dictionary
        

    def extract (self, image):
        """
        """

        if self.col == True:
            img = imp.imread_resize(image, self.maxdim) 
        else:
            img = color.rgb2gray(imp.imread_resize(image, self.maxdim))

        #plt.imshow(hog_image, cmap=plt.cm.gray)
        #plt.show()

        #return fea
        return


    def learn_dictionaries (self, images, npatches, l1_dsize, l2_dsize,
                            l2_iter=5000, visualise=False):
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
        ica = dec.FastICA(l1_dsize)
        ica.fit(patches)
        print('done')

        self.dic_l1 =  ica.unmixing_matrix_ # The level 1 dictionary pseudo-inv
        self.dic_l1_mix = ica.get_mixing_matrix()

        if visualise == True:
            imp.disp_patches(self.dic_l1_mix.T, self.col)

        # ICA Encode training patches (and whiten)
        stdout.write('Encoding and whitening patches...')
        stdout.flush()
        pca = dec.PCA(whiten=True)
        l1_patch = pca.fit_transform(ica.transform(patches))
        print('done')

        # Learn OMP dictionary
        print('Learning sparse code dictionary...')
        sdic = dec.MiniBatchDictionaryLearning(l2_dsize, verbose=True,
                transform_n_nonzero_coefs=self.active, n_iter=l2_iter)
        sdic.fit(l1_patch)
        print('done')

        self.dic_l2 = pca.inverse_transform(sdic.components_) # The level 2 dict
        
        if visualise == True:
            imp.disp_patches(np.dot(self.dic_l2, 
                                ica.get_mixing_matrix().T), self.col)
        
        return
