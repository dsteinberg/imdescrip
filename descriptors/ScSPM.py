# Imdescrip -- a collection of tools to extract descriptors from images.
# Copyright (C) 2013  Daniel M. Steinberg (d.steinberg@acfr.usyd.edu.au)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

""" A modified implementation of Yang et. al.'s ScSPM image descriptor [1]. """

import math
import numpy as np
from sklearn.linear_model import orthogonal_mp_gram as omp
from sklearn.cluster import KMeans
from utils import patch as pch, siftwrap as sw
from descriptor import Descriptor


class ScSPM (Descriptor):
    """ A modified sparse coding spatial pyramid matching image descriptor.

        This class implements a modified version of Yang et. al.'s sparse code
        spatial pyramid match (ScSPM) image descriptor [1]. While the original
        descripor uses sparse code (lasso) dictionaries and image encoding, this
        uses K-means dictionaries and orthogonal matching persuit (OMP)
        encoding. Some classification performance is lost when using OMP, but it
        is far more scalable to a large number of images. 
        
        In addition to this scalability modification, there is an option to save
        compressed ScSPM descriptors instead of the original large-dimensional
        descriptors. Compression is done using random projection, see [2] for
        more details.

        Before using the extract() method, a dictionary must be learned using
        the learn_dictionary() method. I use pickle to save this object once a
        dictionary has been learned, so then I can apply it to multiple
        datasets.

        Arguments:
            maxdim: int (default 320), the maximum dimension the images should 
                be. This will preserve the aspect ratio of the images though.
            psize: int (default 16), the (square) patch size to use for dense
                SIFT descriptor extraction
            pstride: int (default 8), the stride to use for dense SIFT
                extraction.
            active: int (default 10), the number of activations to use for OMP.
            dsize: int (default 1024), the number of dictionary elements to use.
            levels: tuple (default (1,2,4)), the type of spatial pyramid to use.
            compress_dim: int (default None), the dimension of the random
                projection matrix to use for compressing the descriptors. None
                means the descriptors are not compressed.

        Note:
            When using compression, keep the dimensionality quite large. I.e. a
            dictionary size of 1024 will lead to images descriptors of 21,504
            dimensions. To preseve classification accuracy you may want to not
            set compress_dim less than 3000 dimensions.

        References:

        [1] Yang, J.; Yu, K.; Gong, Y. & Huang, T. Linear spatial pyramid
            matching using sparse coding for image classification Computer
            Vision and Pattern Recognition, 2009. CVPR 2009. IEEE Conference on,
            2009, 1794-1801

        [2] Davenport, M. A.; Duarte, M. F.; Eldar, Y. C. & Kutyniok, G.
            Introduction to compressed sensing Chapter 1 Compressed Sensing:
            Theory and Applications, Cambridge University Press, 2011, 93

    """

    def __init__ (self, maxdim=320, psize=16, pstride=8, active=10, dsize=1024,
                    levels=(1,2,4), compress_dim=None):

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
        """ Extract a ScSPM descriptor for an image.
       
        This method will return an ScSPM descriptor for an image.
        learn_dictionary() needs to be run (once) before this method can be
        used.
        
        Arguments:
            impath: str, the path to an image

        Returns:
            a ScSPM descriptor (array) for the image. This array either has
                self.dsize*sum(self.levels**2) elements, or self.compress_dim if
                not None.

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

        # Pyramid pooling and normalisation
        fea = pch.pyramid_pooling(scpatch, cx, cy, img.shape, self.levels)
        fea = fea / math.sqrt((fea**2).sum() + 1e-10)

        if self.compress_dim is not None:
            return np.dot(fea, self.rmat)
        else:
            return fea
        

    def learn_dictionary (self, images, npatches=50000, ntrials=1, njobs=-1):
        """ Learn a K-means dictionary for this ScSPM.

        This method trains a K-means dictionary for the ScSPM descriptor object.
        This only needs to be run once before multiple calls to the extract()
        method can be made.

        Arguments:
            images: list, a list of paths to images to use for training.
            npatches: int (default 50000) number of SIFT patches to extract from
                the images to use for training the dictionary.
            ntrials: int (default 1), the number of random starts of K-means to
                do. The best run will be chosen for the dictionary.
            njobs: int (default -1), the number of threads to use. -1 means the
                number of threads will be equal to the number of cores.

        """

        # Get SIFT training patches 
        patches = sw.training_patches(images, npatches, self.psize, self.maxdim,
                                        verbose=True)
          
        # Learn dictionary
        print('Learning K-means dictionary...')
        sdic = KMeans(n_clusters=self.dsize, verbose=True, n_init=ntrials,
                n_jobs=njobs)
        sdic.fit(patches)
        print('done')

        # Normalise and save the dictionary
        self.dic = pch.norm_patches(sdic.cluster_centers_)
        self.dic_gram = np.dot(self.dic, self.dic.T)
        
