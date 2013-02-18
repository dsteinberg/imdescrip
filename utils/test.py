#! /usr/bin/env python

# Unit test for imgpatches module

import patch as imp
from numpy import array, arange, vstack, ones, zeros, sqrt, abs
import unittest 


class TestImgPatches (unittest.TestCase):
    """ This is a TestCase for the imgpatches module. """

    def setUp (self):
        """ Make a few test inputs and outputs for the tests. """
       
        # Test inputs image, and patch size, stride 
        self.timg = vstack([arange(1,5), arange(5,9), arange(9,13), 
                            arange(13,17)])
        self.psize = 3
        self.pstride = 1
        
        # Test output patches, patch centres
        self.tpatch = array([[ 1,  2,  3,  5,  6,  7,  9,  10, 11],
                             [ 2,  3,  4,  6,  7,  8,  10, 11, 12],
                             [ 5,  6,  7,  9,  10, 11, 13, 14, 15],
                             [ 6,  7,  8,  10, 11, 12, 14, 15, 16]])
        self.tx = array([1, 2, 1, 2])
        self.ty = array([1, 1, 2, 2]) 
        self.tpyr = vstack([self.tpatch[3,:], self.tpatch[0,:], 
                        self.tpatch[1,:], self.tpatch[2,:], self.tpatch[3,:]])

    def test_grid_patches (self):
        """ Test the imgpatches.grid_patches() function. """

        # Make sure the outputs are the right shape and values
        patches, x, y = imp.grid_patches(self.timg, self.psize, self.pstride)
        self.assertTrue((patches == self.tpatch).all())
        self.assertTrue((x == self.tx).all())
        self.assertTrue((y == self.ty).all())

        # Make sure the function errors out when we expect it too
        self.assertRaises(ValueError, imp.grid_patches, array([1, 1, 1, 1]),
                          self.psize, self.pstride)
    

    def test_pyramid_pooling (self):
        """ Test the imgpatches.pyramid_pooling function. """

        pyr = imp.pyramid_pooling(self.tpatch, self.tx, self.ty,
                                  self.timg.shape, (1,2), imp.p_maxabs)
        self.assertTrue((self.tpyr == pyr).all())


    def test_norm_patches (self):
        """ Test patch contrast normalisation/unitisation. """
       
        diff = abs(ones((self.tpatch.shape[0],1)) 
                - sqrt((imp.norm_patches(self.tpatch)**2).sum(axis=1)))
        self.assertTrue((diff < 1e-15).all())
    
        
    def test_centre_patches (self):
        """ Test patch contrast normalisation/unitisation. """
        
        self.assertTrue((zeros((self.tpatch.shape[0],1)) ==
                imp.centre_patches(self.tpatch).mean(axis=1)).all())


if __name__ == '__main__':
    unittest.main()

