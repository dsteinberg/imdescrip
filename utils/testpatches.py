#! /usr/bin/env python

# Unit test for imgpatches module

import imgpatches as imp
from numpy import array, arange, vstack
import unittest 

class TestImgPatches (unittest.TestCase):

    def setUp (self):
        """
        """
       
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


    def test_grid_patches (self):
        """
        """

        # Make sure the outputs are the right shape and values
        patches, x, y = imp.grid_patches(self.timg, self.psize, self.pstride)
        self.assertTrue((patches == self.tpatch).all())
        self.assertTrue((x == self.tx).all())
        self.assertTrue((y == self.ty).all())

        # Make sure the function errors out when we expect it too
        self.assertRaises(ValueError, imp.grid_patches, array([1, 1, 1, 1]),
                          self.psize, self.pstride)


if __name__ == '__main__':
    unittest.main()

