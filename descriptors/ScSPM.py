from descriptor import Descriptor
from skimage import color 

import matplotlib.pyplot as plt

class ScSPM (Descriptor):

    def __init__ (self, maximgdim=320, patchsize=16, patchstride=8):
        
        self.maximgdim   = maximgdim
        self.patchsize   = patchsize
        self.patchstride = patchstride 


    def extract (self, image):
        
        gimg = color.rgb2gray(image)

        #plt.imshow(hog_image, cmap=plt.cm.gray)
        #plt.show()

        return fea
