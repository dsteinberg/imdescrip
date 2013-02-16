import math
import numpy as np
from skimage.color import rgb2gray
from clint.textui import progress
from vlfeat import vl_dsift
from image import imread_resize


def training_patches (imnames, npatches, psize, maxdim=None):
    """ Extract SIFT patches from images for dictionary training

    Arguments:
        imnames: A list of image names from which to extract training patches.
        npatches: The number (int) of patches to extract from the images
        maxdim: The maximum dimension of the image in pixels. The image is
            rescaled if it is larger than this. By default there is no scaling. 
        psize: A int of the size of the square patches to extract

    Returns:
        An np.array (npatches, 128) of SIFT descriptors. NOTE, the actual 
        npatches found may be slightly more or less than that input.

    """

    nimg = len(imnames)
    ppeimg = int(round(float(npatches)/nimg))
    plist = []
    bsize = __patch2bin(psize)

    print('Extracting SIFT patches from images...')
    for ims in progress.bar(imnames):
        
        # Read in and resize the image -- convert to gray if needed
        img = imread_resize(ims, maxdim) 
        if img.ndim > 2:
            img = rgb2gray(img)

        # Extract the patches
        spaceing = int(math.floor(math.sqrt(float(np.prod(img.shape))/ppeimg)))
        xy, desc = vl_dsift(np.float32(img), step=spaceing, size=bsize)
        plist.append(desc.T)

    patches = np.concatenate(plist, axis=0)

    return np.reshape(patches, (patches.shape[0], np.prod(patches.shape[1:])))


def DSIFT_patches (image, psize, pstride):
    """ Extract dense SIFT descriptors from an image.
    """

    if image.ndim > 2:
        image = rgb2gray(image)

    xy, desc = vl_dsift(np.float32(image), step=pstride, 
                        size=__patch2bin(psize))

    return desc.T, xy[0,:], xy[1,:]


def __patch2bin (psize):
    """ Convert image patch size to SIFT bin size as expected by VLFeat. """

    return int(round(float(psize)/4)) + 1
