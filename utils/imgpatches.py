# A few image patch extraction routines

from sklearn.feature_extraction.image import extract_patches_2d
from skimage.color import rgb2gray
from skimage.transform import resize
from scipy.misc import imread, toimage
from numpy import concatenate, prod, reshape, zeros
from clint.textui import progress
from math import sqrt, ceil, floor
from matplotlib import pyplot as plt

def training_patches (imnames, npatches, psize, maxdim=None, colour=False):
    """ Extract patches from images for dictionary training

    Args:
        imnames: A list of image names from which to extract training patches.
        npatches: The number (int) of patches to extract from the images
        maxdim: The maximum dimension of the image in pixels. The image is
            rescaled if it is larger than this. By default there is no scaling. 
        psize: A int of the size of the square patches to extract

    Returns:
        An array (npatches, psize**2*3) for RGB or (npatches, psize**2) for grey
        of flattened image patches. NOTE, the actual npatches found may be less
        than that input.

    """

    nimg = len(imnames)
    ppeimg = int(round(float(npatches)/nimg))
    plist = []

    print('Extracting patches from images...')
    for ims in progress.bar(imnames):
        img = imread_resize(ims) # read in and resize the image
        
        # Extract patches and map to grayscale if necessary
        if (colour == False) and (img.shape[2] == 3):
            imgg = rgb2gray(img)
            plist.append(extract_patches_2d(imgg, (psize, psize), ppeimg))
        else:
            plist.append(extract_patches_2d(img, (psize, psize), ppeimg))

    patches = concatenate(plist, axis=0)

    return reshape(patches, (patches.shape[0], prod(patches.shape[1:])))


def grid_patches (image, psize, pstride):
    """ Extract a grid of (overlapping) patches from an image

    This function extracts square patches from an image in an potentially
    overlapping, dense grid. 

    Arguments:
        image: array (rows, cols, channels) of an image (in memory)
        psize: int the size of the square patches to extract, in pixels.
        pstride: int the stride (in pixels) between successive patches.

    Returns:
        patches: array (npatches, psize**2*channels) the flattened (per row)
            image patches.
        centresx: array (npatches) the centres (column coords) of the patches
        centresy: array (npatches) the centres (row coords) of the patches

    """

    # Check and get image dimensions
    if image.ndim == 3:
        (Iw, Ih, Ic) = image.shape
    elif image.ndim == 2:
        (Iw, Ih) = image.shape
        Ic = 1
    else:
        raise ValueError('image must be a 2D or 3D array')
        

    # Make the overlapping grid
    offsetX = int(floor(float((Iw - psize) % pstride)/2))
    offsetY = int(floor(float((Ih - psize) % pstride)/2))
    spaceX = range(offsetX, Iw-psize+1, pstride)
    spaceY = range(offsetY, Ih-psize+1, pstride)
    npatches = len(spaceX)*len(spaceY)

    # Pre-allocate the returns
    rsize = (psize**2)*Ic
    patches = zeros((npatches, rsize))
    centresy = zeros(npatches)
    centresx = zeros(npatches)

    # Extract the patches and get the patch centres
    cnt = 0
    for sy in spaceY:           # Rows
        gridY = range(sy, sy+psize) 
        
        for sx in spaceX:       # Cols
            gridX = range(sx, sx+psize) 

            patches[cnt,:] = reshape(image[gridY][:,gridX], rsize)
            centresy[cnt] = sy + float(psize)/2 - 0.5;
            centresx[cnt] = sx + float(psize)/2 - 0.5;

            cnt +=1

    return patches, centresx, centresy
    

def disp_patches (patches, colour=False):
    """ Display flattened (square) patches in a grid.

    This function is best for displaying the bases of learned dictionaries.

    Arguments:
        patches: (npatches, pixels) is a numpy array of all of the flattened
            image patches in each row. These will automatically be scalled to be
            displayed as images. It is assumed the original patches are square
        colour: boolean flag indicating whether or not these patches are
            supposed to be colour or not.

    """

    # Argument checking
    if (colour == False) and (sqrt(patches.shape[1])%1 != 0):
        raise ValueError('Gray image has to have square patches')
    elif (colour == True) and (sqrt(float(patches.shape[1])/3)%1 != 0):
        raise ValueError('Colour image has to have square patches')

    # Get patch size
    if colour == False:
        psize = sqrt(patches.shape[1])  
    else: 
        psize = sqrt(patches.shape[1]/3)
    
    ssize = ceil(sqrt(patches.shape[0]))

    # plot filters
    plt.figure()
    for i, p in enumerate(patches):
        plt.subplot(ssize, ssize, i + 1)
        if colour == False:
            plt.imshow(p.reshape(psize, psize), cmap="gray")
        else:
            plt.imshow(toimage(p.reshape(psize, psize, 3), mode='RGB'))
        plt.axis("off")
    plt.show()


def imread_resize(imname, maxdim=None):
    """ Read and resize the and image to a maximum dimension (preserving aspect)

    Arguments:
        imname: string of the full name and path to the image to be read
        maxdim: int of the maximum dimension the image should take (in pixels).
            None if no resize is to take place (same as imread).

    Returns:
        image: (rows, cols, channels) array of the image, if maxdim is not None,
            then {rows,cols} <= maxdim.
    """
    # read in the image
    image = imread(imname)         

    # Resize image if necessary
    imgdim = max(image.shape)
    if (imgdim > maxdim) and (maxdim is not None):
        scaler = float(imgdim)/maxdim
        return resize(image, (round(scaler*image.shape[0]), 
                              round(scaler*image.shape[1])))
    else:
        return image
