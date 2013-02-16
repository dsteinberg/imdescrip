""" TODO
"""

from skimage.transform import resize
from scipy.misc import imread


def imread_resize (imname, maxdim=None):
    """ Read and resize the and image to a maximum dimension (preserving aspect)

    Arguments:
        imname: string of the full name and path to the image to be read
        maxdim: int of the maximum dimension the image should take (in pixels).
            None if no resize is to take place (same as imread).

    Returns:
        image: (rows, cols, channels) np.array of the image, if maxdim is not
            None, then {rows,cols} <= maxdim.  
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
