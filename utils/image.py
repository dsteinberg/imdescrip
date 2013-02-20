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

""" Some useful and generic commonly performed image operations. """

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
        scaler = float(maxdim)/imgdim
        return resize(image, (round(scaler*image.shape[0]), 
                              round(scaler*image.shape[1])))
    else:
        return image
