# Image Descriptor Extraction Module

import os
import cPickle as cpk
from clint.textui import progress
from scipy import misc

def extractor (filelist, savedir, descobj):
    """ Extract features/descriptors from a batch of images. 

    This function calls an image descripor object on a batch of imaged in order
    to extract the images descripor.

    
    Args:
        filelist: A list of files of image names including their paths of images
                  to read and extract descriptors from

        savedir:  A directory in which to save all of the image features. They
                  are pickled objects with the same name as the image file. The
                  object that is pickled is the return from descobj.extract().

        decobj:   An image descriptor object which does the actual extraction
                  work. the method called is descobj.extract(image). See
                  descriptors.Descriptor for an abstract base class. 

    """

    # Try to make the save path
    if not os.path.exists(savedir):
        os.makedev(savedir)
    

    # Iterate through all of the images in filelist and extract features
    for inames in progress.bar(filelist):

        # TODO: check to see if feature file already exists!!!!

        # Read and extract image descriptors
        img = misc.imread(inames) # read in the image
        fea = descobj.extract(img) # extract image descriptor
   
        # Write pickled feature
        imsname = os.path.splitext(os.path.split(inames)[1])[0] # get image name
        with open(os.path.join(savedir, imsname+".p"), 'wb') as f:
            cpk.dump(fea, f)
