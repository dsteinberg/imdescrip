# Image Descriptor Extraction Module

import os
import cPickle as pk
from clint.textui import progress
from scipy import misc

def extractor (filelist, savedir, descobj):
    """ Extract features/descriptors from a batch of images. 

    This function calls an image descripor object on a batch of imaged in order
    to extract the images descripor. If a feature/descriptor file already exists
    for the image, it is skipped.
    
    Arguments:
        filelist: A list of files of image names including their paths of images
                  to read and extract descriptors from

        savedir:  A directory in which to save all of the image features. They
                  are pickled objects (protocol 2) with the same name as the
                  image file. The object that is pickled is the return from
                  descobj.extract().

        decobj:   An image descriptor object which does the actual extraction
                  work. the method called is descobj.extract(image). See
                  descriptors.Descriptor for an abstract base class. 

    """

    # Try to make the save path
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    

    # Iterate through all of the images in filelist and extract features
    for impath in progress.bar(filelist):

        imname = os.path.splitext(os.path.split(impath)[1])[0] # get image name
        feafile = os.path.join(savedir, imname + ".p")
        
        # Check to see if feature file already exists, continue if so
        if os.path.isfile(feafile) == True:
            continue

        # Read and extract image descriptors
        img = misc.imread(impath) # read in the image
        fea = descobj.extract(img) # extract image descriptor
   
        # Write pickled feature
        with open(feafile, 'wb') as f:
            pk.dump(fea, f, protocol=2)
