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

""" Module for single threaded image descriptor extraction. """

import os
import cPickle as pk
from clint.textui import progress

def extractor (filelist, savedir, descobj, verbose=False):
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
        verbose:  bool, display progress?

    Note: If there is a problem extracting any image descriptors, a file
        "errors.log" is created in the savedir directory with a list of file
        names, error number and messages.

    """

    # Try to make the save path
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    errlog = os.path.join(savedir, 'errors.log')
    errflag = False

        # Iterate through all of the images in filelist and extract features
    if verbose == True:
        print('Extracting image descriptors...')

    for impath in progress.bar(filelist, hide=not verbose):

        imname = os.path.splitext(os.path.split(impath)[1])[0] # get image name
        feafile = os.path.join(savedir, imname + ".p")
        
        # Check to see if feature file already exists, continue if so
        if os.path.isfile(feafile) == True:
            continue

        # Extract image descriptors
        try:
            fea = descobj.extract(impath) # extract image descriptor
        except Exception as e:
            with open(errlog, 'a') as l:
                l.write(impath + ' : ' + format(e.errno, e.strerror) + '\n')
            continue
   
        # Write pickled feature
        with open(feafile, 'wb') as f:
            pk.dump(fea, f, protocol=2)
    
    if (verbose == True) and (errflag == False):
        print('done!')
    elif errflag == True:
        print('done with errors. See the "errors.log" file in ' + savedir)
