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

""" Module for image descriptor extraction. """

#TODO: Seems to be a bug where existing files are overwritten OR there is just a
#   thread sync error -- it takes long if files exist or it may recalculate them 

import os, itertools, sys, cPickle, time
import multiprocessing as mp
from clint.textui import progress


def extract (imfile, savedir, descobj):
    """
    """
    
    imname = os.path.splitext(os.path.split(imfile)[1])[0] # get image name
    feafile = os.path.join(savedir, imname + ".p")

    # Check to see if feature file already exists, continue if so
    if os.path.exists(feafile) == True:
        return False 

    # Extract image descriptors
    try:
        fea = descobj.extract(imfile) # extract image descriptor
    except Exception as e:
        with open(os.path.join(savedir, 'errors.log'), 'a') as l:
            l.write(imfile + ' : ' + format(e.errno, e.strerror) + '\n')
        return True

    # Write pickled feature
    with open(feafile, 'wb') as f:
        cPickle.dump(fea, f, protocol=2)

    return False


def extract_batch (filelist, savedir, descobj, verbose=False):
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

    errflag = False

    # Iterate through all of the images in filelist and extract features
    if verbose == True:
        print('Extracting image descriptors...')

    for impath in progress.bar(filelist, hide=not verbose):
        errflag |= extract(impath, savedir, descobj) 
    
    if (verbose == True) and (errflag == False):
        print('done!')
    elif errflag == True:
        print('done with errors. See the "errors.log" file in ' + savedir)


def __extract_star (args):
    """ Covert args to (file, savedir, descobj) arguments. """
    return extract(*args)


def extract_smp (filelist, savedir, descobj, njobs=None, verbose=False):
    """
    """

    # Try to make the save path
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    if verbose == True:
        print('Extracting image descriptors...')
  
    # Set up parallel job
    pool = mp.Pool(processes=njobs)

    # Iterate through all of the images in filelist and extract features
    result = pool.map_async(__extract_star, itertools.izip(filelist, 
                    itertools.repeat(savedir), itertools.repeat(descobj)))

    # Get the status
    while (result.ready() is not True):
        sys.stdout.write('\rNumber of images remaining: ' +
                    str(result._number_left * result._chunksize) + '\t\t') 
        sys.stdout.flush()
        time.sleep(5)

    # Get notification of errors
    errflag = any(result.get())
    pool.close()
    pool.join()

    if (verbose == True) and (errflag == False):
        print('\ndone!')
    elif errflag == True:
        print('\ndone with errors. See the "errors.log" file in ' + savedir)
