Imdescrip
=========

A collection of python tools for extracting descriptors from images (whole and
sub-image descriptors).


*Author*: Daniel Steinberg

*Institute*: Australian Centre for Field Robotics, The University of Sydney

*Date*: 20/02/2013

*License*: GPL v3 (See LICENSE)

*References*:

 [1] Yang, J.; Yu, K.; Gong, Y. & Huang, T. Linear spatial pyramid matching
     using sparse coding for image classification Computer Vision and Pattern
     Recognition, 2009. CVPR 2009. IEEE Conference on, 2009, 1794-1801

 [2] D. M. Steinberg, An Unsupervised Approach to Modelling Visual Data, PhD
     Thesis, 2013.

**If you use this package please consider citing [2]**


Functionality
-------------

It's probably easiest to describe what this package can do by breaking down each
sub-folder/package.


### descriptors:

Actual classes for extracting descriptors/features from images. For instance, a
modified versions of Yang et. al.'s sparse code spatial pyramid matching (ScSPM)
[1] descriptor is implemented here. Also provided is an abstract base class for
implementing new descriptor classes that work with the extractor model.


### extractors:

Routines for batch processing images for descriptor extraction. These routines
call the classes in descriptors to actually extract the descriptors from the
images. The extracted descriptors are then saved to individual binary pickled
objects. 


### utils:

Various utilities used by the other modules. These include:

* spatial pyramid pooling (with arbitrary pooling functions, e.g. max and mean)
* dense grid patch extraction (image and SIFT patches)
* training patch (image and SIFT) extraction from a list of images. Useful for
  training dictionaries.
* patch centring and contrast normalisation.
* image reading and resizing in a single routine.


### test:

Unit tests for this package.


Dependencies
------------

See requirements.txt. Some of the requirements are commented out just because
they are pretty common with a python install (at least for likely users of this
package), such as Scipy and Numpy.


Installation
------------

See requirements.txt for details.

Largely up to you -- there are some executable scripts that can be run from the
root package folder, or you can add this to the python path.

Pip should handle most of the dependencies:
    
    # pip install -r requirements.txt

Where '#' means super-user (use sudo for Ubuntu). Just note that some of the
more common dependencies are commented out. These can be installed with pip,
thought you may have more luck using a maintained package for your system.

This has not been tested on a Windows or Mac system (only Ubuntu Linux)

NOTE: I had difficulty with pyvlfeat being essentially un-maintained, and so
this would not install without error on Ubuntu 12.04. I have instructions in the
requirements.txt file for getting this working.


Usage
-----

Have a look at "example\_extract.py" for some usage examples, and how I would
use the ScSPM descriptor. Typically a work flow consists of:

1. Instantiating and training a descriptor object (e.g. ScSPM)
2. Saving this object with pickle.
3. Loading this descriptor object when a dataset needs to be processed.
4. Calling an extractor routine with this descriptor object on a list of images.

Of course (2) and (3) are optional, but save unnecessary ScSPM dictionary
training.


TODO
----

* Update install instructions when I've tried it out on a new machine
* Remove some dependencies (clint and sklearn) 
* Re-write the python interface for vlfeat DSIFT (pyvlfeat is unmaintained)
* Make image resizing faster
