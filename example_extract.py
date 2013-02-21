#! /usr/bin/env python

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

import glob 
import cPickle as pk
from extractors.extractor import extract_smp
#from descriptors.ScSPM import ScSPM

# Make a list of images
#imgdir   = '/home/dsteinberg/Datasets/Tas2008_5/Images/'
#savdir   = '/home/dsteinberg/Datasets/Tas2008_5/Images/imdesc2/'
#filelist = glob.glob(imgdir + '*.png') 

imgdir   = '/home/dsteinberg/Datasets/outdoor_scenes/images_40ea/'
savdir   = '/home/dsteinberg/Datasets/outdoor_scenes/images_40ea/imdesc/'
filelist = glob.glob(imgdir + '*.jpg') 

# Train a dictionary
#desc = ScSPM(dsize=512, compress_dim=3000)
#desc.learn_dictionary(filelist, 50000)

# Save the dictionary
#with open('sc_auv2.p', 'wb') as f:
    #pk.dump(desc, f, protocol=2)

# OR Load a pre-learned dictionary 
with open('sc_auv.p', 'rb') as f:
    desc = pk.load(f)

extract_smp(filelist, savdir, desc, verbose=True) 
