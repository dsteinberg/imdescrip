#! /usr/bin/env python

import glob 
from extractors.extractor import extractor
from descriptors.testdesc import TestDesc
#from descriptors.ScSPM import ScSPM

imgdir   = '/home/dsteinberg/Datasets/outdoor_scenes/images_40ea/'
savdir   = '/home/dsteinberg/Datasets/outdoor_scenes/images_40ea/tmp/'
filelist = glob.glob(imgdir + '*.jpg') 

desc = TestDesc()
#desc = ScSPM()

extractor(filelist, savdir, desc) 
