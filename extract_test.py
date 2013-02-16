#! /usr/bin/env python

import glob 
import cPickle as pk
from extractors.extractor import extractor
#from descriptors.testdesc import TestDesc
#from descriptors.ScSPM import ScSPM

imgdir   = '/home/dsteinberg/Datasets/outdoor_scenes/Images/'
savdir   = '/home/dsteinberg/Datasets/outdoor_scenes/Images/imdesc/'
filelist = glob.glob(imgdir + '*.jpg') 

#desc = TestDesc()
#desc = ScSPM()

with open('sc.p', 'rb') as f:
    desc = pk.load(f)

extractor(filelist, savdir, desc) 
