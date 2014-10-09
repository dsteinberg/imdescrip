# Imdescrip -- a collection of tools to extract descriptors from images.
# Copyright (C) 2013 Daniel M. Steinberg (daniel.m.steinberg@gmail.com)
#
# This file is part of Imdescrip.
#
# Imdescrip is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# Imdescrip is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License
# for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Imdescrip. If not, see <http://www.gnu.org/licenses/>.

# Simple image descriptor for just testing stuff
from descriptor import Descriptor


class TestDesc (Descriptor):

    def __init__(self):
        pass

    def extract(self, image):
        return (image.mean(axis=0)).mean(axis=0)
