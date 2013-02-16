# Generic (abstract base) descriptor object to inherit

import abc

class Descriptor:
    """ This class is the abstract base class for image descriptor objects. 
    
        If this interface is followed, the image descriptor objects should work
        with the batch image descriptor extractors.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def extract (self, image):
        """ Method required for actual descriptor extraction. 
        
            This method should accept an image file name and should return an
            object/array which is the actual feature.
        """
        pass


