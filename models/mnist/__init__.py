from __future__ import absolute_import

"""The models subpackage contains definitions for the mnist
architectures:

-  `convnet:3C3D`_
- 'fc: 6D'
- 'bn'
- 'toy':3C1D, to plot Fisher inverse
"""

from .convnet import *
from .fc import *
from .bn import *
from .toy import *
from .autoencoder import *