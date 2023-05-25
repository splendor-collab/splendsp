from ._version import version as __version__
from .dsp import *
from .cut import *
from .plot import *

# load seaborn colormaps
from seaborn import cm
del cm
