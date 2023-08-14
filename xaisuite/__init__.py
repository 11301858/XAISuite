#For creating package
from pkg_resources import get_distribution, DistributionNotFound

try:
    dist = get_distribution("XAISuite")
except DistributionNotFound:
    __version__ = "Please install XAISuite with setup.py"
else:
    __version__ = dist.version

#Package Modules
from .dataHandler import*
from .explainableModel import*
from .insightGenerator import*
from .xaisuiteFoundation import*
from models import*
from explainers import*

