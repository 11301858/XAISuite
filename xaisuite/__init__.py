#For creating package
from pkg_resources import get_distribution, DistributionNotFound

try:
    dist = get_distribution("XAISuite")
except DistributionNotFound:
    __version__ = "Please install XAISuite with setup.py"
else:
    __version__ = dist.version

#Package Modules
from .dataLoader import load_data_CSV, load_data_sklearn
from .hub import train_and_explainModel
from .analyzer import compare_explanations, compare_explanationssinglef, maxImportanceScoreGenerator
