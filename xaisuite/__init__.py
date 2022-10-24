from pkg_resources import get_distribution, DistributionNotFound

try:
    dist = get_distribution("XAISuite")
except DistributionNotFound:
    __version__ = "Please install XAISuite with setup.py"
else:
    __version__ = dist.version
    
#Needed Libraries

#Data Storage, Export, and Access
import pandas as pd
import csv
from pathlib import Path 

#Machine Learning Models and Training
from sklearn.svm import*
from sklearn.ensemble import*
from sklearn.gaussian_process import*
from sklearn.isotonic import*
from sklearn.kernel_ridge import*
from sklearn.linear_model import*
from sklearn.mixture import*
from sklearn.multiclass import*
from sklearn.multioutput import*
from sklearn.naive_bayes import*
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import*
from sklearn.tree import*
from sklearn.model_selection import train_test_split

from sklearn.base import is_classifier, is_regressor

#OmniXAI Explanatory Models and Visualization
from omnixai.data.tabular import Tabular
from sklearn.preprocessing import StandardScaler
from omnixai.preprocessing.base import Identity
from omnixai.preprocessing.tabular import TabularTransform
from omnixai.explainers.tabular import TabularExplainer
from omnixai.visualization.dashboard import Dashboard

#For documentation
from typing import Union

#Package Modules
from .dataLoader import load_data_CSV, load_data_sklearn
from .xaichooser import train_and_explainModel
from .analyzer import analyze_explanations
