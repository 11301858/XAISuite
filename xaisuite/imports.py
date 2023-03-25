#Needed Libraries

#Data Storage, Export, Access, and Visualization
import pandas as pd
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import ast
import numpy as np
from sklearn.datasets import*

#Machine Learning Models and Training
from sklearn.model_selection import train_test_split

#I decided to trade-off memory for security. This is a list of all accepted models. Without this, the program works perfectly, but malicious users could hijack the system to execute their own code.
acceptedModels = {"SVC": "sklearn.svm", "NuSVC": "sklearn.svm", "LinearSVC": "sklearn.svm", "SVR": "sklearn.svm", "NuSVR": "sklearn.svm", "LinearSVR": "sklearn.svm", 
                 "AdaBoostClassifier": "sklearn.ensemble", "AdaBoostRegressor": "sklearn.ensemble", "BaggingClassifier": "sklearn.ensemble", "BaggingRegressor": "sklearn.ensemble",
                 "ExtraTreesClassifier": "sklearn.ensemble", "ExtraTreesRegressor": "sklearn.ensemble", 
                 "GradientBoostingClassifier": "sklearn.ensemble", "GradientBoostingRegressor": "sklearn.ensemble",
                 "RandomForestClassifier": "sklearn.ensemble", "RandomForestRegressor": "sklearn.ensemble",
                 "StackingClassifier": "sklearn.ensemble", "StackingRegressor": "sklearn.ensemble",
                 "VotingClassifier": "sklearn.ensemble", "VotingRegressor": "sklearn.ensemble",
                 "HistGradientBoostingClassifier": "sklearn.ensemble", "HistGradientBoostingRegressor": "sklearn.ensemble",
                 "GaussianProcessClassifier": "sklearn.gaussian_process", "GaussianProcessRegressor": "sklearn.gaussian_process", 
                 "IsotonicRegression": "sklearn.isotonic", "KernelRidge": "sklearn.kernel_ridge", 
                 "LogisticRegression": "sklearn.linear_model", "LogisticRegressionCV": "sklearn.linear_model",
                 "PassiveAgressiveClassifier": "sklearn.linear_model", "Perceptron": "sklearn.linear_model", 
                  "RidgeClassifier": "sklearn.linear_model", "RidgeClassifierCV": "sklearn.linear_model",
                  "SGDClassifier": "sklearn.linear_model", "SGDOneClassSVM": "sklearn.linear_model", 
                  "LinearRegression": "sklearn.linear_model", "Ridge": "sklearn.linear_model", 
                  "RidgeCV": "sklearn.linear_model", "SGDRegressor": "sklearn.linear_model",
                  "ElasticNet": "sklearn.linear_model", "ElasticNetCV": "sklearn.linear_model",
                  "Lars": "sklearn.linear_model", "LarsCV": "sklearn.linear_model", 
                  "Lasso": "sklearn.linear_model", "LassoCV": "sklearn.linear_model",
                  "LassoLars": "sklearn.linear_model", "LassoLarsCV": "sklearn.linear_model",
                  "LassoLarsIC": "sklearn.linear_model", "OrthogonalMatchingPursuit": "sklearn.linear_model",
                  "OrthogonalMatchingPursuitCV": "sklearn.linear_model", "ARDRegression": "sklearn.linear_model",
                  "BayesianRidge": "sklearn.linear_model", "MultiTaskElasticNet": "sklearn.linear_model", 
                  "MultiTaskElasticNetCV": "sklearn.linear_model", "MultiTaskLasso": "sklearn.linear_model",
                  "MultiTaskLassoCV": "sklearn.linear_model", "HuberRegressor": "sklearn.linear_model",
                  "QuantileRegressor": "sklearn.linear_model", "RANSACRegressor": "sklearn.linear_model",
                  "TheilSenRegressor": "sklearn.linear_model", "PoissonRegressor": "sklearn.linear_model",
                  "TweedieRegressor": "sklearn.linear_model", "GammaRegressor": "sklearn.linear_model", 
                  "PassiveAggressiveRegressor": "sklearn.linear_model", "BayesianGaussianMixture": "sklearn.mixture",
                  "GaussianMixture": "sklearn.mixture", 
                  "OneVsOneClassifier": "sklearn.multiclass", "OneVsRestClassifier": "sklearn.multiclass", 
                  "OutputCodeClassifier": "sklearn.multiclass", "ClassifierChain": "sklearn.multioutput", 
                   "RegressorChain": "sklearn.multioutput",  "MultiOutputRegressor": "sklearn.multioutput",
                   "MultiOutputClassifier": "sklearn.multioutput", "BernoulliNB": "sklearn.naive_bayes", 
                  "CategoricalNB": "sklearn.naive_bayes", "ComplementNB": "sklearn.naive_bayes", 
                  "GaussianNB": "sklearn.naive_bayes", "MultinomialNB": "sklearn.naive_bayes", 
                  "KNeighborsClassifier": "sklearn.neighbors", "KNeighborsRegressor": "sklearn.neighbors", 
                  "BernoulliRBM": "sklearn.neural_network", "MLPClassifier": "sklearn.neural_network", "MLPRegressor": "sklearn.neural_network",  
                  "DecisionTreeClassifier": "sklearn.tree", "DecisionTreeRegressor": "sklearn.tree",
                  "ExtraTreeClassifier": "sklearn.tree", "ExtraTreeRegressor": "sklearn.tree", "NullRegressor": "models.NullRegressor",
                  "NeuralNetClassifier": "skorch.NeuralNetClassifier", "KerasClassifier": "scikeras.wrappers.KerasClassifier", "KerasRegressor": "scikeras.wrappers.KerasRegressor"
                 }

from sklearn.base import is_classifier, is_regressor #For model type identification. Necessary for explanation generation and to ensure model is not unsupervised

#OmniXAI Explanatory Models and Visualization
from omnixai.data.tabular import Tabular
from sklearn.preprocessing import*
from omnixai.preprocessing.base import Identity
from omnixai.preprocessing.tabular import TabularTransform
from omnixai.explainers.tabular import TabularExplainer
from omnixai.visualization.dashboard import Dashboard

#For documentation
from typing import Union, Dict
