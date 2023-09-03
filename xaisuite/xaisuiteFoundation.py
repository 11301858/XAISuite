#Needed Libraries

#For documentation
from typing import*
del globals()["Text"]

#Data Storage, Export, and Access
import pandas as pd
import csv
import os

from omnixai.data.tabular import*
from omnixai.data.image import*
from omnixai.data.text import*
from omnixai.preprocessing.tabular import*
from omnixai.preprocessing.image import*
from omnixai.preprocessing.text import*


import numpy
from sklearn.datasets import*

#Visualization
import seaborn as sns
import matplotlib.pyplot as plt

#Machine Learning Models and Training
from sklearn.model_selection import train_test_split
from sklearn.metrics import*


linkModels = {"SVC": "sklearn.svm", "NuSVC": "sklearn.svm", "LinearSVC": "sklearn.svm", "SVR": "sklearn.svm", "NuSVR": "sklearn.svm", "LinearSVR": "sklearn.svm", 
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
                  "NeuralNetClassifier": "skorch.NeuralNetClassifier", "KerasClassifier": "scikeras.wrappers", "KerasRegressor": "scikeras.wrappers", 
                  "CCA": "sklearn.cross_decomposition", "DummyRegressor": "sklearn.dummy", "PLSCanonical": "sklearn.cross_decomposition", "PLSRegression": "sklearn.cross_decomposition", 
                  "RadiusNeighborsRegressor": "sklearn.neighbors", "RadiusNeighborsClassifier": "sklearn.neighbors", "TransformedTargetRegressor": "sklearn.compose", "CalibratedClassifierCV": "sklearn.calibration", 
                  "DummyClassifier": "sklearn.dummy", "LinearDiscriminantAnalysis": "sklearn.discriminant_analysis", "Perceptron": "sklearn.linear_model", "QuadraticDiscriminantAnalysis": "sklearn.discriminant_analysis"  
                 }

#OmniXAI Explanatory Models
from omnixai.explainers.tabular import*
from omnixai.explainers.vision import*
from omnixai.explainers.nlp import*

#For calculations
from math import*
