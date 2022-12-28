#Needed Libraries

#Data Storage, Export, Access, and Visualization
import pandas as pd
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import ast

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

acceptedModels = {"SVC": "sklearn.svm.SVC", "NuSVC": "sklearn.svm.NuSVC", "LinearSVC": "sklearn.svm.LinearSVC", "SVR": "sklearn.svm.SVR", "NuSVR": "sklearn.svm.SVR", "LinearSVR": "sklearn.svm.LinearSVR", 
                 "AdaBoostClassifier": "sklearn.ensemble.AdaBoostClassifier", "AdaBoostRegressor": "sklearn.ensemble.AdaBoostRegressor", "BaggingClassifier": "sklearn.ensemble.BaggingClassifier", "BaggingRegressor": "sklearn.ensemble.BaggingRegressor",
                 "ExtraTreesClassifier": "sklearn.ensemble.ExtraTreesClassifier", "ExtraTreesRegressor": "sklearn.ensemble.ExtraTreesRegressor", 
                 "GradientBoostingClassifier": "sklearn.ensemble.GradientBoostingClassifier", "GradientBoostingRegressor": "sklearn.ensemble.GradientBoostingRegressor",
                 "RandomForestClassifier": "sklearn.ensemble.RandomForestClassifier", "RandomForestRegressor": "sklearn.ensemble.RandomForestRegressor",
                 "StackingClassifier": "sklearn.ensemble.StackingClassifier", "StackingRegressor": "sklearn.ensemble.StackingRegressor",
                 "VotingClassifier": "sklearn.ensemble.VotingClassifier", "VotingRegressor": "sklearn.ensemble.VotingRegressor",
                 "HistGradientBoostingClassifier": "sklearn.ensemble.HistGradientBoostingClassifier", "HistGradientBoostingRegressor": "sklearn.ensemble.HistGradientBoostingRegressor",
                 "GaussianProcessClassifier": "sklearn.gaussian_process.GaussianProcessClassifier", "GaussianProcessRegressor": "sklearn.gaussian_process.GaussianProcessRegressor", 
                 "IsotonicRegression": "sklearn.isotonic.IsotonicRegression", "KernelRidge": "sklearn.kernel_ridge.KernelRidge", 
                 "LogisticRegression": "sklearn.linear_model.LogisticRegression", "LogisticRegressionCV": "sklearn.linear_model.LogisticRegressionCV",
                 "PassiveAgressiveClassifier": "sklearn.linear_model.PassiveAggressiveClassifier", "Perceptron": "sklearn.linear_model.Perceptron", 
                  "RidgeClassifier": "sklearn.linear_model.RidgeClassifier", "RidgeClassifierCV": "sklearn.linear_model.RidgeClassifierCV",
                  "SGDClassifier": "sklearn.linear_model.SGDClassifier", "SGDOneClassSVM": "sklearn.linear_model.SGDOneClassSVM", 
                  "LinearRegression": "sklearn.linear_model.LinearRegression", "Ridge": "sklearn.linear_model.Ridge", 
                  "RidgeCV": "sklearn.linear_model.RidgeCV", "SGDRegressor": "sklearn.linear_model.SGDRegressor",
                  "ElasticNet": "sklearn.linear_model.ElasticNet", "ElasticNetCV": "sklearn.linear_model.ElasticNetCV",
                  "Lars": "sklearn.linear_model.Lars", "LarsCV": "sklearn.linear_model.LarsCV", 
                  "Lasso": "sklearn.linear_model.Lasso", "LassoCV": "sklearn.linear_model.LassoCV",
                  "LassoLars": "sklearn.linear_model.LassoLars", "LassoLarsCV": "sklearn.linear_model.LassoLarsCV",
                  "LassoLarsIC": "sklearn.linear_model.LassoLarsIC", "OrthogonalMatchingPursuit": "sklearn.linear_model.OrthogonalMatchingPursuit",
                  "OrthogonalMatchingPursuitCV": "sklearn.linear_model.OrthogonalMatchingPursuitCV", "ARDRegression": "sklearn.linear_model.ARDRegression",
                  "BayesianRidge": "sklearn.linear_model.BayesianRidge"
                 }

from sklearn.base import is_classifier, is_regressor

#OmniXAI Explanatory Models and Visualization
from omnixai.data.tabular import Tabular
from sklearn.preprocessing import*
from omnixai.preprocessing.base import Identity
from omnixai.preprocessing.tabular import TabularTransform
from omnixai.explainers.tabular import TabularExplainer
from omnixai.visualization.dashboard import Dashboard

#For documentation
from typing import Union
