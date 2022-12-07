# Make reproducible - https://stackoverflow.com/questions/61559333/neural-network-why-is-my-code-not-reproducible
import os
import random
import numpy as np
import warnings

SEED = 1
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED'] = str(SEED)
# 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(SEED)
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(SEED)

# Imports
import csv
import os
import sys
import itertools
import pandas as pd
# from pandas.plotting import autocorrelation_plot

import numpy as np

import sklearn
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import plotly.graph_objects as go
import seaborn as sns

from datetime import datetime
import  time
# import mplfinance as mpf
import pickle
from collections import Counter
import pgeocode

from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

import category_encoders as ce

from xgboost import XGBRegressor
