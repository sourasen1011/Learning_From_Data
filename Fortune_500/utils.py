# Make reproducible - https://stackoverflow.com/questions/61559333/neural-network-why-is-my-code-not-reproducible
import os
import random
import numpy as np
seed_value = 1
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED'] = str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

#-----------------------------Keras reproducible------------------#
import tensorflow as tf
import keras
from keras import backend as K

SEED = 1

tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)

session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1, 
    inter_op_parallelism_threads=1
)
sess = tf.compat.v1.Session(
    graph=tf.compat.v1.get_default_graph(), 
    config=session_conf
)
K.set_session(sess)

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

# from statsmodels.tsa.seasonal import seasonal_decompose
# from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm
# from statsmodels.tsa.holtwinters import ExponentialSmoothing
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from statsmodels.graphics.tsaplots import plot_acf
# from statsmodels.graphics.tsaplots import plot_pacf
# from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor


import category_encoders as ce

from xgboost import XGBRegressor

from tensorflow.keras import datasets, layers, models

# Vars
'''
variables and constants go here
'''
horizon = 90 # 1 qtr forecast

# Functions
'''
Going to be using these funcs throughout subsequent notebooks
'''
# Test Train
def train_test(s):
    test = s[-horizon:]
    train = s[:-horizon]
    return train , test

# Symmetric MAPE
def symmetric_mean_absolute_percentage_error(act, forecast):
    return 1/len(act) * np.sum(2 * np.abs(forecast - act) / (np.abs(act) + np.abs(forecast)))

# Plot
def plt_decomp(result):
    '''
    params: result -> statsmodels.tsa.seasonal.DecomposeResult object
    returns: None
    '''
    observed = result.observed
    seasonal = result.seasonal
    trend = result.trend
    resid = result.resid

    fig , ax = plt.subplots(4,1 , figsize = (20 , 8))
    fig.tight_layout()

    ax[0].plot(observed);
    ax[0].set_ylabel('observed');

    ax[1].plot(trend);
    ax[1].set_ylabel('trend');
    
    ax[2].plot(seasonal);
    ax[2].set_ylabel('seasonal');
    
    ax[3].plot(resid);
    ax[3].set_ylabel('resid');

def gap_filler(series):
    # Build an index to fill out all the gaps!
    idx = pd.date_range(series.index.min(), series.index.max() , freq="D")
    all_dt = pd.Series([0]*len(idx) , index = idx , name = 'Helper')

    all_dt_series = pd.concat([all_dt , series] , axis = 1).fillna(method='ffill')
    all_dt_series = all_dt_series.drop('Helper' , axis = 1)
    return all_dt_series

def oos_pred():
    '''
    out of sample forecast
    '''
    pass