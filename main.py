import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score 
from sklearn.preprocessing import PolynomialFeatures, scale, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.base import clone

data = pd.read_csv("future-gc00-daily-prices.csv")
data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = data['Date'].values.astype(np.int64) // 10 ** 9


# Shift the 'Close' column one day back
data['Close'] = data['Close'].shift(-1)

# Drop the last row
data = data[:-1]

# Features
X = data[['Date', 'Open', 'High', 'Low']]

# Target variable
y = data['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)