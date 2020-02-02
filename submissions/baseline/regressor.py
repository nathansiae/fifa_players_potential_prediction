from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator
import numpy as np


class Regressor(BaseEstimator):
    def __init__(self, n_estimators=5, max_depth=30, max_features=10):
        self.reg = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return np.around(self.reg.predict(X))

