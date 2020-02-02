from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator
import numpy as np


class Regressor(BaseEstimator):
    def __init__(self, alpha=1.0):
        self.reg = Ridge(alpha=alpha)

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return np.around(self.reg.predict(X))

