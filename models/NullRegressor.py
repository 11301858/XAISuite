import numpy as np
from sklearn.base import RegressorMixin
class NullRegressor(RegressorMixin):
    def fit(self, X=None, y=None):
        # The prediction will always just be the mean of y
        self.y_bar_ = np.mean(y)
    def predict(self, X=None):
        # Give back the mean of y, in the same
        # length as the number of X observations
        return np.ones(X.shape[0]) * self.y_bar_
