from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd

class StackedRecommender:
    def __init__(self):
        self.model = GradientBoostingRegressor()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)