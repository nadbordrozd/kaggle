import numpy as np
from scipy.optimize import fmin_powell, brute
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression

from commons import logger, cache

# =================== OPTIMIZING OFFSETS
allowed_responses = [1, 1.33, 1.66, 2, 2.33, 2.66, 3]
num_classes = len(allowed_responses)

# this is perhaps unnecessarily convoluted but it doesn't seem to work
# anyway so whatever
def fix_prediction(prediction, offsets):
    actual_offsets = []
    for a, b, o in zip(allowed_responses, allowed_responses[1:], offsets):
        actual_offsets.append((b-a)/(1 + np.exp(-o)) + a)
    for threshold, response in zip(actual_offsets, allowed_responses):
        if prediction < threshold:
            return response
    return allowed_responses[-1]

def apply_offsets(predictions, offsets):
    return [fix_prediction(p, offsets) for p in predictions]

def score_offsets(predictions, y, offsets):
    return np.sqrt(mean_squared_error(apply_offsets(predictions, offsets), y))
    
def optimize_offsets(predictions, y):
    offsets = np.zeros(num_classes-1)
    for i in range(num_classes-1):
        def score(x):
            offsets[i] = x
            return score_offsets(predictions, y, offsets)

        offsets[i] = brute(score, [(-10, 10)])
        offsets[i] = fmin_powell(score, offsets[i])
    return offsets

class OffsetOptimizer(object):
    def __init__(self, model):
        self.model = model
        self.offsets = np.ones(num_classes)
        
    def fit(self, X, y):
        self.model.fit(X, y)
        preds = self.model.predict(X)
        self.offsets = optimize_offsets(preds, y)
        return self
    
    def predict(self, X):
        preds = self.model.predict(X)
        return apply_offsets(preds, self.offsets)
    
    def __str__(self):
        return "optimizer  %s" % self.model
    
############## SOME EXAMPLE MODELS
def rf():
    forest = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
    return BaggingRegressor(forest, n_estimators=45, max_samples=0.1, random_state=25)

def linreg():
    return LinearRegression()
    
