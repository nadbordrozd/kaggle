import numpy as np
from itertools import izip
from functools import partial

from scipy.optimize import fmin_powell, brute
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge, LogisticRegression
from sklearn.svm import SVR
from xgboost.sklearn import XGBRegressor
from sklearn.svm import SVR
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import CountVectorizer

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import np_utils

from commons import logger, cache, get_func_name


#################### STACKER ##############################
@cache
def mini_train_predictions(model_fun, X, y, train_test_folds):
    model = model_fun()
    logger.info("in mini_train_predictions with model %s" % model) 
    ind2pred = {}
    for i, (train, test) in enumerate(train_test_folds):
        model.fit(X[train], y[train])
        preds = model.predict(X[test])
        for i, p in zip(test, preds):
            ind2pred[i] = p
    
    return np.array([ind2pred[i] for i in range(len(y))])
            
      
class Stacker(object):
    def __init__(self, stacker_fun, base_model_funs, folds=4, lazy_stack=False, name="unnamed_stacker"):
        self.folds = folds
        self.base_model_funs = sorted(base_model_funs)
        self.base_models = [m() for m in base_model_funs]
        self.stacker = stacker_fun()
        self.name = name
        self.lazy_stack = lazy_stack

    def fit(self, X, y):
        folds = list(KFold(len(y), n_folds=self.folds, random_state=0))
        base_preds = [mini_train_predictions(m, X, y, folds) for m in self.base_model_funs]
        n = len(y)
            
        XX = np.hstack(([] if self.lazy_stack else [X])
                       + [p.reshape(n, 1) for p in base_preds])
        self.stacker.fit(XX, y)
        for m in self.base_models:
            m.fit(X, y)
        return self
    
    def predict(self, X):
        n = X.shape[0]
        XX = np.hstack(([] if self.lazy_stack else [X])
                       + [m.predict(X).reshape(n, 1) for m in self.base_models])
        return self.stacker.predict(XX)
    
    def __str__(self):
        return self.name

    

################################ neural networks

class KerasWrapper(object):
    def __init__(self, model_maker):
        self.model_maker = model_maker
        self.model = None
        
    def fit(self, X, y):
        _, input_dim = X.shape
        self.model = self.model_maker(input_dim)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
       
    def __str__(self):
        return get_func_name(self.model_maker)
    
def nn1(input_dim):
    model = Sequential()
    model.add(Dense(64, input_shape=(input_dim,)))
    model.add(Activation('sigmoid'))
    model.add(Dense(64, input_shape=(input_dim,)))
    model.add(Activation('sigmoid'))
    model.add(Dense(1))
    model.add(Activation("linear"))
    model.compile(loss='rmse', optimizer='adam')
    return model

############## SOME EXAMPLE MODELS


def rf():
    forest = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
    return BaggingRegressor(forest, n_estimators=45, max_samples=0.1, random_state=25)

def linreg():
    return LinearRegression()
def logreg():
    return LogisticRegression()
def neural_1():
    return KerasWrapper(nn1)
def xgbr():
    return XGBRegressor(objective="reg:linear")
def etr_400():
    return ExtraTreesRegressor(n_estimators=400)
def etr_100():
    return ExtraTreesRegressor(n_estimators=100)
def etr_50():
    return ExtraTreesRegressor(n_estimators=50)
def rfr_400():
    return RandomForestRegressor(n_estimators=400)
def svrsig():
    return SVR(kernel="sigmoid")
def svrrbf():
    return SVR(kernel="rbf")
def lasso():
    return Lasso()
def bayes_ridge():
    return BayesianRidge()
mini_team = sorted([xgbr, etr_50,  svrsig, svrrbf, linreg, rf])
big_team =  sorted([
    xgbr, etr_400, rfr_400,
    svrsig, svrrbf, linreg, rf, lasso, bayes_ridge])
nn1_team =  sorted([
    xgbr, etr_400, rfr_400,
    svrsig, svrrbf, linreg, rf, lasso, bayes_ridge, neural_1])
def stacker_0():
    return Stacker(linreg, [linreg, xgbr], name="lin(lin, xgbr)")
def stacker_1():
    return Stacker(rf, mini_team, name="brf(xgbr, etr_50,  svrsig, svrrbf, linreg, rf)")
def stacker_2():
    return Stacker(linreg, mini_team, name="linreg(xgbr, etr_50,  svrsig, svrrbf, linreg, rf)")
def stacker_3():
    return Stacker(etr_50, mini_team, name="etr_50(xgbr, etr_50,  svrsig, svrrbf, linreg, rf)")
def stacker_4():
    return Stacker(xgbr, mini_team, name="xgbr(xgbr, etr_50,  svrsig, svrrbf, linreg, rf)")
def stacker_5():
    return Stacker(svrsig, mini_team, name="svrsig(xgbr, etr_50,  svrsig, svrrbf, linreg, rf)")
def stacker_7():
    return Stacker(LinearRegression, big_team, 
        name="lin(xgbr, etr200, rfr200, svrsig, svrrbf, linreg, rf, lasso, bayes_ridge")
def stacker_8():
    return Stacker(LinearRegression, nn1_team, name="lin(big_team + neural_1") 
