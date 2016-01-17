

#=========================================LOGGING
import logging
# create logger
logging.basicConfig(filename='log_train.log',level=logging.DEBUG, format="%(asctime)s; %(levelname)s;  %(message)s")
logger = logging.getLogger("trainlo")
logger.setLevel(logging.DEBUG)

def info(msg):
    logger.info(msg.replace("\n", "  "))
#=========================================LOGGING

import sys
import pandas as pd
import numpy as np
from time import time
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier, SGDRegressor, Perceptron
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVC, SVR
from xgboost.sklearn import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from joblib import Memory
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from ml_metrics import quadratic_weighted_kappa

def eval_wrapper(yhat, y):  
    y = np.array(y)
    y = y.astype(int)
    yhat = np.array(yhat)
    yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)   
    return quadratic_weighted_kappa(yhat, y)

#TODO: check which ones are really worth encoding and which can be even dropped (some may contain nonoverlapping values between
#test and training sest)
categorical = {'Product_Info_1', 'Product_Info_2', 'Product_Info_3', 'Product_Info_5', 'Product_Info_6', 
               'Product_Info_7', 'Employment_Info_2', 'Employment_Info_3', 'Employment_Info_5', 
               'InsuredInfo_1', 'InsuredInfo_2', 'InsuredInfo_3', 'InsuredInfo_4', 'InsuredInfo_5', 
               'InsuredInfo_6', 'InsuredInfo_7', 'Insurance_History_1', 'Insurance_History_2', 
               'Insurance_History_3', 'Insurance_History_4', 'Insurance_History_7', 'Insurance_History_8', 
               'Insurance_History_9', 'Family_Hist_1', 'Medical_History_2', 'Medical_History_3', 
               'Medical_History_4', 'Medical_History_5', 'Medical_History_6', 'Medical_History_7', 
               'Medical_History_8', 'Medical_History_9', 'Medical_History_10', 'Medical_History_11', 
               'Medical_History_12', 'Medical_History_13', 'Medical_History_14', 'Medical_History_16', 
               'Medical_History_17', 'Medical_History_18', 'Medical_History_19', 'Medical_History_20', 
               'Medical_History_21', 'Medical_History_22', 'Medical_History_23', 'Medical_History_25', 
               'Medical_History_26', 'Medical_History_27', 'Medical_History_28', 'Medical_History_29', 
               'Medical_History_30', 'Medical_History_31', 'Medical_History_33', 'Medical_History_34', 
               'Medical_History_35', 'Medical_History_36', 'Medical_History_37', 
#                'Medical_History_38', 
               'Medical_History_39', 'Medical_History_40', 'Medical_History_41','Medical_History_1', 
               'Medical_History_15', 'Medical_History_24', 'Medical_History_32'}

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
total = pd.concat([train, test])
median = total.median()
train.fillna(median, inplace=True)
test = test.fillna(median, inplace=True)
encoder = LabelEncoder()
for f in categorical:
    encoder.fit(total[f])
    train[f] = encoder.transform(train[f])
    test[f] = encoder.transform(test[f])

feature_cols = test.columns[1:]
categorical_inds = [i for i, col in enumerate(feature_cols) if col in categorical]
oh_encoder = OneHotEncoder(categorical_features=categorical_inds)

X = np.array(train[test.columns[1:]])
y = np.array(train.Response)
X_actual_test = np.array(test[feature_cols])

oh_encoder.fit(X)
X = oh_encoder.transform(X).todense()
X_actual_test = oh_encoder.transform(X_actual_test).todense()

#debug on
n = 100
X = X[:n, :]
y = y[:n]
#debug off
info("train shape = %s     test shape = %s " % (X.shape, X_actual_test.shape))

train_test_folds = list(StratifiedKFold(y, n_folds=6, random_state=0))
#================================================================================================
train_cache = Memory(cachedir="cache/train", verbose=0)
test_cache = Memory(cachedir="cache/test", verbose=0)

@train_cache.cache
def train_predictions(model):
    ind2pred = {}
    for i, (train, test) in enumerate(train_test_folds):
        info(("fitting fold   "+str(i+1)+" " +  str(model)))
        model.fit(X[train], y[train])
        info(("fold fitted    "+str(i+1)+" " +  str(model)))
        preds = model.predict(X[test])
        for i, p in zip(test, preds):
            ind2pred[i] = p
    
    return np.array([ind2pred[i] for i in range(len(y))])

@test_cache.cache
def test_predictions(model):
    model.fit(X, y)
    return model.predict(X_actual_test)


stacker_train_cache = Memory(cachedir="cache/stacker_train", verbose=0)
stacker_test_cache = Memory(cachedir="cache/stacker_test", verbose=0)

@stacker_train_cache.cache
def stacker_train_predictions(stacker, base_clfs):
    info("start stacker --------------------------")
    n = len(y)
    stacked_X = np.hstack([X] + [train_predictions(clf).reshape(n, 1) for clf in base_clfs])
    info("base regressors done")
    ind2pred = {}
    for i, (train, test) in enumerate(train_test_folds):
        info("fitting stacker fold %s   %s" % (i, str(stacker)))

        stacker.fit(stacked_X[train], y[train])
        info("stacker fitted fold %s    %s " % (i, str(stacker)))
        preds = stacker.predict(stacked_X[test])
        for i, p in zip(test, preds):
            ind2pred[i] = p
    info("stacker done =========================")
    return np.array([ind2pred[i] for i in range(len(y))])

@stacker_test_cache.cache
def stacker_test_predictions(stacker, base_clfs):
    n = len(y)
    stacked_X = np.hstack([X] + [train_predictions(clf).reshape(n, 1) for clf in base_clfs])
    stacker.fit(stacked_X, y)
    n = X_actual_test.shape[0]
    stacked_test_X = np.hstack([X_actual_test] + [test_predictions(clf).reshape(n, 1) for clf in base_clfs])
    return stacker.predict(stacked_test_X)
#============================================================================
def benchmark(model):
    pred = train_predictions(model)
    return eval_wrapper(pred, y)

def make_predictions(model):
    model.fit(X, y)
    return model.predict(X_actual_test)

def benchmark_stacker(model, base_clfs):
    pred = stacker_train_predictions(model, base_clfs)
    result = eval_wrapper(pred, y)
    info("%s   %s, %s" % (result, model, base_clfs))
    return result
#==============================================================================
xgbr = lambda: XGBRegressor(objective="reg:linear", min_child_weight=80, subsample=0.85, colsample_bytree=0.30, silent=1, max_depth=9)
xgbc = lambda: XGBClassifier(objective="reg:linear", min_child_weight=80, subsample=0.85, colsample_bytree=0.30, silent=1, max_depth=9)
rfr = lambda: RandomForestRegressor(n_estimators=400)
etr = lambda: ExtraTreesRegressor(n_estimators=400)
etc = lambda: ExtraTreesClassifier(n_estimators=400)
sgdr = lambda: SGDRegressor()
perceptron = lambda: Perceptron()

dream_team = lambda: sorted([xgbr(), rfr(),  etr(), sgdr(), LinearRegression(), Perceptron(),
              SVR(kernel="linear"), SVR(kernel="poly"), SVR(kernel="sigmoid"),
              SVR(kernel="rbf"), LogisticRegression()])

def make_sub(stacker, base_clfs, filename):
    preds = stacker_test_predictions(stacker, base_clfs)
    
    df = pd.DataFrame()
    df['Id'] = test.Id
    df['Response'] = preds

    df.to_csv(filename, index=False)
info("========================START========================")
info("shape = " + str(X.shape))
train_predictions(etr())

info("extra trees done woohooo")

benchmark_stacker(DummyClassifier(), [DummyRegressor()])
benchmark_stacker(etc(), dream_team())
info("makink submission for extra trees")
make_sub(etc(), dream_team(), "extra_trees.csv")
info("done makink submission for extra trees")
benchmark_stacker(xgbc(), dream_team())
info("makink submission for xgbc")
make_sub(xgbc(), dream_team(), "xgbc.csv")
info("done makink submission for xgbc")
benchmark_stacker(LogisticRegression(), dream_team())
info("makink submission for logistic")
make_sub(xgbc(), dream_team(), "logisticreg.csv")
info("done making submission for logistic")

info("::::::::::::::::::::::::::::::: I'M DONE ::::::::::::::::::::::::::::::::::")
