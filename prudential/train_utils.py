MEMO_PATH = "memo_fe/"
JOBLIB_MEMO_PATH = "train_cache/"
#=========================================LOGGING
import logging
# create logger
logging.basicConfig(filename='newlog.log',level=logging.DEBUG, format="%(asctime)s; %(levelname)s;  %(message)s")
logger = logging.getLogger("trainlo")
logger.setLevel(logging.DEBUG)

def info(msg):
    logger.info(msg)
#=========================================LOGGING

import sys
import pandas as pd
import numpy as np
from time import time
from scipy.optimize import fmin_powell
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier, \
    SGDRegressor, Perceptron, PassiveAggressiveRegressor, BayesianRidge, Lasso
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from xgboost.sklearn import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from persistent_cache import memo, PersistentDict as Perd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from ml_metrics import quadratic_weighted_kappa
from feature_engineering import train_test_sets
from joblib import Memory

def eval_wrapper(yhat, y):  
    y = np.array(y)
    y = y.astype(int)
    yhat = np.array(yhat)
    yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)   
    return quadratic_weighted_kappa(yhat, y)


num_classes = 8


y = np.array(pd.read_csv("train.csv").Response)
test_id = np.array(pd.read_csv("test.csv").Id)
train_test_folds = list(StratifiedKFold(y, n_folds=4, random_state=0))
#================================================================================================

mini_train_memo = Memory(cachedir=JOBLIB_MEMO_PATH+"mini_train", verbose=0)
@mini_train_memo.cache
def mini_train_predictions(model, X, y, train_test_folds):
    ind2pred = {}
    for i, (train, test) in enumerate(train_test_folds):
        info(("fitting MINIfold   "+str(i+1)+ str(model)[:100]))
        model.fit(X[train], y[train])
        info(("MINIfold fitted    "+str(i+1)+  str(model)[:100]))
        preds = model.predict(X[test])
        for i, p in zip(test, preds):
            ind2pred[i] = p
    
    return np.array([ind2pred[i] for i in range(len(y))])

train_pred_memo = Memory(cachedir=JOBLIB_MEMO_PATH+"train_predictions", verbose=0)
@train_pred_memo.cache
def train_predictions(model, fe):
    X, _ = train_test_sets(fe)
    ind2pred = {}
    for i, (train, test) in enumerate(train_test_folds):
        info(("fitting fold   "+str(i+1)+ str(model)[:100]))
        model.fit(X[train], y[train])
        info(("fold fitted    "+str(i+1)+  str(model)[:100]))
        preds = model.predict(X[test])
        for i, p in zip(test, preds):
            ind2pred[i] = p
    
    return np.array([ind2pred[i] for i in range(len(y))])

@memo(Perd(MEMO_PATH + "_test_predictions"))
def test_predictions(model, fe):
    X, X_actual_test = train_test_sets(fe)
    info("fitting (on full train set) %s" % model)
    model.fit(X, y)
    info("done fitting for %s" % model)
    return model.predict(X_actual_test)


@memo(Perd(MEMO_PATH + "_stacker_train_predictions"))
def stacker_train_predictions(stacker, base_clfs, fe):
    X, _ = train_test_sets(fe)
    info("start stacker --------------------------")
    n = len(y)
    stacked_X = np.hstack([X] + [train_predictions(clf, fe).reshape(n, 1) for clf in base_clfs])
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

@memo(Perd(MEMO_PATH + "_lazy_stacker_train_predictions"))
def lazy_stacker_train_predictions(stacker, base_clfs, fe):
    info("start stacker --------------------------")
    n = len(y)
    stacked_X = np.hstack([train_predictions(clf, fe).reshape(n, 1) for clf in base_clfs])
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


@memo(Perd(MEMO_PATH + "_stacker_test_predictions"))
def stacker_test_predictions(stacker, base_clfs, fe):
    n = len(y)
    X, X_actual_test = train_test_sets(fe)
    stacked_X = np.hstack([X] + [train_predictions(clf, fe).reshape(n, 1) for clf in base_clfs])
    stacker.fit(stacked_X, y)
    nn = X_actual_test.shape[0]
    stacked_test_X = np.hstack([X_actual_test] + [test_predictions(clf, fe).reshape(nn, 1) for clf in base_clfs])
    return stacker.predict(stacked_test_X)
#============================================================================
def benchmark(model, fe):
    pred = train_predictions(model, fe)
    return eval_wrapper(pred, y)

def make_predictions(model, fe):
    X, X_actual_test = train_test_sets(fe)
    model.fit(X, y)
    return model.predict(X_actual_test)

def benchmark_stacker(model, base_clfs, fe):
    pred = stacker_train_predictions(model, base_clfs, fe)
    result = eval_wrapper(pred, y)
    info("stacker %.4f   %s, %s, feats = %s" % (result, model, base_clfs, fe))
    return result

def benchmark_lazy_stacker(model, base_clfs, fe):
    pred = lazy_stacker_train_predictions(model, base_clfs, fe)
    result = eval_wrapper(pred, y)
    info("lazy stacker %.4f   %s, %s  feats = %s" % (result, model, base_clfs, fe))
    
    return result
#==============================================================================
# OPTIMISING OFFSETS
# -----------------------------------------------------------------------------
def apply_offset(data, bin_offset, sv, scorer=eval_wrapper):
    # data has the format of pred=0, offset_pred=1, labels=2 in the first dim
    data[1, data[0].astype(int)==sv] = data[0, data[0].astype(int)==sv] + bin_offset
    score = scorer(data[1], data[2])
    return score

offsets_memo = Memory(cachedir=JOBLIB_MEMO_PATH+"offsets", verbose=0)
@offsets_memo.cache
def optimize_offsets(predictions, y):
    # train offsets
    info("optimising offsets %s" % len(y))
    offsets = np.ones(num_classes) * -0.5
    offset_train_preds = np.vstack((predictions, predictions, y))
    
    for j in range(num_classes):
        train_offset = lambda x: -apply_offset(offset_train_preds, x, j)
        offsets[j] = fmin_powell(train_offset, offsets[j], disp=False) 
    info("done optimising offsets %s" % len(y))
    return offsets

def actually_apply_offsets(predictions, offsets):
    data = np.vstack((predictions, predictions, -np.ones(len(predictions))))
    for j in range(num_classes):
        data[1, data[0].astype(int)==j] = data[0, data[0].astype(int)==j] + offsets[j] 

    final_test_preds = np.round(np.clip(data[1], 1, 8)).astype(int)
    return final_test_preds

opt_train_pred_memo = Memory(cachedir=JOBLIB_MEMO_PATH+"opt_train_pred", verbose=0)
@opt_train_pred_memo.cache
def optimized_train_predictions(raw_train_predictions):
    n = len(y)
    ind2pred = {}
    for i, (train, test) in enumerate(train_test_folds):
        train_preds = raw_train_predictions[train]
        offsets = optimize_offsets(train_preds, y[train])
        test_preds = actually_apply_offsets(raw_train_predictions[test], offsets)
        for i, p in zip(test, test_preds):
            ind2pred[i] = p
    return np.array([ind2pred[i] for i in range(len(y))])

def benchmark_model_optimized(model, fe):
    preds = optimized_train_predictions(train_predictions(model, fe))
    result = eval_wrapper(preds, y)
    info("optimized %.4f   %s  %s" % (result, model, fe))
    return result

def benchmark_optimized_stacker(stacker, base_clfs, fe):
    preds = stacker_train_predictions(stacker, base_clfs, fe)
    opreds = optimized_train_predictions(preds)
    result = eval_wrapper(opreds, y)
    info("optimized stacker %.4f   %s, %s,  feats=%s" % (result, stacker, base_clfs, fe))
    return result

def benchmark_optimized_lazy_stacker(stacker, base_clfs, fe):
    preds = lazy_stacker_train_predictions(stacker, base_clfs, fe)
    opreds = optimized_train_predictions(preds)
    result = eval_wrapper(opreds, y)
    info("optimized stacker %.4f   %s, %s   feats=%s" % (result, stacker, base_clfs, fe))
    return result

def optimized_test_predictions(stacker, base_clfs, fe):
    train_preds = stacker_train_predictions(stacker, base_clfs, fe)
    offsets = optimize_offsets(train_preds, y)
    test_preds = stacker_test_predictions(stacker, base_clfs, fe)
    final_test_preds = actually_apply_offsets(test_preds, offsets)
    #print "print len(train_preds), len(test_preds), len(final_test_preds)"
    #print len(train_preds), len(test_preds), len(final_test_preds)
    info("made optimized predictions for stacker %s with features %s and base regressors %s" % (stacker, fe, base_clfs))
    return final_test_preds

def make_sub_optimized(stacker, base_clfs, fe, filename):
    preds = optimized_test_predictions(stacker, base_clfs, fe)
    #print "len(preds)"
    #print len(preds)
    df = pd.DataFrame()
    df['Id'] = test_id
    df['Response'] = preds
    info("made submission to file %s. stacker %s, features %s" % (filename, stacker, fe))
    df.to_csv(filename, index=False)
    
def make_sub(model, fe, filename):
    preds = stacker_test_predictions(stacker, base_clfs, fe)
    
    df = pd.DataFrame()
    df['Id'] = test.Id
    df['Response'] = preds
    info("making stacker %s, nonoptimized submission to file %s " % (stacker, filename))
    df.to_csv(filename, index=False)
    
    
def lazy_benchmark(model, fe):
    X, _ = train_test_sets(fe)
    train_inds, test_inds = train_test_folds[0]
    X_train = X[train_inds]
    X_test = X[test_inds]
    y_train = y[train_inds]
    y_test = y[test_inds]
    model.fit(X_train, y_train)
    test_preds = model.predict(X_test)
    result = eval_wrapper(test_preds, y_test)
    return result
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
class Stacker(object):
    def __init__(self, stacker, base_models, folds=4, lazy_opt=True, lazy_stack=False, name="unnamed_stacker"):
        self.folds = folds
        self.base_models = sorted(base_models)
        self.stacker = stacker
        self.name = name
        self.offsets = None
        self.lazy_stack = lazy_stack
        self.lazy_opt = lazy_opt

    def fit(self, X, y):
        folds = list(StratifiedKFold(y, n_folds=self.folds, random_state=0))
        base_preds = [mini_train_predictions(m, X, y, folds) for m in self.base_models]
        n = len(y)
            
        XX = np.hstack(([] if self.lazy_stack else [X])
                       + [p.reshape(n, 1) for p in base_preds])
        
        for m in self.base_models:
            m.fit(X, y)
        if self.lazy_opt:
            self.stacker.fit(XX, y)
            train_preds = self.stacker.predict(XX)
            self.offsets = optimize_offsets(train_preds, y)
        else:
            buncha_offsets = []
            for train_inds, test_inds in folds:
                train_X = XX[train_inds]
                train_y = y[train_inds]
                test_X = XX[test_inds]
                test_y = y[test_inds]
                self.stacker.fit(train_X, train_y)
                buncha_offsets.append(optimize_offsets(self.stacker.predict(test_X), test_y))
            self.offsets = np.vstack(buncha_offsets).mean(axis=0)
            self.stacker.fit(XX, y)
        return self
    
    def predict(self, X):
        n = X.shape[0]
        XX = np.hstack(([] if self.lazy_stack else [X])
                       + [m.predict(X).reshape(n, 1) for m in self.base_models])
        return actually_apply_offsets(self.stacker.predict(XX), self.offsets)
    
    def __str__(self):
        return self.name
                 
                 

xgbr = lambda: XGBRegressor(objective="reg:linear", min_child_weight=80, subsample=0.85, colsample_bytree=0.30, silent=1, max_depth=9)
xgbc = lambda: XGBClassifier(objective="reg:linear", min_child_weight=80, subsample=0.85, colsample_bytree=0.30, silent=1, max_depth=9)
rfr = lambda: RandomForestRegressor(n_estimators=400)
etr = lambda: ExtraTreesRegressor(n_estimators=400)
etc = lambda: ExtraTreesClassifier(n_estimators=400)
sgdr = lambda: SGDRegressor()
xgbr_poly = lambda: Pipeline([("poly", PolynomialFeatures(degree=2)), ("xgbr", xgbr())])
linreg_poly = lambda: Pipeline([("poly", PolynomialFeatures(degree=2)), ("linreg", LinearRegression())])
linreg = lambda: LinearRegression()
bayes_ridge = lambda: BayesianRidge()
lasso = lambda: Lasso()
svrsig = lambda: SVR(kernel="sigmoid")
svrrbf = lambda: SVR(kernel="rbf")
perc = lambda: Perceptron()
"""
st_lazy_hard = lambda: Stacker(linreg(), [linreg(), xgbr()], 4, lazy_opt=True, lazy_stack=False, name="lin(lin,xgb),4,lazy_opt,full_feat")
st_hard_hard = lambda: Stacker(linreg(), [linreg(), xgbr()], 4, lazy_opt=False, lazy_stack=False, name="lin(lin,xgb),4,hard_opt,full_feat")
st_hard_lazy = lambda: Stacker(linreg(), [linreg(), xgbr()], 4, lazy_opt=False, lazy_stack=True, name="lin(lin,xgb),4,hard_opt,no_feat")
st_lazy_lazy = lambda: Stacker(linreg(), [linreg(), xgbr()], 4, lazy_opt=True, lazy_stack=True, name="lin(lin,xgb),4,lazy_opt,no_feat")
st1_ff = lambda: Stacker(linreg(), [linreg(), xgbr()], 4, lazy_stack=False, name="lin(lin,xgb),4,full_feat")
st1_nf = lambda: Stacker(linreg(), [linreg(), xgbr()], 4, lazy_stack=True, name="lin(lin,xgb),4,no_feat")
st2_ff = lambda: Stacker(xgbr(), [linreg(), xgbr()], 4, lazy_stack=False, name="xgbr(lin,xgb),4,full_feat")
st2_nf = lambda: Stacker(xgbr(), [linreg(), xgbr()], 4, lazy_stack=True, name="xgbr(lin,xgb),4,no_feat")
SEEMS THE DEFAULT SETTING OF THE STACKER lazy_opt=True, lazy_stack=False are indeed the best
"""
lin_lin_xgb = lambda: Stacker(linreg(), [linreg(), xgbr()], name="lin(lin,xgb)")
etr_lin_xgb = lambda: Stacker(linreg(), [linreg(), xgbr()], name="etr(lin,xgb)")

dream_team = lambda: sorted([xgbr(), rfr(),  etr(), LinearRegression(), 
                             #xgbr_poly(), linreg_poly(),
                             bayes_ridge(),
                             lasso(),
                             svrsig(),
                             svrrbf(),
                             perc()
                            ])

mini_team = lambda: sorted([xgbr(), rfr(),  etr(), LinearRegression(), svrsig()])
lin_mini_team = lambda: Stacker(linreg(), mini_team(), name="lin(xgbr,rfr,etr,linreg,svrsig)")
etr_mini_team = lambda: Stacker(linreg(), mini_team(), 4, name="etr(xgbr,rfr,etr,linreg,svrsig)")
xgbr_mini_team = lambda: Stacker(xgbr(), mini_team(), 4, name="xgbr(xgbr,rfr,etr,linreg,svrsig)")
mini_team_plus_bayes = lambda: sorted([xgbr(), rfr(),  etr(), LinearRegression(), svrsig(), bayes_ridge()])
mini_team_plus_lasso = lambda: sorted([xgbr(), rfr(),  etr(), LinearRegression(), svrsig(), lasso()])
mini_team_plus_perc = lambda: sorted([xgbr(), rfr(),  etr(), LinearRegression(), svrsig(), perc()])
mini_team_plus_svrrbf = lambda: sorted([xgbr(), rfr(),  etr(), LinearRegression(), svrsig(), svrrbf()])
lin_mini_bayes = lambda: Stacker(linreg(), mini_team_plus_bayes(), name="lin(xgbr,rfr,etr,lin,svrsig,bayes)")
lin_mini_lasso = lambda: Stacker(linreg(), mini_team_plus_lasso(), name="lin(xgbr,rfr,etr,lin,svrsig,lasso)")
lin_mini_perc = lambda: Stacker(linreg(), mini_team_plus_perc(), name="lin(xgbr,rfr,etr,lin,svrsig,perc)")
lin_mini_svrrbf = lambda: Stacker(linreg(), mini_team_plus_svrrbf(), name="lin(xgbr,rfr,etr,lin,svrsig,svrrbf)")
lin_dream = lambda: Stacker(linreg(), dream_team(), name="lin(xgbr,rfr,etr,lin,svrsig,svrrbf,lasso,bayes,perc)")

