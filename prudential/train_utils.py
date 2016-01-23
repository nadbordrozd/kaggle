MEMO_PATH = "memo_fe"

#=========================================LOGGING
import logging
# create logger
logging.basicConfig(filename='feat_engineered.log',level=logging.DEBUG, format="%(asctime)s; %(levelname)s;  %(message)s")
logger = logging.getLogger("trainlo")
logger.setLevel(logging.DEBUG)

def info(msg):
    logger.info(msg.replace("\n", "  "))
#=========================================LOGGING

import sys
import pandas as pd
import numpy as np
from time import time
from scipy.optimize import fmin_powell
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier, SGDRegressor, Perceptron
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVC, SVR
from xgboost.sklearn import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from persistent_cache import memo, PersistentDict as Perd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from ml_metrics import quadratic_weighted_kappa

def eval_wrapper(yhat, y):  
    y = np.array(y)
    y = y.astype(int)
    yhat = np.array(yhat)
    yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)   
    return quadratic_weighted_kappa(yhat, y)


num_classes = 8


print("Load the data using pandas")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# combine train and test
all_data = train.append(test)

# factorize categorical variables    
all_data['Product_Info_2'] = pd.factorize(all_data['Product_Info_2'])[0]

# FEATURE ENGINEERING
all_data['bmi_ins_age'] = all_data.BMI * all_data.Ins_Age
all_data['nan_count'] = all_data.isnull().sum(axis=1)
all_data['emp_inf_4_sq'] = all_data.Employment_Info_4 ** 2
all_data['fam_hist_4_sq'] = all_data.Family_Hist_4 ** 2
all_data['fam_hist_2_sq'] = all_data.Family_Hist_2 ** 2

mk = [col for col in train.columns if col.startswith("Medical_K")]
all_data['sum_keywords'] = sum(train[col] for col in mk)

all_data.drop('Medical_History_24')
all_data.drop('Medical_History_10')



print('Eliminate missing values')    
# Use -1 for any others
all_data.fillna(-1, inplace=True)

# fix the dtype on the label column
all_data['Response'] = all_data['Response'].astype(int)

# Provide split column
# all_data['Split'] = np.random.randint(5, size=all_data.shape[0])

# split train and test
train = all_data[all_data['Response']>0].copy()
test = all_data[all_data['Response']<1].copy()


X = np.array(train.drop(["Id", "Response"], axis=1))
X_actual_test = np.array(test.drop(["Id", "Response"], axis=1))
y = np.array(train.Response)




train_test_folds = list(StratifiedKFold(y, n_folds=4, random_state=0))
#================================================================================================
@memo(Perd(MEMO_PATH + "_train_predictions"))
def train_predictions(model):
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
def test_predictions(model):
    info("fitting (on full train set) %s" % model)
    model.fit(X, y)
    info("done fitting for %s" % model)
    return model.predict(X_actual_test)


@memo(Perd(MEMO_PATH + "_stacker_train_predictions"))
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

@memo(Perd(MEMO_PATH + "_lazy_stacker_train_predictions"))
def lazy_stacker_train_predictions(stacker, base_clfs):
    info("start stacker --------------------------")
    n = len(y)
    stacked_X = np.hstack([train_predictions(clf).reshape(n, 1) for clf in base_clfs])
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
def stacker_test_predictions(stacker, base_clfs):
    n = len(y)
    print "train length = %s" % n
    stacked_X = np.hstack([X] + [train_predictions(clf).reshape(n, 1) for clf in base_clfs])
    stacker.fit(stacked_X, y)
    nn = X_actual_test.shape[0]
    print "test length = %s" % nn
    stacked_test_X = np.hstack([X_actual_test] + [test_predictions(clf).reshape(nn, 1) for clf in base_clfs])
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
    info("stacker %s   %s, %s" % (result, model, base_clfs))
    return result

def benchmark_lazy_stacker(model, base_clfs):
    pred = lazy_stacker_train_predictions(model, base_clfs)
    result = eval_wrapper(pred, y)
    info("lazy stacker %s   %s, %s" % (result, model, base_clfs))
    
    return result
#==============================================================================
# OPTIMISING OFFSETS
# -----------------------------------------------------------------------------
def apply_offset(data, bin_offset, sv, scorer=eval_wrapper):
    # data has the format of pred=0, offset_pred=1, labels=2 in the first dim
    data[1, data[0].astype(int)==sv] = data[0, data[0].astype(int)==sv] + bin_offset
    score = scorer(data[1], data[2])
    return score


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

def benchmark_model_optimized(model):
    preds = optimized_train_predictions(train_predictions(model))
    result = eval_wrapper(preds, y)
    info("optimized %s   %s" % (result, model))
    return result

def benchmark_optimized_stacker(stacker, base_clfs):
    preds = stacker_train_predictions(stacker, base_clfs)
    opreds = optimized_train_predictions(preds)
    result = eval_wrapper(opreds, y)
    info("optimized stacker %s   %s, %s" % (result, stacker, base_clfs))
    return result

def optimized_test_predictions(stacker, base_clfs):
    train_preds = stacker_train_predictions(stacker, base_clfs)
    offsets = optimize_offsets(train_preds, y)
    test_preds = stacker_test_predictions(stacker, base_clfs)
    final_test_preds = actually_apply_offsets(test_preds, offsets)
    #print "print len(train_preds), len(test_preds), len(final_test_preds)"
    #print len(train_preds), len(test_preds), len(final_test_preds)
    return final_test_preds

def make_sub_optimized(stacker, base_clfs, filename):
    preds = optimized_test_predictions(stacker, base_clfs)
    #print "len(preds)"
    #print len(preds)
    df = pd.DataFrame()
    df['Id'] = test.Id
    df['Response'] = preds

    df.to_csv(filename, index=False)
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

xgbr = lambda: XGBRegressor(objective="reg:linear", min_child_weight=80, subsample=0.85, colsample_bytree=0.30, silent=1, max_depth=9)
xgbc = lambda: XGBClassifier(objective="reg:linear", min_child_weight=80, subsample=0.85, colsample_bytree=0.30, silent=1, max_depth=9)
rfr = lambda: RandomForestRegressor(n_estimators=400)
etr = lambda: ExtraTreesRegressor(n_estimators=400)
etc = lambda: ExtraTreesClassifier(n_estimators=400)
sgdr = lambda: SGDRegressor()
perceptron = lambda: Perceptron()

dream_team = lambda: sorted([xgbr(), rfr(),  etr(), LinearRegression(), Perceptron(),
              #SVR(kernel="linear"), 
              #SVR(kernel="poly"), 
              SVR(kernel="sigmoid"),
              SVR(kernel="rbf")])

def make_sub(stacker, base_clfs, filename):
    preds = stacker_test_predictions(stacker, base_clfs)
    
    df = pd.DataFrame()
    df['Id'] = test.Id
    df['Response'] = preds

    df.to_csv(filename, index=False)
