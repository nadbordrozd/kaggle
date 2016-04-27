import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from commons import logger, cache, get_func_name
from feature_engineering import train_size, y, train_X, test_X, test_id_col
from sklearn.cross_validation import KFold, ShuffleSplit

np.random.seed(0)
inds_2000 = np.random.choice(range(train_size), 2000, False)
train_1000 = inds_2000[:1000]
test_1000 = inds_2000[1000:]

np.random.seed(0)
inds_20000 = np.random.choice(range(train_size), 20000, False)
train_10000 = inds_20000[:10000]
test_10000 = inds_20000[10000:]

@cache
def fit_model(model_fun, X, y, feats):
    model = model_fun()
    logger.info("fitting %s with feat extractor %s on %s examples" 
                % (model, feats, len(y)))
    model.fit(X, y)
    logger.info("DONE fitting %s with feat extractor %s on %s examples" 
                % (model, feats, len(y)))
    return model

def benchmark(model_fun, feature_extractor, train_inds, test_inds):
    X = feature_extractor()
    if type(X) == dict:
        X_train = {}
        for k, v in X.items():
            X_train[k] = v[train_inds]
        X_train = {k: v[train_inds] for k, v in X.items()}
        X_test = {k: v[test_inds] for k, v in X.items()}
    else:
        X_train = X[train_inds]
        X_test = X[test_inds]
    y_train = y[train_inds]
    y_test = y[test_inds]

    model = fit_model(model_fun, X_train, y_train, get_func_name(feature_extractor))
    #model.fit(X_train, y_train)

    return np.sqrt(mean_squared_error(model.predict(X_test), y_test))
    
def tiny_benchmark(model_fun, feature_extractor):
    return benchmark(model_fun, feature_extractor, train_1000, test_1000)

def small_benchmark(model_fun, feature_extractor):
    return benchmark(model_fun, feature_extractor, train_10000, test_10000)

def full_cv(model_fun, feature_extractor, folds=4, return_all_folds=False):
    results = []
    for i, (train_index, test_index) in enumerate(KFold(n=train_size, n_folds=folds, shuffle=True, random_state=0)):
        logger.info("benchmarking fold %s" % i)
        results.append(benchmark(model_fun, feature_extractor, train_index, test_index))
    if return_all_folds:
        return results
    else:
        return np.mean(results)

def benme(model_fun, fun, modelname=None, funname=None):
    # default benchmarking function
    if funname is None:
        funname = get_func_name(fun)
    if modelname is None:
        modelname = str(model)
    result = full_cv(model_fun, fun)
    logger.info("got score %.4f with model %s on feats %s" % (result, modelname, funname))  
    


def make_submission(model_fun, feature_extractor, filename):
    fun_name = get_func_name(feature_extractor)
    logger.info("making submission with %s and %s" % (model, fun_name))
    model = fit_model(model_fun, train_X(feature_extractor), y, fun_name)
    predictions = model.predict(test_X(feature_extractor))
    predictions = [min(max(p, 1), 3) for p in predictions]
    pd.DataFrame({'id': test_id_col, 'relevance': predictions}).to_csv(filename, index=False)
    logger.info("submission made")
    
                                
                                
    