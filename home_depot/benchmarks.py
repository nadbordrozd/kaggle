import numpy as np
from sklearn.metrics import mean_squared_error

from commons import logger, cache, get_func_name
from feature_engineering import train_size, y

np.random.seed(0)
inds_2000 = np.random.choice(range(train_size), 2000, False)
train_1000 = inds_2000[:1000]
test_1000 = inds_2000[1000:]

np.random.seed(0)
inds_20000 = np.random.choice(range(train_size), 20000, False)
train_10000 = inds_20000[:10000]
test_10000 = inds_20000[10000:]


def benchmark(model, feature_extractor, train_inds, test_inds):
    X = feature_extractor()
    X_train, y_train = X[train_1000], y[train_1000]
    X_test, y_test = X[test_1000], y[test_1000]
    logger.info("fitting %s with feat extractor %s on %s examples" 
                % (model, get_func_name(feature_extractor), len(y_train)))
    model.fit(X, y)
    logger.info("DONE fitting %s with feat extractor %s on %s examples" 
                % (model, get_func_name(feature_extractor), len(y_train)))
    return np.sqrt(mean_squared_error(model.predict(X_test), y_test))


def tiny_benchmark(model, feature_extractor):
    return benchmark(model, feature_extractor, train_1000, test_1000)

def small_benchmark(model, feature_extractor):
    return benchmark(model, feature_extractor, train_10000, test_10000)