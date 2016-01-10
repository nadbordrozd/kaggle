# first line: 1
@train_cache.cache
def train_predictions(model):
    ind2pred = {}
    for train, test in train_test_folds:
        model.fit(X[train], y[train])
        preds = model.predict(X[test])
        for i, p in zip(test, preds):
            ind2pred[i] = p
    
    return np.array([ind2pred[i] for i in range(len(y))])
