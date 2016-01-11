# first line: 1
@stacker_train_cache.cache
def stacker_train_predictions(stacker, base_clfs):
    n = len(y)
    stacked_X = np.hstack([X] + [train_predictions(clf).reshape(n, 1) for clf in base_clfs])
    
    ind2pred = {}
    for train, test in train_test_folds:
        stacker.fit(stacked_X[train], y[train])
        preds = stacker.predict(stacked_X[test])
        for i, p in zip(test, preds):
            ind2pred[i] = p
    
    return np.array([ind2pred[i] for i in range(len(y))])
