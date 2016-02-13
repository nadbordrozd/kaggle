from persistent_cache import memo, PersistentDict as Perd
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from joblib import Memory
from sklearn.preprocessing import PolynomialFeatures

def get_y():
    train = pd.read_csv("train.csv")
    return train.Response

def cut_uncorrelated(X_train, X_test, y, threshold):
    corrs = []
    for i in range(X_train.shape[1]):
        corrs.append(np.corrcoef(X_train[:, i], y)[0][1])
    good_corrs = filter(lambda (_, x): abs(x) > threshold, enumerate(corrs))
    good_inds = [i for i, _ in good_corrs]
    return X_train[:, good_inds], X_test[:, good_inds]
    
def oh_med():
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
    #return np.random.random((2*10**5, 10**3)), np.random.random((2*10**4, 10**3))
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
    X = np.array(oh_encoder.transform(X).todense())
    X_actual_test = np.array(oh_encoder.transform(X_actual_test).todense())
    return X, X_actual_test

def oh_med_cut():
    x, xx = train_test_sets("ohmed")
    y = get_y()
    return cut_uncorrelated(x, xx, y, 0.01)

def read_all_data():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    # combine train and test
    all_data = train.append(test)
    all_data.Response.fillna(-1, inplace=True)
    # factorize categorical variables    
    all_data['Product_Info_2'] = pd.factorize(all_data['Product_Info_2'])[0]
    return all_data

def basic_extractor():
    all_data = read_all_data()
    
    print('Eliminate missing values')    
    # Use -1 for any others
    all_data.fillna(-1, inplace=True)

    # fix the dtype on the label column
    all_data['Response'] = all_data['Response'].astype(int)

    # split train and test
    train = all_data[all_data['Response']>0].copy()
    test = all_data[all_data['Response']<1].copy()

    X = np.array(train.drop(["Id", "Response"], axis=1))
    X_actual_test = np.array(test.drop(["Id", "Response"], axis=1))
    y = np.array(train.Response)
    return X, X_actual_test


def fe2():
    """not one-hot encoded
    """
    all_data = read_all_data()

    
    # FEATURE ENGINEERING
    all_data['bmi_ins_age'] = all_data.BMI * all_data.Ins_Age
    all_data['nan_count'] = all_data.isnull().sum(axis=1)
    #all_data['emp_inf_4_sq'] = all_data.Employment_Info_4 ** 2
    #all_data['fam_hist_4_sq'] = all_data.Family_Hist_4 ** 2
    #all_data['fam_hist_2_sq'] = all_data.Family_Hist_2 ** 2

    mk = [col for col in all_data.columns if col.startswith("Medical_K")]
    all_data['sum_keywords'] = sum(all_data[col] for col in mk)

    all_data.drop('Medical_History_24')
    all_data.drop('Medical_History_10')

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
    return X, X_actual_test

def nan_count():
    all_data = read_all_data()
    all_data['nan_count'] = all_data.isnull().sum(axis=1)
    train = all_data[all_data['Response']>0].copy()
    test = all_data[all_data['Response']<1].copy()
    return train.nan_count.reshape(len(train), 1), test.nan_count.reshape(len(test), 1)

def keywords_count():
    all_data = read_all_data()
    mk = [col for col in all_data.columns if col.startswith("Medical_K")]
    all_data['sum_keywords'] = sum(all_data[col] for col in mk)
    train = all_data[all_data['Response']>0].copy()
    test = all_data[all_data['Response']<1].copy()
    return train.sum_keywords.reshape(len(train), 1), test.sum_keywords.reshape(len(test), 1)
    
def fe2_median():
    """not one-hot encoded
    """
    all_data = read_all_data()
    all_data.fillna(all_data.median(), inplace=True)

    # FEATURE ENGINEERING
    all_data['bmi_ins_age'] = all_data.BMI * all_data.Ins_Age
    all_data['nan_count'] = all_data.isnull().sum(axis=1)
    mk = [col for col in all_data.columns if col.startswith("Medical_K")]
    all_data['sum_keywords'] = sum(all_data[col] for col in mk)
    all_data.drop('Medical_History_24')
    all_data.drop('Medical_History_10')
    all_data['Response'] = all_data['Response'].astype(int)
    train = all_data[all_data['Response']>0].copy()
    test = all_data[all_data['Response']<1].copy()
    X = np.array(train.drop(["Id", "Response"], axis=1))
    X_actual_test = np.array(test.drop(["Id", "Response"], axis=1))
    y = np.array(train.Response)
    return X, X_actual_test
 
def kmeans_feats(X, clusters=10):
    X = Normalizer().transform(X)
    kmeans = MiniBatchKMeans(n_clusters=clusters, random_state=0)
    return kmeans.fit_transform(X)
    
def kmns_10():
    X, X_test = train_test_sets("ohmed")
    n, _ = X.shape
    tot = np.vstack([X, X_test])
    kmeans = kmeans_feats(tot, clusters=10)
    return kmeans[:n, :], kmeans[n:, :]
    
def kmns_20():
    X, X_test = train_test_sets("ohmed")
    n, _ = X.shape
    tot = np.vstack([X, X_test])
    kmeans = kmeans_feats(tot, clusters=20)
    return kmeans[:n, :], kmeans[n:, :]

def kmns_40():
    X, X_test = train_test_sets("ohmed")
    n, _ = X.shape
    tot = np.vstack([X, X_test])
    kmeans = kmeans_feats(tot, clusters=40)
    return tot[:n, :], tot[n:, :]

def corr(a, b):
    return np.corrcoef(a, b)[0][1]

def add_poly_feats(X_train, X_test, y, threshold):
    corrs = []
    for i in range(X_train.shape[1]):
        corrs.append(corr(X_train[:, i], y))
    gooduns = [i for i, c in enumerate(corrs) if abs(c) > threshold]
    rest = [i for i, c in enumerate(corrs) if abs(c) <= threshold]
    tot = np.vstack([X_train, X_test])
    good_feats = tot[:, gooduns]
    meh_feats = tot[:, rest]
    poly_feats = PolynomialFeatures(2).fit_transform(good_feats)
    assert set(poly_feats[:, 0]) == {1}
    new_tot = np.hstack([poly_feats, meh_feats[:, 1:]])
    n = len(y)
    return new_tot[:n, :], new_tot[n:, :]


combiner_memo = Memory(cachedir="fecache/combiner", verbose=0)
@combiner_memo.cache
def combine(fextractors):
    trains, tests = [], []
    for f in fextractors:
        train, test = train_test_sets(f)
        trains.append(train)
        tests.append(test)
    return np.hstack(trains), np.hstack(tests)    
    
def oh_kmns10():
    return combine(["ohmed", "kmns10"])
    
def oh_kmns20():
    return combine(["ohmed", "kmns20"])

def oh_kmns40():
    return combine(["ohmed", "kmns40"])


def fe2_kmns10():
    return combine(["feats2", "kmns10"])

def fe2_kmns20():
    return combine(["feats2", "kmns20"])

def fe2_kmns40():
    return combine(["feats2", "kmns40"])

def ohmedcut_kmns10():
    return combine(["oh_med_cut", "kmns10"])

def ohmedcut_kmns20():
    return combine(["oh_med_cut", "kmns20"])

def ohmedcut_kmns40():
    return combine(["oh_med_cut", "kmns40"])

def ohmedcut_nan_count():
    return combine(["oh_med_cut", "nan_count"])

def ohmedcut_keywords_count():
    return combine(["oh_med_cut", "keywords_count"])

def ohmedcut_kw_nan():
    return combine(["oh_med_cut", "keywords_count", "nan_count"])

def ohmedcut_kw_nan_poly02():
    X_train, X_test = ohmedcut_kw_nan()
    return add_poly_feats(X_train, X_test, get_y(), 0.2)

def ohmedcut_kw_nan_poly015():
    X_train, X_test = ohmedcut_kw_nan()
    return add_poly_feats(X_train, X_test, get_y(), 0.15)

def ohmedcut_kw_nan_poly012():
    X_train, X_test = ohmedcut_kw_nan()
    return add_poly_feats(X_train, X_test, get_y(), 0.12)

def ohmedcut_kw_nan_poly01():
    X_train, X_test = ohmedcut_kw_nan()
    return add_poly_feats(X_train, X_test, get_y(), 0.1)

def ohmedcut_kw_nan_poly008():
    X_train, X_test = ohmedcut_kw_nan()
    return add_poly_feats(X_train, X_test, get_y(), 0.08)

def ohmedcut_poly02():
    X_train, X_test = ohmedcut_kw_nan()
    return add_poly_feats(X_train, X_test, get_y(), 0.2)

def ohmedcut_poly015():
    X_train, X_test = ohmedcut_kw_nan()
    return add_poly_feats(X_train, X_test, get_y(), 0.15)

def ohmedcut_poly012():
    X_train, X_test = ohmedcut_kw_nan()
    return add_poly_feats(X_train, X_test, get_y(), 0.12)

def ohmedcut_poly01():
    X_train, X_test = ohmedcut_kw_nan()
    return add_poly_feats(X_train, X_test, get_y(), 0.1)

def ohmedcut_poly008():
    X_train, X_test = ohmedcut_kw_nan()
    return add_poly_feats(X_train, X_test, get_y(), 0.08)


extractors = {
    'basic': basic_extractor,
    'feats2': fe2,
    'ohmed': oh_med,
    'kmns10': kmns_10,
    'kmns20': kmns_20,
    'kmns40': kmns_40,
    'oh_kmns10': oh_kmns10,
    'oh_kmns20': oh_kmns20,
    'oh_kmns40': oh_kmns40,
    'fe2_kmns10': fe2_kmns10,
    'fe2_kmns20': fe2_kmns20,
    'fe2_kmns40': fe2_kmns40,
    "fe2_median": fe2_median,
    'oh_med_cut': oh_med_cut,
    'ohmedcut_kmns10': ohmedcut_kmns10,
    'ohmedcut_kmns20': ohmedcut_kmns20,
    'ohmedcut_kmns40': ohmedcut_kmns40,
    'nan_count': nan_count,
    'keywords_count': keywords_count,
    'ohmedcut_kwcount': ohmedcut_keywords_count,
    'ohmedcut_nancount': ohmedcut_nan_count,
    'ohmedcut_kw_nan': ohmedcut_kw_nan,
    'ohmedcut_kw_nan_poly02': ohmedcut_kw_nan_poly02,
    'ohmedcut_kw_nan_poly015': ohmedcut_kw_nan_poly015,
    'ohmedcut_kw_nan_poly012': ohmedcut_kw_nan_poly012,
    'ohmedcut_kw_nan_poly01': ohmedcut_kw_nan_poly01,
    'ohmedcut_kw_nan_poly008': ohmedcut_kw_nan_poly008,
    'ohmedcut_poly02': ohmedcut_poly02,
    'ohmedcut_poly015': ohmedcut_poly015,
    'ohmedcut_poly012': ohmedcut_poly012,
    'ohmedcut_poly01': ohmedcut_poly01,
    'ohmedcut_poly008': ohmedcut_poly008,
}

memo = Memory(cachedir="fecache/traintestset", verbose=0)
@memo.cache
def train_test_sets(extractor_name):
    """extractor_name one of:
    'basic'
    'feats1'
    returns X_train, X_test
    """
    extractor = extractors[extractor_name]
    return extractor()