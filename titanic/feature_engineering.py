from __future__ import division

from collections import Counter
import numpy as np
import pandas as pd
import scipy
from scipy.spatial.distance import euclidean, cosine, norm
from functools import partial
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from spacy.en import English
from itertools import izip
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder

from commons import logger, cache, DATA_DIR


train_path = DATA_DIR + "train.csv"
test_path = DATA_DIR + "test.csv"
descriptions_path = DATA_DIR + "product_descriptions.csv"

nlp = English()

df_train = pd.read_csv('data/act_train.csv')
y = np.array(df_train.outcome)
df_test = pd.read_csv('data/act_test.csv')
test_id_col = df_test.activity_id
df_peeps = pd.read_csv('data/people.csv')

df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)


total_size = len(df_all)
train_size = len(df_train)
test_size = len(df_test)


def test_X(feature_extractor):
    tot_X = feature_extractor()
    return feature_extractor()[train_size:, :]


def train_X(feature_extractor):
    return feature_extractor()[:train_size, :]

def oh_activity_cat():
    return LabelBinarizer().fit_transform(df_all.activity_category.map(str))
def oh_char1():
    return LabelBinarizer().fit_transform(df_all.char_1.map(str))
def oh_char2():
    return LabelBinarizer().fit_transform(df_all.char_2.map(str))
def oh_char3():
    return LabelBinarizer().fit_transform(df_all.char_3.map(str))
def oh_char4():
    return LabelBinarizer().fit_transform(df_all.char_4.map(str))
def oh_char5():
    return LabelBinarizer().fit_transform(df_all.char_5.map(str))
def oh_char6():
    return LabelBinarizer().fit_transform(df_all.char_6.map(str))
def oh_char7():
    return LabelBinarizer().fit_transform(df_all.char_7.map(str))
def oh_char8():
    return LabelBinarizer().fit_transform(df_all.char_8.map(str))
def oh_char9():
    return LabelBinarizer().fit_transform(df_all.char_9.map(str))
def oh_char10():
    return LabelBinarizer().fit_transform(df_all.char_10.map(str))


def f1():
    return np.hstack([oh_char1(), oh_char2(), oh_char3(), oh_char4(), 
                      oh_char5(), oh_char6(), oh_char7(), oh_char8(),
                      oh_char9(), oh_char10()])