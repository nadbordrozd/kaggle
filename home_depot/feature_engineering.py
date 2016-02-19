import numpy as np
import pandas as pd

from commons import logger, cache
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

df_train = pd.read_csv("data/train.csv", encoding="ISO-8859-1")
y = np.array(df_train.relevance)
train_size = len(y)
df_test = pd.read_csv("data/test.csv", encoding="ISO-8859-1")
df_pro_desc = pd.read_csv("data/product_descriptions.csv")
# temporarily commented out, extracting feature from the entire set was taking too long
# playing with just the training set will be enough for now
#df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = df_train
df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')

total_size = len(df_all)

stemmer = SnowballStemmer('english')

def str_stemmer(s):
    return [stemmer.stem(word) for word in s.lower().split()]

@cache
def search_tokens():
    logger.info("tokenizing search terms, should be over soon")
    result = df_all.search_term.map(str_stemmer)
    logger.info("done tokenizing search terms")
    return result

@cache
def description_tokens():
    logger.info("tokenizing product descriptions, this may take a LONG while")
    result = df_all.product_description.map(str_stemmer)
    logger.info("done tokenizing, no more tears")
    return result


@cache
def title_tokens():
    logger.info("tokenizing product titles, this shouldn't take long at all")
    result = df_all.product_description.map(str_stemmer)
    logger.info("done tokenizing titles")
    return result


@cache
def descr_same_tokens_count():
    search_count = np.array([len(set(a).intersection(set(b))) 
                             for a, b in zip(search_tokens(), description_tokens())]
                           ).reshape(total_size, 1)
    return search_count

@cache
def title_same_tokens_count():
    title_count = np.array([len(set(a).intersection(set(b))) 
                             for a, b in zip(search_tokens(), title_tokens())]
                           ).reshape(total_size, 1)
    return title_count

def identity(x):
    return x

@cache
def tfidf_descr_vectorizer():
    vectorizer = TfidfVectorizer(analyzer=identity)
    tfidf_descr = vectorizer.fit_transform(description_tokens())
    return tfidf_descr, vectorizer

@cache
def tfidf_descr():
    descr, _ = tfidf_descr_vectorizer()
    return descr

@cache
def tfidf_search():
    _, vectorizer = tfidf_descr_vectorizer()
    return vectorizer.transform(search_tokens())

@cache
def tfidf_title():
    _, vectorizer = tfidf_descr_vectorizer()
    return vectorizer.transform(title_tokens())

@cache
def tfidf_search_times_descr():
    search = tfidf_search()
    descr = tfidf_descr()
    return np.array([np.float(a.dot(b.T).todense())
                     for a, b in zip(search, descr)]).reshape(total_size, 1)

@cache
def tfidf_search_times_title():
    search = tfidf_search()
    title = tfidf_title()
    return np.array([np.float(a.dot(b.T).todense())
                     for a, b in zip(search, title)]).reshape(total_size, 1)

@cache
def len_of_query():
    return np.array([len(x) for x in search_tokens()]).reshape(total_size, 1)


@cache
def features_0():
    return np.hstack([descr_same_tokens_count(), title_same_tokens_count(),
                      len_of_query()])

@cache
def features_1():
    return np.hstack([tfidf_search_times_descr(), tfidf_search_times_title(),
                      descr_same_tokens_count(), title_same_tokens_count(),
                      len_of_query()])
                      
                      
