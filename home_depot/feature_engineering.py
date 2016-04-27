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
from sklearn.preprocessing import PolynomialFeatures

from commons import logger, cache, DATA_DIR


train_path = DATA_DIR + "train.csv"
test_path = DATA_DIR + "test.csv"
descriptions_path = DATA_DIR + "product_descriptions.csv"

nlp = English()

df_train = pd.read_csv(train_path, encoding="ISO-8859-1")
y = np.array(df_train.relevance)
df_test = pd.read_csv(test_path, encoding="ISO-8859-1")
test_id_col = df_test.id
df_pro_desc = pd.read_csv(descriptions_path, encoding="ISO-8859-1")
attributes = pd.read_csv("data/attributes.csv", encoding="ISO-8859-1")

df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')

total_size = len(df_all)
train_size = len(df_train)
test_size = len(df_test)


def test_X(feature_extractor):
    tot_X = feature_extractor()
    if type(tot_X) == dict:
        X = {}
        for k, v in tot_X.items():
            X[k] = v[train_size:, :]
        return X
    else:
        return tot_X[train_size:, :]


def train_X(feature_extractor):
    tot_X = feature_extractor()
    if type(tot_X) == dict:
        X = {}
        for k, v in tot_X.items():
            X[k] = v[:train_size, :]
        return X
    else:
        return tot_X[:train_size, :]

stemmer = SnowballStemmer('english')


@cache
def brand():
    brand_rows = attributes[attributes.name == "MFG Brand Name"]
    prod2brand = dict(zip(brand_rows.product_uid, brand_rows.value))
    return df_all.product_uid.map(lambda x: prod2brand.get(x)).map(lambda x: None if type(x) != unicode else x)


@cache
def brand_lhood_mnb():
    sle = search_lemmas()
    vectorizer = TfidfVectorizer(analyzer=lambda x: x)
    vectorizer.fit(sle)
    brands = [x if type(x) == str else "PLACEHOLDER" for x in brand()]
    brand_set = list(set(brands))
    model = MultinomialNB()
    # have to fit in steps, otherwise running out of memory
    step = 40000
    n = len(sle)
    for i in range(int(n / step) + 1):
        model.partial_fit(vectorizer.transform(
            sle[step * i:step * (i + 1)]), brands[step * i:step * (i + 1)], classes=brand_set)

    class2ind = {cls: ind for ind, cls in enumerate(model.classes_)}
    likelihoods = []
    for i in range(int(n / step) + 1):
        proba_preds = model.predict_proba(
            vectorizer.transform(sle[step * i:step * (i + 1)]))
        for propre, actual_brand in izip(proba_preds, brands[step * i:step * (i + 1)]):
            lhood = propre[class2ind[actual_brand]]
            likelihoods.append(lhood)
    return np.array(likelihoods).reshape(total_size, 1)


def attribute_present(attr):
    prods = set(attributes[attributes.name == attr].product_uid)
    return np.array(df_all.product_uid.map(lambda x: x in prods)).reshape(total_size, 1)


def attr_lhood(attr):
    vectorizer = TfidfVectorizer(analyzer=lambda x: x)
    tfidf = vectorizer.fit_transform(search_lemmas())
    model = MultinomialNB()
    model.fit(tfidf, attribute_present(attr).ravel())
    return model.predict_proba(tfidf)[:, 1].reshape(total_size, 1)


def random_features():
    return np.random.random((total_size, 10))


def str_stemmer(s):
    return [stemmer.stem(word) for word in s.lower().split()]


def product2attributes():
    bad_count = 0
    prod2attrs = {}
    for prodid, name, value in zip(attributes.product_uid, attributes.name, attributes.value):
        if np.isnan(prodid):
            bad_count += 1
            continue
        prod2attrs.setdefault(int(prodid), []).append(unicode(value))
    logger.info("found %s empty (malformed?) rows in attributes")
    return prod2attrs


@cache
def attributes_texts():
    prod2attrs = product2attributes()
    return df_all.product_uid.map(lambda x: "\n".join(prod2attrs.get(x, [])))


def attributes_spacy_iterator():
    for attrs in attributes_texts():
        yield nlp(unicode(attrs))


def description_spacy_iterator():
    for description in df_all.product_description:
        yield nlp(unicode(description))


def search_spacy_iterator():
    for text in df_all.search_term:
        yield nlp(unicode(text))


def title_spacy_iterator():
    for text in df_all.product_title:
        yield nlp(unicode(text))


def lemma_tokens(spacy_tokens):
    return [t.lemma_ for t in spacy_tokens]


@cache
def attributes_lemmas():
    return map(lemma_tokens, attributes_spacy_iterator())


@cache
def description_lemmas():
    return map(lemma_tokens, description_spacy_iterator())


@cache
def title_lemmas():
    return map(lemma_tokens, title_spacy_iterator())


@cache
def search_lemmas():
    return map(lemma_tokens, search_spacy_iterator())


@cache
def title_same_lemmas():
    title_count = np.array([len(set(a).intersection(set(b)).difference({"'"}))
                            for a, b in izip(search_lemmas(), title_lemmas())]
                           ).reshape(total_size, 1)
    return title_count


@cache
def descr_same_lemmas():
    search_count = np.array([len(set(a).intersection(set(b)).difference({"'"}))
                             for a, b in izip(search_lemmas(), description_lemmas())]
                            ).reshape(total_size, 1)
    return search_count


@cache
def attr_same_lemmas():
    search_count = np.array([len(set(a).intersection(set(b)).difference({"'"}))
                             for a, b in izip(search_lemmas(), attributes_lemmas())]
                            ).reshape(total_size, 1)
    return search_count


@cache
def search_title_same_lemmas_list():
    return np.array([list(set(search).intersection(set(title)))
                     for search, title in zip(search_lemmas(), title_lemmas())]).reshape(total_size, 1)


@cache
def search_descr_same_lemmas_list():
    return np.array([list(set(search).intersection(set(title)))
                     for search, title in zip(search_lemmas(), description_lemmas())]).reshape(total_size, 1)


@cache
def search_attr_same_lemmas_list():
    return np.array([list(set(search).intersection(set(attr)))
                     for search, attr in zip(search_lemmas(), attributes_lemmas())]).reshape(total_size, 1)


@cache
def search_title_description_spacy_similarities():
    st = []
    sd = []
    td = []
    for s, t, d in izip(search_spacy_iterator(), title_spacy_iterator(), description_spacy_iterator()):
        st.append(s.similarity(t))
        sd.append(s.similarity(d))
        td.append(t.similarity(d))
    return st, sd, td


@cache
def search_title_spacy_similarity():
    st = []
    for s, t in izip(search_spacy_iterator(), title_spacy_iterator()):
        st.append(s.similarity(t))
    return np.array(st).reshape(total_size, 1)


@cache
def search_descr_spacy_similarity():
    sd = []
    for s, d in izip(search_spacy_iterator(), description_spacy_iterator()):
        sd.append(s.similarity(d))
    return np.array(sd).reshape(total_size, 1)


@cache
def search_attr_spacy_similarity():
    sd = []
    for s, a in izip(search_spacy_iterator(), attributes_spacy_iterator()):
        sd.append(s.similarity(a))
    return np.array(sd).reshape(total_size, 1)


@cache
def search_title_bigrams_count():
    counts = []
    for search, title in izip(search_lemmas(), title_lemmas()):
        search_bigrams = set([(a, b) for a, b in izip(search, search[1:])])
        title_bigrams = set([(a, b) for a, b in izip(title, title[1:])])
        counts.append(len(search_bigrams.intersection(title_bigrams)))
    return np.array(counts).reshape(total_size, 1)


@cache
def search_descr_bigrams_count():
    counts = []
    for search, descr in izip(search_lemmas(), description_lemmas()):
        search_bigrams = set([(a, b) for a, b in izip(search, search[1:])])
        descr_bigrams = set([(a, b) for a, b in izip(descr, descr[1:])])
        counts.append(len(search_bigrams.intersection(descr_bigrams)))
    return np.array(counts).reshape(total_size, 1)


@cache
def search_attr_bigrams_count():
    counts = []
    for search, attr in izip(search_lemmas(), attributes_lemmas()):
        search_bigrams = set([(a, b) for a, b in izip(search, search[1:])])
        attr_bigrams = set([(a, b) for a, b in izip(attr, attr[1:])])
        counts.append(len(search_bigrams.intersection(attr_bigrams)))
    return np.array(counts).reshape(total_size, 1)


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
    result = df_all.product_title.map(str_stemmer)
    logger.info("done tokenizing titles")
    return result


@cache
def attributes_tokens():
    logger.info("tokenizing product attributes")
    result = attributes_texts().map(str_stemmer)
    logger.info("done tokenizing, no more teara")
    return result


@cache
def descr_same_tokens_count():
    search_count = np.array([len(set(a).intersection(set(b)))
                             for a, b in izip(search_tokens(), description_tokens())]
                            ).reshape(total_size, 1)
    return search_count


@cache
def title_same_tokens_count():
    title_count = np.array([len(set(a).intersection(set(b)))
                            for a, b in izip(search_tokens(), title_tokens())]
                           ).reshape(total_size, 1)
    return title_count


@cache
def attr_same_tokens_count():
    attr_count = np.array([len(set(a).intersection(set(b)))
                           for a, b in izip(search_tokens(), attributes_tokens())]
                          ).reshape(total_size, 1)
    return attr_count


@cache
def last_search_token_in_title():
    return np.array([a[-1] in b
                     for a, b in izip(search_tokens(), title_tokens())]
                    ).reshape(total_size, 1)


@cache
def last_search_token_in_descr_count():
    return np.array([b.count(a[-1])
                     for a, b in izip(search_tokens(), title_tokens())]
                    ).reshape(total_size, 1)


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
def tfidf_attr():
    _, vectorizer = tfidf_descr_vectorizer()
    return vectorizer.transform(attributes_tokens())


@cache
def tfidf_search_times_descr():
    search = tfidf_search()
    descr = tfidf_descr()
    return np.array([np.float(a.dot(b.T).todense())
                     for a, b in izip(search, descr)]).reshape(total_size, 1)


@cache
def tfidf_search_times_title():
    search = tfidf_search()
    title = tfidf_title()
    return np.array([np.float(a.dot(b.T).todense())
                     for a, b in izip(search, title)]).reshape(total_size, 1)


@cache
def tfidf_search_times_attr():
    search = tfidf_search()
    attr = tfidf_attr()
    return np.array([np.float(a.dot(b.T).todense())
                     for a, b in izip(search, attr)]).reshape(total_size, 1)


@cache
def len_of_query():
    return np.array([len(x) for x in search_tokens()]).reshape(total_size, 1)


@cache
def unique_desc_tokens():
    return list(df_pro_desc.product_description.map(str_stemmer))


@cache
def unique_descr_lemmas():
    return df_pro_desc.product_description.map(lambda x: lemma_tokens(nlp(x)))


@cache
def stemmed_descr_w2v():
    logger.info("tokenizing and stemming descriptions")
    sentences = unique_desc_tokens()
    logger.info(
        "done tokenizing and stemming descriptions. now on to wordtovecing")
    w2v = Word2Vec(sentences)
    logger.info("done word2vecing")
    return w2v


@cache
def lemma_descr_w2v():
    logger.info("lemmatizing descriptions")
    sentences = unique_descr_lemmas()
    logger.info("done lemmatizing descriptions. now on to wordtovecing")
    w2v = Word2Vec(sentences)
    logger.info("done word2vecing")
    return w2v


def mean_sentence_vec(sentence, w2v):
    vectors = [w2v[t] for t in sentence if t in w2v]
    if not vectors:
        n = w2v[w2v.index2word[0]].shape[0]
        return np.zeros(n)
    else:
        return np.mean(vectors, axis=0)


@cache
def title_vecs():
    w2v = stemmed_descr_w2v()
    return [mean_sentence_vec(title, w2v) for title in title_tokens()]


@cache
def search_vecs():
    logger.info("in search_vecs function. loading w2v")
    w2v = stemmed_descr_w2v()
    logger.info("done loading w2v. now taking means of word vectors")
    result = [mean_sentence_vec(search, w2v) for search in search_tokens()]
    logger.info("done taking means of word vectors")
    return result


@cache
def descr_vecs():
    w2v = stemmed_descr_w2v()
    return [mean_sentence_vec(descr, w2v) for descr in description_tokens()]


@cache
def attr_vecs():
    w2v = stemmed_descr_w2v()
    return [mean_sentence_vec(descr, w2v) for descr in attributes_tokens()]


@cache
def title_vecs_l():
    w2v = lemma_descr_w2v()
    return [mean_sentence_vec(title, w2v) for title in title_lemmas()]


@cache
def search_vecs_l():
    logger.info("in search_vecs_l function. loading w2v")
    w2v = lemma_descr_w2v()
    logger.info("done loading w2v. now taking means of word vectors")
    result = [mean_sentence_vec(search, w2v) for search in search_lemmas()]
    logger.info("done taking means of word vectors")
    return result


@cache
def descr_vecs_l():
    w2v = lemma_descr_w2v()
    return [mean_sentence_vec(descr, w2v) for descr in description_lemmas()]


@cache
def attr_vecs_l():
    w2v = lemma_descr_w2v()
    return [mean_sentence_vec(descr, w2v) for descr in attributes_lemmas()]


@cache
def search_title_cossim():
    return np.array([cosine(a, b) if np.linalg.norm(a) * np.linalg.norm(b) > 0 else 2
                     for a, b in izip(search_vecs(), title_vecs())
                     ]).reshape(total_size, 1)


@cache
def search_descr_cossim():
    return np.array([cosine(a, b) if np.linalg.norm(a) * np.linalg.norm(b) > 0 else 2
                     for a, b in izip(search_vecs(), descr_vecs())
                     ]).reshape(total_size, 1)


@cache
def search_attr_cossim():
    return np.array([cosine(a, b) if np.linalg.norm(a) * np.linalg.norm(b) > 0 else 2
                     for a, b in izip(search_vecs(), attr_vecs())
                     ]).reshape(total_size, 1)


def glove_w2v():
    logger.info(
        "loading glove vectors trained 300d, 2m vectors, this may take a while")
    w2v = {}
    with open("data/glove.840B.300d.txt", "rb") as lines:
        for line in lines:
            parts = line.split()
            w = parts[0]
            nums = parts[1:]
            vec = np.array(map(float, nums))
            w2v[w] = vec
    logger.info("done loading glove vectors wooohooo")
    return w2v


def is_good_token(t):
    return not (t.is_punct or t.is_stop or t.is_digit or t.is_space or t.is_bracket or t.is_quote)


def text_to_vec_orth(spacy_text, w2v, dim):
    vectors = [w2v[t.orth_]
               for t in spacy_text if t.orth_ in w2v and is_good_token(t)]
    return np.mean(vectors, axis=0) if vectors else np.zeros(dim)

@cache
def token_counts():
    return Counter([c for lems in unique_descr_lemmas() for c in lems])

def text_to_vec_tfidf(spacy_text, w2v, dim, wordcount):
    N = len(df_pro_desc)
    vectors = [w2v[t.lemma_] * np.log(N / min(N, wordcount.get(t.lemma_, 0) + 5))
               for t in spacy_text if t.lemma_ in w2v and is_good_token(t)]
    return np.mean(vectors, axis=0) if vectors else np.zeros(dim)


def text_to_vec_lemma(spacy_text, w2v, dim):
    vectors = [w2v[t.lemma_]
               for t in spacy_text if t.lemma_ in w2v and is_good_token(t)]
    return np.mean(vectors, axis=0) if vectors else np.zeros(dim)


def text_vec_dists(spacy_iterator_A, spacy_iterator_B, w2v, comparator, text_to_vec):
    dists = []
    if type(w2v) == dict:
        dim = len(w2v.itervalues().next())
    else:
        _, dim = w2v.syn0.shape
    for a, b in izip(spacy_iterator_A, spacy_iterator_B):
        v1 = text_to_vec(a, w2v, dim)
        v2 = text_to_vec(b, w2v, dim)
        n1, n2 = norm(v1), norm(v2)
        if n1 * n2 == 0:
            dists.append(-1)
        else:
            dists.append(comparator(v1, v2))
    return np.array(dists).reshape(total_size, 1)


@cache
def search_title_cossim_l():
    return text_vec_dists(title_spacy_iterator(),
                          search_spacy_iterator(),
                          lemma_descr_w2v(),
                          cosine,
                          text_to_vec_lemma)


@cache
def search_title_cossim_l_tfidf():
    return text_vec_dists(title_spacy_iterator(),
                          search_spacy_iterator(),
                          lemma_descr_w2v(),
                          cosine,
                          partial(text_to_vec_tfidf, wordcount=token_counts()))


@cache
def search_descr_cossim_l():
    return text_vec_dists(description_spacy_iterator(),
                          search_spacy_iterator(),
                          lemma_descr_w2v(),
                          cosine,
                          text_to_vec_lemma)


@cache
def search_descr_cossim_l_tfidf():
    return text_vec_dists(description_spacy_iterator(),
                          search_spacy_iterator(),
                          lemma_descr_w2v(),
                          cosine,
                          partial(text_to_vec_tfidf, wordcount=token_counts()))


@cache
def search_attr_cossim_l():
    return text_vec_dists(attributes_spacy_iterator(),
                          search_spacy_iterator(),
                          lemma_descr_w2v(),
                          cosine,
                          text_to_vec_lemma)


@cache
def search_descr_euclid_l():
    logger.info("starting search_descr_euclidean_2")
    return text_vec_dists(search_spacy_iterator(),
                          description_spacy_iterator(),
                          lemma_descr_w2v(),
                          euclidean,
                          text_to_vec_lemma)


@cache
def search_title_euclid_l():
    logger.info("starting search_title_euclidean_2")
    return text_vec_dists(search_spacy_iterator(),
                          title_spacy_iterator(),
                          lemma_descr_w2v(),
                          euclidean,
                          text_to_vec_lemma)


@cache
def glove_euclid_search_descr():
    logger.info("starting glove_euclid_search_descr")
    result = text_vec_dists(description_spacy_iterator(),
                            search_spacy_iterator(),
                            glove_w2v(),
                            euclidean,
                            text_to_vec_orth)
    logger.info("done glove_euclid_search_descr")
    return result


@cache
def glove_euclid_search_title():
    logger.info("starting glove_euclid_search_title")
    result = text_vec_dists(title_spacy_iterator(),
                            search_spacy_iterator(),
                            glove_w2v(),
                            euclidean,
                            text_to_vec_orth)
    logger.info("done glove_euclid_search_title")
    return result


@cache
def glove_euclid_search_attr():
    return text_vec_dists(attributes_spacy_iterator(),
                          search_spacy_iterator(),
                          glove_w2v(),
                          euclidean,
                          text_to_vec_orth)


@cache
def glove_cossim_search_descr():
    logger.info("starting glove_cossim_search_descr")
    result = text_vec_dists(description_spacy_iterator(),
                            search_spacy_iterator(),
                            glove_w2v(),
                            cosine,
                            text_to_vec_orth)
    logger.info("done glove_cossim_search_descr")
    return result

@cache
def glove_cossim_search_descr_tfidf():
    return text_vec_dists(description_spacy_iterator(),
                            search_spacy_iterator(),
                            glove_w2v(),
                            cosine,
                            partial(text_to_vec_tfidf, wordcount=token_counts()))

@cache
def glove_cossim_search_title():
    logger.info("starting glove_cossim_search_descr")
    result = text_vec_dists(title_spacy_iterator(),
                            search_spacy_iterator(),
                            glove_w2v(),
                            cosine,
                            text_to_vec_orth)
    logger.info("done glove_cossim_search_descr")
    return result

@cache
def glove_cossim_search_title_tfidf():
    return text_vec_dists(title_spacy_iterator(),
                            search_spacy_iterator(),
                            glove_w2v(),
                            cosine,
                            partial(text_to_vec_tfidf, wordcount=token_counts()))

@cache
def attribute_lhoods():
    attribute_names = [
        'Product Width (in.)', 'Color Family', 'Material',
        'Indoor/Outdoor', 'Commercial / Residential', 'ENERGY STAR Certified',
        'Hardware Included', 'Package Quantity', 'Flooring Product Type',
        'Tools Product Type', 'Voltage (volts)']
    return np.hstack(
        [attr_lhood(some_attr) for some_attr in attribute_names] +
        [attribute_present(some_attr) for some_attr in attribute_names])


@cache
def brand_same_tokens():
    result = []
    for b, s in izip(brand(), df_all.search_term):
        if b is None:
            result.append(0)
            continue
        result.append(
            len(set(s.lower().split()).intersection(b.lower().split())))
    return np.array(result).reshape(total_size, 1)


@cache
def title_ratio():
    return title_same_lemmas() / len_of_query()


def llas_adapter(feature_function):
    return lambda: {'X': feature_function(),
                    "search_title_same_lemmas_list": search_title_same_lemmas_list(),
                    "search_descr_same_lemmas_list": search_descr_same_lemmas_list()
                    }


@cache
def features_0():
    return np.hstack([descr_same_tokens_count(), title_same_tokens_count(),
                      len_of_query()])


@cache
def features_1():
    return np.hstack([tfidf_search_times_descr(), tfidf_search_times_title(),
                      descr_same_tokens_count(), title_same_tokens_count(),
                      len_of_query()])


@cache
def features_2():
    return np.hstack([tfidf_search_times_descr(), tfidf_search_times_title(),
                      descr_same_tokens_count(), title_same_tokens_count(),
                      len_of_query(), search_title_cossim(), search_descr_cossim()])


@cache
def features_3():
    return np.hstack([tfidf_search_times_descr(), tfidf_search_times_title(),
                      descr_same_tokens_count(), title_same_tokens_count(),
                      len_of_query(), search_title_cossim(), search_descr_cossim(),
                      last_search_token_in_title(), last_search_token_in_descr_count()
                      ])


@cache
def features_4():
    return np.hstack([descr_same_lemmas(), title_same_lemmas(),
                      len_of_query()])


@cache
def features_5():
    return np.hstack([tfidf_search_times_descr(), tfidf_search_times_title(),
                      descr_same_tokens_count(), title_same_tokens_count(),
                      len_of_query(), search_title_cossim(), search_descr_cossim(),
                      descr_same_lemmas(), title_same_lemmas(),
                      len_of_query()])


@cache
def features_6():
    return np.hstack([tfidf_search_times_descr(), tfidf_search_times_title(),
                      descr_same_tokens_count(), title_same_tokens_count(),
                      len_of_query(), search_title_cossim(), search_descr_cossim(),
                      descr_same_lemmas(), title_same_lemmas(),
                      len_of_query(),
                      search_title_bigrams_count(), search_descr_bigrams_count(),
                      search_title_spacy_similarity(), search_descr_spacy_similarity()])


@cache
def features_7():
    return np.hstack([
        tfidf_search_times_descr(), tfidf_search_times_title(),
        descr_same_tokens_count(), title_same_tokens_count(),
        len_of_query(), search_title_cossim(), search_descr_cossim(),
        descr_same_lemmas(), title_same_lemmas(),
        len_of_query(),
        search_title_bigrams_count(), search_descr_bigrams_count(),
        search_title_spacy_similarity(), search_descr_spacy_similarity(),
        glove_euclid_search_descr(), glove_euclid_search_title()])


@cache
def features_8():
    return np.hstack([features_7(), brand_lhood_mnb()])


def parrot_feats_8():
    return {'X': features_8(), 'search_lemmas': np.array(map(tuple, search_lemmas())), 'prod': np.array(df_all.product_uid)}


@cache
def features_9():
    return np.hstack([features_8(), attribute_lhoods()])


@cache
def features_10():
    return np.hstack([
        tfidf_search_times_descr(), tfidf_search_times_title(),
        descr_same_tokens_count(), title_same_tokens_count(),
        len_of_query(),
        descr_same_lemmas(), title_same_lemmas(),
        len_of_query(),
        search_title_bigrams_count(), search_descr_bigrams_count(),
        search_title_spacy_similarity(), brand_lhood_mnb(),
        search_title_cossim_l(), search_descr_cossim_l(), search_attr_cossim_l(),
        glove_cossim_search_descr(), glove_cossim_search_title(),
        attribute_lhoods()]
    )


@cache
def features_11():
    return np.hstack([features_10(), brand_same_tokens(), title_ratio()])

#experimental feature sets, disregard
def features_exp():
    return np.hstack([features_11(), search_title_cossim_l_tfidf(), search_descr_cossim_l_tfidf()])


def features_square():
    good_ones = np.hstack([tfidf_search_times_descr(),
                           tfidf_search_times_title(),
                           descr_same_tokens_count(),
                           title_same_tokens_count(),
                           title_ratio()])
    return np.hstack([features_11(), PolynomialFeatures(2, interaction_only=True).fit_transform(good_ones)])

def features_exp_glove():
    return np.hstack([features_11(), glove_cossim_search_descr_tfidf(), glove_cossim_search_title_tfidf()])
#end of experimental feature sets

def features_12():
    return np.hstack([features_11(), 
                       search_title_cossim_l_tfidf(), search_descr_cossim_l_tfidf(),
                       glove_cossim_search_descr_tfidf(), glove_cossim_search_title_tfidf()
                      ])

