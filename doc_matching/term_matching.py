import os
import zipfile
from functools import partial
import nltk
import requests
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from ipywidgets import widgets
from IPython.display import display, HTML
from collections import namedtuple
import logging

from doc_matching.preprocess import *
from doc_matching.evaluate import *


def bow_search(query: str, index_set: int) -> List[str]:
    """
    Perform a search over all documents with the given query.

    :param query: a (unprocessed) query
    :param index_set: the index to use

    :return:  a list of (document_id, score), sorted in descending relevance to the given query
    """

    index = get_index(index_set)
    processed_query = preprocess_query(query, index_set)

    bag_dict = {}
    for q in processed_query:

        if q not in index:
            continue

        for doc_id, tf in index[q]:
            if doc_id not in bag_dict:
                bag_dict[doc_id] = 0.0

            bag_dict[doc_id] += 1.0

    sorted_result = sorted(bag_dict.items(), key=lambda tup: tup[1], reverse=True)

    return sorted_result


def build_tf_index(documents):
    """
    Build an inverted index (with counts). The output is a dictionary which takes in a token
    and returns a list of (doc_id, count) where 'count' is the count of the 'token' in 'doc_id'

    :param documents: a list of documents - (doc_id, tokens)
    :return: An inverted index. [token] -> [(doc_id, token_count)]
    """
    tf_index = {}
    for doc in documents:
        for token in np.unique(doc[1]):
            doc_list = (doc[0], doc[1].count(token))

            if token in tf_index.keys():
                tf_index[token].append(doc_list)
            else:
                tf_index[token] = [doc_list]
    return tf_index


def get_index(index_set):
    assert index_set in {1, 2}
    return {1: tf_index_1, 2: tf_index_2}[index_set]


def preprocess_query(text, index_set):
    """Preprocessed query based on the two configs"""
    assert index_set in {1, 2}
    if index_set == 1:
        return process_text(text, **config_1)
    elif index_set == 2:
        return process_text(text, **config_2)


def compute_df(documents: List[str]) -> Dict:
    """
    Compute the document frequency of all terms in the vocabulary

    :param documents: A list of documents
    :return:  A dictionary with {token: document frequency)
    """
    doc_freq = {}
    for i in range(len(documents)):
        tokens = documents[i]
        for token in tokens:
            if token not in doc_freq:
                doc_freq[token] = {i}
            else:
                doc_freq[token].add(i)
    for token in doc_freq:
        doc_freq[token] = len(doc_freq[token])
    return doc_freq


def get_df(index_set):
    assert index_set in {1, 2}
    return {1: df_1, 2: df_2}[index_set]


def tfidf_search(query, index_set):
    """
    Perform a search over all documents with the given query using tf-idf.

    :param query: a (unprocessed) query
    :param index_set: the index to use
    :return: a list of (document_id, score), sorted in descending relevance to the given query
    """

    index = get_index(index_set)
    df = get_df(index_set)
    processed_query = preprocess_query(query, index_set)
    n_doc = len(doc_repr_1) if index_set == 1 else len(doc_repr_2)

    tfidf_dict = {}
    for q in processed_query:
        if q not in index:
            continue
        for doc_id, tf in index[q]:
            if doc_id not in tfidf_dict:
                tfidf_dict[doc_id] = 0
            tfidf_dict[doc_id] += tf * np.log(n_doc / df[q])

    sorted_result = sorted(tfidf_dict.items(), key=lambda tup: tup[1], reverse=True)
    return sorted_result


def doc_lengths(documents):
    doc_lengths = {doc_id: len(doc) for (doc_id, doc) in documents}
    return doc_lengths


doc_lengths_1 = doc_lengths(doc_repr_1)
doc_lengths_2 = doc_lengths(doc_repr_2)


def get_doc_lengths(index_set):
    assert index_set in {1, 2}
    return {1: doc_lengths_1, 2: doc_lengths_2}[index_set]


def naive_ql_search(query, index_set):
    """
    Perform a search over all documents with the given query using a naive QL model.

    :param query: a (unprocessed) query
    :param index_set: the index to use
    :return: a list of (document_id, score), sorted in descending relevance to the given query
    """

    index = get_index(index_set)
    doc_lengths = get_doc_lengths(index_set)
    processed_query = preprocess_query(query, index_set)
    unigram_probs = {}

    for i, q in enumerate(processed_query):
        if q not in index:
            continue

        if i > 0:
            tf_dicts = dict(index[q])
            for doc_id in unigram_probs:
                if doc_id in tf_dicts:
                    unigram_probs[doc_id] *= (
                        1.0 * tf_dicts[doc_id] / doc_lengths[doc_id]
                    )
                else:
                    unigram_probs[doc_id] = 0

        else:
            for doc_id, tf in index[processed_query[0]]:
                unigram_probs[doc_id] = tf / doc_lengths[doc_id]

    sorted_rank = sorted(unigram_probs.items(), key=lambda d: d[1], reverse=True)
    return sorted_rank


def get_doc_ids(query, index_set):
    """
    Return doc_id list of documents that contain the query terms

    :param query: a (unprocessed) query
    :param index_set: the index to use
    :return: a list of doc_id that contain the query terms
    """
    index = get_index(index_set)
    doc_ids = []
    for q in query:
        if q not in index:
            continue
        for doc_id, _ in index[q]:
            if doc_id not in doc_ids:
                doc_ids.append(doc_id)
    return doc_ids


def ql_search(query, index_set):
    """
    Perform a search over all documents with the given query using a QL model with Jelinek-Mercer Smoothing (set smoothing=0.1).

    :param query: a (unprocessed) query
    :param index_set: the index to use
    :return: a list of (document_id, score), sorted in descending relevance to the given query
    """

    index = get_index(index_set)
    doc_lengths = get_doc_lengths(index_set)
    processed_query = preprocess_query(query, index_set)

    doc_ids = get_doc_ids(processed_query, index_set)
    cl = sum(doc_lengths.values())
    lamb = 0.1
    unigram_probs = dict(zip(doc_ids, np.zeros(len(doc_ids))))

    for i, q in enumerate(processed_query):
        if q not in index:
            continue

        tf_dict = dict(index[q])
        cf = sum(tf_dict.values())

        for doc_id in doc_ids:
            tf = tf_dict[doc_id] if doc_id in tf_dict else 0
            unigram_probs[doc_id] += np.log(
                (1 - lamb) * tf / doc_lengths[doc_id] + lamb * cf / cl
            )

    sorted_rank = sorted(unigram_probs.items(), key=lambda d: d[1], reverse=True)
    return sorted_rank


def bm25_search(query, index_set):
    """
    Perform a search over all documents with the given query using BM25. Use k_1 = 1.5 and b = 0.75

    :param query: a (unprocessed) query
    :param index_set: the index to use
    :return: a list of (document_id, score), sorted in descending relevance to the given query
    """
    index = get_index(index_set)
    df = get_df(index_set)
    doc_lengths = get_doc_lengths(index_set)
    processed_query = preprocess_query(query, index_set)

    k_1, b = 1.5, 0.75
    bm25_dict = {}
    dl_avg = 1.0 * sum(doc_lengths.values()) / len(doc_lengths)

    for q in processed_query:
        if q not in index:
            continue
        for doc_id, tf in index[q]:
            if doc_id not in bm25_dict:
                bm25_dict[doc_id] = 0
            idf = np.log(len(doc_lengths) / df[q])
            bm25_dict[doc_id] += (
                idf
                * (k_1 + 1)
                * tf
                / (k_1 * (1 - b + b * doc_lengths[doc_id] / dl_avg) + tf)
            )
    sorted_rank = sorted(bm25_dict.items(), key=lambda d: d[1], reverse=True)
    return sorted_rank


def print_results(docs, len_limit=50):
    for i, (doc_id, score) in enumerate(docs):
        doc_content = (
            docs_by_id[doc_id].strip().replace("\n", "\\n")[:len_limit] + "..."
        )
        print(f"Rank {i}({score:.2}): {doc_content}")


if __name__ == "__main__":

    config_1 = {"stem": False, "remove_stopwords": False, "lowercase_text": True}

    config_2 = {
        "stem": True,
        "remove_stopwords": True,
        "lowercase_text": True,
    }

    # Prepate data
    download_dataset()
    docs = read_cacm_docs()
    queries = read_queries()
    qrels = read_qrels()
    stopwords = load_stopwords()

    doc_repr_1 = []
    doc_repr_2 = []
    for (doc_id, document) in docs:
        doc_repr_1.append((doc_id, process_text(document, **config_1)))
        doc_repr_2.append((doc_id, process_text(document, **config_2)))
    docs_by_id = dict(docs)

    # Bag of Words
    test_bow = bow_search("report", index_set=1)[:5]
    print(f"BOW Results:")
    print_results(test_bow)

    # TF-IDF
    tf_index_1 = build_tf_index(doc_repr_1)
    tf_index_2 = build_tf_index(doc_repr_2)
    df_1 = compute_df([d[1] for d in doc_repr_1])
    df_2 = compute_df([d[1] for d in doc_repr_2])

    test_tfidf = tfidf_search("report", index_set=1)[:5]
    print(f"TFIDF Results:")
    print_results(test_tfidf)

    # Naive QL
    test_naiveql = naive_ql_search("report", index_set=1)[:5]
    print(f"Naive QL Results:")
    print_results(test_naiveql)

    # QL
    test_ql_results = ql_search("report", index_set=1)[:5]
    print_results(test_ql_results)
    print()
    test_ql_results_long = ql_search("report " * 10, index_set=1)[:5]
    print_results(test_ql_results_long)

    # BM25
    test_bm25_results = bm25_search("report", index_set=1)[:5]
    print_results(test_bm25_results)

    # precision
    qid = queries[0][0]
    qtext = queries[0][1]
    print(f'query:{qtext}')
    results = bm25_search(qtext, 2)
    precision = precision_k(results, qrels[qid], 10)
    print(f'precision@10 = {precision}')

    # recall
    qid = queries[10][0]
    qtext = queries[10][1]
    print(f'query:{qtext}')
    results = bm25_search(qtext, 2)
    recall = recall_k(results, qrels[qid], 10)
    print(f'recall@10 = {recall}')

    # MAP
    qid = queries[20][0]
    qtext = queries[20][1]
    print(f'query:{qtext}')
    results = bm25_search(qtext, 2)
    mean_ap = average_precision(results, qrels[qid])
    print(f'MAP = {mean_ap}')

    # ERR
    qid = queries[30][0]
    qtext = queries[30][1]
    print(f'query:{qtext}')
    results = bm25_search(qtext, 2)
    ERR = err(results, qrels[qid])
    print(f'ERR = {ERR}')

    # Evaluate the fn
    recall_at_1 = partial(recall_k, k=1)
    recall_at_5 = partial(recall_k, k=5)
    recall_at_10 = partial(recall_k, k=10)
    precision_at_1 = partial(precision_k, k=1)
    precision_at_5 = partial(precision_k, k=5)
    precision_at_10 = partial(precision_k, k=10)
