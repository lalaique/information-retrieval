#!/usr/bin/env python
# coding: utf-8


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

from preprocess import *


def build_tf_index(documents):
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
    return {
        1: tf_index_1,
        2: tf_index_2
    }[index_set]


def preprocess_query(text, index_set):
    assert index_set in {1, 2}
    if index_set == 1:
        return process_text(text, **config_1)
    elif index_set == 2:
        return process_text(text, **config_2)


def bow_search(query, index_set):

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


def compute_df(documents):

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


df_1 = compute_df([d[1] for d in doc_repr_1])
df_2 = compute_df([d[1] for d in doc_repr_2])


def get_df(index_set):
    assert index_set in {1, 2}
    return {
        1: df_1,
        2: df_2
    }[index_set]


def tfidf_search(query, index_set):

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
    return {
        1: doc_lengths_1,
        2: doc_lengths_2
    }[index_set]


def naive_ql_search(query, index_set):

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
                    unigram_probs[doc_id] *= 1.0 * tf_dicts[doc_id] / doc_lengths[doc_id]

                else:
                    unigram_probs[doc_id] = 0

        else:
            for doc_id, tf in index[processed_query[0]]:
                unigram_probs[doc_id] = tf / doc_lengths[doc_id]

    sorted_rank = sorted(unigram_probs.items(), key=lambda d: d[1], reverse=True)

    return sorted_rank


def get_doc_ids(query, index_set):

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
            unigram_probs[doc_id] += np.log((1 - lamb) * tf / doc_lengths[doc_id] + lamb * cf / cl)

    sorted_rank = sorted(unigram_probs.items(), key=lambda d: d[1], reverse=True)
    return sorted_rank


def bm25_search(query, index_set):

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
            bm25_dict[doc_id] += idf * (k_1 + 1) * tf / (k_1 * (1 - b + b * doc_lengths[doc_id] / dl_avg) + tf)

    sorted_rank = sorted(bm25_dict.items(), key=lambda d: d[1], reverse=True)
    return sorted_rank
