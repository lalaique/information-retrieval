import os
import zipfile
from functools import partial
import nltk
import requests
import numpy as np
from ipywidgets import widgets
from collections import namedtuple
import logging
import tqdm
from typing import List, Optional, Dict, Tuple


def download_dataset() -> None:
    """
    This func downloads the dataset and unzip to a local dir.
    """
    folder_path = os.environ.get("IR1_DATA_PATH")
    if not folder_path:
        folder_path = "./datasets/"
    os.makedirs(folder_path)

    file_location = os.path.join(folder_path, "cacm.zip")

    if not os.path.exists(file_location):

        url = "https://surfdrive.surf.nl/files/index.php/s/M0FGJpX2p8wDwxR/download"

        with open(file_location, "wb") as handle:
            print(f"Downloading file from {url} to {file_location}")
            response = requests.get(url, stream=True)
            for data in response.iter_content():
                handle.write(data)
            print("Finished downloading file")

    if not os.path.exists(os.path.join(folder_path, "train.txt")):
        # unzip file
        with zipfile.ZipFile(file_location, "r") as zip_ref:
            zip_ref.extractall(folder_path)


def read_cacm_docs(root_folder="./datasets/"):
    """
    Reads in the CACM documents. The dataset is assumed to be in the folder "./datasets/" by default
    Returns: A list of 2-tuples: (doc_id, document), where 'document' is a single string created by
        appending the title and abstract (separated by a "\n").
        In case the record doesn't have an abstract, the document is composed only by the title
    """
    file_dir = root_folder + "cacm.all"
    keeping = False
    temp = ""
    doc_list = []

    with open(file_dir) as cacm_file:
        for line in cacm_file:
            line_begin = line[0:2]

            if line_begin == ".I":
                doc_index = line.split(" ")[1].replace("\n", "")
                temp = ""

            elif line_begin == ".T":
                keeping = True

            elif line_begin == ".W":
                keeping = True
                temp += "\n "

            elif line_begin in [".B", ".A", ".N", ".K"]:
                keeping = False

            elif line_begin not in [".I", ".T", ".W", ".B", ".A", ".N", ".X", ".K"]:
                if keeping:
                    temp += line
            else:
                doc_list.append((doc_index, temp))
                keeping = False

    return doc_list


def read_queries(root_folder="./datasets/"):
    """
    Reads in the CACM queries. The dataset is assumed to be in the folder "./datasets/" by default
    Returns: A list of 2-tuples: (query_id, query)
    """
    file_dir = root_folder + "query.text"
    keeping = False
    temp = ""
    query_list = []

    with open(file_dir, "r") as query_file:
        for line in query_file:
            line_begin = line[0:2]

            if line_begin == ".I":
                doc_index = line.split(" ")[1].replace("\n", "")
                temp = ""

            elif line_begin == ".W":
                keeping = True

            elif line_begin == ".A":
                keeping = False

            elif line_begin not in [".I", ".T", ".W", ".B", ".A", ".N", ".X", ".K"]:
                if keeping:
                    temp += line
            else:
                query_list.append((doc_index, temp))
                keeping = False

    return query_list


def load_stopwords(root_folder="./datasets/"):
    """
    Loads the stopwords. The dataset is assumed to be in the folder "./datasets/" by default

    :param root_folder:
    :return: A set of stopwords
    """
    file_dir = root_folder + "common_words"
    with open(file_dir, "r") as f:
        stopwords = [line.strip() for line in f]

    return set(stopwords)


def tokenize(text: str) -> List[str]:
    """
    Tokenizes the input text. Use the WordPunctTokenizer

    :param text: text - a string
    :return: a list of tokens
    """
    return nltk.WordPunctTokenizer().tokenize(text)


def stem_token(token):
    """
    Stems the given token using the PorterStemmer from the nltk library

    :param token: a single token
    :return: the stem of the token
    """
    ps = nltk.stem.PorterStemmer()
    return ps.stem(token)


def process_text(
    text: str,
    stem: bool = False,
    remove_stopwords: str = False,
    lowercase_text: str = False,
):
    """
    Given a string, this func tokenizes it and processes it according to the flags that you set.

    :param text: input text
    :param stem: to stem or not
    :param remove_stopwords: to remove stopwords or not
    :param lowercase_text: to lowercase the text or not
    :return:
    """
    tokens = []
    for token in tokenize(text):
        if remove_stopwords and token.lower() in stopwords:
            continue
        if stem:
            token = stem_token(token)
        if lowercase_text:
            token = token.lower()
        tokens.append(token)
    return tokens


def read_qrels(root_folder="./datasets/"):
    query_f = open(os.path.join(root_folder, "qrels.text"), "r")
    query_dic = {}

    for line in query_f:
        q_id, doc_id, _, _ = line.split()
        q_id = str(int(q_id))
        if q_id not in query_dic:
            query_dic[q_id] = []
        query_dic[q_id].append(doc_id)

    return query_dic
