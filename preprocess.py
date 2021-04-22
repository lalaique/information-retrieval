#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import zipfile
from functools import partial
import nltk
import requests
import numpy as np
from ipywidgets import widgets
from collections import namedtuple
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# In[2]:


def download_dataset():
    folder_path = os.environ.get("IR1_DATA_PATH")
    if not folder_path:
        folder_path = "./datasets/"
    os.makedirs(folder_path, exist_ok=True)
    
    file_location = os.path.join(folder_path, "cacm.zip")
    
    # download file if it doesn't exist
    if not os.path.exists(file_location):
        
        url = "https://surfdrive.surf.nl/files/index.php/s/M0FGJpX2p8wDwxR/download"

        with open(file_location, "wb") as handle:
            print(f"Downloading file from {url} to {file_location}")
            response = requests.get(url, stream=True)
            for data in tqdm(response.iter_content()):
                handle.write(data)
            print("Finished downloading file")
    
    if not os.path.exists(os.path.join(folder_path, "train.txt")):
        
        # unzip file
        with zipfile.ZipFile(file_location, 'r') as zip_ref:
            zip_ref.extractall(folder_path)
        
download_dataset()


# In[5]:


def read_cacm_docs(root_folder = "./datasets/"):
    """
        Reads in the CACM documents. The dataset is assumed to be in the folder "./datasets/" by default
        Returns: A list of 2-tuples: (doc_id, document), where 'document' is a single string created by 
            appending the title and abstract (separated by a "\n"). 
            In case the record doesn't have an abstract, the document is composed only by the title
    """

    file_dir = root_folder + "cacm.all"
    keeping = False
    temp = ""
    doc_list =[]

    with open(file_dir) as cacm_file:
        for line in cacm_file:
            line_begin = line[0:2]

            if line_begin == ".I":
                doc_index = line.split(" ")[1].replace("\n","")
                temp = ""

            elif line_begin == ".T":
                keeping = True

            elif line_begin == ".W":
                keeping = True
                temp += "\n "

            elif line_begin in ['.B', '.A', '.N', '.K']:
                keeping = False

            elif (line_begin not in ['.I', '.T', '.W', '.B', '.A', '.N', '.X', '.K']):
                if keeping:
                    temp += line
            else:
                doc_list.append((doc_index, temp))
                keeping = False

    return doc_list


# In[8]:


def read_queries(root_folder = "./datasets/"):
    """
        Reads in the CACM queries. The dataset is assumed to be in the folder "./datasets/" by default
        Returns: A list of 2-tuples: (query_id, query)
    """
    
    file_dir = root_folder + "query.text"
    keeping = False
    temp = ""
    query_list =[]

    with open(file_dir, "r") as query_file:
        for line in query_file:
            line_begin = line[0:2]

            if line_begin == ".I":
                doc_index = line.split(" ")[1].replace("\n","")
                temp = ""

            elif line_begin == ".W":
                keeping = True   

            elif line_begin == ".A":
                keeping = False

            elif (line_begin not in ['.I', '.T', '.W', '.B', '.A', '.N', '.X', '.K']):
                if keeping:
                    temp += line
            else:
                query_list.append((doc_index, temp))
                keeping = False
                
    return query_list


# In[11]:


def load_stopwords(root_folder = "./datasets/"):
    """
        Loads the stopwords. The dataset is assumed to be in the folder "./datasets/" by default
        Output: A set of stopwords
    """
    file_dir = root_folder + "common_words"

    with open(file_dir , 'r') as f:
        stopwords = [line.strip() for line in f]
        
    return set(stopwords)


# In[13]:


def tokenize(text):
    """
        Tokenizes the input text. Use the WordPunctTokenizer
        Input: text - a string
        Output: a list of tokens
    """
    tokenized = nltk.WordPunctTokenizer().tokenize(text)
    return tokenized
    


# In[15]:


def stem_token(token):
    """
        Stems the given token using the PorterStemmer from the nltk library
        Input: a single token
        Output: the stem of the token
    """
    ps = nltk.stem.PorterStemmer() 
    return ps.stem(token)
    


# In[17]:


def process_text(text, stem=False, remove_stopwords=False, lowercase_text=False):
    
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


# In[18]:


# In this configuration:
# Don't preprocess the text, except to tokenize 
config_1 = {
  "stem": False,
  "remove_stopwords" : False,
  "lowercase_text": True
} 


# In this configuration:
# Preprocess the text, stem and remove stopwords
config_2 = {
  "stem": True,
  "remove_stopwords" : True,
  "lowercase_text": True, 
} 

####
doc_repr_1 = []
doc_repr_2 = []
for (doc_id, document) in docs:
    doc_repr_1.append((doc_id, process_text(document, **config_1)))
    doc_repr_2.append((doc_id, process_text(document, **config_2)))

####


# In[20]:


#### Indexed documents based on the two configs

# Create the 2 indices
tf_index_1 = build_tf_index(doc_repr_1)
tf_index_2 = build_tf_index(doc_repr_2)

# This function returns the tf_index of the corresponding config
def get_index(index_set):
    assert index_set in {1, 2}
    return {
        1: tf_index_1,
        2: tf_index_2
    }[index_set]

####
#### Preprocessed query based on the two configs

# This function preprocesses the text given the index set, according to the specified config
def preprocess_query(text, index_set):
    assert index_set in {1, 2}
    if index_set == 1:
        return process_text(text, **config_1)
    elif index_set == 2:
        return process_text(text, **config_2)

#### 


# In[59]:


def read_qrels(root_folder = "./datasets/"):
    """
        Reads the qrels.text file. 
        Output: A dictionary: query_id -> [list of relevant documents]
    """
    # YOUR CODE HERE
    query_f = open(os.path.join(root_folder, "qrels.text"), 'r')
    query_dic = {}

    for line in query_f:
      q_id, doc_id, _, _ = line.split()
      
      # make queries and qrels consistent in query_id ('01'->'1')
      q_id = str(int(q_id))
      
      if q_id not in query_dic:
        query_dic[q_id] = []
      query_dic[q_id].append(doc_id)
    
    return query_dic

