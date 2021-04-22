#!/usr/bin/env python
# coding: utf-8


import nltk
import numpy as np
from tqdm import tqdm
from ipywidgets import widgets
from IPython.display import display, HTML
from collections import namedtuple
from gensim.corpora import Dictionary
from gensim.models import LdaModel, LsiModel, Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim import downloader as g_downloader
import itertools


def dot(vec_1,vec_2):
    return sum( [vec_1[i][1]*vec_2[i][1] for i in range(len(vec_1))] )


def cosine_sim(vec_1, vec_2):
    return dot(vec_1, vec_2)/((np.sqrt(dot(vec_1, vec_1)) * np.sqrt(dot(vec_2, vec_2)))+1e-6)


def jenson_shannon_divergence(vec_1, vec_2, assert_prob=False):
    _vec_1 = np.asarray(vec_1) / sum(n for _, n in vec_1)
    _vec_2 = np.asarray(vec_2) / sum(n for _, n in vec_2)
    _avg = 0.5 * (_vec_1 + _vec_2)

    def KL(a, b):
        a = np.asarray([x[1] for x in a], dtype=np.float)
        b = np.asarray([x[1] for x in b], dtype=np.float)

        return np.sum(np.where(a != 0, a * np.log2(a / b), 0))

    return 0.5 * (KL(_vec_1, _avg) + KL(_vec_2, _avg))


def jenson_shannon_sim(vec_1, vec_2, assert_prob=False):
    return 1 - jenson_shannon_divergence(vec_1, vec_2)


class VectorSpaceRetrievalModel:

    def __init__(self, doc_repr):
        self.doc_repr = doc_repr
        self.documents = [_[1] for _ in self.doc_repr]
        self.dictionary = Dictionary(self.documents)
        self.dictionary.filter_extremes(no_below=10)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in self.documents]
        temp = self.dictionary[0]
        self.id2word = self.dictionary.id2token
        self.model = None
        
        
    def vectorize_documents(self):
        vectors = {}
        for (doc_id, _), cc in zip(self.doc_repr, self.corpus):
            vectors[doc_id] = self.model[cc]
        return vectors

    def vectorize_query(self, query):
        query = process_text(query, **config_2)
        query_vector = self.dictionary.doc2bow(query)
        return self.model[query_vector]
    
    def train_model(self):
        raise NotImplementedError()


class LsiRetrievalModel(VectorSpaceRetrievalModel):
    def __init__(self, doc_repr):
        super().__init__(doc_repr)
        
        self.num_topics = 100
        self.chunksize = 2000
    
    def train_model(self):
        self.model = LsiModel(self.corpus, num_topics=self.num_topics, id2word=self.id2word, chunksize=self.chunksize)


class DenseRetrievalRanker:
    def __init__(self, vsrm, similarity_fn):
        self.vsrm = vsrm 
        self.vectorized_documents = self.vsrm.vectorize_documents()
        self.similarity_fn = similarity_fn
    
    def _compute_sim(self, query_vector):

        empty_list = []

        for key in self.vectorized_documents:
          doc_id = key
          document_single = self.vectorized_documents.get(key)

          if document_single != [] and query_vector != []:
            score = self.similarity_fn(document_single,query_vector) # compute similary between the query vector to each vectorized document
            empty_list.append(tuple((doc_id, score)))
          else:
            score = 0
            empty_list.append(tuple((doc_id, score)))


        return empty_list
    
    def search(self, query):
        scores = self._compute_sim(self.vsrm.vectorize_query(query))
        scores.sort(key=lambda _:-_[1])
        return scores


class LdaRetrievalModel(VectorSpaceRetrievalModel):
    def __init__(self, doc_repr):
        super().__init__(doc_repr)
        self.num_topics = 100
        self.chunksize = 2000
        self.passes = 20
        self.iterations = 400
        self.eval_every = 10
        self.minimum_probability=0.0
        self.alpha='auto'
        self.eta='auto'
    
    
    def train_model(self):
        self.model = LdaModel(self.corpus, self.num_topics, id2word=self.id2word, chunksize=self.chunksize,
                              passes=self.passes, iterations=self.iterations, eval_every=self.eval_every, 
                              minimum_probability=self.minimum_probability, alpha=self.alpha, eta=self.eta)



class W2VRetrievalModel(VectorSpaceRetrievalModel):
    def __init__(self, doc_repr):
        super().__init__(doc_repr)
        
        self.size = 100 
        self.min_count = 1
    
    def train_model(self):
        self.model = Word2Vec(self.documents, size=self.size, min_count = self.min_count)
        self.model.save("word2vec-google-news-300.model")
        
    def vectorize_documents(self):
        vectors = {}
        for (doc_id, _), cc in zip(self.doc_repr, self.documents):
          vector_dim = self.model.vector_size
          arr = np.empty((0,vector_dim), dtype='f')

          for wrd in cc:
            if wrd in self.model.wv.vocab:
              word_array = self.model.wv[wrd]
              norm = np.linalg.norm(word_array)
              word_array = (word_array/norm).reshape(1, -1)
              arr = np.append(arr,np.array(word_array), axis=0)
            else:
              word_array = np.zeros(self.size).reshape(1, -1)
              arr = np.append(arr,np.array(word_array), axis=0)

          list1 = np.mean(arr, axis=0)
          list2 = list(range(self.size))
          vectors[doc_id] = list(zip(list2, list1)) # save vectorized query for each doc
        return vectors

    def vectorize_query(self, query):
        query = process_text(query, **config_2)
        vector_dim = self.model.vector_size
        arr = np.empty((0,vector_dim), dtype='f')
        
        for wrd in query:
          if wrd in self.model.wv.vocab:
            word_array = self.model.wv[wrd] # infer vector for each word

            norm = np.linalg.norm(word_array)
            word_array = (word_array/norm).reshape(1, -1) # normalize the inferred vector

            arr = np.append(arr,np.array(word_array), axis=0)
          else:
            word_array = np.zeros(self.size).reshape(1, -1) # if the word is not present, return 0 for all dimension

            arr = np.append(arr,np.array(word_array), axis=0)

        list1 = np.mean(arr, axis=0) # average over each dimension
        list2 = list(range(self.size))

        return list(zip(list2, list1))
      
    
class W2VPretrainedRetrievalModel(W2VRetrievalModel):
    def __init__(self, doc_repr):
        super().__init__(doc_repr)
        self.model_name = "word2vec-google-news-300"
        self.size = 300
    
    def train_model(self):
        self.model = g_downloader.load(self.model_name)


class D2VRetrievalModel(VectorSpaceRetrievalModel):
    def __init__(self, doc_repr):
        super().__init__(doc_repr)
        
        self.vector_size= 100
        self.min_count = 1
        self.epochs = 20
        
        self.taggedDocument = [TaggedDocument(doc, [i]) for i, doc in enumerate(self.documents)]
        
    def train_model(self):
        self.model = Doc2Vec(self.taggedDocument, size=self.vector_size, min_count = self.min_count, epochs=self.epochs)
    
    def vectorize_documents(self):
        vectors = {}

        for (doc_id, _), cc in zip(self.doc_repr, self.taggedDocument):
          list1 = self.model.infer_vector(cc[0]) # infer vector for the query
          list2 = list(range(self.vector_size))

          vectors[doc_id] = list(zip(list2, list1))
        return vectors


    def vectorize_query(self, query):
        query = process_text(query, **config_2)

        list1 = self.model.infer_vector(query) # infer vector for the query
        list2 = list(range(self.vector_size))
        
        return list(zip(list2, list1))


class DenseRerankingModel:
    def __init__(self, initial_retrieval_fn, vsrm, similarity_fn):
        self.ret = initial_retrieval_fn
        self.vsrm = vsrm
        self.similarity_fn = similarity_fn
        self.vectorized_documents = vsrm.vectorize_documents()
        
        assert len(self.vectorized_documents) == len(doc_repr_2)
    
    def search(self, query, K=50):
        scores = self.ret(query)
        doc_ids = [i[0] for i in scores][0:K]
        newdict = {k: self.vectorized_documents[k] for k in doc_ids}

        empty_list  = []
        query_vector = self.vsrm.vectorize_query(query)
        
        for key in newdict:
          doc_id = key
          document_single = newdict.get(key)

          if document_single !=[] and query_vector != []:
            score = self.similarity_fn(document_single, query_vector) # compute similary between the query vector to each vectorized document
            empty_list.append(tuple((doc_id, score)))
          else:
            score = 0
            empty_list.append(tuple((doc_id, score)))

        empty_list.sort(key=lambda _:-_[1])
        return empty_list



bm25_search_2 = partial(bm25_search, index_set=2)
lsi_rerank = DenseRerankingModel(bm25_search_2, lsi, cosine_sim)
lda_rerank = DenseRerankingModel(bm25_search_2, lda, jenson_shannon_sim)
w2v_rerank = DenseRerankingModel(bm25_search_2, w2v, cosine_sim)
w2v_pretrained_rerank = DenseRerankingModel(bm25_search_2, w2v_pretrained, cosine_sim)
d2v_rerank = DenseRerankingModel(bm25_search_2, d2v, cosine_sim)


