#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[61]:


def precision_k(results, relevant_docs, k):
    """
        Compute Precision@K
        Input: 
            results: A sorted list of 2-tuples (document_id, score), 
                    with the most relevant document in the first position
            relevant_docs: A set of relevant documents. 
            k: the cut-off
        Output: Precision@K
    """
    relevant_cnt = 0

    for i, (doc_id, _) in enumerate(results):
      if doc_id in relevant_docs:
        relevant_cnt += 1
      if i == k - 1:
        break

    return relevant_cnt / k


# In[63]:


def recall_k(results, relevant_docs, k):
    """
        Compute Recall@K
        Input: 
            results: A sorted list of 2-tuples (document_id, score), with the most relevant document in the first position
            relevant_docs: A set of relevant documents. 
            k: the cut-off
        Output: Recall@K
    """
    relevant_cnt = 0

    for i, (doc_id, _) in enumerate(results):
      if doc_id in relevant_docs:
        relevant_cnt += 1
      if i == k - 1:
        break

    return relevant_cnt / len(relevant_docs)


# In[65]:


def average_precision(results, relevant_docs):
    """
        Compute Average Precision (for a single query - the results are 
        averaged across queries to get MAP in the next few cells)
        Hint: You can use the recall_k and precision_k functions here!
        Input: 
            results: A sorted list of 2-tuples (document_id, score), with the most 
                    relevant document in the first position
            relevant_docs: A set of relevant documents. 
        Output: Average Precision
    """
    relevant_cnt = 0
    sum = 0
    search_cnt = 0

    while relevant_cnt < len(relevant_docs) and search_cnt < len(results):
      doc_id, _ = results[search_cnt]
      search_cnt += 1
      if doc_id in relevant_docs:
        relevant_cnt += 1
        sum += relevant_cnt / search_cnt

    return sum / len(relevant_docs)


# In[67]:


def err(results, relevant_docs):
    """
        Compute the expected reciprocal rank.
        Input: 
            results: A sorted list of 2-tuples (document_id, score), with the most 
                    relevant document in the first position
            relevant_docs: A set of relevant documents. 
        Output: ERR
        
    """
    # YOUR CODE HERE
    err_score = 0
    r_prod = 1

    for i, (doc_id, _) in enumerate(results):
      if doc_id in relevant_docs:
        err_score += 1/(i+1) * r_prod * 0.5
        r_prod *= 1 - 0.5
    
    return err_score


# In[69]:


#### metrics@k functions

recall_at_1 = partial(recall_k, k=1)
recall_at_5 = partial(recall_k, k=5)
recall_at_10 = partial(recall_k, k=10)
precision_at_1 = partial(precision_k, k=1)
precision_at_5 = partial(precision_k, k=5)
precision_at_10 = partial(precision_k, k=10)

list_of_metrics = [
    ("ERR", err),
    ("MAP", average_precision),
    ("Recall@1",recall_at_1),
    ("Recall@5", recall_at_5),
    ("Recall@10", recall_at_10),
    ("Precision@1", precision_at_1),
    ("Precision@5", precision_at_5),
    ("Precision@10", precision_at_10)]
####


# In[70]:


#### Evaluate a search function

list_of_search_fns = [
    ("BOW", bow_search),
    ("TF-IDF", tfidf_search),
    ("NaiveQL", naive_ql_search),
    ("QL", ql_search),
    ("BM25", bm25_search)
]

def evaluate_search_fn(search_fn, metric_fns, index_set=None):
    # build a dict query_id -> query 
    queries_by_id = dict((q[0], q[1]) for q in queries)
    
    metrics = {}
    for metric, metric_fn in metric_fns:
        metrics[metric] = np.zeros(len(qrels), dtype=np.float32)
    
    for i, (query_id, relevant_docs) in enumerate(qrels.items()):
        query = queries_by_id[query_id]
        if index_set:
            results = search_fn(query, index_set)
        else:
            results = search_fn(query)
        
        for metric, metric_fn in metric_fns:
            metrics[metric][i] = metric_fn(results, relevant_docs)
    
    
    final_dict = {}
    for metric, metric_vals in metrics.items():
        final_dict[metric] = metric_vals.mean()
    
    return final_dict

