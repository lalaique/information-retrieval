from doc_matching.term_matching import *


def precision_k(results, relevant_docs, k: int) -> float:
    """
    Compute Precision@K

    :param results: A sorted list of 2-tuples (document_id, score), with the most relevant document in the first position
    :param relevant_docs: A set of relevant documents.
    :param k: the cut-off
    :return: Precision@K
    """
    relevant_cnt = 0
    for i, (doc_id, _) in enumerate(results):
        if doc_id in relevant_docs:
            relevant_cnt += 1
        if i == k - 1:
            break
    return relevant_cnt / k


def recall_k(results, relevant_docs, k: int) -> float:
    """
    Compute Recall@K

    :param results: A sorted list of 2-tuples (document_id, score), with the most relevant document in the first position
    :param relevant_docs: A set of relevant documents.
    :param k: the cut-off
    :return: Recall@K
    """
    relevant_cnt = 0
    for i, (doc_id, _) in enumerate(results):
        if doc_id in relevant_docs:
            relevant_cnt += 1
        if i == k - 1:
            break
    return relevant_cnt / len(relevant_docs)


def average_precision(results, relevant_docs) -> float:
    """
    Compute Average Precision (for a single query - the results are averaged across queries to get MAP in the next few cells)

    :param results: A sorted list of 2-tuples (document_id, score), with the most relevant document in the first position
    :param relevant_docs: A set of relevant documents.
    :return: Average Precision
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


def err(results, relevant_docs):
    """
    Compute the expected reciprocal rank.

    :param results: A sorted list of 2-tuples (document_id, score), with the most relevant document in the first position
    :param relevant_docs: A set of relevant documents.
    :return: ERR
    """
    err_score = 0
    r_prod = 1
    for i, (doc_id, _) in enumerate(results):
        if doc_id in relevant_docs:
            err_score += 1 / (i + 1) * r_prod * 0.5
            r_prod *= 1 - 0.5
    return err_score


list_of_search_fns = [
    ("BOW", bow_search),
    ("TF-IDF", tfidf_search),
    ("NaiveQL", naive_ql_search),
    ("QL", ql_search),
    ("BM25", bm25_search),
]


def evaluate_search_fn(search_fn, metric_fns, index_set=None) -> Dict:
    """
    Build a dict query_id -> query

    :param search_fn: the name of searching function
    :param metric_fns: the name of evaluation metrics
    :param index_set: the corresponding configuration index
    :return:
    """
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
