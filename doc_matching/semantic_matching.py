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
from doc_matching.preprocess import *
from doc_matching.term_matching import *


def dot(vec_1, vec_2):
    """
    vec_1 and vec_2 are of the form: [(int, float), (int, float), ...]
    Return the dot product of two such vectors, computed only on the floats

    :param vec_1: vectorized input
    :param vec_2: vectorized input
    :return: the doc product
    """
    return sum([vec_1[i][1] * vec_2[i][1] for i in range(len(vec_1))])


def cosine_sim(vec_1, vec_2):
    return dot(vec_1, vec_2) / (
        (np.sqrt(dot(vec_1, vec_1)) * np.sqrt(dot(vec_2, vec_2))) + 1e-6
    )


def jenson_shannon_divergence(vec_1, vec_2, assert_prob=False):
    """
    The Jenson-Shannon divergence is a symmetric and finite measure on two probability distributions (unlike the KL, which is neither).
    Computes the Jensen-Shannon divergence between two probability distributions.

    :param vec_1: vectorized input
    :param vec_2: vectorized input
    :param assert_prob:
    :return: Jensen-Shannon divergence
    """
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
        self.model = LsiModel(
            self.corpus,
            num_topics=self.num_topics,
            id2word=self.id2word,
            chunksize=self.chunksize,
        )


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
                score = self.similarity_fn(
                    document_single, query_vector
                )  # compute similary between the query vector to each vectorized document
                empty_list.append(tuple((doc_id, score)))
            else:
                score = 0
                empty_list.append(tuple((doc_id, score)))

        return empty_list

    def search(self, query):
        scores = self._compute_sim(self.vsrm.vectorize_query(query))
        scores.sort(key=lambda _: -_[1])
        return scores


class LdaRetrievalModel(VectorSpaceRetrievalModel):
    def __init__(self, doc_repr):
        super().__init__(doc_repr)
        self.num_topics = 100
        self.chunksize = 2000
        self.passes = 20
        self.iterations = 400
        self.eval_every = 10
        self.minimum_probability = 0.0
        self.alpha = "auto"
        self.eta = "auto"

    def train_model(self):
        self.model = LdaModel(
            self.corpus,
            self.num_topics,
            id2word=self.id2word,
            chunksize=self.chunksize,
            passes=self.passes,
            iterations=self.iterations,
            eval_every=self.eval_every,
            minimum_probability=self.minimum_probability,
            alpha=self.alpha,
            eta=self.eta,
        )


class W2VRetrievalModel(VectorSpaceRetrievalModel):
    def __init__(self, doc_repr):
        super().__init__(doc_repr)

        self.size = 100
        self.min_count = 1

    def train_model(self):
        self.model = Word2Vec(self.documents, size=self.size, min_count=self.min_count)
        self.model.save("word2vec-google-news-300.model")

    def vectorize_documents(self):
        vectors = {}
        for (doc_id, _), cc in zip(self.doc_repr, self.documents):
            vector_dim = self.model.vector_size
            arr = np.empty((0, vector_dim), dtype="f")

            for wrd in cc:
                if wrd in self.model.wv.vocab:
                    word_array = self.model.wv[wrd]
                    norm = np.linalg.norm(word_array)
                    word_array = (word_array / norm).reshape(1, -1)
                    arr = np.append(arr, np.array(word_array), axis=0)
                else:
                    word_array = np.zeros(self.size).reshape(1, -1)
                    arr = np.append(arr, np.array(word_array), axis=0)

            list1 = np.mean(arr, axis=0)
            list2 = list(range(self.size))
            vectors[doc_id] = list(
                zip(list2, list1)
            )  # save vectorized query for each doc
        return vectors

    def vectorize_query(self, query):
        query = process_text(query, **config_2)
        vector_dim = self.model.vector_size
        arr = np.empty((0, vector_dim), dtype="f")

        for wrd in query:
            if wrd in self.model.wv.vocab:
                word_array = self.model.wv[wrd]  # infer vector for each word

                norm = np.linalg.norm(word_array)
                word_array = (word_array / norm).reshape(
                    1, -1
                )  # normalize the inferred vector

                arr = np.append(arr, np.array(word_array), axis=0)
            else:
                word_array = np.zeros(self.size).reshape(
                    1, -1
                )  # if the word is not present, return 0 for all dimension

                arr = np.append(arr, np.array(word_array), axis=0)

        list1 = np.mean(arr, axis=0)  # average over each dimension
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

        self.vector_size = 100
        self.min_count = 1
        self.epochs = 20

        self.taggedDocument = [
            TaggedDocument(doc, [i]) for i, doc in enumerate(self.documents)
        ]

    def train_model(self):
        self.model = Doc2Vec(
            self.taggedDocument,
            size=self.vector_size,
            min_count=self.min_count,
            epochs=self.epochs,
        )

    def vectorize_documents(self):
        vectors = {}

        for (doc_id, _), cc in zip(self.doc_repr, self.taggedDocument):
            list1 = self.model.infer_vector(cc[0])  # infer vector for the query
            list2 = list(range(self.vector_size))

            vectors[doc_id] = list(zip(list2, list1))
        return vectors

    def vectorize_query(self, query):
        query = process_text(query, **config_2)

        list1 = self.model.infer_vector(query)  # infer vector for the query
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

        empty_list = []
        query_vector = self.vsrm.vectorize_query(query)

        for key in newdict:
            doc_id = key
            document_single = newdict.get(key)

            if document_single != [] and query_vector != []:
                score = self.similarity_fn(
                    document_single, query_vector
                )  # compute similary between the query vector to each vectorized document
                empty_list.append(tuple((doc_id, score)))
            else:
                score = 0
                empty_list.append(tuple((doc_id, score)))

        empty_list.sort(key=lambda _: -_[1])
        return empty_list


def plot_evaluation_metric(list_of_sem_search_fns: List,
                           list_of_metrics: List) -> None:
    ERR = {}
    MAP = {}
    Precision1 = {}
    Precision10 = {}
    Precision5 = {}
    Recall1 = {}
    Recall10 = {}
    Recall5 = {}
    for x in list_of_sem_search_fns:
        print(x[0])
        sem_evaluation = evaluate_search_fn(x[1], list_of_metrics)
        ERR[x[0]] = sem_evaluation.get('ERR')
        MAP[x[0]] = sem_evaluation.get('MAP')
        Precision1[x[0]] = sem_evaluation.get('Precision@1')
        Precision10[x[0]] = sem_evaluation.get('Precision@10')
        Precision5[x[0]] = sem_evaluation.get('Precision@5')
        Recall1[x[0]] = sem_evaluation.get('Recall@1')
        Recall10[x[0]] = sem_evaluation.get('Recall@10')
        Recall5[x[0]] = sem_evaluation.get('Recall@5')

    # for semantic models
    labels = ['ERR', 'MAP', 'Precision@1', 'Precision@10', 'Precision@5', 'Recall@1', 'Recall@10', 'Recall@5']

    lda_list = [ERR.get("lda"), MAP.get("lda"), Precision1.get("lda"), Precision10.get("lda"), Precision5.get("lda"),
                Recall1.get("lda"), Recall10.get("lda"), Recall5.get("lda")]
    lsi_list = [ERR.get("lsi"), MAP.get("lsi"), Precision1.get("lsi"), Precision10.get("lsi"), Precision5.get("lsi"),
                Recall1.get("lsi"), Recall10.get("lsi"), Recall5.get("lsi")]
    w2v_list = [ERR.get("w2v"), MAP.get("w2v"), Precision1.get("w2v"), Precision10.get("w2v"), Precision5.get("w2v"),
                Recall1.get("w2v"), Recall10.get("w2v"), Recall5.get("w2v")]
    w2v_pretrained_list = [ERR.get("w2v_pretrained"), MAP.get("w2v_pretrained"), Precision1.get("w2v_pretrained"),
                           Precision10.get("w2v_pretrained"), Precision5.get("w2v_pretrained"),
                           Recall1.get("w2v_pretrained"), Recall10.get("w2v_pretrained"), Recall5.get("w2v_pretrained")]
    d2v_list = [ERR.get("d2v"), MAP.get("d2v"), Precision1.get("d2v"), Precision10.get("d2v"), Precision5.get("d2v"),
                Recall1.get("d2v"), Recall10.get("d2v"), Recall5.get("d2v")]

    x = np.arange(len(labels)) * 2  # the label locations

    width = 0.1  # the width of the bars

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.bar(x - 2 * width, lda_list, width, label='lda')
    ax.bar(x - 1 * width, lsi_list, width, label='lsi')
    ax.bar(x, w2v_list, width, label='w2v')
    ax.bar(x + 1 * width, w2v_pretrained_list, width, label='w2v_pretrained')
    ax.bar(x + 2 * width, d2v_list, width, label='d2v')

    # notation
    ax.set_ylabel('Scores')
    ax.set_title('Scores by different evaluation metrics for semantic-based models')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
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

    # LDA
    lda = LdaRetrievalModel(doc_repr_2)
    lda.train_model()

    # LSI
    lsi = LsiRetrievalModel(doc_repr_2)
    lsi.train_model()

    # Word2Vec
    w2v = W2VRetrievalModel(doc_repr_2)
    w2v.train_model()

    # Word2Vec with pretrained
    w2v_pretrained = W2VPretrainedRetrievalModel(doc_repr_2)
    w2v_pretrained.train_model()

    # Word2Vec
    d2v = D2VRetrievalModel(doc_repr_2)
    d2v.train_model()

    drm_lda = DenseRetrievalRanker(lda, jenson_shannon_sim)
    drm_lsi = DenseRetrievalRanker(lsi, cosine_sim)
    drm_w2v = DenseRetrievalRanker(w2v, cosine_sim)
    drm_w2v_pretrained = DenseRetrievalRanker(w2v_pretrained, cosine_sim)
    drm_d2v = DenseRetrievalRanker(d2v, cosine_sim)

    query = "algebraic functions"
    print("BM25: ")
    bm25_search(query, 2)
    print("LSI: ")
    drm_lsi.search(query)
    print("LDA: ")
    drm_lda.search(query)
    print("W2V: ")
    drm_w2v.search(query)
    print("W2V(Pretrained): ")
    drm_w2v_pretrained.search(query)
    print("D2V:")
    drm_d2v.search(query)

    # Rerank with BM25
    bm25_search_2 = partial(bm25_search, index_set=2)
    lsi_rerank = DenseRerankingModel(bm25_search_2, lsi, cosine_sim)
    lda_rerank = DenseRerankingModel(bm25_search_2, lda, jenson_shannon_sim)
    w2v_rerank = DenseRerankingModel(bm25_search_2, w2v, cosine_sim)
    w2v_pretrained_rerank = DenseRerankingModel(
        bm25_search_2, w2v_pretrained, cosine_sim
    )
    d2v_rerank = DenseRerankingModel(bm25_search_2, d2v, cosine_sim)


    query = "algebraic functions"
    print("BM25: ")
    bm25_search(query, 2)
    print("LSI: ")
    lsi_rerank.search(query)
    print("LDA: ")
    lda_rerank.search(query)
    print("W2V: ")
    w2v_rerank.search(query)
    print("W2V(Pretrained): ")
    w2v_pretrained_rerank.search(query)
    print("D2V:")
    d2v_rerank.search(query)
