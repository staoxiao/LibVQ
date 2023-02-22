from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import joblib
import pandas as pd
import faiss
import os
import torch
import re
from math import fabs
from typing import Tuple
from collections.abc import Iterable
from LibVQ.base_index import FaissIndex
# pip install scikit-learn

class TopicModel:
    def __init__(self,
                 language: str = "english",
                 top_n_words: int = 10,
                 n_gram_range: Tuple[int, int] = (1, 1)
                 ):
        """

        :param language: The main language used in your documents
        :param top_n_words: The number of words per topic to extract. Setting this
                            too high can negatively impact topic embeddings as topics
                            are typically best represented by at most 10 words.
        :param n_gram_range:The n-gram range for the CountVectorizer.
                            Advised to keep high values between 1 and 3.
                            More would likely lead to memory issues.
        """
        # Topic-based parameters
        if top_n_words > 30:
            raise ValueError("top_n_words should be lower or equal to 30. The preferred value is 10.")
        self.language = language
        self.top_n_words = top_n_words
        self.n_gram_range = n_gram_range

        self.vectorizer_model = CountVectorizer(ngram_range=self.n_gram_range, stop_words=self.language)

        self.topic_name = list()
        self.topic_count = list()
        self.topic_info = dict()
        self.length = 0
        self.single_documents = None
        self.correspodent_id = None

    def fit(self, document_info):
        documents, documents_count, single_documents, correspodent_id = \
            document_info[0], document_info[1], document_info[2], document_info[3]
        check_documents_type(documents)
        self.length = len(documents)
        documents = remove_stop_words(documents) if self.language == 'english' else documents
        self.single_documents = single_documents
        self.correspodent_id = correspodent_id
        tf_idf, words, count = c_tf_idf(documents, self.vectorizer_model)
        indices = extract_top_n_words_per_topic(tf_idf)
        self.__information(words, count, indices, tf_idf, self.top_n_words, documents_count)

    def __information(self, words, count, indices, tf_idf, n, documents_count):
        # print(words)
        # t = count.toarray()
        # w = t.sum(axis=1)

        if len(indices[0]) > n:
            indices = indices[:, :n]
        tf_idf_transposed = tf_idf.T

        for i in range(self.length):
            name = str(i)
            info = {}
            for j in range(n):
                label = indices[i][j]
                if fabs(tf_idf_transposed[i][label]) < 1e-6:
                    break
                if j < 4:
                    name = name + '_' + words[label]
                info[words[label]] = tf_idf_transposed[i][label]
            # self.topic_count.append(w[i])
            self.topic_count.append(documents_count[i])
            self.topic_name.append(name)
            self.topic_info[i] = info

    def get_topic_info(self, topic: int = None):
        self.check_is_fit()
        data = {'Topic': range(self.length),
                'Count': self.topic_count,
                'Name': self.topic_name}
        info = pd.DataFrame(data)
        if topic is None:
            return info
        else:
            return info.loc[info.Topic == topic]

    def get_topic(self, i: int):
        self.check_is_fit()
        if i > self.length-1:
            print('Error: topic info %d is to large' % i)
        return self.topic_info[i]

    def get_topics(self):
        self.check_is_fit()
        return self.topic_info

    def get_document_info(self, doc: int = None):
        self.check_is_fit()
        if self.single_documents and self.correspodent_id:
            data = {'Document': self.single_documents,
                    'Document_id': range(len(self.single_documents)),
                    'Topic': self.correspodent_id,
                    'Name': [self.topic_name[i] for i in self.correspodent_id]}
            info = pd.DataFrame(data)
            if doc is None:
                return info
            else:
                return info.loc[info.Document_id == doc]

    def save(self,
             path: str) -> None:
        with open(path, 'wb') as file:
            joblib.dump(self, file)

    @classmethod
    def load(cls,
             path: str):
        with open(path, 'rb') as file:
            return joblib.load(file)

    def check_is_fit(self):
        if self.length == 0:
            raise ValueError('Pleas fit your documents first')

    def find_nearest_topic(self, doc_emb, index_file):
        index = faiss.read_index(index_file)
        if isinstance(index, faiss.IndexPreTransform):
            ivf_index = faiss.downcast_index(index.index)
        else:
            ivf_index = index

        coarse_quantizer = faiss.downcast_index(ivf_index.quantizer)
        coarse_embeds = faiss.vector_to_array(coarse_quantizer.xb)
        center_vecs = coarse_embeds.reshape((-1, ivf_index.d))
        topic_id = []
        for j in range(len(doc_emb)):
            loc = -1
            value = float('inf')
            for i in range(len(center_vecs)):
                if np.linalg.norm(center_vecs[i] - doc_emb[j]) < value:
                    value = np.linalg.norm(center_vecs[i] - doc_emb[j])
                    loc = i
            topic_id.append(loc)
        if len(topic_id) == 1:
            return topic_id[0]
        return topic_id

def c_tf_idf(documents, vectorizer):
    """
    :param documents: (n, 1)  n rows documents, one lines content
    :param ngram_range:
    :return:
    """
    count = vectorizer.fit_transform(documents)
    t = count.toarray()
    w = t.sum(axis=1)
    tf_tc = np.divide(t.T, w)
    # A = t.sum() / len(documents)
    # tf_t = t.sum(axis=0) / t.sum()
    # idf = np.log(1 + np.divide(A, tf_t)).reshape(-1, 1)

    df = np.squeeze(np.asarray(t.sum(axis=0)))
    avg_nr_samples = int(t.sum(axis=1).mean())
    idf = np.log((avg_nr_samples / df) + 1).reshape(-1, 1)

    tf_idf = np.multiply(tf_tc, idf)
    return tf_idf, vectorizer.get_feature_names_out(), count

def extract_top_n_words_per_topic(tf_idf):
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -1::-1]
    return indices


def get_documents(docs_path, index_path):
    docsFile = open(docs_path, 'r', encoding='UTF-8')
    id2text = dict()
    count = 0
    for line in docsFile:
        content = ' '.join(line.strip('\n').split('\t')[1:])
        id2text[count] = content
        count += 1
    docsFile.close()
    documents_id = load_ivf_list_from_faiss_index(index_path)
    documents = []
    single_documents = []
    correspodent_id = []
    documents_count = []
    # rmove = u'[0-9!"#$%&()*+,-./:;<=>?@?[\\]^_`{|}~]+'
    for i in range(len(documents_id)):
        temp = ''
        num = 0
        for c_id in documents_id[i]:
            # temp = temp + ' ' + re.sub(rmove, ' ', id2text[c_id])
            temp = temp + ' ' + id2text[c_id]
            num += 1
            single_documents.append(id2text[c_id])
            correspodent_id.append(i)
        documents.append(temp)
        documents_count.append(num)
    return [documents, documents_count, single_documents, correspodent_id]

def remove_stop_words(documents):
    # stop_words = stopwords.words('english')
    if not os.path.exists('topic_modeling/stop_words.txt'):
        print('Please make sure exists file topic_modeling/stop_words.txt')
        return documents
    rmove = u'[0-9!"#$%&()*+,-./:;<=>?@?[\\]^_`{|}~]+'
    stop_words = []
    f = open('topic_modeling/stop_words.txt', 'r', encoding='utf-8')
    for line in f:
        stop_word = line.strip('\n')
        stop_words.append(stop_word)
    f.close()
    for i in range(len(documents)):
        content = re.sub(rmove, ' ', documents[i]).lower().split()
        documents[i] = ' '.join([word for word in content if word not in stop_words])
    return documents

def get_doc_emb(docs, parameters_path):
    encoder_config_file = os.path.join(parameters_path, 'encoder_config.json')
    if not os.path.exists(encoder_config_file):
        print('You should provide ' + encoder_config_file)
        return None
    index = FaissIndex.load_all(parameters_path)
    if isinstance(docs, str):
        docs = [docs]
    input_data = index.model.text_tokenizer(docs, padding=True, truncation=True)
    input_ids = torch.LongTensor(input_data['input_ids'])
    attention_mask = torch.LongTensor(input_data['attention_mask'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    index.model.to(device)
    docs_emb = index.model.encoder.query_emb(input_ids, attention_mask).detach().cpu().numpy()
    if index.pooler:
        with torch.no_grad():
            docs_emb = torch.Tensor(docs_emb.copy()).to(device)
            index.pooler.to(device)
            docs_emb = index.pooler(docs_emb).detach().cpu().numpy()
    return docs_emb

def load_ivf_list_from_faiss_index(index_file):
    index = faiss.read_index(index_file)
    if isinstance(index, faiss.IndexPreTransform):
        ivf_index = faiss.downcast_index(index.index)
    else:
        ivf_index = index

    coarse_quantizer = faiss.downcast_index(ivf_index.quantizer)
    coarse_embeds = faiss.vector_to_array(coarse_quantizer.xb)
    center_vecs = coarse_embeds.reshape((-1, ivf_index.d))

    invlists = ivf_index.invlists
    documents_id = []
    for i in range(len(center_vecs)):
        ls = invlists.list_size(i)
        list_ids = faiss.rev_swig_ptr(invlists.get_ids(i), ls)
        documents_id.append(list_ids.copy())
    return documents_id


def check_documents_type(documents):
    """ Check whether the input documents are indeed a list of strings """
    if isinstance(documents, Iterable) and not isinstance(documents, str):
        if not any([isinstance(doc, str) for doc in documents]):
            raise TypeError("Make sure that the iterable only contains strings.")

    else:
        raise TypeError("Make sure that the documents variable is an iterable containing strings only.")

# documents = list()
# documents.append('we one two 1992 - - - --+}]p-- it\'s are family, and would would would the we want to talk with you')
# documents.append('nice to see you, my dear baby The')
# documents.append('you looks like so beautiful, can you marry with me')
# topic = TopicModel()
# topic.fit(documents)
# print(topic.get_topic_info())
# print(topic.get_topic_info(1))
# print(topic.get_topics())
# topic.get_topic(2)
# count = CountVectorizer(ngram_range=(1, 1), stop_words='english')
# print(count.stop_words)