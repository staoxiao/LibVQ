from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import faiss
import re
from math import fabs
# import datasets
import json
import gc
# from nltk.corpus import stopwords
# pip install scikit-learn

class TopicModel:
    def __init__(self, classes):
        self.topic_name = list()
        self.topic_count = list()
        self.topic_info = list()
        self.length = len(classes)
        tf_idf, words, count = c_tf_idf(classes)
        indices = extract_top_n_words_per_topic(tf_idf)
        self.__information(words, count, indices, tf_idf)

    def __information(self, words, count, indices, tf_idf, n=10):
        print(words)
        t = count.toarray()
        w = t.sum(axis=1)

        if len(indices[0]) > n:
            indices = indices[:, :n]
        tf_idf_transposed = tf_idf.T
        # tf_idf_transposed = tf_idf
        # t_num = np.zeros(len(t))
        # for i in range(len(t)):
        #     for j in range(len(t[0])):
        #         if t[i][j] > 0:
        #             t_num[i] += 1

        for i in range(self.length):
            name = str(i-1)
            info = {}
            for j in range(n):
                label = indices[i][j]
                if fabs(tf_idf_transposed[i][label]) < 1e-6:
                    break
                if j < 4:
                    name = name + '_' + words[label]
                info[words[label]] = tf_idf_transposed[i][label]
            self.topic_count.append(w[i])
            self.topic_name.append(name)
            self.topic_info.append(info)

    def get_all_topic_info(self):
        print('%10s\t%10s\t%20s' % ('Topic', 'Count', 'Name'))
        for i in range(self.length):
            if i < 5 or i > self.length - 5:
                print('%10d\t%10d\t%20s' % (i-1, self.topic_count[i], self.topic_name[i]))
            elif i == 10:
                print('...................')

    def get_topic_info(self, i: int):
        if i > self.length-1 or i < -1:
            print('Error: topic info %d is to large' % i)
        print('%10s\t%10s\t%20s' % ('Topic', 'Count', 'Name'))
        print('%10d\t%10d\t%20s' % (i, self.topic_count[i+1], self.topic_name[i+1]))

    def get_topic(self, i: int):
        if i > self.length-1 or i < -1:
            print('Error: topic info %d is to large' % i)
        print(self.topic_info[i+1])


def c_tf_idf(classes, ngram_range=(1, 1)):
    """
    :param classes: (n, 1)  n rows classes, one lines content
    :param ngram_range:
    :return:
    """
    vectorizer = CountVectorizer(ngram_range=ngram_range)
    count = vectorizer.fit_transform(classes)
    t = count.toarray()
    w = t.sum(axis=1)
    tf_tc = np.divide(t.T, w)
    A = t.sum() / len(classes)
    tf_t = t.sum(axis=0) / t.sum()
    idf = np.log(1 + np.divide(A, tf_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf_tc, idf)
    return tf_idf, vectorizer.get_feature_names_out(), count


def extract_top_n_words_per_topic(tf_idf):
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -1::-1]
    return indices


def get_classes(docs_path, index_path):
    docsFile = open(docs_path, 'r', encoding='UTF-8')
    id2text = dict()
    count = 0
    for line in docsFile:
        content = ' '.join(line.strip('\n').split('\t')[1:])
        id2text[count] = content
        count += 1
    docsFile.close()
    classes_id = load_ivf_list_from_faiss_index(index_path)
    classes = []
    rmove = u'[0-9!"#$%&()*+,-./:;<=>?@?[\\]^_`{|}~]+'
    for i in range(len(classes_id)):
        temp = ''
        num = 0
        for c_id in classes_id[i]:
            temp = temp + ' ' + re.sub(rmove, ' ', id2text[c_id]).lower()
            num += 1
        classes.append(temp)
    # return classes
    return remove_stop_words(classes)

def remove_stop_words(classes):
    # stop class
    stop_class = ''
    # stop_words = stopwords.words('english')
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                  "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she'search2",
                  'her', 'hers', 'herself', 'it', "it'search2", 'its', 'itself', 'they', 'them', 'their', 'theirs',
                  'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is',
                  'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
                  'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
                  'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
                  'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                  'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
                  'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
                  'so', 'than', 'too', 'very', 'search2', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've",
                  'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
                  "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
                  'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn',
                  "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    # rmove = u'[0-9!"#$%&()*+,-./:;<=>?@?[\\]^_`{|}~]+'
    for i in range(len(classes)):
        content = classes[i].split()
        # content = re.sub(rmove, ' ', classes[i]).split()
        classes[i] = ' '.join([word for word in content if word not in stop_words])
        stop_class = stop_class + ' ' + ' '.join([word for word in content if word in stop_words])
    classes.insert(0, stop_class)
    return classes


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
    classes_id = []
    for i in range(len(center_vecs)):
        ls = invlists.list_size(i)
        list_ids = faiss.rev_swig_ptr(invlists.get_ids(i), ls)
        classes_id.append(list_ids.copy())
    return classes_id


# def save_list_to_json(output_file, content):
#     with open(output_file, 'w+', encoding='utf-8') as f:
#         for line in content:
#             data = {'content': line}
#             f.write(json.dumps(data) + '\n')
#
#
# def load_list_from_json(input_file):
#     datas = datasets.load_dataset('json', data_files=input_file, split='train', cache_dir='./temp_datas')
#     return datas['content']
#
#
# def save_numpy_to_json(output_file, content_memmap):
#     with open(output_file, 'w+', encoding='utf-8') as f:
#         for line in content_memmap:
#             data = {'content': [float(x) for x in line]}
#             f.write(json.dumps(data) + '\n')
#
#
# def load_numpy_from_json(input_file):
#     datas = datasets.load_dataset('json', data_files=input_file, split='train', cache_dir='./temp_datas')
#     return datas['content']

# classes = list()
# classes.append('we are family, and the we want to talk with you')
# classes.append('nice to see you, my dear baby The')
# classes.append('you looks like so beautiful, can you marry with me')
# classes = remove_stop_words(classes)
# topic = TopicModel(classes)
# topic.get_topic(-1)
# topic.get_topic(0)
# topic.get_topic(1)
# topic.get_topic(2)