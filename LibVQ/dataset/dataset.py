import json
import os
import pickle
import random
from collections import defaultdict
from typing import List, Union, Dict

import numpy
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class DatasetForVQ(Dataset):
    def __init__(self,
                 rel_data: Union[str, Dict[int, List[int]]] = None,
                 query_data_dir: str = None,
                 max_query_length: int = 32,
                 doc_data_dir: str = None,
                 max_doc_length: int = 256,
                 per_query_neg_num: int = 1,
                 neg_data: Union[str, Dict[int, List[int]]] = None,
                 query_embeddings: Union[str, numpy.ndarray] = None,
                 doc_embeddings: Union[str, numpy.ndarray] = None,
                 emb_size: int = None):
        """
        :param rel_data: positive doc ids for each search2: {query_id:[doc_id1, doc_id2,...]}, or a tsv file which save the relevance relationship: qeury_id \t doc_id \n.
        :param query_data_dir: path to the preprocessed tokens data (needed for jointly training search2 encoder).
        :param max_query_length: max length of search2 tokens sequence.
        :param doc_data_dir: path to the preprocessed tokens data (needed for jointly training doc encoder).
        :param max_doc_length: max length of doc tokens sequence.
        :param per_query_neg_num: the number of negatives for each search2.
        :param neg_data: negative doc ids for each search2: {query_id:[doc_id1, doc_id2,...]}, or a pickle file which save the query2neg.
                        if set None, it will randomly sample negative.
        :param query_embeddings: embeddings for each search2, also support pass a filename('.npy', '.memmap').
        :param doc_embeddings: embeddigns for each doc, also support pass a filename('.npy', '.memmap').
        :param emb_size: dim of embeddings.
        """
        if isinstance(rel_data, str):
            self.query2pos = load_rel(rel_data)
        else:
            self.query2pos = rel_data

        self.query_dataset, self.doc_dataset = None, None
        if query_data_dir is not None:
            self.query_dataset = TokensCache(data_dir=query_data_dir, prefix="train-queries",
                                             max_length=max_query_length)
        if doc_data_dir is not None:
            self.doc_dataset = TokensCache(data_dir=doc_data_dir, prefix="docs", max_length=max_doc_length)

        assert per_query_neg_num > 0
        self.per_query_neg_num = per_query_neg_num
        self.query_length = max_query_length
        self.doc_length = max_doc_length

        self.doc_embeddings = self.init_embedding(doc_embeddings, emb_size=emb_size)
        self.query_embeddings = self.init_embedding(query_embeddings, emb_size=emb_size)

        if self.doc_embeddings is not None:
            self.docs_list = list(range(len(self.doc_embeddings)))
        else:
            self.docs_list = list(range(len(self.doc_dataset)))
        self.query_list = list(self.query2pos.keys())

        if neg_data is not None:
            if isinstance(neg_data, str):
                self.query2neg = self.get_query2neg_from_file(neg_data)
            else:
                self.query2neg = neg_data
        else:
            self.query2neg = self.random_negative_sample(self.query_list)

    def get_query2neg_from_file(self, neg_file):
        query2neg = pickle.load(open(neg_file, 'rb'))
        return query2neg

    def random_negative_sample(self, queries):
        query2neg = {}
        for q in queries:
            neg = random.sample(self.docs_list, 100)
            query2neg[q] = list(neg)
        return query2neg

    def init_embedding(self, emb, emb_size):
        if isinstance(emb, str):
            assert 'npy' in emb or 'memmap' in emb
            if 'memmap' in emb:
                embeddings = np.memmap(emb, dtype=np.float32, mode="r")
                return embeddings.reshape(-1, emb_size)
            elif 'npy' in emb:
                return np.load(emb)
        else:
            return emb

    def __getitem__(self, index):
        query = self.query_list[index]

        pos = random.sample(list(self.query2pos[query]), 1)[0]
        if self.per_query_neg_num > len(self.query2neg[query]):
            negs = self.query2neg[query] + random.sample(self.docs_list,
                                                         self.per_query_neg_num - len(self.query2neg[query]))
        else:
            negs = random.sample(self.query2neg[query], self.per_query_neg_num)

        query_tokens, pos_tokens, negs_tokens = None, None, None
        if self.query_dataset is not None:
            query_tokens = torch.LongTensor(self.query_dataset[query])
        if self.doc_dataset is not None:
            pos_tokens = torch.LongTensor(self.doc_dataset[pos])
            negs_tokens = [torch.LongTensor(self.doc_dataset[n]) for n in negs]

        q_emb, d_emb, n_emb = None, None, None
        if self.doc_embeddings is not None:
            d_emb = self.doc_embeddings[pos]
            n_emb = self.doc_embeddings[negs]
        if self.query_embeddings is not None:
            q_emb = self.query_embeddings[query]

        return query_tokens, pos_tokens, negs_tokens, q_emb, d_emb, n_emb, pos, negs

    def __len__(self):
        return len(self.query2pos)

class DataCollatorForVQ():
    def __call__(self, examples):
        query_token_ids, query_attention_mask = [], []
        doc_token_ids, doc_attention_mask = [], []
        neg_token_ids, neg_attention_mask = [], []
        origin_q_emb, origin_d_emb, origin_n_emb = [], [], []
        doc_ids, neg_ids = [], []

        for query_tokens, pos_tokens, negs_tokens, q_emb, d_emb, n_emb, pos, negs in examples:
            if query_tokens is not None:
                query_token_ids.append(query_tokens)
                query_attention_mask.append(torch.tensor([1] * len(query_tokens)))

            if pos_tokens is not None and negs_tokens is not None:
                doc_token_ids.append(pos_tokens)
                doc_attention_mask.append(torch.tensor([1] * len(pos_tokens)))
                neg_token_ids.extend(negs_tokens)
                neg_attention_mask.extend([torch.tensor([1] * len(x)) for x in negs_tokens])

            if q_emb is not None: origin_q_emb.append(q_emb)
            if d_emb is not None: origin_d_emb.append(d_emb)
            if n_emb is not None: origin_n_emb.extend(n_emb)

            doc_ids.append(pos)
            neg_ids.extend(negs)

        if len(query_token_ids) > 0:
            query_token_ids = tensorize_batch(query_token_ids, 0)
            query_attention_mask = tensorize_batch(query_attention_mask, 0)
        else:
            query_token_ids, query_attention_mask = None, None

        if len(doc_token_ids) > 0 and len(neg_token_ids) > 0:
            doc_token_ids = tensorize_batch(doc_token_ids, 0)
            doc_attention_mask = tensorize_batch(doc_attention_mask, 0)
            neg_token_ids = tensorize_batch(neg_token_ids, 0)
            neg_attention_mask = tensorize_batch(neg_attention_mask, 0)
        else:
            doc_token_ids, doc_attention_mask, neg_token_ids, neg_attention_mask = None, None, None, None

        origin_q_emb = torch.FloatTensor(origin_q_emb) if len(origin_q_emb)>0 else None
        origin_d_emb = torch.FloatTensor(origin_d_emb) if len(origin_d_emb)>0  else None
        origin_n_emb = torch.FloatTensor(origin_n_emb) if len(origin_n_emb)>0 else None

        batch = {
            "query_token_ids": query_token_ids,
            "query_attention_mask": query_attention_mask,
            "doc_token_ids": doc_token_ids,
            "doc_attention_mask": doc_attention_mask,
            "neg_token_ids": neg_token_ids,
            "neg_attention_mask": neg_attention_mask,
            "origin_q_emb": origin_q_emb,
            "origin_d_emb": origin_d_emb,
            "origin_n_emb": origin_n_emb,
            "doc_ids": doc_ids,
            "neg_ids": neg_ids
        }
        return batch


def tensorize_batch(sequences: List[torch.Tensor], padding_value, align_right=False) -> torch.Tensor:
    max_len_1 = max([s.size(0) for s in sequences])
    out_dims = (len(sequences), max_len_1)
    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length_1 = tensor.size(0)
        if align_right:
            out_tensor[i, -length_1:] = tensor
        else:
            out_tensor[i, :length_1] = tensor
    return out_tensor


class TokensCache:
    def __init__(self, data_dir, prefix, max_length):
        meta = json.load(open(f"{data_dir}/{prefix}_meta"))
        self.total_number = meta['total_number']
        self.max_seq_len = meta['max_seq_length']

        self.tokens_memmap = np.memmap(f"{data_dir}/{prefix}.memmap",
                                       shape=(self.total_number, self.max_seq_len),
                                       dtype=np.dtype(meta['type']), mode="r")
        self.lengths_memmap = np.load(f"{data_dir}/{prefix}_length.npy")

        assert len(self.lengths_memmap) == self.total_number
        self.max_length = max_length

    def __len__(self):
        return self.total_number

    def __getitem__(self, item):
        tokens = self.tokens_memmap[item, :self.lengths_memmap[item]].tolist()
        seq_length = min(self.max_length - 1, len(tokens) - 1)
        input_ids = [tokens[0]] + tokens[1:seq_length] + [tokens[-1]]
        return input_ids


class DatasetForEncoding(Dataset):
    def __init__(self, data_dir, prefix, max_length):
        self.tokens_cache = TokensCache(data_dir=data_dir, prefix=prefix, max_length=max_length)
        self.max_length = max_length

    def __len__(self):
        return len(self.tokens_cache)

    def __getitem__(self, item):
        input_ids = self.tokens_cache[item]

        attention_mask = [1] * len(input_ids) + [0] * (self.max_length - len(input_ids))
        input_ids = input_ids + [0] * (self.max_length - len(input_ids))

        return torch.LongTensor(input_ids), torch.LongTensor(attention_mask)


def load_rel(rel_file):
    reldict = defaultdict(set)
    for line in tqdm(open(rel_file), desc=os.path.split(rel_file)[1]):
        qid, pid = line.split()[:2]
        qid, pid = int(qid), int(pid)
        reldict[qid].add(pid)
    return reldict


def write_rel(rel_file, reldict):
    with open(rel_file, 'w', encoding='utf-8') as f:
        for q, ds in reldict.items():
            for d in ds:
                f.write(str(q) + '\t' + str(d) + '\n')

class Datasets():
    def __init__(self, file_path, emb_size: int = None, max_doc_length: int = 256, max_query_length: int = 32,
                 preprocess_dir: str = None, embedding_dir: str = None):
        """

        :param file_path: path to load file
        :param emb_size: embedding size of your docs.memmap/train-queries.memmap/dev-queries.memmap
        :param max_doc_length: max doc length
        :param max_query_length: max search2 length
        :param preprocess_dir: path to save preprocessed files
        :param embedding_dir: path to save embedding files
        """
        # if file_type not in ['text2text', 'text2img', 'img2img']:
        #     raise ValueError("your file_type must in 'text2text, text2img, img2img'")
        # self.file_type = file_type
        self.file_path = file_path
        self.emb_size = emb_size
        self.max_doc_length = max_doc_length
        self.max_query_length = max_query_length
        self.preprocess_dir = os.path.join(file_path, 'preprecessed_dataset') if preprocess_dir is None else preprocess_dir
        self.embedding_dir = os.path.join(file_path, 'embedding') if embedding_dir is None else embedding_dir

        if file_path in ['MSMARCO']:
            if not os.path.exists(file_path):
                raise ValueError("Please read example/MSMARCO/REARME.md to prepare_data the dataset MSMARCO")

        dirs = os.listdir(file_path)
        self.docs_path = os.path.join(file_path, 'collection.tsv') if 'collection.tsv' in dirs else None
        self.train_queries_path = os.path.join(file_path, 'train-queries.tsv') if 'train-queries.tsv' in dirs else None
        self.train_rels_path = os.path.join(file_path, 'train-rels.tsv') if 'train-rels.tsv' in dirs else None
        self.dev_queries_path = os.path.join(file_path, 'dev-queries.tsv') if 'dev-queries.tsv' in dirs else None
        self.dev_rels_path = os.path.join(file_path, 'dev-rels.tsv') if 'dev-rels.tsv' in dirs else None
        self.doc_embeddings_dir = os.path.join(
            os.path.join(file_path, 'docs.memmap')) if 'docs.memmap' in dirs else None
        self.train_queries_embedding_dir = os.path.join(
            os.path.join(file_path, 'train-queries.memmap')) if 'train-queries.memmap' in dirs else None
        self.dev_queries_embedding_dir = os.path.join(
            os.path.join(file_path, 'dev-queries.memmap')) if 'dev-queries.memmap' in dirs else None
        self.emb_size = emb_size

        if os.path.exists(embedding_dir) and os.path.exists(preprocess_dir):
            preprocess_dirs = os.listdir(preprocess_dir)
            embedding_dirs = os.listdir(embedding_dir)
            self.train_rels_path = os.path.join(
                preprocess_dir, 'train-rels.tsv') if 'train-rels.tsv' in preprocess_dirs else None
            self.dev_rels_path = os.path.join(
                preprocess_dir, 'dev-rels.tsv') if 'dev-rels.tsv' in preprocess_dirs else None
            self.doc_embeddings_dir = os.path.join(
                os.path.join(embedding_dir, 'docs.memmap')) if 'docs.memmap' in embedding_dirs else None
            self.train_queries_embedding_dir = os.path.join(os.path.join(
                embedding_dir, 'train-queries.memmap')) if 'train-queries.memmap' in embedding_dirs else None
            self.dev_queries_embedding_dir = os.path.join(
                os.path.join(embedding_dir, 'dev-queries.memmap')) if 'dev-queries.memmap' in embedding_dirs else None

        if self.docs_path is None and self.doc_embeddings_dir is None:
            raise ValueError("you must have at least one doc file 'collection.tsv' or 'docs.memmap'")

        self.generate_query()

    def generate_query(self):
        if self.train_queries_embedding_dir is None and self.doc_embeddings_dir is not None:
            docs_embedding = np.memmap(self.doc_embeddings_dir,
                                       dtype=np.float32, mode="r")
            docs_embedding = docs_embedding.reshape(-1, self.emb_size)
            count = len(docs_embedding)
            needNum = np.random.randint(count, size=count // 1000)
            needNum = np.unique(needNum)
            temp_train = docs_embedding[needNum]
            size = len(temp_train) * self.emb_size
            temp_train = temp_train.reshape(size, )
            self.train_queries_embedding_dir = os.path.join(os.path.join(self.file_path, 'train-queries.memmap'))
            train_queries_embedding = np.memmap(self.train_queries_embedding_dir,
                                    dtype=np.float32, mode='w+', shape=(size,))
            train_queries_embedding[:] = temp_train[:]
        elif self.train_queries_path is None and self.docs_path is not None:
            self.train_queries_path = os.path.join(self.file_path, 'train-queries.tsv')
            docs = open(self.docs_path, 'r')
            query = open(self.train_queries_path, 'w+')
            for count, line in enumerate(docs):
                count += 1
            docs.close()

            needNum = np.random.randint(count, size=count // 1000)
            needNum = np.unique(needNum)
            docs = open(self.docs_path, 'r')
            for count, line in enumerate(docs):
                count += 1
                if needNum.__contains__(count):
                    query.write(line)
            docs.close()
            query.close()

