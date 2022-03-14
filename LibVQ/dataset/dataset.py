import json
import logging
import numpy
import numpy as np
import os
import pickle
import random
import torch
from collections import defaultdict
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import List, Union, Dict


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
                 emb_size: int = 768):
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

        self.docs_list = list(range(len(self.doc_embeddings)))
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
            if emb is not None:
                embeddings = np.memmap(emb, dtype=np.float32, mode="r")
                return embeddings.reshape(-1, emb_size)
            else:
                return None
        else:
            return emb

    def __getitem__(self, index):
        query = self.query_list[index]

        pos = random.sample(self.query2pos[query], 1)[0]
        if self.per_query_neg_num > len(self.query2neg[query]):
            negs = self.query2neg[query] + random.sample(self.docs_list, self.per_query_neg_num - len(self.query2neg[query]))
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

            origin_q_emb.append(q_emb)
            origin_d_emb.append(d_emb)
            origin_n_emb.extend(n_emb)

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

        origin_q_emb = torch.FloatTensor(origin_q_emb) if origin_q_emb[0] is not None else None
        origin_d_emb = torch.FloatTensor(origin_d_emb) if origin_d_emb[0] is not None else None
        origin_n_emb = torch.FloatTensor(origin_n_emb) if origin_n_emb[0] is not None else None

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
