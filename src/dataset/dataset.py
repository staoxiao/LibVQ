import os
import json
import logging
import numpy as np
from tqdm import tqdm
import random
from collections import defaultdict
from torch.utils.data import Dataset
from typing import List
import torch
logger = logging.getLogger(__name__)

class DatasetForVQ(Dataset):
    def __init__(self,
                 data_dir,
                 max_query_length,
                 max_doc_length,
                 rel_file,
                 per_query_neg_num,
                 neg_file=None,
                 doc_embeddings_file=None,
                 query_embeddings_file=None):
        self.query2pos = load_rel(rel_file)
        self.query_dataset = TokensCache(data_dir=data_dir, prefix="train-query", max_length=max_query_length)
        self.doc_dataset = TokensCache(data_dir=data_dir, prefix="passages", max_length=max_doc_length)
        self.docs_list = list(range(len(self.doc_dataset)))
        self.query_list = list(self.query2pos.keys())

        if neg_file is not None:
            self.query2neg = self.get_query2neg_from_file(neg_file)
        else:
            self.query2neg = self.random_negative_sample(self.query_list)

        assert per_query_neg_num > 0
        self.per_query_neg_num = per_query_neg_num
        self.query_length = max_query_length
        self.doc_length = max_doc_length

        self.doc_embeddings = self.init_embedding(doc_embeddings_file, emb_num=len(self.doc_dataset))
        self.query_embeddings = self.init_embedding(query_embeddings_file, emb_num=len(self.query_dataset))

    def get_query2neg_from_file(self, neg_file):
        query2neg = json.load(open(neg_file))
        return query2neg

    def random_negative_sample(self, queries):
        query2neg = {}
        for q in queries:
            neg = random.sample(self.docs_list, 100)
            query2neg[q] = set(neg)
        return query2neg

    def init_embedding(self, emb_file, emb_num):
        if emb_file is not None:
            embeddings = np.memmap(emb_file, dtype=np.float32, mode="r")
            return embeddings.reshape(emb_num, -1)
        else:
            return None

    def __getitem__(self, index):
        query = self.query_list[index]
        pos = random.sample(self.query2pos[query], 1)[0]
        negs = random.sample(self.query2neg[query], self.per_query_neg_num)

        query_tokens = torch.LongTensor(self.query_dataset[query])
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
            query_token_ids.append(query_tokens)
            query_attention_mask.append(torch.tensor([1] * len(query_tokens)))
            doc_token_ids.append(torch.LongTensor(pos_tokens))
            doc_attention_mask.append(torch.tensor([1] * len(pos_tokens)))
            neg_token_ids.extend(negs_tokens)
            neg_attention_mask.extend([torch.tensor([1] * len(x)) for x in negs_tokens])

            origin_q_emb.append(q_emb)
            origin_d_emb.append(d_emb)
            origin_n_emb.extend(n_emb)

            doc_ids.append(pos)
            neg_ids.extend(negs)

        query_token_ids = tensorize_batch(query_token_ids, 0)
        query_attention_mask = tensorize_batch(query_attention_mask, 0)
        doc_token_ids = tensorize_batch(doc_token_ids, 0)
        doc_attention_mask = tensorize_batch(doc_attention_mask, 0)
        neg_token_ids = tensorize_batch(neg_token_ids, 0)
        neg_attention_mask = tensorize_batch(neg_attention_mask, 0)

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
            "neg_ids":neg_ids
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
        self.max_seq_len = meta['embedding_size']

        self.tokens_memmap = np.memmap(f"{data_dir}/{prefix}.memmap",
            shape=(self.total_number, self.max_seq_len),
            dtype=np.dtype(meta['type']), mode="r")
        self.lengths_memmap = np.load(f"{data_dir}/{prefix}_length.npy")

        assert len(self.lengths_memmap) == self.total_number
        self.max_length = max_length
        
    def __len__(self):
        return self.total_number
    
    def __getitem__(self, item):
        tokens = list(self.tokens_memmap[item, :self.lengths_memmap[item]])
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
        input_ids = self.tokens_cache[item].tolist()
        input_ids = input_ids + [0]*(self.max_length - len(input_ids))
        attention_mask = [1]*len(input_ids) + [0]*(self.max_length - len(input_ids))
        return torch.LongTensor(input_ids), torch.LongTensor(attention_mask)


def load_rel(rel_path):
    reldict = defaultdict(set)
    for line in tqdm(open(rel_path), desc=os.path.split(rel_path)[1]):
        qid, _, pid, _ = line.split()
        qid, pid = int(qid), int(pid)
        reldict[qid].add(pid)
    return reldict


def load_rank(rank_path):
    rankdict = defaultdict(list)
    for line in tqdm(open(rank_path), desc=os.path.split(rank_path)[1]):
        qid, pid, _ = line.split()
        qid, pid = int(qid), int(pid)
        rankdict[qid].append(pid)
    return dict(rankdict)






