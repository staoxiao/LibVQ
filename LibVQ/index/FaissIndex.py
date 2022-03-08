import sys
import os
import torch
import faiss
import argparse
import logging
import numpy as np
import math
from tqdm import tqdm
import json
import time

from LibVQ.dataset.dataset import load_rel
from LibVQ.index.BaseIndex import BaseIndex


class FaissIndex(BaseIndex):
    def __init__(self, index_method='ivfpq', emb_size=768, ivf_centers=10000, subvector_num=32, subvector_bits=8, dist_mode='ip'):

        assert dist_mode in ('ip', 'l2')
        self.index_metric = faiss.METRIC_INNER_PRODUCT if dist_mode == 'ip' else faiss.METRIC_L2

        if index_method == 'flat':
            self.index = faiss.IndexFlatIP(emb_size) if dist_mode == 'ip' else faiss.IndexFlatL2(emb_size)
        elif index_method == 'ivf':
            quantizer = faiss.IndexFlatIP(emb_size) if dist_mode == 'ip' else faiss.IndexFlatL2(emb_size)
            self.index = faiss.IndexIVFFlat(quantizer, emb_size, ivf_centers, self.index_metric)
        elif index_method == 'ivf_opq':
            self.index = faiss.index_factory(emb_size, f"OPQ{subvector_num}, IVF{ivf_centers}, PQ{subvector_num}x{subvector_bits}", self.index_metric)
        elif index_method == 'ivf_pq':
            self.index = faiss.index_factory(emb_size, f"IVF{ivf_centers}, PQ{subvector_num}x{subvector_bits}", self.index_metric)
        elif index_method == 'opq':
            self.index = faiss.index_factory(emb_size, f"OPQ{subvector_num}, PQ{subvector_num}x{subvector_bits}", self.index_metric)
        elif index_method == 'pq':
            self.index = faiss.index_factory(emb_size, f"PQ{subvector_num}x{subvector_bits}", self.index_metric)

        self.index_method = index_method
        self.ivf_centers =ivf_centers
        self.subvector_num = subvector_num
        self.is_trained = False

    def fit(self, embeddings):
        if self.index_method != 'flat':
            self.index.train(embeddings)
        self.is_trained = True

    def add(self, embeddings):
        if self.is_trained:
            self.index.add(embeddings)
        else:
            raise RuntimeError("The index need to be trained")

    def load_index(self, index_file):
        self.index = faiss.read_index(index_file)

    def save_index(self, index_file):
        faiss.write_index(self.index, index_file)

    def CPU_to_GPU(self, gpu_index):
        res = faiss.StandardGpuResources()
        res.setTempMemory(1024 * 1024 * 1024)
        co = faiss.GpuClonerOptions()
        self.index = faiss.index_cpu_to_gpu(res, gpu_index, self.index, co)

    def GPU_to_CPU(self):
        self.index = faiss.index_gpu_to_cpu(self.index)

    def set_nprobe(self, nprobe):
        if isinstance(self.index, faiss.IndexPreTransform):
            ivf_index = faiss.downcast_index(self.index.index)
            ivf_index.nprobe = nprobe
        else:
            self.index.nprobe = nprobe

    def search(self, query_embeddings, topk=1000, batch_size=None):
        start_time = time.time()
        if batch_size:
            batch_num = math.ceil(len(query_embeddings) / batch_size)
            all_scores = []
            all_search_results = []
            for step in tqdm(range(batch_num)):
                start = batch_size * step
                end = min(batch_size * (step + 1), len(query_embeddings))
                batch_emb = np.array(query_embeddings[start:end])
                score, batch_results = self.index.search(batch_emb, topk)
                all_search_results.extend(batch_results.tolist())
                all_scores.extend(score.tolist())
        else:
            all_scores, all_search_results = self.index.search(query_embeddings, topk)
        search_time = time.time() - start_time
        print(f'number of query:{len(query_embeddings)},  searching time per query: {search_time / len(query_embeddings)}')
        return all_scores, all_search_results

    def get_ivf_listnum(self):
        pass

