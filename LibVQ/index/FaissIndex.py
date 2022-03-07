import sys
sys.path.append('./src')
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
from dataset.dataset import load_rel
from index.BaseIndex import BaseIndex

logger = logging.Logger(__name__, level=logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s- %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gpu_search", action='store_true')

    parser.add_argument("--index_method", type=str, required=True)
    parser.add_argument("--subvector_num", type=int, default=64)
    parser.add_argument("--subvector_bits", type=int, default=8)
    parser.add_argument("--ivf_centers", type=int, default=10000)
    parser.add_argument("--dist_mode", type=str, required=True)
    parser.add_argument("--emb_size", type=int, required=True)

    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument("--nprobe", type=int, default=100)
    parser.add_argument("--mode", type=str, default='dev')
    parser.add_argument("--save_hardneg_to_json", action='store_true')

    parser.add_argument("--doc_file", type=str, default=None)
    parser.add_argument("--query_file", type=str, default=None)
    parser.add_argument("--init_index_path", type=str, default=None)
    parser.add_argument("--rel_file", type=str, default=None)

    parser.add_argument("--MRR_cutoffs", type=int, nargs='+', default=[10, 100])
    parser.add_argument("--Recall_cutoffs", type=int, nargs='+', default=[5, 10, 30, 50, 100])

    args = parser.parse_args()
    logger.info(args)

    os.makedirs(args.output_dir, exist_ok=True)
    if args.doc_file is None:
        args.doc_embed_path = os.path.join(args.output_dir, "passages.memmap")
    else:
        args.doc_embed_path = args.doc_file
    if args.query_file is None:
        args.query_embed_path = os.path.join(args.output_dir, f"{args.mode}-query.memmap")
    else:
        args.query_embed_path = args.query_file

    doc_embeddings = np.memmap(args.doc_embed_path,
        dtype=np.float32, mode="r")
    doc_embeddings = doc_embeddings.reshape(-1, args.emb_size)

    query_embeddings = np.memmap(args.query_embed_path,
                                 dtype=np.float32, mode="r")
    query_embeddings = query_embeddings.reshape(-1, args.emb_size)

    faiss.omp_set_num_threads(32)
    index = FaissIndex(index_method=args.index_method,
                       emb_size=args.emb_size,
                       ivf_centers=args.ivf_centers,
                       subvector_num=args.subvector_num,
                       subvector_bits=args.subvector_bits,
                       dist_mode=args.dist_mode)
    # index.fit(doc_embeddings)
    # index.add(doc_embeddings)
    index.load_index(os.path.join(args.output_dir, f'{args.index_method}.index'))

    # index.save_index(os.path.join(args.output_dir, f'{args.index_method}.index'))
    index.set_nprobe(args.nprobe)

    ground_truths = load_rel(args.rel_file)
    if args.mode != 'train':
        qids = list(range(len(query_embeddings)))
        index.test(query_embeddings, qids, ground_truths, topk=args.topk, batch_size=64,
                   MRR_cutoffs=args.MRR_cutoffs, Recall_cutoffs=args.Recall_cutoffs)

    if args.save_hardneg_to_json:
        score, search_results = index.search(query_embeddings, topk=1000, batch_size=64)
        neg_dict = {}
        for qid, neighbors in enumerate(search_results):
            neg = list(filter(lambda x: x not in ground_truths[qid], neighbors))
            neg_dict[qid] = neg
        json.dump(neg_dict, open(os.path.join(args.output_dir, f"{args.mode}_hardneg.json"), 'w'))

if __name__ == "__main__":
    main()

