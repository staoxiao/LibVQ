# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
sys.path.append('./')
import os
import torch
import faiss
import argparse
import logging
import numpy as np
from tqdm import tqdm
import json
import time
from utils.msmarco_eval import compute_metrics_from_files
from src.dataset.dataset import load_rel

logger = logging.Logger(__name__, level=logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s- %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def load_index(index_path, use_cuda, faiss_gpu_index, centroid_embeds=None, doc_embeddings=None):
    index = faiss.read_index(index_path)

    if centroid_embeds is not None:
        ivf_index = faiss.downcast_index(index.index)
        faiss.copy_array_to_vector(
            centroid_embeds.detach().cpu().numpy().ravel(),
            ivf_index.pq.centroids)
        index.remove_ids(faiss.IDSelectorRange(0, len(doc_embeddings)))
        index.add(doc_embeddings)

    if use_cuda:
        res = faiss.StandardGpuResources()
        res.setTempMemory(1024*1024*1024)
        co = faiss.GpuClonerOptions()
        if isinstance(index, faiss.IndexPreTransform):
            subvec_num = faiss.downcast_index(index.index).pq.M
        else:
            subvec_num = index.pq.M
        if int(subvec_num) >= 56:
            co.useFloat16 = True
        else:
            co.useFloat16 = False
        logger.info(f"subvec_num: {subvec_num}; useFloat16: {co.useFloat16}")
        if co.useFloat16:
            logger.warning("If the number of subvectors >= 56 and gpu search is turned on, Faiss uses float16 and therefore there is very little performance loss. You can use cpu search to obtain the best ranking effectiveness")
        index = faiss.index_cpu_to_gpu(res, faiss_gpu_index, index, co)
    return index


def search(index, args, embedding, batch_size):
    import math
    batch_num = math.ceil(len(embedding) / batch_size)
    all_scores = []
    all_search_results = []
    search_time = 0.0
    for step in tqdm(range(batch_num)):
        start = batch_size * step
        end = min(batch_size * (step + 1), len(embedding))
        batch_emb = np.array(embedding[start:end])
        temp_t = time.time()
        score, batch_results = index.search(batch_emb, args.topk)
        search_time += time.time() - temp_t
        all_search_results.extend(batch_results.tolist())
        all_scores.extend(score.tolist())
    print('searching time:', search_time,  search_time/ len(embedding))
    return all_scores, all_search_results


def faiss_search(args, doc_embeddings, query_embeddings, query_ids, embed_size=768):
    save_index_path = os.path.join(args.output_dir, f"OPQ{args.subvector_num},PQ{args.subvector_num}x8.index")
    if args.index == 'pq':
        save_index_path = os.path.join(args.output_dir, f"PQ{args.subvector_num}x8.index")

    if args.init_index_path is not None:
        ckpt = torch.load(os.path.join(args.ckpt_path, 'pytorch_model.bin'))
        centroid_embeds = ckpt['codebook']
        index = load_index(args.init_index_path, use_cuda=args.gpu_search, faiss_gpu_index=0,
                           centroid_embeds=centroid_embeds, doc_embeddings=doc_embeddings)
        # index = faiss.index_gpu_to_cpu(index)
        faiss.write_index(index, save_index_path)

    if not os.path.exists(save_index_path):
        print('creating index------------')
        res = faiss.StandardGpuResources()
        res.setTempMemory(1024 * 1024 * 512)
        co = faiss.GpuClonerOptions()
        co.useFloat16 = args.subvector_num >= 56

        faiss.omp_set_num_threads(32)
        dim = embed_size
        if args.index == 'pq':
            index = faiss.index_factory(dim,
                                        f"PQ{args.subvector_num}x8", faiss.METRIC_INNER_PRODUCT)
        else:
            index = faiss.index_factory(dim,
                                        f"OPQ{args.subvector_num},PQ{args.subvector_num}x8", faiss.METRIC_INNER_PRODUCT)
        index.verbose = True
        index = faiss.index_cpu_to_gpu(res, 0, index, co)
        index.train(doc_embeddings)

        index.add(doc_embeddings)
        index = faiss.index_gpu_to_cpu(index)
        faiss.write_index(index, save_index_path)

    index = load_index(save_index_path, use_cuda=args.gpu_search, faiss_gpu_index=0)
    if not args.gpu_search:
        faiss.omp_set_num_threads(32)

    stme = time.time()
    scores, topk = search(index, args, query_embeddings, batch_size=32)
    print('searching costs:', time.time() - stme, (time.time() - stme) / len(query_embeddings))

    file_name = os.path.join(args.output_dir, f"{args.mode}.rank_{args.topk}_score_faiss_{args.index}.tsv")
    with open(file_name, 'w') as outputfile:
        for qid, score, neighbors in zip(query_ids, scores, topk):
            for idx, pid in enumerate(neighbors):
                outputfile.write(f"{qid}\t{pid}\t{idx + 1}\t{score[idx]}\n")

    path_to_reference = f'./data/{args.data_type}/preprocess/{args.mode}-qrel.tsv'

    if args.mode != 'train':
        MRR, Recalls = compute_metrics_from_files(path_to_reference, file_name, args.MRR_cutoff, args.Recall_cutoff)

    if args.save_hardneg_to_json:
        rel_dict = load_rel(path_to_reference)
        neg_dict = {}
        for qid, neighbors in zip(query_ids, topk):
            neg = list(filter(lambda x: x not in rel_dict[qid], neighbors))
            neg_dict[qid] = neg
        json.dump(neg_dict, open(os.path.join(args.output_dir, f"{args.mode}_hardneg.json"), 'w'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--data_type", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gpu_search", action='store_true')

    parser.add_argument("--index", type=str, default='opq')
    parser.add_argument("--subvector_num", type=int, required=True)
    parser.add_argument("--subvector_num", type=int, required=True)
    parser.add_argument("--subvector_num", type=int, required=True)

    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument("--mode", type=str, default='dev')
    parser.add_argument("--save_hardneg_to_json", action='store_true')

    parser.add_argument("--doc_file", type=str, default=None)
    parser.add_argument("--query_file", type=str, default=None)
    parser.add_argument("--init_index_path", type=str, default=None)

    parser.add_argument("--MRR_cutoff", type=int, default=10)
    parser.add_argument("--Recall_cutoff", type=int, nargs='+', default=[5, 10, 30, 50, 100])

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

    embed_size = 768

    doc_embeddings = np.memmap(args.doc_embed_path,
        dtype=np.float32, mode="r")
    doc_embeddings = doc_embeddings.reshape(-1, embed_size)

    query_embeddings = np.memmap(args.query_embed_path,
                                 dtype=np.float32, mode="r")
    query_embeddings = query_embeddings.reshape(-1, 768)
    query_ids = list(range(len(query_embeddings)))

    faiss_search(args, doc_embeddings, query_embeddings, query_ids, embed_size=768)

if __name__ == "__main__":
    main()

