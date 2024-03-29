import math
import time
from typing import Dict, List

import faiss
import numpy
import numpy as np
from tqdm import tqdm

from LibVQ.base_index import BaseIndex


class FaissIndex(BaseIndex):
    def __init__(self,
                 index_method: str = 'ivf_opq',
                 emb_size: int = 768,
                 ivf_centers_num: int = 10000,
                 subvector_num: int = 32,
                 subvector_bits: int = 8,
                 dist_mode: str = 'ip',
                 doc_embeddings: np.ndarray = None):
        """
        Index based on Faiss Library

        :param index_method: Type of index, support flat, ivf, ivf_opq, ivf_pq, opq and pq.
        :param emb_size: Dim of embeddings.
        :param ivf_centers_num: The number of post lists
        :param subvector_num: The number of codebooks
        :param subvector_bits: The number of codewords in each codebook
        :param dist_mode: The metric to compute distance
        :param doc_embeddings: Embedding of docs.
        """
        BaseIndex.__init__(self, )

        assert dist_mode in ('ip', 'l2')
        self.index_metric = faiss.METRIC_INNER_PRODUCT if dist_mode == 'ip' else faiss.METRIC_L2

        if doc_embeddings is not None:
            emb_size = np.shape(doc_embeddings)[-1]

        if index_method == 'flat':
            self.index = faiss.IndexFlatIP(emb_size) if dist_mode == 'ip' else faiss.IndexFlatL2(emb_size)
        elif index_method == 'ivf':
            quantizer = faiss.IndexFlatIP(emb_size) if dist_mode == 'ip' else faiss.IndexFlatL2(emb_size)
            self.index = faiss.IndexIVFFlat(quantizer, emb_size, ivf_centers_num, self.index_metric)
        elif index_method == 'ivf_opq':
            self.index = faiss.index_factory(emb_size,
                                             f"OPQ{subvector_num},IVF{ivf_centers_num},PQ{subvector_num}x{subvector_bits}",
                                             self.index_metric)
        elif index_method == 'ivf_pq':
            self.index = faiss.index_factory(emb_size, f"IVF{ivf_centers_num},PQ{subvector_num}x{subvector_bits}",
                                             self.index_metric)
        elif index_method == 'opq':
            self.index = faiss.index_factory(emb_size, f"OPQ{subvector_num},PQ{subvector_num}x{subvector_bits}",
                                             self.index_metric)
        elif index_method == 'pq':
            self.index = faiss.index_factory(emb_size, f"PQ{subvector_num}x{subvector_bits}", self.index_metric)

        self.index_method = index_method
        self.ivf_centers_num = ivf_centers_num
        self.subvector_num = subvector_num
        self.is_trained = False

        if doc_embeddings is not None:
            self.fit(doc_embeddings)
            self.add(doc_embeddings)
            self.is_trained = True

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

    def CPU_to_GPU(self, gpu_index=0):
        res = faiss.StandardGpuResources()
        res.noTempMemory()
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        self.index = faiss.index_cpu_to_gpu(res, gpu_index, self.index, co)

    def GPU_to_CPU(self):
        self.index = faiss.index_gpu_to_cpu(self.index)


    def set_nprobe(self, nprobe):
        if isinstance(self.index, faiss.IndexPreTransform):
            ivf_index = faiss.downcast_index(self.index.index)
            ivf_index.nprobe = nprobe
        else:
            self.index.nprobe = nprobe

    def search(self,
               query_embeddings: numpy.ndarray,
               topk: int = 1000,
               nprobe: int = None,
               batch_size: int = 64):
        if nprobe is not None:
            self.set_nprobe(nprobe)

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
                all_search_results.extend([list(x) for x in batch_results])
                all_scores.extend([list(x) for x in score])
        else:
            all_scores, all_search_results = self.index.search(query_embeddings, topk)
        search_time = time.time() - start_time
        print(
            f'number of query:{len(query_embeddings)},  searching time per query: {search_time / len(query_embeddings)}')
        return all_scores, all_search_results

    def test(self,
             query_embeddings: numpy.ndarray,
             ground_truths: Dict[int, List[int]],
             topk: int,
             MRR_cutoffs: List[int],
             Recall_cutoffs: List[int],
             nprobe: int = 1,
             qids: List[int] = None,
             batch_size: int = 64):
        assert max(max(MRR_cutoffs), max(Recall_cutoffs)) <= topk
        scores, retrieve_results = self.search(query_embeddings, topk, nprobe, batch_size)
        return self.evaluate(retrieve_results, ground_truths, MRR_cutoffs, Recall_cutoffs, qids)

    def get_rotate_matrix(self):
        assert isinstance(self.index, faiss.IndexPreTransform)
        vt = faiss.downcast_VectorTransform(self.index.chain.at(0))
        rotate = faiss.vector_to_array(vt.A).reshape(vt.d_out, vt.d_in)
        return rotate

    def get_codebook(self):
        if isinstance(self.index, faiss.IndexPreTransform):
            pq_index = faiss.downcast_index(self.index.index)
        else:
            pq_index = self.index

        centroid_embeds = faiss.vector_to_array(pq_index.pq.centroids)
        codebook = centroid_embeds.reshape(pq_index.pq.M, pq_index.pq.ksub, pq_index.pq.dsub)
        return codebook
