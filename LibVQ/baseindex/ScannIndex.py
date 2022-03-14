import scann
import time
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional

from LibVQ.baseindex.BaseIndex import BaseIndex


class ScaNNIndex(BaseIndex):
    def __init__(self,
                 doc_embeddings,
                 ivf_centers_num=10000,
                 subvector_num=32,
                 hash_type='lut256',
                 anisotropic_quantization_threshold=0.2):
        self.index = scann.scann_ops_pybind.builder(doc_embeddings, 100, "dot_product").tree(
             num_leaves=ivf_centers_num, num_leaves_to_search=1,
             training_sample_size=min(len(doc_embeddings), ivf_centers_num * 256)).score_ah(
             len(doc_embeddings[0]) // subvector_num, anisotropic_quantization_threshold=anisotropic_quantization_threshold, hash_type=hash_type).build()
        # self.index.set_n_training_threads(threads_num)

    def search(self,
               query_embeddings,
               topk=1000,
               nprobe=1):
        start_time = time.time()
        all_search_results, all_scores = self.index.search_batched(query_embeddings, leaves_to_search=nprobe,
                                                                   final_num_neighbors=topk)
        search_time = time.time() - start_time
        print(
            f'number of query:{len(query_embeddings)},  searching time per query: {search_time / len(query_embeddings)}')
        return all_scores, all_search_results

    def test(self, query_embeddings, ground_truths, topk, nprobe, MRR_cutoffs, Recall_cutoffs, qids=None):
        assert max(max(MRR_cutoffs), max(Recall_cutoffs)) <= topk
        scores, retrieve_results = self.search(query_embeddings, topk, nprobe)
        return self.evaluate(retrieve_results, ground_truths, MRR_cutoffs, Recall_cutoffs, qids)
