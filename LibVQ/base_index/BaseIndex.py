import json
from typing import Dict, List

import numpy

from LibVQ.utils import evaluate

class BaseIndex():
    def fit(self):
        raise NotImplementedError

    def add(self, embeddings):
        raise NotImplementedError

    def save_index(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def CPU_to_GPU(self, gpu_index):
        raise NotImplementedError

    def GPU_to_CPU(self):
        raise NotImplementedError

    def test(self, query_embeddings, qids, ground_truths, topk, batch_size, MRR_cutoffs, Recall_cutoffs):
        raise NotImplementedError

    def search(self,
               query_embeddings: numpy.ndarray,
               topk: int,
               nprobe: int):
        """
        search2 nearest neighbors for queries
        :param query_embeddings: queries' embeddings
        :param topk: the number of neighbors
        :param nprobe: the number of ivf post list to search2 for each search2
        :return:
        """
        raise NotImplementedError

    def hard_negative(self,
                      query_embeddings: numpy.ndarray,
                      ground_truths: Dict[int, List[int]] = None,
                      topk: int = 400,
                      batch_size: int = None,
                      nprobe: int = None) -> Dict[int, List[int]]:
        """
        search2 topk docs for search2 embeddings and filter the positive items in ground truths; the results can be viewed as queries' hard negatives
        :return: query2hardneg: hard negatives for each search2
        """
        score, search_results = self.search(query_embeddings, topk=topk, batch_size=batch_size, nprobe=nprobe)
        query2hardneg = {}
        for qid, neighbors in enumerate(search_results):
            neg = list(filter(lambda x: x not in ground_truths[qid] and x != -1, neighbors))
            query2hardneg[qid] = neg
        return query2hardneg

    def generate_virtual_traindata(self,
                                   query_embeddings: numpy.ndarray,
                                   topk: int = 400,
                                   batch_size: int = None,
                                   nprobe: int = None):
        """
        search2 topk docs for search2 embeddings, then use the top-1 doc as the positive item to form query2pos, and use the rest to form the query2neg
        """
        score, search_results = self.search(query_embeddings, topk=topk, batch_size=batch_size, nprobe=nprobe)
        query2pos = {}
        query2neg = {}
        for qid, neighbors in enumerate(search_results):
            query2pos[qid] = neighbors[:1]
            query2neg[qid] = [x for x in neighbors[1:] if x != -1]
        return query2pos, query2neg

    def evaluate(self,
                 retrieve_results: List[List[int]],
                 ground_truths: Dict[int, List[int]],
                 MRR_cutoffs: List[int] = [10],
                 Recall_cutoffs: List[int] = [5, 10, 50],
                 qids: List[int] = None):
        evaluate(retrieve_results,
                 ground_truths,
                 MRR_cutoffs,
                 Recall_cutoffs,
                 qids)
