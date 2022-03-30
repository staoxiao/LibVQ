import json
from typing import Dict, List

import numpy


class IndexConfig():
    def __init__(self,
                 index_method: str = 'ivfopq',
                 emb_size: int = 768,
                 ivf_centers_num: int = 10000,
                 subvector_num: int = 32,
                 subvector_bits: int = 8,
                 nprobe: int = 100,
                 dist_mode: str = 'ip',
                 **kwargs):
        self.index_method = index_method
        self.emb_size = emb_size
        self.ivf_centers_num = ivf_centers_num
        self.subvector_num = subvector_num
        self.subvector_bits = subvector_bits
        self.nprobe = nprobe
        self.dist_mode = dist_mode

    @classmethod
    def from_config_json(cls, config_josn):
        config_dict = json.load(open(config_josn, 'r'))
        return cls(**config_dict)


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
        search nearest neighbors for queries
        :param query_embeddings: queries' embeddings
        :param topk: the number of neighbors
        :param nprobe: the number of ivf post list to search for each query
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
        search topk docs for query embeddings and filter the positive items in ground truths; the results can be viewed as queries' hard negatives
        :return: query2hardneg: hard negatives for each query
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
        search topk docs for query embeddings, then use the top-1 doc as the positive item to form query2pos, and use the rest to form the query2neg
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
        """
        calculate MRR and Recall
        """
        MRR = [0.0] * len(MRR_cutoffs)
        Recall = [0.0] * len(Recall_cutoffs)
        ranking = []
        if qids is None:
            qids = list(range(len(retrieve_results)))
        for qid, candidate_pid in zip(qids, retrieve_results):
            if qid in ground_truths:
                target_pid = ground_truths[qid]
                ranking.append(-1)

                for i in range(0, max(MRR_cutoffs)):
                    if candidate_pid[i] in target_pid:
                        ranking.pop()
                        ranking.append(i + 1)
                        for inx, cutoff in enumerate(MRR_cutoffs):
                            if i <= cutoff - 1:
                                MRR[inx] += 1 / (i + 1)
                        break

                for i, k in enumerate(Recall_cutoffs):
                    Recall[i] += (len(set.intersection(set(target_pid), set(candidate_pid[:k]))) / len(set(target_pid)))

        if len(ranking) == 0:
            raise IOError("No matching QIDs found. Are you sure you are scoring the evaluation set?")

        print(f"{len(ranking)} matching queries found")
        MRR = [x / len(ranking) for x in MRR]
        for i, k in enumerate(MRR_cutoffs):
            print(f'MRR@{k}:{MRR[i]}')

        Recall = [x / len(ranking) for x in Recall]
        for i, k in enumerate(Recall_cutoffs):
            print(f'Recall@{k}:{Recall[i]}')

        return MRR, Recall
