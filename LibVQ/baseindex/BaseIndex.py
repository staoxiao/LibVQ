import json

class IndexConfig():
    def __init__(self,
                 index_method: str ='ivfopq',
                 emb_size: int = 768,
                 ivf_centers: int = 10000,
                 subvector_num: int = 32,
                 subvector_bits: int = 8,
                 nprobe: int = 100,
                 dist_mode: str = 'ip',
                 **kwargs):
        self.index_method = index_method
        self.emb_size = emb_size
        self.ivf_centers = ivf_centers
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

    def search(self, query_embeddings, topk, batch_size):
        raise NotImplementedError

    def hard_negative(self, query_embeddings, ground_truths=None, topk=400, batch_size=None):
        score, search_results = self.search(query_embeddings, topk=topk, batch_size=batch_size)
        query2hardneg = {}
        for qid, neighbors in enumerate(search_results):
            neg = list(filter(lambda x: x not in ground_truths[qid], neighbors))
            query2hardneg[qid] = neg
        return query2hardneg

    def virtual_data(self, query_embeddings, topk=400, batch_size=None):
        score, search_results = self.search(query_embeddings, topk=topk, batch_size=batch_size)
        query2pos = {}
        query2neg = {}
        for qid, neighbors in enumerate(search_results):
            query2pos[qid] = neighbors[:1]
            query2neg[qid] = neighbors[1:]
        return query2pos, query2neg

    def evaluate(self, retrieve_results, ground_truths, MRR_cutoffs, Recall_cutoffs, qids):
        MRR = [0.0] * len(MRR_cutoffs)
        Recall = [0.0] * len(Recall_cutoffs)
        ranking = []
        for qid, candidate_pid in zip(qids, retrieve_results):
            if qid in ground_truths:
                target_pid = ground_truths[qid]
                ranking.append(-1)

                for i in range(0, max(MRR_cutoffs)):
                    if candidate_pid[i] in target_pid:
                        ranking.pop()
                        ranking.append(i+1)
                        for inx, cutoff in enumerate(MRR_cutoffs):
                            if i <= cutoff-1:
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