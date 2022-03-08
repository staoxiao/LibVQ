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
        assert max(max(MRR_cutoffs), max(Recall_cutoffs)) <= topk
        scores, retrieve_results = self.search(query_embeddings, topk, batch_size)
        return self.evaluate(retrieve_results, ground_truths, MRR_cutoffs, Recall_cutoffs, qids)

    def search(self, query_embeddings, topk, batch_size):
        raise NotImplementedError

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

        print(len(ranking))
        MRR = [x / len(ranking) for x in MRR]
        for i, k in enumerate(MRR_cutoffs):
            print(f'MRR@{k}:{MRR[i]}')

        Recall = [x / len(ranking) for x in Recall]
        for i, k in enumerate(Recall_cutoffs):
            print(f'Recall@{k}:{Recall[i]}')

        return MRR, Recall