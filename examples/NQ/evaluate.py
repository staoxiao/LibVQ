

def get_recall(args, output_rank_file, test_questions, test_answers, passage_text):
    similar_scores, nearest_neighbors = [],[]
    with open(output_rank_file, 'r') as outputfile:
        for line in outputfile:
            qid, pid, rank, score = line.strip('\n').split('\t')
            qid, pid, rank, score = int(qid), int(pid), int(rank), float(score)
            if rank == 1:
                similar_scores.append([score])
                nearest_neighbors.append([pid])
            else:
                similar_scores[-1].append(score)
                nearest_neighbors[-1].append(pid)
    similar_scores = np.array(similar_scores)[:, :100]
    nearest_neighbors = np.array(nearest_neighbors)[:, :100]
    print(similar_scores.shape, nearest_neighbors.shape)

    logger.info("***** Begin test validate *****")
    top_k_hits, scores, result_dict, result_dict_list = \
        validate(test_questions, passage_text, test_answers, nearest_neighbors,
                 similar_scores)
    logger.info("***** Done test validate *****")

    logger.info("***** Done test validate *****")
    # ndcg_output_path = os.path.join(args.output_dir, "test_eval_result.json")
    # with open(ndcg_output_path, 'w') as f:
    #     json.dump({'top1': top_k_hits[0], 'top5': top_k_hits[4],
    #                'top20': top_k_hits[19], 'top100': top_k_hits[99], 'result_dict': result_dict}, f, indent=2)
    print(f"top5:{top_k_hits[4]}, top10:{top_k_hits[9]}, top20:{top_k_hits[19]}, top50:{top_k_hits[49]}, top100:{top_k_hits[99]}")
    print(result_dict['MRR_n@_10'], result_dict['MRR_n@_100'])
    # output_path = os.path.join(args.output_dir, 'test_result_dict_list.json')
    # with open(output_path, 'w') as f:
    #     json.dump(result_dict_list, f, indent=2)



def validate(questions, passages, answers, closest_docs, similar_scores):
    v_dataset = V_dataset(questions, passages, answers, closest_docs, similar_scores)
    v_dataloader = DataLoader(v_dataset,128,shuffle=False,num_workers=20, collate_fn=V_dataset.collect_fn())
    final_scores = []
    final_result_list = []
    final_result_dict_list = []
    for k, (scores, result_list, result_dict_list) in enumerate(tqdm(v_dataloader, total=len(v_dataloader))):
        final_scores.extend(scores)  # 等着func的计算结果
        final_result_list.extend(result_list)
        final_result_dict_list.extend(result_dict_list)
    logger.info('Per question validation results len=%d', len(final_scores))
    n_docs = len(closest_docs[0])
    top_k_hits = [0] * n_docs
    for question_hits in final_scores:
        best_hit = next((i for i, x in enumerate(question_hits) if x), None)
        if best_hit is not None:
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]

    logger.info('Validation results: top k documents hits %s', top_k_hits)
    top_k_hits = [v / len(closest_docs) for v in top_k_hits]
    logger.info('Validation results: top k documents hits accuracy %s', top_k_hits)

    return top_k_hits, final_scores, Eval_Tool.get_matrics(final_scores), final_result_dict_list


class V_dataset(Dataset):
    def __init__(self, questions, passages, answers, closest_docs,
                 similar_scores):
        self.questions = questions
        self.passages = passages
        self.answers = answers
        self.closest_docs = closest_docs
        self.similar_scores = similar_scores
        tok_opts = {}
        self.tokenizer = SimpleTokenizer(**tok_opts)
    def __getitem__(self, query_id):
        doc_ids = [pidx for pidx in self.closest_docs[query_id]]
        hits = []
        temp_result_dict = {}
        temp_result_dict['id'] = str(query_id)
        temp_result_dict['question'] = self.questions[query_id]
        temp_result_dict['answers'] = self.answers[query_id]
        temp_result_dict['ctxs'] = []
        for i, doc_id in enumerate(doc_ids):
            if doc_id == -1:
                hits.append(False)
                text, title = '', ''
            else:
                text, title = self.passages[doc_id]
                hits.append(has_answer(self.answers[query_id], text, self.tokenizer))
            temp_result_dict['ctxs'].append({'d_id': str(doc_id),
                                             'text': text,
                                             'title': title,
                                             'score': str(self.similar_scores[query_id, i]),
                                             'hit': str(hits[-1])})
        return hits, [query_id, doc_ids], temp_result_dict

    def __len__(self):
        return self.closest_docs.shape[0]

    @classmethod
    def collect_fn(cls,):
        def create_biencoder_input2(features):
            scores = [feature[0] for feature in features]
            result_list = [feature[1] for feature in features]
            result_dict_list = [feature[2] for feature in features]
            return scores,result_list,result_dict_list
        return create_biencoder_input2


class Eval_Tool:
    @classmethod
    def MRR_n(cls, results_list, n):
        mrr_100_list = []
        for hits in results_list:
            score = 0
            for rank, item in enumerate(hits[:n]):
                if item:
                    score = 1.0 / (rank + 1.0)
                    break
            mrr_100_list.append(score)
        return sum(mrr_100_list) / len(mrr_100_list)

    @classmethod
    def MAP_n(cls, results_list, n):
        MAP_n_list = []
        for predict in results_list:
            ap = 0
            hit_num = 1
            for rank, item in enumerate(predict[:n]):
                if item:
                    ap += hit_num / (rank + 1.0)
                    hit_num += 1
            ap /= n
            MAP_n_list.append(ap)
        return sum(MAP_n_list) / len(MAP_n_list)

    @classmethod
    def DCG_n(cls, results_list, n):
        DCG_n_list = []
        for predict in results_list:
            DCG = 0
            for rank, item in enumerate(predict[:n]):
                if item:
                    DCG += 1 / math.log2(rank + 2)
            DCG_n_list.append(DCG)
        return sum(DCG_n_list) / len(DCG_n_list)

    @classmethod
    def nDCG_n(cls, results_list, n):
        nDCG_n_list = []
        for predict in results_list:
            nDCG = 0
            for rank, item in enumerate(predict[:n]):
                if item:
                    nDCG += 1 / math.log2(rank + 2)
            nDCG /= sum([math.log2(i + 2) for i in range(n)])
            nDCG_n_list.append(nDCG)
        return sum(nDCG_n_list) / len(nDCG_n_list)

    @classmethod
    def P_n(cls, results_list, n):
        p_n_list = []
        for predict in results_list:
            true_num = 0
            for rank, item in enumerate(predict[:n]):
                if item:
                    true_num += 1
            p = true_num / n
            p_n_list.append(p)
        return sum(p_n_list) / len(p_n_list)

    @classmethod
    def get_matrics(cls, results_list):
        p_list = [1, 5, 10, 20, 50, 100]
        metrics = {'MRR_n': cls.MRR_n,
                   'MAP_n': cls.MAP_n,
                   'DCG_n': cls.DCG_n, 'nDCG_n': cls.nDCG_n, 'P_n': cls.P_n}
        result_dict = {}
        for metric_name, fuction in metrics.items():
            for p in p_list:
                temp_result = fuction(results_list, p)
                result_dict[metric_name + '@_' + str(p)] = temp_result
        return result_dict