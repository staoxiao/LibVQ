from LibVQ.dataset import Datasets
from LibVQ.base_index import IndexConfig
from LibVQ.models import EncoderConfig
from LibVQ.learnable_index import AutoIndex
import numpy as np
from LibVQ.dataset.dataset import load_rel

# get index and train
data = Datasets('text2text', 'MSMARCO')
index_config = IndexConfig(index_method='ivf_pq', ivf_centers_num=5000)
encoder_config = EncoderConfig(is_finetune=True, doc_encoder_name_or_path='Shitao/msmarco_doc_encoder', query_encoder_name_or_path='Shitao/msmarco_query_encoder')

index = AutoIndex.get_index(index_config, encoder_config, data)
index.train(data)

# MRR and RECALL
if data.dev_queries_embedding_dir is not None:
    dev_query = np.memmap(data.dev_queries_embedding_dir, dtype=np.float32, mode="r")
    dev_query = dev_query.reshape(-1, index_config.emb_size)
    ground_truths = load_rel(data.dev_rels_path)
    index.test(dev_query, ground_truths, topk=1000, batch_size=64,
               MRR_cutoffs=[5,10,20], Recall_cutoffs=[5,10,20], nprobe=index_config.nprobe)
# query search
indexDocDict = dict()
docsFile = open(data.docs_path, 'r', encoding='UTF-8')
count = 0
for line in docsFile:
    indexDocDict[count] = line.replace('\n', '').replace(line.split('\t')[0] + '\t', '')
    count += 1
docsFile.close()
answer, answer_id = index.search_query(['what is paranoid sc', 'what is mean streaming'], indexDocDict)
print(answer)
print(answer_id)