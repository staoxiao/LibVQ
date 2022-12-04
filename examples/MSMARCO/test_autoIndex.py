from LibVQ.dataset import Datasets
from LibVQ.base_index import IndexConfig
from LibVQ.models import EncoderConfig
from LibVQ.learnable_index import AutoIndex
from LibVQ.dataset.dataset import load_rel
import os

# get index and train
data = Datasets('MSMARCO')
index_config = IndexConfig(index_method='ivf_pq', ivf_centers_num=5000, emb_size=640)
encoder_config = EncoderConfig(is_finetune=True, doc_encoder_name_or_path='Shitao/msmarco_doc_encoder', query_encoder_name_or_path='Shitao/msmarco_query_encoder')

index = AutoIndex.get_index(index_config, encoder_config, data)
index.train(data)

saved_index_file = os.path.join(data.embedding_dir, 'distillLearnableIndexWithEncoder.index')
index.save_index(saved_index_file)
saved_pooler_file = os.path.join(data.embedding_dir, 'pooler.pth')
index.pooler.save(saved_pooler_file)

# MRR and RECALL
if data.dev_queries_embedding_dir is not None:
    dev_query = index.get(data, 'dev-queries.memmap')
    ground_truths = load_rel(data.dev_rels_path)
    index.test(dev_query, ground_truths, topk=1000, batch_size=64,
               MRR_cutoffs=[5, 10, 20], Recall_cutoffs=[5, 10, 20], nprobe=index_config.nprobe)
# query search
answer, answer_id = index.search_query(['what is paranoid sc', 'what is mean streaming'], data)
print(answer)
print(answer_id)