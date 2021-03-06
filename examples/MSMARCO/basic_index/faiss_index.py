import sys
sys.path.append('./')
import os

import faiss
import numpy as np
from arguments import IndexArguments, DataArguments, ModelArguments, TrainingArguments
from transformers import HfArgumentParser

from LibVQ.base_index import FaissIndex
from LibVQ.dataset.dataset import load_rel



if __name__ == '__main__':
    parser = HfArgumentParser((IndexArguments, DataArguments, ModelArguments, TrainingArguments))
    index_args, data_args, model_args, training_args = parser.parse_args_into_dataclasses()

    # Load embeddings
    emb_size = 768
    doc_embeddings = np.memmap(os.path.join(data_args.embeddings_dir, 'docs.memmap'),
                               dtype=np.float32, mode="r")
    doc_embeddings = doc_embeddings.reshape(-1, emb_size)
    query_embeddings = np.memmap(os.path.join(data_args.embeddings_dir, 'dev-queries.memmap'),
                                 dtype=np.float32, mode="r")
    query_embeddings = query_embeddings.reshape(-1, emb_size)

    # Creat Faiss index
    faiss.omp_set_num_threads(32)
    index = FaissIndex(index_method=index_args.index_method,
                       emb_size=len(doc_embeddings[0]),
                       ivf_centers_num=index_args.ivf_centers_num,
                       subvector_num=index_args.subvector_num,
                       subvector_bits=index_args.subvector_bits,
                       dist_mode=index_args.dist_mode)

    print('Training the index with doc embeddings')
    index.fit(doc_embeddings)
    index.add(doc_embeddings)

    index_file = os.path.join(data_args.embeddings_dir, f'{index_args.index_method}_ivf{index_args.ivf_centers_num}_pq{index_args.subvector_num}x{index_args.subvector_bits}.index')
    index.save_index(index_file)
    # index.load_index(index_file)

    # Test the performance
    ground_truths = load_rel(os.path.join(data_args.preprocess_dir, 'dev-rels.tsv'))
    index.test(query_embeddings, ground_truths, topk=1000, batch_size=64,
               MRR_cutoffs=[10, 100], Recall_cutoffs=[10, 30, 50, 100], nprobe=index_args.nprobe)
