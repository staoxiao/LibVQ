import sys
sys.path.append('./')
import os

import numpy as np
from arguments import IndexArguments, DataArguments, ModelArguments, TrainingArguments
from transformers import HfArgumentParser

from LibVQ.baseindex.ScannIndex import ScaNNIndex
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

    # Creat ScaNN index
    index = ScaNNIndex(doc_embeddings,
                       ivf_centers_num=index_args.ivf_centers_num,
                       subvector_num=index_args.subvector_num,
                       hash_type='lut256',
                       anisotropic_quantization_threshold=0.2)

    # Test the performance
    ground_truths = load_rel(os.path.join(data_args.preprocess_dir, 'dev-rels.tsv'))
    index.test(query_embeddings, ground_truths, topk=1000, nprobe=index_args.nprobe,
               MRR_cutoffs=[10, 100], Recall_cutoffs=[10, 30, 50, 100])

    ground_truths = load_rel(os.path.join(data_args.preprocess_dir, 'dev-rels.tsv'))
    index.test(query_embeddings, ground_truths, topk=1000, nprobe=index_args.ivf_centers_num,
               MRR_cutoffs=[10, 100], Recall_cutoffs=[10, 30, 50, 100])