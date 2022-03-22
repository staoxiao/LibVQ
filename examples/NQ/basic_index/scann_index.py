import sys
sys.path.append('./')
import os

import numpy as np
from arguments import IndexArguments, DataArguments, ModelArguments, TrainingArguments
from transformers import HfArgumentParser
from evaluate import validate, load_test_data

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
    query_embeddings = np.memmap(os.path.join(data_args.embeddings_dir, 'test-queries.memmap'),
                                 dtype=np.float32, mode="r")
    query_embeddings = query_embeddings.reshape(-1, emb_size)

    # Creat ScaNN index
    index = ScaNNIndex(doc_embeddings,
                       ivf_centers_num=index_args.ivf_centers_num,
                       subvector_num=index_args.subvector_num,
                       hash_type='lut256',
                       anisotropic_quantization_threshold=0.2)

    # Test the performance
    scores, ann_items = index.search(query_embeddings, topk=100, nprobe=index_args.nprobe)
    test_questions, test_answers, collections = load_test_data(
        query_andwer_file='./data/NQ/raw_dataset/nq-test.qa.csv',
        collections_file='./data/NQ/dataset/collection.tsv')
    validate(ann_items, test_questions, test_answers, collections)