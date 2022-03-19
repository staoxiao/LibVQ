import sys
sys.path.append('./')
import os

import faiss
import numpy as np
from evaluate import validate, load_test_data
from transformers import HfArgumentParser

from LibVQ.baseindex import FaissIndex

from arguments import IndexArguments, DataArguments, ModelArguments, TrainingArguments

faiss.omp_set_num_threads(32)

if __name__ == '__main__':
    parser = HfArgumentParser((IndexArguments, DataArguments, ModelArguments, TrainingArguments))
    index_args, data_args, model_args, training_args = parser.parse_args_into_dataclasses()

    os.makedirs(data_args.output_dir, exist_ok=True)

    # Load embeddings
    emb_size = 768
    doc_embeddings = np.memmap(os.path.join(data_args.output_dir, 'docs.memmap'),
                               dtype=np.float32, mode="r")
    doc_embeddings = doc_embeddings.reshape(-1, emb_size)
    query_embeddings = np.memmap(os.path.join(data_args.output_dir, 'test-queries.memmap'),
                                 dtype=np.float32, mode="r")
    query_embeddings = query_embeddings.reshape(-1, emb_size)

    # Creat Faiss index
    index = FaissIndex(index_method=index_args.index_method,
                       emb_size=len(doc_embeddings[0]),
                       ivf_centers_num=index_args.ivf_centers_num,
                       subvector_num=index_args.subvector_num,
                       subvector_bits=index_args.subvector_bits,
                       dist_mode=index_args.dist_mode)

    print('Training the index with doc embeddings')

    # res = faiss.StandardGpuResources()
    # res.setTempMemory(30 * 1024 * 1024 * 1024)
    # co = faiss.GpuClonerOptions()
    # co.useFloat16 = index_args.subvector_num >= 56
    # gpu_index = faiss.index_cpu_to_gpu(res, 0, index.index, co)
    # gpu_index.train(doc_embeddings)
    # gpu_index.add(doc_embeddings)
    # print('gpu ----------------------------')

    # index.fit(doc_embeddings)
    # index.add(doc_embeddings)
    print('cpu ------------------------------')

    # index.save_index(os.path.join(data_args.output_dir, f'{index_args.index_method}.index'))
    index.load_index(os.path.join(data_args.output_dir, f'{index_args.index_method}.index'))

    # Test the performance
    scores, ann_items = index.search(query_embeddings, topk=100, nprobe=index_args.nprobe)

    test_questions, test_answers, passage_text = load_test_data(
        query_andwer_file='./data/NQ/raw_dataset/nq-test.qa.csv',
        collections_file='./data/NQ/dataset/collection.tsv')
    validate(ann_items, test_questions, test_answers, passage_text)
