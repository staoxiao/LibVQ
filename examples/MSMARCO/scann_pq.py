import os
import numpy as np
from transformers import HfArgumentParser

from LibVQ.baseindex.ScannIndex import ScaNNIndex
from LibVQ.dataset.dataset import load_rel, write_rel
from LibVQ.dataset.preprocess import preprocess_data

from arguments import IndexArguments, DataArguments, ModelArguments, TrainingArguments


if __name__ == '__main__':
    parser = HfArgumentParser((IndexArguments, DataArguments, ModelArguments, TrainingArguments))
    index_args, data_args, model_args, training_args = parser.parse_args_into_dataclasses()

    os.makedirs(data_args.output_dir, exist_ok=True)


    # Load embeddings
    emb_size = 768
    doc_embeddings = np.memmap(os.path.join(data_args.output_dir, 'docs.memmap'),
                               dtype=np.float32, mode="r")
    doc_embeddings = doc_embeddings.reshape(-1, emb_size)
    query_embeddings = np.memmap(os.path.join(data_args.output_dir, 'dev-queries.memmap'),
                                 dtype=np.float32, mode="r")
    query_embeddings = query_embeddings.reshape(-1, emb_size)


    # Creat ScaNN index
    index = ScaNNIndex(doc_embeddings,
                       ivf_centers=index_args.ivf_centers_num,
                       subvector_num=index_args.subvector_num,
                       hash_type='lut256',
                       anisotropic_quantization_threshold=0.2)

    # Test the performance
    ground_truths = load_rel(os.path.join(data_args.preprocess_dir, 'dev-rels.tsv'))
    qids = list(range(len(query_embeddings)))
    index.test(query_embeddings, qids, ground_truths, topk=1000, nprobe=index_args.nprobe,
               MRR_cutoffs=[10, 100], Recall_cutoffs=[10, 30, 50, 100])

    index.test(query_embeddings, qids, ground_truths, topk=1000, nprobe=1,
               MRR_cutoffs=[10, 100], Recall_cutoffs=[10, 30, 50, 100])
    index.test(query_embeddings, qids, ground_truths, topk=1000, nprobe=100,
               MRR_cutoffs=[10, 100], Recall_cutoffs=[10, 30, 50, 100])
