import sys

sys.path.append('../')

import os
import numpy as np
import faiss
from transformers import HfArgumentParser

from LibVQ.baseindex import ScaNNIndex
from LibVQ.inference import inference
from LibVQ.models import Encoder, EncoderConfig
from LibVQ.dataset.dataset import load_rel, write_rel
from LibVQ.dataset.preprocess import preprocess_data

from arguments import IndexArguments, DataArguments, ModelArguments, TrainingArguments

faiss.omp_set_num_threads(32)



def inference_embeddings(text_encoder, data_args):
    inference(data_dir=data_args.preprocess_dir,
              is_query=False,
              encoder=text_encoder,
              prefix=f'docs',
              max_length=data_args.max_doc_length,
              output_dir=data_args.output_dir,
              batch_size=10240)
    inference(data_dir=data_args.preprocess_dir,
              is_query=True,
              encoder=text_encoder,
              prefix=f'dev-queries',
              max_length=data_args.max_query_length,
              output_dir=data_args.output_dir,
              batch_size=10240)
    inference(data_dir=data_args.preprocess_dir,
              is_query=True,
              encoder=text_encoder,
              prefix=f'train-queries',
              max_length=data_args.max_query_length,
              output_dir=data_args.output_dir,
              batch_size=10240)



if __name__ == '__main__':
    parser = HfArgumentParser((IndexArguments, DataArguments, ModelArguments, TrainingArguments))
    index_args, data_args, model_args, training_args = parser.parse_args_into_dataclasses()

    os.makedirs(data_args.output_dir, exist_ok=True)

    # preprocess
    if not os.path.exists(data_args.preprocess_dir):
        preprocess_data(data_dir=data_args.data_dir,
                        output_dir=data_args.preprocess_dir,
                        tokenizer_name=model_args.pretrained_model_name,
                        max_doc_length=data_args.max_doc_length,
                        max_query_length=data_args.max_query_length,
                        workers_num=64)

    # Load encoder
    config = EncoderConfig.from_pretrained(model_args.pretrained_model_name)
    config.pretrained_model_name = model_args.pretrained_model_name
    config.use_two_encoder = model_args.use_two_encoder
    config.sentence_pooling_method = model_args.sentence_pooling_method
    text_encoder = Encoder(config)

    emb_size = text_encoder.output_embedding_size

    # Generate embeddings of queries and docs
    inference_embeddings(text_encoder, data_args)

    doc_embeddings = np.memmap(os.path.join(data_args.output_dir, 'docs.memmap'),
                               dtype=np.float32, mode="r")
    doc_embeddings = doc_embeddings.reshape(-1, emb_size)
    query_embeddings = np.memmap(os.path.join(data_args.output_dir, 'dev-queries.memmap'),
                                 dtype=np.float32, mode="r")
    query_embeddings = query_embeddings.reshape(-1, emb_size)

    # Creat Faiss IVFOPQ index
    index = ScaNNIndex(doc_embeddings,
                       ivf_centers=index_args.ivf_centers_num,
                       subvector_num=index_args.subvector_num,
                       hash_type='lut256',
                       anisotropic_quantization_threshold=0.2,
                       threads_num=32)

    # Test the performance
    ground_truths = load_rel(os.path.join(data_args.preprocess_dir, 'dev-rels.tsv'))
    qids = list(range(len(query_embeddings)))
    index.test(query_embeddings, qids, ground_truths, topk=1000, nprobe=index_args.nprobe,
               MRR_cutoffs=[10, 100], Recall_cutoffs=[10, 30, 50, 100])



