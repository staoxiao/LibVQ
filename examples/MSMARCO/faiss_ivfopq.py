import sys

sys.path.append('../')

import os
import numpy as np
import faiss
import torch
from transformers import HfArgumentParser

from LibVQ.index import FaissIndex
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
    index = FaissIndex(index_method=index_args.index_method,
                       emb_size=len(doc_embeddings[0]),
                       ivf_centers=index_args.ivf_centers_num,
                       subvector_num=index_args.subvector_num,
                       subvector_bits=index_args.subvector_bits,
                       dist_mode=index_args.dist_mode)

    print('Training the index with doc embeddings')
    # if faiss.get_num_gpus() > 0:
    #     index.CPU_to_GPU(0)
    index.fit(doc_embeddings)
    index.add(doc_embeddings)
    # if faiss.get_num_gpus() > 0:
    #     index.GPU_to_CPU()
    index.save_index(os.path.join(data_args.output_dir, f'{index_args.index_method}.index'))
    index.load_index(os.path.join(data_args.output_dir, f'{index_args.index_method}.index'))

    # Test the performance
    index.set_nprobe(index_args.nprobe)
    ground_truths = load_rel(os.path.join(data_args.preprocess_dir, 'dev-rels.tsv'))
    qids = list(range(len(query_embeddings)))
    index.test(query_embeddings, qids, ground_truths, topk=1000, batch_size=64,
               MRR_cutoffs=[10, 100], Recall_cutoffs=[10, 30, 50, 100])




    # train_query_embeddings = np.memmap(os.path.join(data_args.output_dir, 'train-queries.memmap'),
    #                              dtype=np.float32, mode="r")
    # train_query_embeddings = train_query_embeddings.reshape(-1, emb_size)
    # train_ground_truths = load_rel(os.path.join(data_args.preprocess_dir, 'train-rels.tsv'))
    #
    # index.set_nprobe(index_args.ivf_centers_num)
    # trainquery2hardneg = index.hard_negative(train_query_embeddings, train_ground_truths, topk=400, batch_size=64)
    # pickle.dump(trainquery2hardneg, open(os.path.join(data_args.output_dir, f"train-queries_hardneg.json"), 'wb'))
    #
    # q2n = pickle.load(open(os.path.join(data_args.output_dir, f"train-queries_hardneg.json"), 'rb'))
    # for k,v in q2n.items():
    #     print(k, type(k))
    #     print(v, type(v[0]))
    #     break
    #
    # query2pos, query2neg = trainquery2hardneg = index.virtual_data(train_query_embeddings, topk=400, batch_size=64)
    # write_rel(os.path.join(data_args.output_dir, 'train-virtual_rel.tsv'), query2pos)
    # pickle.dump(query2neg, open(os.path.join(data_args.output_dir, f"train-queries-virtual_hardneg.json"), 'wb'))

    # Creat LearnableIndex
    # learnable_index = LearnableIndex(encoder=text_encoder,
    #                                  index_file=os.path.join(data_args.output_dir, f'{index_args.index_method}.index'))
    #
    # # Train LearnableIndex
    # learnable_index.fit_with_multi_gpus(data_dir=data_args.preprocess_dir,
    #                                     max_query_length=data_args.max_query_length,
    #                                     max_doc_length=data_args.max_doc_length,
    #                                     query_embeddings_file = data_args.query_embeddings_file,
    #                                     doc_embeddings_file = data_args.doc_embeddings_file,
    #                                     checkpoint_path = data_args.save_ckpt_dir,
    #                                     per_device_train_batch_size = training_args.per_device_train_batch_size,
    #                                     epochs = training_args.num_train_epochs,
    #                                     loss_weight = {'encoder_weight': 1.0, 'pq_weight': 1.0, 'ivf_weight': 0.012})
    #
    # # Update LearnableIndex and query embeddings
    # ckpt_path = os.path.join(data_args.save_ckpt_dir, 'epoch_9_step_2460')
    # data_args.output_dir = f'./data/passage/evaluate/test_model_{ckpt_path}'
    #
    # learnable_index.update_encoder(f'{ckpt_path}/encoder.bin')
    # learnable_index.encode(data_dir=data_args.preprocess_dir,
    #                        prefix='dev-queries',
    #                        max_length=data_args.max_query_length,
    #                        output_dir=data_args.output_dir,
    #                        batch_size=8196,
    #                        is_query=True)
    # query_embeddings = np.memmap(f'{data_args.output_dir}/dev-queries.memmap', dtype=np.float32,
    #                              mode="r")
    # query_embeddings = query_embeddings.reshape(-1, 768)
    #
    # print('Updating the index with new ivf and pq')
    # learnable_index.update_index_with_ckpt(ckpt_path=ckpt_path, doc_embeddings=doc_embeddings)
    #
    # # Test
    # learnable_index.set_nprobe(index_args.nprobe)
    # qids = list(range(len(query_embeddings)))
    # learnable_index.test(query_embeddings, qids, ground_truths, topk=1000, batch_size=64,
    #                      MRR_cutoffs=[10, 100], Recall_cutoffs=[10, 30, 50, 100])
    #
