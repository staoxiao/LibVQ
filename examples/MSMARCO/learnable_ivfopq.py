import sys

sys.path.append('../')

import os
import torch
import numpy as np
import faiss
import pickle
from transformers import HfArgumentParser

from LibVQ.baseindex import FaissIndex
from LibVQ.inference import inference
from LibVQ.models import Encoder, EncoderConfig
from LibVQ.dataset.dataset import load_rel, write_rel
from LibVQ.learnable_index import LearnableIndex
from LibVQ.dataset.preprocess import preprocess_data
from LibVQ.utils import setuplogging

from arguments import IndexArguments, DataArguments, ModelArguments, TrainingArguments

faiss.omp_set_num_threads(32)


if __name__ == '__main__':
    setuplogging()
    parser = HfArgumentParser((IndexArguments, DataArguments, ModelArguments, TrainingArguments))
    index_args, data_args, model_args, training_args = parser.parse_args_into_dataclasses()

    # Load encoder
    config = EncoderConfig.from_pretrained(model_args.pretrained_model_name)
    config.pretrained_model_name = model_args.pretrained_model_name
    config.use_two_encoder = model_args.use_two_encoder
    config.sentence_pooling_method = model_args.sentence_pooling_method
    text_encoder = Encoder(config)

    emb_size = text_encoder.output_embedding_size

    # Load embeddings of queries and docs
    doc_embeddings = np.memmap(os.path.join(data_args.output_dir, 'docs.memmap'),
                               dtype=np.float32, mode="r")
    doc_embeddings = doc_embeddings.reshape(-1, emb_size)

    query_embeddings = np.memmap(os.path.join(data_args.output_dir, 'dev-queries.memmap'),
                                 dtype=np.float32, mode="r")
    query_embeddings = query_embeddings.reshape(-1, emb_size)
    train_query_embeddings = np.memmap(os.path.join(data_args.output_dir, 'train-queries.memmap'),
                                       dtype=np.float32, mode="r")
    train_query_embeddings = train_query_embeddings.reshape(-1, emb_size)

    # Load Faiss IVFOPQ index
    index = FaissIndex(index_method=index_args.index_method,
                       emb_size=len(doc_embeddings[0]),
                       ivf_centers=index_args.ivf_centers_num,
                       subvector_num=index_args.subvector_num,
                       subvector_bits=index_args.subvector_bits,
                       dist_mode=index_args.dist_mode)
    index.load_index(os.path.join(data_args.output_dir, f'{index_args.index_method}.index'))

    index.set_nprobe(index_args.ivf_centers_num)

    # generate hard negative if need
    # train_ground_truths = load_rel(os.path.join(data_args.preprocess_dir, 'train-rels.tsv'))
    # trainquery2hardneg = index.hard_negative(train_query_embeddings, train_ground_truths, topk=400, batch_size=64)
    # pickle.dump(trainquery2hardneg, open(os.path.join(data_args.output_dir, f"train-queries_hardneg.pickle"), 'wb'))

    # q2n = pickle.load(open(os.path.join(data_args.output_dir, f"train-queries_hardneg.json"), 'rb'))
    # for k, v in q2n.items():
    #     print(k, type(k))
    #     print(v, type(v[0]))
    #     break


    # contrastive learning
    if training_args.loss_method == 'contras':
        data_args.save_ckpt_dir = './saved_ckpts/contras/'
        learnable_index = LearnableIndex(encoder=text_encoder,
                                         index_file=os.path.join(data_args.output_dir, f'{index_args.index_method}.index'))

        learnable_index.fit_with_multi_gpus(rel_file=os.path.join(data_args.preprocess_dir, 'train-rels.tsv'),
                                            neg_file=os.path.join(data_args.output_dir, f"train-queries_hardneg.pickle"),
                                            query_data_dir=data_args.preprocess_dir,
                                            max_query_length=data_args.max_query_length,
                                            doc_embeddings_file=data_args.doc_embeddings_file,
                                            emb_size = emb_size,
                                            checkpoint_path=data_args.save_ckpt_dir,
                                            logging_steps=training_args.logging_steps,
                                            per_device_train_batch_size=training_args.per_device_train_batch_size,
                                            loss_weight={'encoder_weight': 1.0, 'pq_weight': 1.0, 'ivf_weight': 'scaled_to_pqloss'},
                                            lr_params = {'encoder_lr': 1e-5, 'pq_lr':1e-4, 'ivf_lr':1e-3},
                                            loss_method = 'contras',
                                            fix_emb = 'doc',
                                            epochs=5)



    # distill learning
    if training_args.loss_method == 'distill':
        data_args.save_ckpt_dir = './saved_ckpts/distill/'
        learnable_index = LearnableIndex(encoder=text_encoder,
                                         index_file=os.path.join(data_args.output_dir, f'{index_args.index_method}.index'))

        learnable_index.fit_with_multi_gpus(rel_file=os.path.join(data_args.preprocess_dir, 'train-rels.tsv'),
                                            neg_file=os.path.join(data_args.output_dir, f"train-queries_hardneg.pickle"),
                                            query_data_dir=data_args.preprocess_dir,
                                            max_query_length=data_args.max_query_length,
                                            query_embeddings_file=data_args.query_embeddings_file,
                                            doc_embeddings_file=data_args.doc_embeddings_file,
                                            emb_size = emb_size,
                                            checkpoint_path=data_args.save_ckpt_dir,
                                            logging_steps=training_args.logging_steps,
                                            per_device_train_batch_size=training_args.per_device_train_batch_size,
                                            loss_weight={'encoder_weight': 1.0, 'pq_weight': 1.0, 'ivf_weight': 'scaled_to_pqloss'},
                                            lr_params={'encoder_lr': 1e-5, 'pq_lr': 1e-4, 'ivf_lr': 1e-3},
                                            loss_method='distill',
                                            fix_emb='doc',
                                            epochs=12)


    # distill with no label data
    '''
     # generate train data from retrieve results if need
    # query2pos, query2neg = trainquery2hardneg = index.virtual_data(train_query_embeddings, topk=400, batch_size=64)
    # write_rel(os.path.join(data_args.output_dir, 'train-virtual_rel.tsv'), query2pos)
    # pickle.dump(query2neg, open(os.path.join(data_args.output_dir, f"train-queries-virtual_hardneg.pickle"), 'wb'))

    learnable_index = LearnableIndex(encoder=text_encoder,
                                     index_file=os.path.join(data_args.output_dir, f'{index_args.index_method}.index'))

    learnable_index.fit_with_multi_gpus(rel_file=os.path.join(data_args.output_dir, 'train-virtual_rel.tsv'),
                                        neg_file=os.path.join(data_args.output_dir, f"train-queries-virtual_hardneg.pickle"),
                                        query_data_dir=data_args.preprocess_dir,
                                        max_query_length=data_args.max_query_length,
                                        query_embeddings_file=data_args.query_embeddings_file,
                                        doc_embeddings_file=data_args.doc_embeddings_file,
                                        emb_size = emb_size,
                                        per_query_neg_num=100,
                                        checkpoint_path=data_args.save_ckpt_dir,
                                        logging_steps=training_args.logging_steps,
                                        per_device_train_batch_size=training_args.per_device_train_batch_size,
                                        loss_weight={'encoder_weight': 1.0, 'pq_weight': 1.0, 'ivf_weight': 0.012},
                                        lr_params={'encoder_lr': 1e-5, 'pq_lr': 1e-4, 'ivf_lr': 1e-3},
                                        loss_method='distill',
                                        fix_emb='doc',
                                        epochs=2)
    '''



    # distill based on fixed embeddigns of queries and docs
    '''
    query2pos, query2neg = trainquery2hardneg = index.virtual_data(train_query_embeddings, topk=400, batch_size=64)
    write_rel(os.path.join(data_args.output_dir, 'train-virtual_rel.tsv'), query2pos)
    pickle.dump(query2neg, open(os.path.join(data_args.output_dir, f"train-queries-virtual_hardneg.pickle"), 'wb'))

    learnable_index = LearnableIndex(encoder=text_encoder,
                                     index_file=os.path.join(data_args.output_dir, f'{index_args.index_method}.index'))

    learnable_index.fit_with_multi_gpus(rel_file=os.path.join(data_args.output_dir, 'train-virtual_rel.tsv'),
                                        neg_file=os.path.join(data_args.output_dir, f"train-queries-virtual_hardneg.pickle"),
                                        query_embeddings_file=data_args.query_embeddings_file,
                                        doc_embeddings_file=data_args.doc_embeddings_file,
                                        emb_size = emb_size,
                                        per_query_neg_num=100,
                                        checkpoint_path=data_args.save_ckpt_dir,
                                        logging_steps=training_args.logging_steps,
                                        per_device_train_batch_size=training_args.per_device_train_batch_size,
                                        loss_weight={'encoder_weight': 1.0, 'pq_weight': 1.0, 'ivf_weight': 0.012},
                                        lr_params={'encoder_lr': 1e-5, 'pq_lr': 1e-4, 'ivf_lr': 1e-3},
                                        loss_method='distill',
                                        fix_emb='query, doc',
                                        epochs=12)
    '''




    # Creat LearnableIndex
    # learnable_index = LearnableIndex(encoder=text_encoder,
    #                                  index_file=os.path.join(data_args.output_dir, f'{index_args.index_method}.index'))
    #
    # # Train LearnableIndex
    # learnable_index.fit_with_multi_gpus(data_dir=data_args.preprocess_dir,
    #                                     max_query_length=data_args.max_query_length,
    #                                     max_doc_length=data_args.max_doc_length,
    #                                     query_embeddings_file=data_args.query_embeddings_file,
    #                                     doc_embeddings_file=data_args.doc_embeddings_file,
    #                                     checkpoint_path=data_args.save_ckpt_dir,
    #                                     per_device_train_batch_size=training_args.per_device_train_batch_size,
    #                                     epochs=training_args.num_train_epochs,
    #                                     loss_weight={'encoder_weight': 1.0, 'pq_weight': 1.0, 'ivf_weight': 0.012})

    # select a ckpt
    # or set saved_ckpts_path in function "update_encoder“ and "update_index_with_ckpt",  which will select the latest ckpt to update
    ckpt_path = learnable_index.get_latest_ckpt(data_args.save_ckpt_dir)

    # update query embeddings when re-training the query encoder
    data_args.output_dir = f'./data/passage/evaluate/LearnableIndex_{training_args.loss_method}_{training_args.fix_emb}/'
    learnable_index.update_encoder(encoder_file=f'{ckpt_path}/encoder.bin')
    learnable_index.encode(data_dir=data_args.preprocess_dir,
                           prefix='dev-queries',
                           max_length=data_args.max_query_length,
                           output_dir=data_args.output_dir,
                           batch_size=8196,
                           is_query=True)
    query_embeddings = np.memmap(f'{data_args.output_dir}/dev-queries.memmap', dtype=np.float32,
                                 mode="r")
    query_embeddings = query_embeddings.reshape(-1, 768)


    # update index
    print('Updating the index with new ivf and pq')
    learnable_index.update_index_with_ckpt(ckpt_path=ckpt_path, doc_embeddings=doc_embeddings)

    # Test
    ground_truths = load_rel(os.path.join(data_args.preprocess_dir, 'dev-rels.tsv'))
    learnable_index.set_nprobe(index_args.nprobe)
    qids = list(range(len(query_embeddings)))
    learnable_index.test(query_embeddings, qids, ground_truths, topk=1000, batch_size=64,
                         MRR_cutoffs=[10, 100], Recall_cutoffs=[10, 30, 50, 100])

