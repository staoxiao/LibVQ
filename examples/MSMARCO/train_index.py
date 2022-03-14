import os
import torch
import numpy as np
import faiss
import pickle
from transformers import HfArgumentParser

from LibVQ.baseindex import FaissIndex
from LibVQ.dataset.dataset import load_rel, write_rel
from LibVQ.learnable_index import LearnableIndex
from LibVQ.utils import setuplogging

from arguments import IndexArguments, DataArguments, ModelArguments, TrainingArguments

faiss.omp_set_num_threads(32)


if __name__ == '__main__':
    setuplogging()
    parser = HfArgumentParser((IndexArguments, DataArguments, ModelArguments, TrainingArguments))
    index_args, data_args, model_args, training_args = parser.parse_args_into_dataclasses()


    # Load embeddings of queries and docs
    emb_size = 768
    doc_embeddings = np.memmap(os.path.join(data_args.output_dir, 'docs.memmap'),
                               dtype=np.float32, mode="r")
    doc_embeddings = doc_embeddings.reshape(-1, emb_size)

    query_embeddings = np.memmap(os.path.join(data_args.output_dir, 'dev-queries.memmap'),
                                 dtype=np.float32, mode="r")
    query_embeddings = query_embeddings.reshape(-1, emb_size)
    train_query_embeddings = np.memmap(os.path.join(data_args.output_dir, 'train-queries.memmap'),
                                       dtype=np.float32, mode="r")
    train_query_embeddings = train_query_embeddings.reshape(-1, emb_size)


    if training_args.training_mode == 'contras_index':
        data_args.save_ckpt_dir = f'./saved_ckpts/{training_args.training_mode}/'
        learnable_index = LearnableIndex(index_file=os.path.join(data_args.output_dir, f'{index_args.index_method}.index'))

        learnable_index.fit_with_multi_gpus(rel_file=os.path.join(data_args.preprocess_dir, 'train-rels.tsv'),
                                            neg_file=os.path.join(data_args.output_dir, f"train-queries_hardneg.pickle"),
                                            query_embeddings_file=data_args.query_embeddings_file,
                                            doc_embeddings_file=data_args.doc_embeddings_file,
                                            emb_size = emb_size,
                                            checkpoint_path=data_args.save_ckpt_dir,
                                            logging_steps=training_args.logging_steps,
                                            per_device_train_batch_size=training_args.per_device_train_batch_size,
                                            loss_weight={'encoder_weight': 1.0, 'pq_weight': 1.0, 'ivf_weight': 'scaled_to_pqloss'},
                                            lr_params={'encoder_lr': 1e-5, 'pq_lr': 1e-4, 'ivf_lr': 1e-3},
                                            loss_method='contras',
                                            fix_emb='query, doc',
                                            epochs=16)


    # distill based on fixed embeddigns of queries and docs
    if training_args.training_mode == 'distill_index':



        data_args.save_ckpt_dir = f'./saved_ckpts/{training_args.training_mode}/'
        learnable_index = LearnableIndex(index_file=os.path.join(data_args.output_dir, f'{index_args.index_method}.index'))
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
                                            epochs=3)



    # select a ckpt
    # or set saved_ckpts_path in function "update_encoder“ and "update_index_with_ckpt",  which will select the latest ckpt to update
    ckpt_path = learnable_index.get_latest_ckpt(data_args.save_ckpt_dir)
    print(ckpt_path)

    # ckpt_path = './saved_ckpts/distill_512/epoch_28_step_3567'

    # update query embeddings when re-training the query encoder
    data_args.output_dir = f'./data/passage/evaluate/LearnableIndex_{training_args.loss_method}_{training_args.fix_emb}'

    # update index
    print('Updating the index with new ivf and pq')
    learnable_index.update_index_with_ckpt(ckpt_path=ckpt_path, doc_embeddings=doc_embeddings)

    # Test
    ground_truths = load_rel(os.path.join(data_args.preprocess_dir, 'dev-rels.tsv'))
    # learnable_index.set_nprobe(index_args.nprobe)
    # qids = list(range(len(new_query_embeddings)))
    # learnable_index.test(new_query_embeddings, qids, ground_truths, topk=1000, batch_size=64,
    #                      MRR_cutoffs=[10, 100], Recall_cutoffs=[10, 30, 50, 100])

    learnable_index.set_nprobe(1)
    qids = list(range(len(new_query_embeddings)))
    learnable_index.test(new_query_embeddings, qids, ground_truths, topk=1000, batch_size=64,
                         MRR_cutoffs=[10, 100], Recall_cutoffs=[10, 30, 50, 100])

    learnable_index.set_nprobe(100)
    qids = list(range(len(new_query_embeddings)))
    learnable_index.test(new_query_embeddings, qids, ground_truths, topk=1000, batch_size=64,
                         MRR_cutoffs=[10, 100], Recall_cutoffs=[5, 10, 20, 30, 50, 100])
