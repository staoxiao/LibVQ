import os
import torch
import numpy as np
import faiss
import pickle
from transformers import HfArgumentParser, AdamW

from LibVQ.dataset.dataset import load_rel, write_rel
from LibVQ.learnable_index import LearnableIndex
from LibVQ.baseindex import FaissIndex
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

    # Create Index
    # if there is a faiss index in init_index_file, it will creat learnable_index based on it;
    # if no, it will creat and save a faiss index in init_index_file
    learnable_index = LearnableIndex(index_method=index_args.index_method,
                                     init_index_file=os.path.join(data_args.output_dir, f'{index_args.index_method}.index'),
                                     doc_embeddings=doc_embeddings,
                                     ivf_centers_num=index_args.ivf_centers_num,
                                     subvector_num=index_args.subvector_num,
                                     subvector_bits=index_args.subvector_bits)


    # The class randomly sample the negative from corpus by default. You also can assgin speficed negative for each query (set --neg_file)
    neg_file = os.path.join(data_args.output_dir, f"train-queries_hardneg.pickle")
    if not os.path.exists(neg_file):
        train_ground_truths = load_rel(os.path.join(data_args.preprocess_dir, 'train-rels.tsv'))
        trainquery2hardneg = learnable_index.hard_negative(train_query_embeddings, train_ground_truths, topk=400, batch_size=64)
        pickle.dump(trainquery2hardneg, open(neg_file, 'wb'))

    # contrastive learning
    if training_args.training_mode == 'contrastive_index':
        data_args.save_ckpt_dir = f'./saved_ckpts/{training_args.training_mode}/'
        learnable_index.fit_with_multi_gpus(rel_file=os.path.join(data_args.preprocess_dir, 'train-rels.tsv'),
                                            neg_file=os.path.join(data_args.output_dir, f"train-queries_hardneg.pickle"),
                                            query_embeddings_file=data_args.query_embeddings_file,
                                            doc_embeddings_file=data_args.doc_embeddings_file,
                                            emb_size = emb_size,
                                            checkpoint_path=data_args.save_ckpt_dir,
                                            logging_steps=training_args.logging_steps,
                                            per_device_train_batch_size=training_args.per_device_train_batch_size,
                                            checkpoint_save_steps=training_args.checkpoint_save_steps,
                                            max_grad_norm=training_args.max_grad_norm,
                                            temperature=training_args.temperature,
                                            optimizer_class=AdamW,
                                            loss_weight={'encoder_weight': 1.0, 'pq_weight': 1.0, 'ivf_weight': 'scaled_to_pqloss'},
                                            lr_params={'encoder_lr': 1e-5, 'pq_lr': 1e-4, 'ivf_lr': 1e-3},
                                            loss_method='contras',
                                            fix_emb='query, doc',
                                            epochs=16)


    # distill based on fixed embeddigns of queries and docs
    if training_args.training_mode == 'distill_index':
        data_args.save_ckpt_dir = f'./saved_ckpts/{training_args.training_mode}/'
        learnable_index.fit_with_multi_gpus(rel_file=os.path.join(data_args.preprocess_dir, 'train-rels.tsv'),
                                            neg_file=os.path.join(data_args.output_dir, f"train-queries_hardneg.pickle"),
                                            query_embeddings_file=data_args.query_embeddings_file,
                                            doc_embeddings_file=data_args.doc_embeddings_file,
                                            emb_size = emb_size,
                                            per_query_neg_num=1,
                                            checkpoint_path=data_args.save_ckpt_dir,
                                            logging_steps=training_args.logging_steps,
                                            per_device_train_batch_size=training_args.per_device_train_batch_size,
                                            checkpoint_save_steps=training_args.checkpoint_save_steps,
                                            max_grad_norm=training_args.max_grad_norm,
                                            temperature=training_args.temperature,
                                            optimizer_class=AdamW,
                                            loss_weight={'encoder_weight': 1.0, 'pq_weight': 1.0, 'ivf_weight': 'scaled_to_pqloss'},
                                            lr_params={'encoder_lr': 1e-5, 'pq_lr': 1e-4, 'ivf_lr': 1e-3},
                                            loss_method='distill',
                                            fix_emb='query, doc',
                                            epochs=30)


    # distill with no label data
    if training_args.training_mode == 'distill_virtual-data_index':

        # generate train data by brute-force or the index which should has similar performance with brute force
        if not os.path.exists(os.path.join(data_args.output_dir, 'train-virtual_rel.tsv')):
            flat_index = FaissIndex(doc_embeddings=doc_embeddings, index_method='flat', dist_mode='ip')
            # query2pos, query2neg = trainquery2hardneg = flat_index.generate_virtual_traindata(train_query_embeddings,
            #                                                                                        topk=400, batch_size=64)
            # or
            query2pos, query2neg = learnable_index.generate_virtual_traindata(train_query_embeddings,
                                                                                                   topk=400, batch_size=64)

            write_rel(os.path.join(data_args.output_dir, 'train-virtual_rel.tsv'), query2pos)
            pickle.dump(query2neg, open(os.path.join(data_args.output_dir, f"train-queries-virtual_hardneg.pickle"), 'wb'))


        data_args.save_ckpt_dir = f'./saved_ckpts/{training_args.training_mode}/'
        learnable_index.fit_with_multi_gpus(rel_file=os.path.join(data_args.output_dir, 'train-virtual_rel.tsv'),
                                            neg_file=os.path.join(data_args.output_dir, f"train-queries-virtual_hardneg.pickle"),
                                            query_embeddings_file=data_args.query_embeddings_file,
                                            doc_embeddings_file=data_args.doc_embeddings_file,
                                            emb_size = emb_size,
                                            per_query_neg_num=1,
                                            checkpoint_path=data_args.save_ckpt_dir,
                                            logging_steps=training_args.logging_steps,
                                            per_device_train_batch_size=training_args.per_device_train_batch_size,
                                            checkpoint_save_steps=training_args.checkpoint_save_steps,
                                            max_grad_norm=training_args.max_grad_norm,
                                            temperature=training_args.temperature,
                                            optimizer_class=AdamW,
                                            loss_weight={'encoder_weight': 1.0, 'pq_weight': 1.0, 'ivf_weight': 'scaled_to_pqloss'},
                                            lr_params={'encoder_lr': 1e-5, 'pq_lr': 1e-4, 'ivf_lr': 1e-3},
                                            loss_method='distill',
                                            fix_emb='query, doc',
                                            epochs=30)



    # select a latest ckpt
    # ckpt_path = learnable_index.get_latest_ckpt(data_args.save_ckpt_dir)
    # data_args.output_dir = f'./data/passage/evaluate/LearnableIndex_{training_args.training_mode}'

    # update index
    # print('Updating the index with new ivf and pq')
    # learnable_index.update_index_with_ckpt(ckpt_path=ckpt_path, doc_embeddings=doc_embeddings)

    # Test
    ground_truths = load_rel(os.path.join(data_args.preprocess_dir, 'dev-rels.tsv'))
    learnable_index.test(query_embeddings=query_embeddings,
                         ground_truths=ground_truths,
                         topk=1000,
                         batch_size=64,
                         MRR_cutoffs=[10, 100], Recall_cutoffs=[10, 30, 50, 100],
                         nprobe=index_args.nprobe)

    learnable_index.save_index(f'{data_args.output_dir}/learnable_index.index')