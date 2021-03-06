import sys
sys.path.append('./')
import os
import pickle
import gc

import faiss
import numpy as np
from transformers import HfArgumentParser
from torch.optim import AdamW

from LibVQ.base_index import FaissIndex
from LibVQ.dataset.dataset import load_rel, write_rel
from LibVQ.learnable_index import LearnableIndex
from LibVQ.utils import setuplogging

from arguments import IndexArguments, DataArguments, ModelArguments, TrainingArguments
from evaluate import validate, load_test_data

faiss.omp_set_num_threads(32)

if __name__ == '__main__':
    setuplogging()
    parser = HfArgumentParser((IndexArguments, DataArguments, ModelArguments, TrainingArguments))
    index_args, data_args, model_args, training_args = parser.parse_args_into_dataclasses()

    # Load embeddings of queries and docs
    emb_size = 768
    doc_embeddings_file = os.path.join(data_args.embeddings_dir, 'docs.memmap')
    query_embeddings_file = os.path.join(data_args.embeddings_dir, 'train-queries.memmap')

    doc_embeddings = np.memmap(doc_embeddings_file,
                               dtype=np.float32, mode="r")
    doc_embeddings = doc_embeddings.reshape(-1, emb_size)

    train_query_embeddings = np.memmap(query_embeddings_file,
                                       dtype=np.float32, mode="r")
    train_query_embeddings = train_query_embeddings.reshape(-1, emb_size)

    test_query_embeddings = np.memmap(os.path.join(data_args.embeddings_dir, 'test-queries.memmap'),
                                 dtype=np.float32, mode="r")
    test_query_embeddings = test_query_embeddings.reshape(-1, emb_size)


    # Create Index
    # if there is a faiss index in init_index_file, it will creat learnable_index based on it;
    # if no, it will creat and save a faiss index in init_index_file
    init_index_file = os.path.join(data_args.embeddings_dir, f'{index_args.index_method}_ivf{index_args.ivf_centers_num}_pq{index_args.subvector_num}x{index_args.subvector_bits}.index')
    learnable_index = LearnableIndex(index_method=index_args.index_method,
                                     init_index_file=init_index_file,
                                     doc_embeddings=doc_embeddings,
                                     ivf_centers_num=index_args.ivf_centers_num,
                                     subvector_num=index_args.subvector_num,
                                     subvector_bits=index_args.subvector_bits)

    # The class randomly sample the negative from corpus by default. You also can assgin speficed negative for each query (set --neg_file)
    neg_file = os.path.join(data_args.embeddings_dir, f"train-queries_hardneg.pickle")
    if not os.path.exists(neg_file):
        print('generating hard negatives for train queries ...')
        train_ground_truths = load_rel(os.path.join(data_args.preprocess_dir, 'train-rels.tsv'))
        trainquery2hardneg = learnable_index.hard_negative(train_query_embeddings,
                                                           train_ground_truths,
                                                           topk=400,
                                                           batch_size=64,
                                                           nprobe=index_args.ivf_centers_num)
        pickle.dump(trainquery2hardneg, open(neg_file, 'wb'))

        del trainquery2hardneg
        gc.collect()

    data_args.save_ckpt_dir = f'./saved_ckpts/{training_args.training_mode}_{index_args.index_method}/'

    # contrastive learning
    if training_args.training_mode == 'contrastive_index':
        learnable_index.fit_with_multi_gpus(query_embeddings_file=query_embeddings_file,
                                            doc_embeddings_file=doc_embeddings_file,
                                            rel_file=os.path.join(data_args.preprocess_dir, 'train-rels.tsv'),
                                            neg_file=os.path.join(data_args.embeddings_dir,
                                                                  f"train-queries_hardneg.pickle"),
                                            emb_size=emb_size,
                                            per_query_neg_num=1,
                                            checkpoint_path=data_args.save_ckpt_dir,
                                            logging_steps=training_args.logging_steps,
                                            per_device_train_batch_size=512,
                                            checkpoint_save_steps=training_args.checkpoint_save_steps,
                                            max_grad_norm=training_args.max_grad_norm,
                                            temperature=training_args.temperature,
                                            optimizer_class=AdamW,
                                            loss_weight={'encoder_weight': 1.0, 'pq_weight': 1.0,
                                                         'ivf_weight': 'scaled_to_pqloss'},
                                            lr_params={'encoder_lr': 5e-6, 'pq_lr': 1e-4, 'ivf_lr': 1e-3},
                                            loss_method='contras',
                                            epochs=16)


    # distill based on fixed embeddigns of queries and docs
    if training_args.training_mode == 'distill_index':
        learnable_index.fit_with_multi_gpus(query_embeddings_file=query_embeddings_file,
                                            doc_embeddings_file=doc_embeddings_file,
                                            rel_file=os.path.join(data_args.preprocess_dir, 'train-rels.tsv'),
                                            neg_file=os.path.join(data_args.embeddings_dir,
                                                                  f"train-queries_hardneg.pickle"),
                                            emb_size=emb_size,
                                            per_query_neg_num=1,
                                            checkpoint_path=data_args.save_ckpt_dir,
                                            logging_steps=training_args.logging_steps,
                                            per_device_train_batch_size=128,
                                            checkpoint_save_steps=training_args.checkpoint_save_steps,
                                            max_grad_norm=training_args.max_grad_norm,
                                            temperature=training_args.temperature,
                                            optimizer_class=AdamW,
                                            loss_weight={'encoder_weight': 1.0, 'pq_weight': 1.0,
                                                         'ivf_weight': 'scaled_to_pqloss'},
                                            lr_params={'encoder_lr': 5e-6, 'pq_lr': 1e-4, 'ivf_lr': 1e-3},
                                            loss_method='distill',
                                            epochs=10)


    if 'nolabel' in training_args.training_mode:
        '''
                If there is not relevance data, you can set the rel_file/rel_data to None, and it will automatically generate the data for training.
                You also can manually generate the data as following:
                '''
        # generate train data by brute-force or the index which should has similar performance with brute force
        if not os.path.exists(os.path.join(data_args.embeddings_dir, 'train-virtual_rel.tsv')):
            print('generating relevance labels for train queries ...')
            # flat_index = FaissIndex(doc_embeddings=doc_embeddings, index_method='flat', dist_mode='ip')
            # query2pos, query2neg = flat_index.generate_virtual_traindata(train_query_embeddings,
            #                                                                                        topk=400, batch_size=64)
            # or
            query2pos, query2neg = trainquery2hardneg = learnable_index.generate_virtual_traindata(
                train_query_embeddings, topk=400, batch_size=64, nprobe=index_args.ivf_centers_num)

            write_rel(os.path.join(data_args.embeddings_dir, 'train-virtual_rel.tsv'), query2pos)
            pickle.dump(query2neg,
                        open(os.path.join(data_args.embeddings_dir, f"train-queries-virtual_hardneg.pickle"), 'wb'))

            del query2neg, query2pos
            gc.collect()


    # distill with no label data
    if training_args.training_mode == 'distill_index_nolabel':
        learnable_index.fit_with_multi_gpus(query_embeddings_file=query_embeddings_file,
                                            doc_embeddings_file=doc_embeddings_file,
                                            rel_file=os.path.join(data_args.embeddings_dir, 'train-virtual_rel.tsv'),
                                            neg_file=os.path.join(data_args.embeddings_dir,
                                                                  f"train-queries-virtual_hardneg.pickle"),
                                            emb_size=emb_size,
                                            per_query_neg_num=100,
                                            checkpoint_path=data_args.save_ckpt_dir,
                                            logging_steps=training_args.logging_steps,
                                            per_device_train_batch_size=64,
                                            checkpoint_save_steps=training_args.checkpoint_save_steps,
                                            max_grad_norm=training_args.max_grad_norm,
                                            temperature=training_args.temperature,
                                            optimizer_class=AdamW,
                                            loss_weight={'encoder_weight': 1.0, 'pq_weight': 1.0,
                                                         'ivf_weight': 'scaled_to_pqloss'},
                                            lr_params={'encoder_lr': 5e-6, 'pq_lr': 1e-4, 'ivf_lr': 1e-3},
                                            loss_method='distill',
                                            epochs=10)

    # Test
    scores, ann_items = learnable_index.search(test_query_embeddings, topk=100, nprobe=index_args.nprobe)
    test_questions, test_answers, collections = load_test_data(
        query_andwer_file='./data/NQ/raw_dataset/nq-test.qa.csv',
        collections_file='./data/NQ/dataset/collection.tsv')
    validate(ann_items, test_questions, test_answers, collections)

    data_args.output_dir = f'./data/NQ/evaluate/LearnableIndex_{training_args.training_mode}'
    os.makedirs(data_args.output_dir, exist_ok=True)
    saved_index_file = os.path.join(data_args.output_dir,
                                    f'LibVQ_{training_args.training_mode}_{index_args.index_method}_ivf{index_args.ivf_centers_num}_pq{index_args.subvector_num}x{index_args.subvector_bits}.index')
    learnable_index.save_index(saved_index_file)
    learnable_index.load_index(saved_index_file)

    # get the faiss index and then you can use the faiss API.
    '''
    index = learnable_index.index 
    index = faiss.read_index(saved_index_file)
    index = faiss.index_gpu_to_cpu(index)
    '''