import sys
sys.path.append('./')
import os
import pickle
import gc

import faiss
import numpy as np
from transformers import HfArgumentParser
from torch.optim import AdamW

from LibVQ.dataset.dataset import load_rel, write_rel
from LibVQ.learnable_index import LearnableIndexWithEncoder
from LibVQ.models import Encoder
from LibVQ.utils import setuplogging

from arguments import IndexArguments, DataArguments, ModelArguments, TrainingArguments
from prepare_data.get_embeddings import MS_Encoder

faiss.omp_set_num_threads(32)

if __name__ == '__main__':
    setuplogging()
    parser = HfArgumentParser((IndexArguments, DataArguments, ModelArguments, TrainingArguments))
    index_args, data_args, model_args, training_args = parser.parse_args_into_dataclasses()

    # Load encoder
    query_encoder = MS_Encoder(pretrained_model_name='Luyu/co-condenser-marco-retriever')
    doc_encoder = MS_Encoder(pretrained_model_name='Luyu/co-condenser-marco-retriever')
    emb_size = doc_encoder.output_embedding_size

    text_encoder = Encoder(query_encoder=query_encoder,
                           doc_encoder=doc_encoder)

    # Load embeddings of queries and docs
    doc_embeddings_file = os.path.join(data_args.embeddings_dir, 'docs.memmap')
    query_embeddings_file = os.path.join(data_args.embeddings_dir, 'train-queries.memmap')

    doc_embeddings = np.memmap(doc_embeddings_file,
                               dtype=np.float32, mode="r")
    doc_embeddings = doc_embeddings.reshape(-1, emb_size)

    train_query_embeddings = np.memmap(query_embeddings_file,
                                       dtype=np.float32, mode="r")
    train_query_embeddings = train_query_embeddings.reshape(-1, emb_size)

    test_query_embeddings = np.memmap(os.path.join(data_args.embeddings_dir, 'dev-queries.memmap'),
                                 dtype=np.float32, mode="r")
    test_query_embeddings = test_query_embeddings.reshape(-1, emb_size)



    # Create Index
    # if there is a faiss index in init_index_file, it will creat learnable_index based on it;
    # if no, it will creat and save a faiss index in init_index_file
    init_index_file = os.path.join(data_args.embeddings_dir, f'{index_args.index_method}_ivf{index_args.ivf_centers_num}_pq{index_args.subvector_num}x{index_args.subvector_bits}.index')
    learnable_index = LearnableIndexWithEncoder(index_method=index_args.index_method,
                                     encoder=text_encoder,
                                     init_index_file=init_index_file,
                                     doc_embeddings=doc_embeddings,
                                     ivf_centers_num=index_args.ivf_centers_num,
                                     subvector_num=index_args.subvector_num,
                                     subvector_bits=index_args.subvector_bits)

    # The dataloader randomly sample the negative from corpus by default. You also can assgin speficed negative for each query (set --neg_file)
    neg_file = os.path.join(data_args.embeddings_dir, f"train-queries_hardneg.pickle")
    if not os.path.exists(neg_file):
        print('generating hard negatives for train queries ...')
        train_ground_truths = load_rel(os.path.join(data_args.preprocess_dir, 'train-rels.tsv'))
        trainquery2hardneg = learnable_index.hard_negative(train_query_embeddings,
                                                           train_ground_truths,
                                                           topk=400,
                                                           batch_size=64,
                                                           nprobe=min(index_args.ivf_centers_num, 500))
        pickle.dump(trainquery2hardneg, open(neg_file, 'wb'))

        del trainquery2hardneg
        gc.collect()

    data_args.save_ckpt_dir = f'./saved_ckpts/{training_args.training_mode}_{index_args.index_method}/'

    # contrastive learning
    if training_args.training_mode == 'contrastive_index-and-query-encoder':
        learnable_index.fit_with_multi_gpus(rel_file=os.path.join(data_args.preprocess_dir, 'train-rels.tsv'),
                                            neg_file=os.path.join(data_args.embeddings_dir,
                                                                  f"train-queries_hardneg.pickle"),
                                            query_data_dir=data_args.preprocess_dir,
                                            max_query_length=data_args.max_query_length,
                                            doc_embeddings_file=doc_embeddings_file,
                                            emb_size=emb_size,
                                            checkpoint_path=data_args.save_ckpt_dir,
                                            logging_steps=training_args.logging_steps,
                                            per_device_train_batch_size=training_args.per_device_train_batch_size,
                                            checkpoint_save_steps=training_args.checkpoint_save_steps,
                                            max_grad_norm=training_args.max_grad_norm,
                                            temperature=training_args.temperature,
                                            optimizer_class=AdamW,
                                            loss_weight={'encoder_weight': 1.0, 'pq_weight': 1.0,
                                                         'ivf_weight': 'scaled_to_pqloss'},
                                            lr_params={'encoder_lr': 1e-5, 'pq_lr': 1e-4, 'ivf_lr': 1e-3},
                                            loss_method='contras',
                                            fix_emb='doc',
                                            epochs=16)

    # distill learning
    if training_args.training_mode == 'distill_index-and-query-encoder':
        learnable_index.fit_with_multi_gpus(rel_file=os.path.join(data_args.preprocess_dir, 'train-rels.tsv'),
                                            neg_file=os.path.join(data_args.embeddings_dir,
                                                                  f"train-queries_hardneg.pickle"),
                                            query_data_dir=data_args.preprocess_dir,
                                            max_query_length=data_args.max_query_length,
                                            query_embeddings_file=query_embeddings_file,
                                            doc_embeddings_file=doc_embeddings_file,
                                            emb_size=emb_size,
                                            checkpoint_path=data_args.save_ckpt_dir,
                                            logging_steps=training_args.logging_steps,
                                            per_device_train_batch_size=training_args.per_device_train_batch_size,
                                            checkpoint_save_steps=training_args.checkpoint_save_steps,
                                            max_grad_norm=training_args.max_grad_norm,
                                            temperature=training_args.temperature,
                                            optimizer_class=AdamW,
                                            loss_weight={'encoder_weight': 1.0, 'pq_weight': 1.0,
                                                         'ivf_weight': 'scaled_to_pqloss'},
                                            lr_params={'encoder_lr': 1e-5, 'pq_lr': 1e-4, 'ivf_lr': 1e-3},
                                            loss_method='distill',
                                            fix_emb='doc',
                                            epochs=30)

        # learnable_index.fit(rel_data=os.path.join(data_args.preprocess_dir, 'train-rels.tsv'),
        #                                     neg_data=os.path.join(data_args.embeddings_dir,
        #                                                           f"train-queries_hardneg.pickle"),
        #                                     query_data_dir=data_args.preprocess_dir,
        #                                     max_query_length=data_args.max_query_length,
        #                                     query_embeddings=query_embeddings_file,
        #                                     doc_embeddings=doc_embeddings_file,
        #                                     emb_size=emb_size,
        #                                     checkpoint_path=data_args.save_ckpt_dir,
        #                                     logging_steps=training_args.logging_steps,
        #                                     per_device_train_batch_size=training_args.per_device_train_batch_size,
        #                                     checkpoint_save_steps=training_args.checkpoint_save_steps,
        #                                     max_grad_norm=training_args.max_grad_norm,
        #                                     temperature=training_args.temperature,
        #                                     optimizer_class=AdamW,
        #                                     loss_weight={'encoder_weight': 1.0, 'pq_weight': 1.0,
        #                                                  'ivf_weight': 'scaled_to_pqloss'},
        #                                     lr_params={'encoder_lr': 1e-5, 'pq_lr': 1e-4, 'ivf_lr': 1e-3},
        #                                     loss_method='distill',
        #                                     fix_emb='doc',
        #                                     epochs=10)

    # distill learning and train both query encoder and doc encoder, which only can be used when ivf is disabled
    if training_args.training_mode == 'distill_index-and-two-encoders':
        assert 'ivf' not in index_args.index_method
        learnable_index.fit_with_multi_gpus(rel_file=os.path.join(data_args.preprocess_dir, 'train-rels.tsv'),
                                            neg_file=os.path.join(data_args.embeddings_dir,
                                                                  f"train-queries_hardneg.pickle"),
                                            query_data_dir=data_args.preprocess_dir,
                                            max_query_length=data_args.max_query_length,
                                            doc_data_dir=data_args.preprocess_dir,
                                            max_doc_length=128,
                                            query_embeddings_file=query_embeddings_file,
                                            doc_embeddings_file=doc_embeddings_file,
                                            emb_size=emb_size,
                                            checkpoint_path=data_args.save_ckpt_dir,
                                            logging_steps=training_args.logging_steps,
                                            per_device_train_batch_size=training_args.per_device_train_batch_size,
                                            checkpoint_save_steps=training_args.checkpoint_save_steps,
                                            max_grad_norm=training_args.max_grad_norm,
                                            temperature=training_args.temperature,
                                            optimizer_class=AdamW,
                                            loss_weight={'encoder_weight': 1.0, 'pq_weight': 1.0, 'ivf_weight': 0.0},
                                            lr_params={'encoder_lr': 1e-5, 'pq_lr': 1e-4, 'ivf_lr': 0.0},
                                            loss_method='distill',
                                            fix_emb='',
                                            epochs=30)


    if training_args.training_mode == 'contras_index-and-two-encoders':
        assert 'ivf' not in index_args.index_method
        learnable_index.fit_with_multi_gpus(rel_file=os.path.join(data_args.preprocess_dir, 'train-rels.tsv'),
                                            neg_file=os.path.join(data_args.embeddings_dir,
                                                                  f"train-queries_hardneg.pickle"),
                                            query_data_dir=data_args.preprocess_dir,
                                            max_query_length=data_args.max_query_length,
                                            doc_data_dir=data_args.preprocess_dir,
                                            max_doc_length=128,
                                            query_embeddings_file=query_embeddings_file,
                                            doc_embeddings_file=doc_embeddings_file,
                                            emb_size=emb_size,
                                            checkpoint_path=data_args.save_ckpt_dir,
                                            logging_steps=training_args.logging_steps,
                                            per_device_train_batch_size=training_args.per_device_train_batch_size,
                                            checkpoint_save_steps=training_args.checkpoint_save_steps,
                                            max_grad_norm=training_args.max_grad_norm,
                                            temperature=training_args.temperature,
                                            optimizer_class=AdamW,
                                            loss_weight={'encoder_weight': 1.0, 'pq_weight': 1.0, 'ivf_weight': 0.0},
                                            lr_params={'encoder_lr': 1e-5, 'pq_lr': 1e-4, 'ivf_lr': 0.0},
                                            loss_method='contras',
                                            fix_emb='',
                                            epochs=30)

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
            query2pos, query2neg = learnable_index.generate_virtual_traindata(train_query_embeddings,
                                                                              topk=400,
                                                                              batch_size=64,
                                                                              nprobe=min(index_args.ivf_centers_num,
                                                                                         500))

            write_rel(os.path.join(data_args.embeddings_dir, 'train-virtual_rel.tsv'), query2pos)
            pickle.dump(query2neg,
                        open(os.path.join(data_args.embeddings_dir, f"train-queries-virtual_hardneg.pickle"), 'wb'))

            del query2pos, query2neg
            gc.collect()


    # distill with no label data
    if training_args.training_mode == 'distill_index-and-query-encoder_nolabel':
        learnable_index.fit_with_multi_gpus(rel_file=os.path.join(data_args.embeddings_dir, 'train-virtual_rel.tsv'),
                                            neg_file=os.path.join(data_args.embeddings_dir,
                                                                  f"train-queries-virtual_hardneg.pickle"),
                                            query_data_dir=data_args.preprocess_dir,
                                            max_query_length=data_args.max_query_length,
                                            query_embeddings_file=query_embeddings_file,
                                            doc_embeddings_file=doc_embeddings_file,
                                            emb_size=emb_size,
                                            per_query_neg_num=1,
                                            checkpoint_path=data_args.save_ckpt_dir,
                                            logging_steps=training_args.logging_steps,
                                            per_device_train_batch_size=training_args.per_device_train_batch_size,
                                            checkpoint_save_steps=training_args.checkpoint_save_steps,
                                            max_grad_norm=training_args.max_grad_norm,
                                            temperature=training_args.temperature,
                                            optimizer_class=AdamW,
                                            loss_weight={'encoder_weight': 1.0, 'pq_weight': 1.0,
                                                         'ivf_weight': 'scaled_to_pqloss'},
                                            lr_params={'encoder_lr': 1e-5, 'pq_lr': 1e-4, 'ivf_lr': 1e-3},
                                            loss_method='distill',
                                            fix_emb='doc',
                                            epochs=30)


    # update query embeddings when re-training the query encoder
    data_args.output_dir = f'./data/passage/evaluate/LearnableIndex_{training_args.training_mode}'
    new_query_embeddings = learnable_index.encode(data_dir=data_args.preprocess_dir,
                                                  prefix='dev-queries',
                                                  max_length=data_args.max_query_length,
                                                  output_dir=data_args.output_dir,
                                                  batch_size=8196,
                                                  is_query=True,
                                                  return_vecs=True
                                                  )

    # Test
    ground_truths = load_rel(os.path.join(data_args.preprocess_dir, 'dev-rels.tsv'))
    learnable_index.test(new_query_embeddings, ground_truths, topk=1000, batch_size=64,
                         MRR_cutoffs=[10, 100], Recall_cutoffs=[10, 30, 50, 100],
                         nprobe=index_args.nprobe)

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