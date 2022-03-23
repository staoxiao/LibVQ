import sys
sys.path.append('./')
import os
import pickle

import faiss
import numpy as np
from transformers import HfArgumentParser, AdamW, AutoConfig, DPRContextEncoder, DPRQuestionEncoder

from LibVQ.dataset.dataset import load_rel, write_rel
from LibVQ.learnable_index import LearnableIndex
from LibVQ.models import Encoder
from LibVQ.utils import setuplogging

from arguments import IndexArguments, DataArguments, ModelArguments, TrainingArguments
from prepare_data.get_embeddings import DPR_Encoder
from evaluate import validate, load_test_data

faiss.omp_set_num_threads(32)

if __name__ == '__main__':
    setuplogging()
    parser = HfArgumentParser((IndexArguments, DataArguments, ModelArguments, TrainingArguments))
    index_args, data_args, model_args, training_args = parser.parse_args_into_dataclasses()

    # Load encoder
    # doc_encoder = DPR_Encoder(DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base"))
    # query_encoder = DPR_Encoder(DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base'))
    # config = AutoConfig.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    # emb_size = config.hidden_size
    #
    # text_encoder = Encoder(query_encoder=query_encoder,
    #                        doc_encoder=doc_encoder)
    from prepare_data.get_embeddings import get_ARG_encoder
    text_encoder = get_ARG_encoder()
    emb_size = 768


    # Load embeddings of queries and docs
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
    learnable_index = LearnableIndex(index_method=index_args.index_method,
                                     encoder=text_encoder,
                                     init_index_file=os.path.join(data_args.embeddings_dir,
                                                                  f'{index_args.index_method}.index'),
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

    # contrastive learning
    if training_args.training_mode == 'contrastive_index-and-query-encoder':
        data_args.save_ckpt_dir = f'./saved_ckpts/{training_args.training_mode}_{index_args.index_method}/'
        learnable_index.fit_with_multi_gpus(rel_file=os.path.join(data_args.preprocess_dir, 'train-rels.tsv'),
                                            neg_file=os.path.join(data_args.embeddings_dir,
                                                                  f"train-queries_hardneg.pickle"),
                                            query_data_dir=data_args.preprocess_dir,
                                            max_query_length=data_args.max_query_length,
                                            doc_embeddings_file=doc_embeddings_file,
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
                                            fix_emb='doc',
                                            epochs=16)

    # distill learning
    if training_args.training_mode == 'distill_index-and-query-encoder':
        data_args.save_ckpt_dir = f'./saved_ckpts/{training_args.training_mode}_{index_args.index_method}/'
        learnable_index.fit_with_multi_gpus(rel_file=os.path.join(data_args.preprocess_dir, 'train-rels.tsv'),
                                            neg_file=os.path.join(data_args.embeddings_dir,
                                                                  f"train-queries_hardneg.pickle"),
                                            query_data_dir=data_args.preprocess_dir,
                                            max_query_length=data_args.max_query_length,
                                            query_embeddings_file=query_embeddings_file,
                                            doc_embeddings_file=doc_embeddings_file,
                                            emb_size=emb_size,
                                            per_query_neg_num=1,
                                            checkpoint_path=data_args.save_ckpt_dir,
                                            logging_steps=training_args.logging_steps,
                                            per_device_train_batch_size=160,
                                            checkpoint_save_steps=training_args.checkpoint_save_steps,
                                            max_grad_norm=training_args.max_grad_norm,
                                            temperature=training_args.temperature,
                                            optimizer_class=AdamW,
                                            loss_weight={'encoder_weight': 1.0, 'pq_weight': 1.0,
                                                         'ivf_weight': 0.56},
                                            lr_params={'encoder_lr': 5e-6, 'pq_lr': 1e-4, 'ivf_lr': 1e-3},
                                            loss_method='distill',
                                            fix_emb='doc',
                                            epochs=9)

    # distill learning and train both query encoder and doc encoder, which only can be used when ivf is disabled
    if training_args.training_mode == 'distill_index-and-two-encoders':
        data_args.save_ckpt_dir = f'./saved_ckpts/{training_args.training_mode}_{index_args.index_method}/'
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
                                            per_query_neg_num=12,
                                            checkpoint_path=data_args.save_ckpt_dir,
                                            logging_steps=training_args.logging_steps,
                                            per_device_train_batch_size=16,
                                            checkpoint_save_steps=training_args.checkpoint_save_steps,
                                            max_grad_norm=training_args.max_grad_norm,
                                            temperature=training_args.temperature,
                                            optimizer_class=AdamW,
                                            loss_weight={'encoder_weight': 1.0, 'pq_weight': 1.0, 'ivf_weight': 0.0},
                                            lr_params={'encoder_lr': 5e-6, 'pq_lr': 1e-4, 'ivf_lr': 0.0},
                                            loss_method='distill',
                                            fix_emb='',
                                            epochs=20)

    # distill with no label data
    if training_args.training_mode == 'distill_index-and-query-encoder_nolabel':

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

        data_args.save_ckpt_dir = f'./saved_ckpts/{training_args.training_mode}_{index_args.index_method}/'
        learnable_index.fit_with_multi_gpus(rel_file=os.path.join(data_args.embeddings_dir, 'train-virtual_rel.tsv'),
                                            neg_file=os.path.join(data_args.embeddings_dir,
                                                                  f"train-queries-virtual_hardneg.pickle"),
                                            query_data_dir=data_args.preprocess_dir,
                                            max_query_length=data_args.max_query_length,
                                            query_embeddings_file=query_embeddings_file,
                                            doc_embeddings_file=doc_embeddings_file,
                                            emb_size=emb_size,
                                            per_query_neg_num=100,
                                            checkpoint_path=data_args.save_ckpt_dir,
                                            logging_steps=training_args.logging_steps,
                                            per_device_train_batch_size=64,    #training_args.per_device_train_batch_size,
                                            checkpoint_save_steps=training_args.checkpoint_save_steps,
                                            max_grad_norm=training_args.max_grad_norm,
                                            temperature=training_args.temperature,
                                            optimizer_class=AdamW,
                                            loss_weight={'encoder_weight': 1.0, 'pq_weight': 1.0,
                                                         'ivf_weight': 'scaled_to_pqloss'},
                                            lr_params={'encoder_lr': 5e-6, 'pq_lr': 1e-4, 'ivf_lr': 1e-3},
                                            loss_method='distill',
                                            fix_emb='doc',
                                            epochs=10)


    if training_args.training_mode == 'distill_index-and-two-encoders_nolabel':
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

        data_args.save_ckpt_dir = f'./saved_ckpts/{training_args.training_mode}_{index_args.index_method}/'
        learnable_index.fit_with_multi_gpus(rel_file=os.path.join(data_args.embeddings_dir, 'train-virtual_rel.tsv'),
                                            neg_file=os.path.join(data_args.embeddings_dir,
                                                                  f"train-queries-virtual_hardneg.pickle"),
                                            query_data_dir=data_args.preprocess_dir,
                                            max_query_length=data_args.max_query_length,
                                            doc_data_dir=data_args.preprocess_dir,
                                            max_doc_length=128,
                                            query_embeddings_file=query_embeddings_file,
                                            doc_embeddings_file=doc_embeddings_file,
                                            emb_size=emb_size,
                                            per_query_neg_num=12,
                                            checkpoint_path=data_args.save_ckpt_dir,
                                            logging_steps=training_args.logging_steps,
                                            per_device_train_batch_size=16,    #training_args.per_device_train_batch_size,
                                            checkpoint_save_steps=training_args.checkpoint_save_steps,
                                            max_grad_norm=training_args.max_grad_norm,
                                            temperature=training_args.temperature,
                                            optimizer_class=AdamW,
                                            loss_weight={'encoder_weight': 1.0, 'pq_weight': 1.0,
                                                         'ivf_weight': 'scaled_to_pqloss'},
                                            lr_params={'encoder_lr': 5e-6, 'pq_lr': 1e-4, 'ivf_lr': 1e-3},
                                            loss_method='distill',
                                            fix_emb='',
                                            epochs=20)



    # update query embeddings when re-training the query encoder
    data_args.output_dir = f'./data/NQ/evaluate/LearnableIndex_{training_args.training_mode}_{index_args.index_method}'
    new_query_embeddings = learnable_index.encode(data_dir=data_args.preprocess_dir,
                                                  prefix='test-queries',
                                                  max_length=data_args.max_query_length,
                                                  output_dir=data_args.output_dir,
                                                  batch_size=8196,
                                                  is_query=True,
                                                  return_vecs=True
                                                  )


    # Test
    scores, ann_items = learnable_index.search(test_query_embeddings, topk=100, nprobe=index_args.nprobe)
    test_questions, test_answers, collections = load_test_data(
        query_andwer_file='./data/NQ/raw_dataset/nq-test.qa.csv',
        collections_file='./data/NQ/dataset/collection.tsv')
    validate(ann_items, test_questions, test_answers, collections)

    learnable_index.save_index(f'{data_args.output_dir}/learnable_index{training_args.training_mode}_{index_args.index_method}.index')
    learnable_index.load_index(f'{data_args.output_dir}/learnable_index{training_args.training_mode}_{index_args.index_method}.index')

