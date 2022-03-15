import os
import torch
import numpy as np
import faiss
import pickle
from transformers import HfArgumentParser

from LibVQ.models import Encoder, EncoderConfig
from LibVQ.dataset.dataset import load_rel, write_rel
from LibVQ.learnable_index import LearnableIndex
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


    # Create Index
    # if there is a faiss index in init_index_file, it will creat learnable_index based on it;
    # if no, it will creat and save a faiss index in init_index_file
    learnable_index = LearnableIndex(index_method=index_args.index_method,
                                     encoder=text_encoder,
                                     init_index_file=os.path.join(data_args.output_dir,
                                                                  f'{index_args.index_method}.index'),
                                     doc_embeddings=doc_embeddings,
                                     ivf_centers_num=index_args.ivf_centers_num,
                                     subvector_num=index_args.subvector_num,
                                     subvector_bits=index_args.subvector_bits)


    # The class randomly sample the negative from corpus by default. You also can assgin speficed negative for each query (set --neg_file)
    neg_file = os.path.join(data_args.output_dir, f"train-queries_hardneg.pickle")
    if not os.path.exists(neg_file):
        train_ground_truths = load_rel(os.path.join(data_args.preprocess_dir, 'train-rels.tsv'))
        trainquery2hardneg = learnable_index.hard_negative(train_query_embeddings, train_ground_truths, topk=400,
                                                           batch_size=64)
        pickle.dump(trainquery2hardneg, open(neg_file, 'wb'))


    # contrastive learning
    if training_args.training_mode == 'contrastive_jointly':
        data_args.save_ckpt_dir = f'./saved_ckpts/{training_args.training_mode}/'
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
                                            epochs=16)


    # distill learning
    if training_args.training_mode == 'distill_jointly':
        data_args.save_ckpt_dir = f'./saved_ckpts/{training_args.training_mode}/'
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
                                            epochs=30)


    # distill learning and train both query encoder and doc encoder, which only can be used when ivf is disabled
    if training_args.training_mode == 'distill_jointly_v2':
        data_args.save_ckpt_dir = f'./saved_ckpts/{training_args.training_mode}/'
        assert  'ivf' not in index_args.index_method
        learnable_index.fit_with_multi_gpus(rel_file=os.path.join(data_args.preprocess_dir, 'train-rels.tsv'),
                                            neg_file=os.path.join(data_args.output_dir, f"train-queries_hardneg.pickle"),
                                            query_data_dir=data_args.preprocess_dir,
                                            max_query_length=data_args.max_query_length,
                                            doc_data_dir=data_args.preprocess_dir,
                                            max_doc_length=128,
                                            query_embeddings_file=data_args.query_embeddings_file,
                                            doc_embeddings_file=data_args.doc_embeddings_file,
                                            emb_size = emb_size,
                                            checkpoint_path=data_args.save_ckpt_dir,
                                            logging_steps=training_args.logging_steps,
                                            per_device_train_batch_size=training_args.per_device_train_batch_size,
                                            loss_weight={'encoder_weight': 1.0, 'pq_weight': 1.0, 'ivf_weight': 0.0},
                                            lr_params={'encoder_lr': 1e-5, 'pq_lr': 1e-4, 'ivf_lr': 0.0},
                                            loss_method='distill',
                                            fix_emb=None,
                                            epochs=30)

    # distill with no label data
    if training_args.training_mode == 'distill_virtual-data_jointly':

        # generate train data by brute-force or the index which should has similar performance with brute force
        if not os.path.exists(os.path.join(data_args.output_dir, 'train-virtual_rel.tsv')):
            # flat_index = FaissIndex(doc_embeddings=doc_embeddings, index_method='flat', dist_mode='ip')
            # query2pos, query2neg = trainquery2hardneg = flat_index.generate_virtual_traindata(train_query_embeddings,
            #                                                                                        topk=400, batch_size=64)
            # or
            query2pos, query2neg = trainquery2hardneg = learnable_index.generate_virtual_traindata(train_query_embeddings, topk=400, batch_size=64)

            write_rel(os.path.join(data_args.output_dir, 'train-virtual_rel.tsv'), query2pos)
            pickle.dump(query2neg, open(os.path.join(data_args.output_dir, f"train-queries-virtual_hardneg.pickle"), 'wb'))


        data_args.save_ckpt_dir = f'./saved_ckpts/{training_args.training_mode}_{training_args.per_device_train_batch_size}/'
        learnable_index.fit_with_multi_gpus(rel_file=os.path.join(data_args.output_dir, 'train-virtual_rel.tsv'),
                                            neg_file=os.path.join(data_args.output_dir, f"train-queries-virtual_hardneg.pickle"),
                                            query_data_dir=data_args.preprocess_dir,
                                            max_query_length=data_args.max_query_length,
                                            query_embeddings_file=data_args.query_embeddings_file,
                                            doc_embeddings_file=data_args.doc_embeddings_file,
                                            emb_size = emb_size,
                                            per_query_neg_num = 1,
                                            checkpoint_path=data_args.save_ckpt_dir,
                                            logging_steps=training_args.logging_steps,
                                            per_device_train_batch_size=training_args.per_device_train_batch_size,
                                            loss_weight={'encoder_weight': 1.0, 'pq_weight': 1.0, 'ivf_weight': 'scaled_to_pqloss'},
                                            lr_params={'encoder_lr': 1e-5, 'pq_lr': 1e-4, 'ivf_lr': 1e-3},
                                            loss_method='distill',
                                            fix_emb='doc',
                                            epochs=30)

    # select a latest ckpt
    ckpt_path = learnable_index.get_latest_ckpt(data_args.save_ckpt_dir)

    # update query embeddings when re-training the query encoder
    data_args.output_dir = f'./data/passage/evaluate/LearnableIndex_{training_args.training_mode}'
    learnable_index.update_encoder(encoder_file=f'{ckpt_path}/encoder.bin')
    learnable_index.encode(data_dir=data_args.preprocess_dir,
                           prefix='dev-queries',
                           max_length=data_args.max_query_length,
                           output_dir=data_args.output_dir,
                           batch_size=8196,
                           is_query=True,
                           )
    print(f'{data_args.output_dir}/dev-queries.memmap')
    new_query_embeddings = np.memmap(f'{data_args.output_dir}/dev-queries.memmap', dtype=np.float32,
                                 mode="r")
    new_query_embeddings = new_query_embeddings.reshape(-1, emb_size)

    # update doc embeddings when re-training the doc encoder
    if training_args.training_mode == 'distill_jointly_v2':
        learnable_index.encode(data_dir=data_args.preprocess_dir,
                               prefix='docs',
                               max_length=data_args.max_doc_length,
                               output_dir=data_args.output_dir,
                               batch_size=8196,
                               is_query=False,
                               )
        print(f'{data_args.output_dir}/docs.memmap')
        doc_embeddings = np.memmap(f'{data_args.output_dir}/docs.memmap', dtype=np.float32,
                                         mode="r")
        doc_embeddings = doc_embeddings.reshape(-1, emb_size)

    # update index
    print('Updating the index with new ivf and pq')
    learnable_index.update_index_with_ckpt(ckpt_path=ckpt_path, doc_embeddings=doc_embeddings)

    # Test
    ground_truths = load_rel(os.path.join(data_args.preprocess_dir, 'dev-rels.tsv'))
    learnable_index.test(new_query_embeddings, ground_truths, topk=1000, batch_size=64,
                         MRR_cutoffs=[10, 100], Recall_cutoffs=[10, 30, 50, 100],
                         nprobe=index_args.nprobe)
