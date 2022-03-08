import os
import torch
import numpy as np
import faiss

from LibVQ.index.FaissIndex import FaissIndex
from LibVQ.inference import inference
from LibVQ.models.encoder import Encoder, EncoderConfig
from LibVQ.dataset.dataset import load_rel
from LibVQ.learnable_index import LearnableIndex


if __name__ == '__main__':

    ckpt_path = '../../saved_ckpts/AR_G/0'
    data_dir = './data/passage/preprocess_bert-base-uncased'
    output_dir = './data/passage/evaluate/AR_G_0'
    index_method = 'ivf_opq'

    # Train encoder

    # Generate the embeddings of queries and docs
    os.makedirs(output_dir, exist_ok=True)
    config = EncoderConfig.from_pretrained(os.path.join(ckpt_path, 'config.json'))
    text_encoder = Encoder(config)
    text_encoder.load_state_dict(torch.load(os.path.join(ckpt_path, 'encoder.bin')))
    emb_size = text_encoder.output_embedding_size
    inference(data_dir=data_dir,
              is_query=False,
              encoder=text_encoder,
              prefix=f'docs',
              max_length=256,
              output_dir=output_dir,
              batch_size=10240)
    inference(data_dir=data_dir,
              is_query=True,
              encoder=text_encoder,
              prefix=f'dev-queries',
              max_length=32,
              output_dir=output_dir,
              batch_size=10240)
    inference(data_dir=data_dir,
              is_query=True,
              encoder=text_encoder,
              prefix=f'train-queries',
              max_length=32,
              output_dir=output_dir,
              batch_size=10240)

    # Creat Faiss IVFOPQ index
    doc_embeddings = np.memmap(os.path.join(output_dir, 'docs.memmap'),
                               dtype=np.float32, mode="r")
    doc_embeddings = doc_embeddings.reshape(-1, emb_size)
    query_embeddings = np.memmap(os.path.join(output_dir, 'dev-queries.memmap'),
                                 dtype=np.float32, mode="r")
    query_embeddings = query_embeddings.reshape(-1, emb_size)
    faiss.omp_set_num_threads(32)
    index = FaissIndex(index_method='ivf_opq',
                       emb_size=text_encoder.output_embedding_size,
                       ivf_centers=10000,
                       subvector_num=32,
                       subvector_bits=8,
                       dist_mode='ip')
    print('Training the index with doc embeddings')
    # index.fit(doc_embeddings)
    # index.add(doc_embeddings)
    # index.save_index(os.path.join(output_dir, f'{index_method}.index'))
    index.load_index(os.path.join(output_dir, f'{index_method}.index'))
    index.set_nprobe(10)

    # Test the performance
    ground_truths = load_rel(os.path.join(data_dir, 'dev-rels.tsv'))
    qids = list(range(len(query_embeddings)))
    index.test(query_embeddings, qids, ground_truths, topk=1000, batch_size=64,
               MRR_cutoffs=[5, 100], Recall_cutoffs=[10, 30, 50, 100])


    # Creat LearnableIndex
    learnable_index = LearnableIndex(trained_encoder_ckpt=ckpt_path,
                                     index_file=os.path.join(output_dir, f'{index_method}.index'))

    # Train LearnableIndex
    learnable_index.fit_with_multi_gpus(data_dir=data_dir,
                                        max_query_length=32,
                                        max_doc_length=128,
                                        query_embeddings_file = './data/passage/evaluate/AR_G_0/train-queries.memmap',
                                        doc_embeddings_file = './data/passage/evaluate/AR_G_0/docs.memmap',
                                        checkpoint_path='./saved_ckpts/test_model',
                                        per_device_train_batch_size=256,
                                        epochs=10,
                                        loss_weight = {'encoder_weight': 1.0, 'pq_weight': 1.0, 'ivf_weight': 0.012})

    # Update LearnableIndex and query embeddings
    ckpt_path = 'epoch_19_step_9840'
    learnable_index.update_encoder(f'./saved_ckpts/test_model/{ckpt_path}/encoder.bin')
    learnable_index.encode(data_dir=data_dir,
                           prefix='dev-query',
                           max_length=32,
                           output_dir=f'./data/passage/evaluate/test_model_{ckpt_path}',
                           batch_size=8196,
                           is_query=True)
    query_embeddings = np.memmap(f'./data/passage/evaluate/try_{ckpt_path}/dev-queries.memmap', dtype=np.float32, mode="r")
    query_embeddings = query_embeddings.reshape(-1, 768)

    print('Updating the index with new ivf and pq')
    learnable_index.update_index_with_ckpt(ckpt_path=f'./saved_ckpts/try/{ckpt_path}/', doc_embeddings=doc_embeddings)

    # Test
    learnable_index.set_nprobe(10)
    qids = list(range(len(query_embeddings)))
    learnable_index.test(query_embeddings, qids, ground_truths, topk=1000, batch_size=64,
                   MRR_cutoffs=[10, 100], Recall_cutoffs=[10, 30, 50, 100])