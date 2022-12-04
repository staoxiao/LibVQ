import logging
import os
import pickle
import shutil
from typing import List, Dict, Type, Union

import numpy
import torch
import torch.multiprocessing as mp
from torch.optim import AdamW, Optimizer

from LibVQ.train import train_model
from LibVQ.dataset import write_rel
from LibVQ.learnable_index import LearnableIndexWithEncoder
from LibVQ.base_index import IndexConfig
from LibVQ.models import EncoderConfig
from LibVQ.dataset import Datasets


class DistillLearnableIndexWithEncoder(LearnableIndexWithEncoder):
    def __init__(self,
                 index_config: IndexConfig = None,
                 encoder_config: EncoderConfig = None,
                 init_index_file: str = None,
                 init_pooler_file: str = None
                 ):
        """
        finetune the distill index

        :param index_config: Config of index. Default is None.
        :param encoder_config: Config of Encoder. Default is None.
        :param init_index_file: Create the learnable idex from the faiss index file; if is None, it will create a faiss index and save it
        :param init_pooler_file: Create the pooler layer from the faiss index file
        """
        LearnableIndexWithEncoder.__init__(self, index_config,
                                           encoder_config,
                                           init_index_file,
                                           init_pooler_file)

    def train(self,
              data: Datasets = None,
              per_query_neg_num: int = 1,
              save_ckpt_dir: str = None,
              logging_steps: int = 100,
              per_device_train_batch_size: int = 512,
              loss_weight: Dict[str, object] = {'encoder_weight': 1.0, 'pq_weight': 1.0,
                                                'ivf_weight': 'scaled_to_pqloss'},
              lr_params: Dict[str, object] = {'encoder_lr': 1e-5, 'pq_lr': 1e-4, 'ivf_lr': 1e-3},
              epoch: int = 10
              ):
        # if self.index_config.emb_size is not None and data.emb_size is not None:
        #     if self.index_config.emb_size != data.emb_size:
        #         raise ValueError("Your index config emb_size is not equal to your dataset emb_size")

        if not self.model:
            raise ValueError("Due to the lack of encoder, you can't train encoder")
        elif self.encoder_config.is_finetune is False:
            raise ValueError("Due to your encoder is not finetune, you can't use distill")

        if data.docs_path is None:
            raise ValueError("Due to the lack of docs file, you can't train encoder")

        self.model.encode(datasets=data)
        if self.is_trained is False:
            self.faiss_train(data)
        if self.index_config.index_method != 'flat':
            world_size = torch.cuda.device_count()
            if world_size <= 1:
                self.fit(rel_data=data.train_rels_path,
                         query_embeddings=data.train_queries_embedding_dir,
                         doc_embeddings=data.doc_embeddings_dir,
                         query_data_dir=data.preprocess_dir,
                         max_query_length=data.max_query_length,
                         doc_data_dir=data.preprocess_dir,
                         max_doc_length=data.max_doc_length,
                         emb_size=self.model.encoder.query_encoder.encoder.config.hidden_size,
                         per_query_neg_num=per_query_neg_num,
                         checkpoint_path=save_ckpt_dir,
                         logging_steps=logging_steps,
                         per_device_train_batch_size=per_device_train_batch_size,
                         loss_weight=loss_weight,
                         lr_params=lr_params,
                         epochs=epoch)
            else:
                self.fit_with_multi_gpus(rel_file=data.train_rels_path,
                                         query_embeddings_file=data.train_queries_embedding_dir,
                                         doc_embeddings_file=data.doc_embeddings_dir,
                                         query_data_dir=data.preprocess_dir,
                                         max_query_length=data.max_query_length,
                                         doc_data_dir=data.preprocess_dir,
                                         max_doc_length=data.max_doc_length,
                                         emb_size=self.model.encoder.query_encoder.encoder.config.hidden_size,
                                         per_query_neg_num=per_query_neg_num,
                                         checkpoint_path=save_ckpt_dir,
                                         logging_steps=logging_steps,
                                         per_device_train_batch_size=per_device_train_batch_size,
                                         loss_weight=loss_weight,
                                         lr_params=lr_params,
                                         epochs=epoch)
        if data.dev_queries_path is not None:
            os.remove(data.dev_queries_embedding_dir)
            self.encode(data_dir=data.preprocess_dir,
                        prefix='dev-queries',
                        max_length=data.max_query_length,
                        output_dir=data.embedding_dir,
                        batch_size=2048,
                        is_query=True,
                        return_vecs=True
                        )

    def fit(self,
            query_embeddings: Union[str, numpy.ndarray] = None,
            doc_embeddings: Union[str, numpy.ndarray] = None,
            query_data_dir: str = None,
            max_query_length: int = 32,
            doc_data_dir: str = None,
            max_doc_length: int = None,
            rel_data: Union[str, Dict[int, List[int]]] = None,
            neg_data: Union[str, Dict[int, List[int]]] = None,
            epochs: int = 5,
            per_device_train_batch_size: int = 128,
            per_query_neg_num: int = 1,
            emb_size: int = None,
            warmup_steps_ratio: float = 0.1,
            optimizer_class: Type[Optimizer] = AdamW,
            lr_params: Dict[str, float] = {'encoder_lr': 1e-5, 'pq_lr': 1e-4, 'ivf_lr': 1e-3},
            loss_weight: Dict[str, object] = {'encoder_weight': 1.0, 'pq_weight': 1.0,
                                              'ivf_weight': 'scaled_to_pqloss'},
            temperature: float = 1.0,
            fix_emb: str = 'doc',
            weight_decay: float = 0.01,
            max_grad_norm: float = -1,
            show_progress_bar: bool = True,
            checkpoint_path: str = None,
            checkpoint_save_steps: int = None,
            logging_steps: int = 100,
            ):
        """
        Train the index and encoder

        :param query_embeddings: Embeddings for each query, also support pass a file('.npy', '.memmap').
        :param doc_embeddings: Embeddigns for each doc, also support pass a filename('.npy', '.memmap').
        :param query_data_dir: Path to the preprocessed tokens data (needed for jointly training query encoder).
        :param max_query_length: Max length of query tokens sequence.
        :param doc_data_dir: Path to the preprocessed tokens data (needed for jointly training doc encoder).
        :param max_doc_length: Max length of doc tokens sequence.
        :param rel_data: Positive doc ids for each query: {query_id:[doc_id1, doc_id2,...]}, or a tsv file which save the relevance relationship: qeury_id \t doc_id \n.
                         If set None, it will automatically generate the data for training based on the retrieval results.
        :param neg_data: Negative doc ids for each query: {query_id:[doc_id1, doc_id2,...]}, or a pickle file which save the query2neg.
                         If set None, it will randomly sample negative.
        :param epochs: The epochs of training
        :param per_device_train_batch_size: The number of query-doc positive pairs in a batch
        :param per_query_neg_num: The number of negatives for each query
        :param emb_size: Dim of embeddings.
        :param warmup_steps_ratio: The ration of warmup steps
        :param optimizer_class: torch.optim.Optimizer
        :param lr_params: Learning rate for encoder, ivf, and pq
        :param loss_weight: Wight for loss of encoder, ivf, and pq. "scaled_to_pqloss"" means that make the weighted loss closed to the loss of pq module.
        :param temperature: Temperature for softmax
        :param fix_emb: Fix the embeddings of query or doc. 'doc' means to fix the embeddings of doc; 'query' means to fix the embeddings of query; 'query,doc' means to  fix both embeddings.
        :param weight_decay: Hyper-parameter for Optimizer
        :param max_grad_norm: Used for gradient normalization
        :param checkpoint_path: Folder to save checkpoints during training. If set None, it will create a temp folder.
        :param checkpoint_save_steps: Will save a checkpoint after so many steps
        :param logging_steps: Will show the loss information after so many steps
        :return:
        """

        temp_checkpoint_path = None
        if checkpoint_path is None:
            temp_checkpoint_path = self.get_temp_checkpoint_save_path()
            logging.info(f"The model will be saved into {temp_checkpoint_path}")
            checkpoint_path = temp_checkpoint_path

        query_embeddings = self.load_embedding(query_embeddings, emb_size=emb_size)
        doc_embeddings = self.load_embedding(doc_embeddings, emb_size=emb_size)

        if rel_data is None:
            # generate train data
            logging.info("generating relevance data...")
            rel_data, neg_data = self.generate_virtual_traindata(query_embeddings=query_embeddings, topk=400,
                                                                 nprobe=self.ivf_centers_num)

        train_model(model=self.learnable_vq,
                    rel_data=rel_data,
                    query_data_dir=query_data_dir,
                    max_query_length=max_query_length,
                    doc_data_dir=doc_data_dir,
                    max_doc_length=max_doc_length,
                    epochs=epochs,
                    per_device_train_batch_size=per_device_train_batch_size,
                    per_query_neg_num=per_query_neg_num,
                    neg_data=neg_data,
                    query_embeddings=query_embeddings,
                    doc_embeddings=doc_embeddings,
                    emb_size=emb_size,
                    warmup_steps_ratio=warmup_steps_ratio,
                    optimizer_class=optimizer_class,
                    lr_params=lr_params,
                    loss_weight=loss_weight,
                    temperature=temperature,
                    loss_method='distill',
                    fix_emb=fix_emb,
                    weight_decay=weight_decay,
                    max_grad_norm=max_grad_norm,
                    show_progress_bar=show_progress_bar,
                    checkpoint_path=checkpoint_path,
                    checkpoint_save_steps=checkpoint_save_steps,
                    logging_steps=logging_steps
                    )
        if fix_emb is None or 'query' not in fix_emb or 'doc' not in fix_emb:
            # update encoder
            assert self.learnable_vq.encoder is not None
            self.update_encoder(saved_ckpts_path=checkpoint_path)

        if fix_emb is None or 'doc' not in fix_emb:
            # update doc_embeddings
            logging.info(f"updating doc embeddings and saving it to {checkpoint_path}")
            doc_embeddings = self.encode(data_dir=doc_data_dir,
                                         prefix='docs',
                                         max_length=max_doc_length,
                                         output_dir=checkpoint_path,
                                         batch_size=8196,
                                         is_query=False,
                                         return_vecs=True
                                         )

        # update index
        if self.index is not None:
            self.update_index_with_ckpt(saved_ckpts_path=checkpoint_path,
                                        doc_embeddings=doc_embeddings)

        # delete temp folder
        if temp_checkpoint_path is not None:
            shutil.rmtree(temp_checkpoint_path)

    def fit_with_multi_gpus(
            self,
            query_embeddings_file: str = None,
            doc_embeddings_file: str = None,
            query_data_dir: str = None,
            max_query_length: int = 32,
            doc_data_dir: str = None,
            max_doc_length: int = None,
            rel_file: str = None,
            neg_file: str = None,
            epochs: int = 5,
            per_device_train_batch_size: int = 128,
            per_query_neg_num: int = 1,
            cross_device_sample: bool = True,
            emb_size: int = None,
            warmup_steps_ratio: float = 0.1,
            optimizer_class: Type[Optimizer] = AdamW,
            lr_params: Dict[str, float] = {'encoder_lr': 1e-5, 'pq_lr': 1e-4, 'ivf_lr': 1e-3},
            loss_weight: Dict[str, object] = {'encoder_weight': 1.0, 'pq_weight': 1.0,
                                              'ivf_weight': 'scaled_to_pqloss'},
            temperature: float = 1.0,
            fix_emb: str = 'doc',
            weight_decay: float = 0.01,
            max_grad_norm: float = -1,
            show_progress_bar: bool = True,
            checkpoint_path: str = None,
            checkpoint_save_steps: int = None,
            logging_steps: int = 100,
            master_port: str = '12345'
    ):
        """
        Train the index and encoder with multi GPUs

        :param query_embeddings_file: Filename('.npy', '.memmap') to query embeddings.
        :param doc_embeddings_file: Filename('.npy', '.memmap') to doc embeddings.
        :param query_data_dir: Path to the preprocessed tokens data (needed for jointly training query encoder).
        :param max_query_length: Max length of query tokens sequence.
        :param doc_data_dir: Path to the preprocessed tokens data (needed for jointly training doc encoder).
        :param max_doc_length: Max length of doc tokens sequence.
        :param rel_file: A tsv file which save the relevance relationship: qeury_id \t doc_id \n.
                         If set None, it will automatically generate the data for training based on the retrieval results.
        :param neg_file: A pickle file which save the query2neg. if set None, it will randomly sample negative.
                         If set None, it will randomly sample negative.
        :param epochs: The epochs of training
        :param per_device_train_batch_size: The number of query-doc positive pairs in a batch
        :param per_query_neg_num: The number of negatives for each query
        :param emb_size: Dim of embeddings.
        :param warmup_steps_ratio: The ration of warmup steps
        :param optimizer_class: torch.optim.Optimizer
        :param lr_params: Learning rate for encoder, ivf, and pq
        :param loss_weight: Wight for loss of encoder, ivf, and pq. "scaled_to_pqloss"" means that make the weighted loss closed to the loss of pq module.
        :param temperature: Temperature for softmax
        :param fix_emb: Fix the embeddings of query or doc. 'doc' means to fix the embeddings of doc; 'query' means to fix the embeddings of query; 'query,doc' means to  fix both embeddings.
        :param weight_decay: Hyper-parameter for Optimizer
        :param max_grad_norm: Used for gradient normalization
        :param checkpoint_path: Folder to save checkpoints during training. If set None, it will create a temp folder.
        :param checkpoint_save_steps: Will save a checkpoint after so many steps
        :param logging_steps: Will show the loss information after so many steps
        :param master_port: setting for distributed training
        :return:
        """
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = master_port
        world_size = torch.cuda.device_count()

        temp_checkpoint_path = None
        if checkpoint_path is None:
            temp_checkpoint_path = self.get_temp_checkpoint_save_path()
            logging.info(f"The model will be saved into {temp_checkpoint_path}")
            checkpoint_path = temp_checkpoint_path

        if rel_file is None:
            # generate train data
            logging.info("generating relevance data...")
            query_embeddings = self.load_embedding(query_embeddings_file, emb_size=emb_size)
            doc_embeddings = self.load_embedding(doc_embeddings_file, emb_size=emb_size)
            rel_data, neg_data = self.generate_virtual_traindata(query_embeddings=query_embeddings, topk=400,
                                                                 nprobe=self.ivf_centers_num)

            logging.info(f"saving relevance data to {checkpoint_path}...")
            rel_file = os.path.join(checkpoint_path, 'train-virtual_rel.tsv')
            neg_file = os.path.join(checkpoint_path, f"train-queries-virtual_hardneg.pickle")
            write_rel(rel_file, rel_data)
            pickle.dump(neg_data, open(neg_file, 'wb'))

        mp.spawn(train_model,
                 args=(self.learnable_vq,
                       None,
                       rel_file,
                       query_data_dir,
                       max_query_length,
                       doc_data_dir,
                       max_doc_length,
                       world_size,
                       epochs,
                       per_device_train_batch_size,
                       per_query_neg_num,
                       cross_device_sample,
                       neg_file,
                       query_embeddings_file,
                       doc_embeddings_file,
                       emb_size,
                       warmup_steps_ratio,
                       optimizer_class,
                       lr_params,
                       loss_weight,
                       temperature,
                       'distill',
                       fix_emb,
                       weight_decay,
                       max_grad_norm,
                       show_progress_bar,
                       checkpoint_path,
                       checkpoint_save_steps,
                       logging_steps
                       ),
                 nprocs=world_size,
                 join=True)

        if fix_emb is None or 'query' not in fix_emb or 'doc' not in fix_emb:
            # update encoder
            assert self.learnable_vq.encoder is not None
            self.update_encoder(saved_ckpts_path=checkpoint_path)

        if fix_emb is None or 'doc' not in fix_emb:
            # update doc_embeddings
            logging.info(f"updating doc embeddings and saving it to {checkpoint_path}")
            doc_embeddings = self.encode(data_dir=doc_data_dir,
                                         prefix='docs',
                                         max_length=max_doc_length,
                                         output_dir=checkpoint_path,
                                         batch_size=8196,
                                         is_query=False,
                                         return_vecs=True
                                         )
        else:
            assert 'npy' in doc_embeddings_file or 'memmap' in doc_embeddings_file
            doc_embeddings = self.load_embedding(doc_embeddings_file, emb_size)

        # update index
        if self.index is not None:
            self.update_index_with_ckpt(saved_ckpts_path=checkpoint_path,
                                        doc_embeddings=doc_embeddings)

        # delete temp folder
        if temp_checkpoint_path is not None:
            shutil.rmtree(temp_checkpoint_path)