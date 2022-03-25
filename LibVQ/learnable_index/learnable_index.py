import logging
import os
import pickle
import time
from typing import List, Dict, Type, Union
import shutil

import faiss
import numpy
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.optim import Optimizer
from transformers import AdamW

from LibVQ.base_index import FaissIndex, IndexConfig
from LibVQ.dataset import write_rel
from LibVQ.models import LearnableVQ
from LibVQ.train import train_model


class LearnableIndex(FaissIndex):
    def __init__(self,
                 index_method: str,
                 init_index_file: str = None,
                 ivf_centers_num: int = 10000,
                 subvector_num: int = 32,
                 subvector_bits: int = 8,
                 dist_mode: str = 'ip',
                 doc_embeddings: np.ndarray = None,
                 emb_size: int = 768,
                 config: IndexConfig = None
                 ):
        """
        finetune the index

        :param index_method: The type of index, e.g., ivf_pq, ivf_opq, pq, opq
        :param init_index_file: Create the learnable idex from the faiss index file; if is None, it will create a faiss index and save it
        :param ivf_centers_num: The number of post lists
        :param subvector_num: The number of codebooks
        :param subvector_bits: The number of codewords for each codebook
        :param dist_mode: Metric to calculate the distance between query and doc
        :param doc_embeddings: Embeddings of docs, needed when there is no a trained index in init_index_file
        :param emb_size: Dim of embeddings
        :param config: Config of index. Default is None.
        """
        super(LearnableIndex).__init__()

        if init_index_file is None or not os.path.exists(init_index_file):
            logging.info(f"generating the init index by faiss")
            self.index = FaissIndex(doc_embeddings=doc_embeddings,
                                    emb_size=emb_size,
                                    ivf_centers_num=ivf_centers_num,
                                    subvector_num=subvector_num,
                                    subvector_bits=subvector_bits,
                                    index_method=index_method,
                                    dist_mode=dist_mode)

            if init_index_file is None:
                init_index_file = f'./temp/{index_method}.index'
                os.makedirs('./temp', exist_ok=True)
            logging.info(f"save the init index to {init_index_file}")
            self.index.save_index(init_index_file)
        else:
            logging.info(f"loading the init index from {init_index_file}")
            self.index = faiss.read_index(init_index_file)

        self.learnable_vq = LearnableVQ(config, index_file=init_index_file, index_method=index_method)
        self.check_index_parameters(self.learnable_vq, ivf_centers_num, subvector_num, subvector_bits, init_index_file)

    def check_index_parameters(self,
                               vq_model: LearnableVQ,
                               ivf_centers_num: int,
                               subvector_num: int,
                               subvector_bits: int,
                               init_index_file: str):
        if ivf_centers_num is not None:
            if vq_model.ivf.ivf_centers_num != ivf_centers_num:
                raise ValueError(
                    f"The ivf_centers_num :{vq_model.ivf.ivf_centers_num} of index from {init_index_file} is not equal to you set: {ivf_centers_num}. "
                    f"please use the correct saved index or set it None to create a new faiss index")
        if subvector_num is not None:
            if vq_model.pq.subvector_num != subvector_num:
                raise ValueError(
                    f"The subvector_num :{vq_model.pq.subvector_num} of index from {init_index_file} is not equal to you set: {subvector_num}. "
                    f"please use the correct saved index or set it None to create a new faiss index")
        if subvector_bits is not None:
            if vq_model.pq.subvector_bits != subvector_bits:
                raise ValueError(
                    f"The subvector_bits :{vq_model.pq.subvector_bits} of index from {init_index_file} is not equal to you set: {subvector_bits}. "
                    f"please use the correct saved index or set it None to create a new faiss index")

    def update_ivf(self,
                   center_vecs: numpy.ndarray):
        if isinstance(self.index, faiss.IndexPreTransform):
            ivf_index = faiss.downcast_index(self.index.index)
            coarse_quantizer = faiss.downcast_index(ivf_index.quantizer)
        else:
            coarse_quantizer = faiss.downcast_index(self.index.quantizer)

        faiss.copy_array_to_vector(
            center_vecs.ravel(),
            coarse_quantizer.xb)

    def update_pq(self,
                  codebook: numpy.ndarray,
                  doc_embeddings: numpy.ndarray):
        if isinstance(self.index, faiss.IndexPreTransform):
            ivf_index = faiss.downcast_index(self.index.index)
            faiss.copy_array_to_vector(
                codebook.ravel(),
                ivf_index.pq.centroids)
        else:
            faiss.copy_array_to_vector(
                codebook.ravel(),
                self.index.pq.centroids)

        logging.info(f"updating the quantized results of docs' embeddings")
        self.index.remove_ids(faiss.IDSelectorRange(0, len(doc_embeddings)))
        self.index.add(doc_embeddings)

    def update_index_with_ckpt(self,
                               ckpt_path: str = None,
                               saved_ckpts_path: str = None,
                               doc_embeddings: numpy.ndarray = None):
        if ckpt_path is None:
            assert saved_ckpts_path is not None
            ckpt_path = self.get_latest_ckpt(saved_ckpts_path)

        logging.info(f"updating index based on {ckpt_path}")

        ivf_file = os.path.join(ckpt_path, 'ivf_centers.npy')
        if os.path.exists(ivf_file):
            logging.info(f"loading ivf centers from {ivf_file}")
            center_vecs = np.load(ivf_file)
            self.update_ivf(center_vecs)

        codebook_file = os.path.join(ckpt_path, 'codebook.npy')
        if os.path.exists(codebook_file):
            logging.info(f"loading codebook from {codebook_file}")
            codebook = np.load(codebook_file)
            self.update_pq(codebook=codebook, doc_embeddings=doc_embeddings)

    def get_latest_ckpt(self, saved_ckpts_path: str):
        if len(os.listdir(saved_ckpts_path)) == 0: raise IOError(f"There is no ckpt in path: {saved_ckpts_path}")

        latest_epoch, latest_step = 0, 0
        for ckpt in os.listdir(saved_ckpts_path):
            if 'epoch' in ckpt and 'step' in ckpt:
                name = ckpt.split('_')
                epoch, step = int(name[1]), int(name[3])
                if epoch > latest_epoch:
                    latest_epoch, latest_step = epoch, step
                elif epoch == latest_epoch:
                    latest_step = max(latest_step, step)
        assert latest_epoch > 0 or latest_step > 0
        return os.path.join(saved_ckpts_path, f"epoch_{latest_epoch}_step_{latest_step}")


    def get_temp_checkpoint_save_path(self):
        time_str = time.strftime('%m_%d-%H-%M-%S', time.localtime(time.time()))
        return f'./temp/{time_str}'

    def load_embedding(self,
                       emb: Union[str, numpy.ndarray],
                       emb_size: int):
        if isinstance(emb, str):
            assert 'npy' in emb or 'memmap' in emb
            if 'memmap' in emb:
                embeddings = np.memmap(emb, dtype=np.float32, mode="r")
                return embeddings.reshape(-1, emb_size)
            elif 'npy' in emb:
                return np.load(emb)
        else:
            return emb

    def fit(self,
            query_embeddings: Union[str, numpy.ndarray],
            doc_embeddings: Union[str, numpy.ndarray],
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
            loss_method: str = 'distill',
            weight_decay: float = 0.01,
            max_grad_norm: float = -1,
            show_progress_bar: bool = True,
            checkpoint_path: str = None,
            checkpoint_save_steps: int = None,
            logging_steps: int = 100,
            ):
        """
        Train the index

        :param query_embeddings: Embeddings for each query, also support pass a file('.npy', '.memmap').
        :param doc_embeddings: Embeddigns for each doc, also support pass a filename('.npy', '.memmap').
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
        :param loss_method: We provide two loss: 'contrastive' and 'distill'
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
                                                                 nprobe=self.learnable_vq.ivf.ivf_centers_num)

        train_model(model=self.learnable_vq,
                    rel_data=rel_data,
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
                    loss_method=loss_method,
                    fix_emb=fix_emb,
                    weight_decay=weight_decay,
                    max_grad_norm=max_grad_norm,
                    show_progress_bar=show_progress_bar,
                    checkpoint_path=checkpoint_path,
                    checkpoint_save_steps=checkpoint_save_steps,
                    logging_steps=logging_steps
                    )

        # update index
        self.update_index_with_ckpt(saved_ckpts_path=checkpoint_path,
                                    doc_embeddings=doc_embeddings)

        # delete temp folder
        if temp_checkpoint_path is not None:
            shutil.rmtree(temp_checkpoint_path)


    def fit_with_multi_gpus(
            self,
            query_embeddings_file: str,
            doc_embeddings_file: str,
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
            loss_method: str = 'distill',
            weight_decay: float = 0.01,
            max_grad_norm: float = -1,
            show_progress_bar: bool = True,
            checkpoint_path: str = None,
            checkpoint_save_steps: int = None,
            logging_steps: int = 100,
            master_port: str = '12345'
    ):
        """
        Train the VQ model with multi GPUs and update the index

        :param query_embeddings_file: Filename('.npy', '.memmap') to query embeddings.
        :param doc_embeddings_file: Filename('.npy', '.memmap') to doc embeddings.
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
        :param loss_method: We provide two loss: 'contrastive' and 'distill'
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
                                                                 nprobe=self.learnable_vq.ivf.ivf_centers_num)

            logging.info(f"saving relevance data to {checkpoint_path}...")
            rel_file = os.path.join(checkpoint_path, 'train-virtual_rel.tsv')
            neg_file = os.path.join(checkpoint_path, f"train-queries-virtual_hardneg.pickle")
            write_rel(rel_file, rel_data)
            pickle.dump(neg_data, open(neg_file, 'wb'))

        mp.spawn(train_model,
                 args=(self.learnable_vq,
                       None,
                       rel_file,
                       None,
                       None,
                       None,
                       None,
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
                       loss_method,
                       '',
                       weight_decay,
                       max_grad_norm,
                       show_progress_bar,
                       checkpoint_path,
                       checkpoint_save_steps,
                       logging_steps
                       ),
                 nprocs=world_size,
                 join=True)


        # update index
        self.update_index_with_ckpt(saved_ckpts_path=checkpoint_path,
                                    doc_embeddings=doc_embeddings)

        # delete temp folder
        if temp_checkpoint_path is not None:
            shutil.rmtree(temp_checkpoint_path)