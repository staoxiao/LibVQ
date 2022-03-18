import faiss
import logging
import numpy
import numpy as np
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import traceback
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.data import RandomSampler, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.autonotebook import trange
from transformers import AdamW, get_linear_schedule_with_warmup, AutoConfig
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional

from LibVQ.baseindex import FaissIndex
from LibVQ.baseindex import IndexConfig
from LibVQ.dataset import DatasetForVQ, DataCollatorForVQ
from LibVQ.inference import inference
from LibVQ.learnable_vq import LearnableVQ
from LibVQ.models import Encoder
from LibVQ.utils import setup_worker, setuplogging, dist_gather_tensor
from LibVQ.train import train_model

class LearnableIndex(FaissIndex):
    def __init__(self,
                 index_method: str,
                 encoder: Encoder = None,
                 config: IndexConfig = None,
                 init_index_file: str = None,
                 emb_size: int = 768,
                 ivf_centers_num: int = 10000,
                 subvector_num: int = 32,
                 subvector_bits: int = 8,
                 dist_mode: str = 'ip',
                 doc_embeddings: np.ndarray = None
                 ):
        """
        finetune the index and encoder
        :param index_method: the type of index, e.g., ivf_pq, pq, opq
        :param encoder: the encoder for query and doc
        :param config: config of index
        :param init_index_file: the faiss index file, if is None, it will create a faiss index and save it
        :param emb_size: dim of embeddings
        :param ivf_centers_num: the number of post lists
        :param subvector_num: the number of codebooks
        :param subvector_bits: the number of codewords for each codebook
        :param dist_mode: metric to calculate the distance between query and doc
        :param doc_embeddings: embeddings of docs, needed when there is no a trained index in init_index_file
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
                index_file = f'./temp/{index_method}.index'
                os.makedirs('./temp', exist_ok=True)
            logging.info(f"save the init index to {init_index_file}")
            self.index.save_index(init_index_file)
        else:
            logging.info(f"loading the init index from {init_index_file}")
            self.index = faiss.read_index(init_index_file)

        self.learnable_vq = LearnableVQ(config, encoder=encoder, index_file=init_index_file, index_method=index_method)

    def update_encoder(self, encoder_file=None, saved_ckpts_path=None):
        if encoder_file is None:
            assert saved_ckpts_path is not None
            ckpt_path = self.get_latest_ckpt(saved_ckpts_path)
            encoder_file = os.path.join(ckpt_path, 'encoder.bin')

        self.learnable_vq.encoder.load_state_dict(torch.load(encoder_file, map_location='cpu'))

    def update_ivf(self, center_vecs):
        if isinstance(self.index, faiss.IndexPreTransform):
            ivf_index = faiss.downcast_index(self.index.index)
            coarse_quantizer = faiss.downcast_index(ivf_index.quantizer)
        else:
            coarse_quantizer = faiss.downcast_index(self.index.quantizer)

        faiss.copy_array_to_vector(
            center_vecs.ravel(),
            coarse_quantizer.xb)

    def update_pq(self, codebook, doc_embeddings):
        if isinstance(self.index, faiss.IndexPreTransform):
            ivf_index = faiss.downcast_index(self.index.index)
            faiss.copy_array_to_vector(
                codebook.ravel(),
                ivf_index.pq.centroids)
        else:
            faiss.copy_array_to_vector(
                codebook.ravel(),
                self.index.pq.centroids)

        self.index.remove_ids(faiss.IDSelectorRange(0, len(doc_embeddings)))
        self.index.add(doc_embeddings)

    def update_index_with_ckpt(self, ckpt_path=None, saved_ckpts_path=None, doc_embeddings=None):
        if ckpt_path is None:
            assert saved_ckpts_path is not None
            ckpt_path = self.get_latest_ckpt(saved_ckpts_path)

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

    def get_latest_ckpt(self, saved_ckpts_path):
        if len(os.listdir(saved_ckpts_path)) == 0: raise IOError(f"There is no ckpt in path: {saved_ckpts_path}")

        latest_epoch, latest_step = 0, 0
        for ckpt in os.listdir(saved_ckpts_path):
            name = ckpt.split('_')
            epoch, step = int(name[1]), int(name[3])
            if epoch > latest_epoch:
                latest_epoch, latest_step = epoch, step
            elif epoch == latest_epoch:
                latest_step = max(latest_step, step)
        return os.path.join(saved_ckpts_path, f"epoch_{latest_epoch}_step_{latest_step}")

    def encode(self,
               data_dir: str,
               prefix: str,
               max_length: int,
               output_dir: str,
               batch_size: int,
               is_query: bool,
               return_vecs: bool = False):
        os.makedirs(output_dir, exist_ok=True)
        vecs = inference(data_dir=data_dir,
                  is_query=is_query,
                  encoder=self.learnable_vq.encoder,
                  prefix=prefix,
                  max_length=max_length,
                  output_dir=output_dir,
                  batch_size=batch_size,
                  return_vecs=return_vecs)
        return vecs


    def fit_with_multi_gpus(
            self,
            rel_file: str = None,
            query_data_dir: str = None,
            max_query_length: int = 32,
            doc_data_dir: str = None,
            max_doc_length: int = None,
            epochs: int = 5,
            per_device_train_batch_size: int = 128,
            per_query_neg_num: int = 1,
            cross_device_sample: bool = True,
            neg_file: str = None,
            query_embeddings_file: str = None,
            doc_embeddings_file: str = None,
            emb_size: int = None,
            warmup_steps_ratio: float = 0.1,
            optimizer_class: Type[Optimizer] = AdamW,
            lr_params: Dict[str, float] = {'encoder_lr': 1e-5, 'pq_lr': 1e-4, 'ivf_lr': 1e-3},
            loss_weight: Dict[str, object] = {'encoder_weight': 1.0, 'pq_weight': 1.0,
                                              'ivf_weight': 'scaled_to_pqloss'},
            temperature: float = 1.0,
            loss_method: str = 'distill',
            fix_emb: str = 'doc',
            weight_decay: float = 0.01,
            max_grad_norm: float = -1,
            show_progress_bar: bool = True,
            checkpoint_path: str = None,
            checkpoint_save_steps: int = None,
            logging_steps: int = 100,
            master_port: str = '12345'
    ):

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = master_port
        world_size = torch.cuda.device_count()
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
                       loss_method,
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

        self.update_index_with_ckpt(checkpoint_path, doc_embeddings)

    def fit(self,
            local_rank: int = -1,
            model: LearnableVQ = None,
            dataset: DatasetForVQ = None,
            rel_data: Union[str, Dict[int, List[int]]] = None,
            query_data_dir: str = None,
            max_query_length: int = 32,
            doc_data_dir: str = None,
            max_doc_length: int = None,
            world_size: int = 1,
            epochs: int = 5,
            per_device_train_batch_size: int = 128,
            per_query_neg_num: int = 1,
            cross_device_sample: bool = True,
            neg_data: Union[str, Dict[int, List[int]]] = None,
            query_embeddings: Union[str, numpy.ndarray] = None,
            doc_embeddings: Union[str, numpy.ndarray] = None,
            emb_size: int = None,
            warmup_steps_ratio: float = 0.1,
            optimizer_class: Type[Optimizer] = AdamW,
            lr_params: Dict[str, float] = {'encoder_lr': 1e-5, 'pq_lr': 1e-4, 'ivf_lr': 1e-3},
            loss_weight: Dict[str, object] = {'encoder_weight': 1.0, 'pq_weight': 1.0,
                                              'ivf_weight': 'scaled_to_pqloss'},
            temperature: float = 1.0,
            loss_method: str = 'distill',
            fix_emb: str = 'doc',
            weight_decay: float = 0.01,
            max_grad_norm: float = -1,
            show_progress_bar: bool = True,
            checkpoint_path: str = './temp/',
            checkpoint_save_steps: int = None,
            logging_steps: int = 100,
            **kwargs
    ):


        train_model(kwargs)
        self.update_index_with_ckpt(checkpoint_path, doc_embeddings)




    def prepare_data(self,
                     dataset_dir,
                     preprocess_dir,
                     max_query_length,
                     max_doc_length,
                     rel_data,
                     query_embeddings,
                     doc_embeddings,
                     output_dir,
                     ):

        if dataset_dir:


        if emb:
            inference()

        if rel_data is None:
            self.generate_virtual_traindata(query_embeddings=query_embeddings,
                                            topk = 400,
                                            batch_size = None,
                                            nprobe = None)

        preprocess_data(data_dir=dataset_dir,
                        output_dir=preprocess_dir,
                        tokenizer_name=pretrained_model_name,
                        max_doc_length=max_doc_length,
                        max_query_length=max_query_length,
                        workers_num=64)

        inference(data_dir=preprocess_dir,
                  is_query=is_query,
                  encoder=encoder,
                  prefix=f'docs',
                  max_length=max_length,
                  output_dir=output_dir,
                  batch_size=10240)

