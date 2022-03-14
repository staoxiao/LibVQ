import argparse
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
from LibVQ.dataset.dataset import DatasetForVQ, DataCollatorForVQ
from LibVQ.inference import inference
from LibVQ.learnable_vq import LearnableVQ
from LibVQ.models import Encoder
from LibVQ.utils import setup_worker, setuplogging, dist_gather_tensor


class LearnableIndex(FaissIndex):
    def __init__(self,
                 index_method: str,
                 encoder: Type[Encoder] = None,
                 config: Type[IndexConfig] = None,
                 index_file: str = None,
                 emb_size: int = 768,
                 ivf_centers_num: int = 10000,
                 subvector_num: int = 32,
                 subvector_bits: int = 8,
                 dist_mode: str = 'ip',
                 doc_embeddings: np.ndarray = None
                 ):
        """

        :param index_method:
        :param encoder:
        :param config:
        :param index_file:
        :param emb_size:
        :param ivf_centers_num:
        :param subvector_num:
        :param subvector_bits:
        :param dist_mode:
        :param doc_embeddings:
        """
        super(LearnableIndex).__init__()

        if index_file is None or not os.path.exists(index_file):
            logging.info(f"generating the init index by faiss")
            self.index = FaissIndex(doc_embeddings=doc_embeddings,
                                    emb_size=emb_size,
                                    ivf_centers_num=ivf_centers_num,
                                    subvector_num=subvector_num,
                                    subvector_bits=subvector_bits,
                                    index_method=index_method,
                                    dist_mode=dist_mode)

            if index_file is None:
                index_file = f'./temp/{index_method}.index'
                os.makedirs('./temp', exist_ok=True)
            logging.info(f"save the init index to {index_file}")
            self.index.save_index(index_file)
        else:
            logging.info(f"loading the init index from {index_file}")
            self.index = faiss.read_index(index_file)

        self.learnable_vq = LearnableVQ(config, encoder=encoder, index_file=index_file, index_method=index_method)

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
        mp.spawn(LearnableIndex.fit,
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


    @staticmethod
    def fit(
            local_rank: int = -1,
            model: Type[LearnableVQ] = None,
            dataset: Type[DatasetForVQ] = None,
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
            checkpoint_path: str = None,
            checkpoint_save_steps: int = None,
            logging_steps: int = 100
    ):
        try:
            setuplogging()
            if dataset is None:
                dataset = DatasetForVQ(rel_data=rel_data,
                                       query_data_dir=query_data_dir,
                                       doc_data_dir=doc_data_dir,
                                       max_query_length=max_query_length,
                                       max_doc_length=max_doc_length,
                                       per_query_neg_num=per_query_neg_num,
                                       neg_data=neg_data,
                                       doc_embeddings=doc_embeddings,
                                       query_embeddings=query_embeddings,
                                       emb_size=emb_size)

            if world_size > 1:
                setup_worker(local_rank, world_size)
                sampler = DistributedSampler(dataset=dataset)
            else:
                sampler = RandomSampler(dataset)
            dataloader = DataLoader(dataset, sampler=sampler,
                                    batch_size=per_device_train_batch_size,
                                    collate_fn=DataCollatorForVQ())
            device = torch.device("cuda" if torch.cuda.is_available() else 'cpu',
                                  local_rank if local_rank >= 0 else 0)

            model = model.to(device)

            use_ivf = True if model.ivf is not None else False

            # Prepare optimizers
            num_train_steps = len(dataloader) * epochs
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            pq_parameter = ['rotate', 'codebook']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if
                            not any(nd in n for nd in no_decay) and not any(nd in n for nd in pq_parameter)],
                 'weight_decay': weight_decay,
                 "lr": lr_params['encoder_lr']},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and not any(nd in n for nd in pq_parameter)],
                 'weight_decay': 0.0,
                 "lr": lr_params['encoder_lr']},
                {
                    "params": [p for n, p in param_optimizer if any(nd in n for nd in pq_parameter)],
                    "weight_decay": 0.0,
                    "lr": lr_params['pq_lr']
                },
            ]

            optimizer = optimizer_class(optimizer_grouped_parameters)
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps= int(warmup_steps_ratio * num_train_steps), num_training_steps=num_train_steps
            )

            if world_size > 1:
                model = DDP(model,
                            device_ids=[local_rank],
                            output_device=local_rank,
                            find_unused_parameters=True)

            # train
            loss, dense_loss, ivf_loss, pq_loss = 0., 0., 0., 0.
            global_step = 0
            for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
                model.train()
                if world_size > 1:
                    sampler.set_epoch(epoch)
                for step, sample in enumerate(dataloader):
                    sample = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
                    batch_dense_loss, batch_ivf_loss, batch_pq_loss = model(temperature=temperature,
                                                                            loss_method=loss_method,
                                                                            fix_emb=fix_emb,
                                                                            world_size=world_size,
                                                                            cross_device_sample=cross_device_sample,
                                                                            **sample)

                    if loss_weight['ivf_weight'] == 'scaled_to_pqloss':
                        if use_ivf:
                            weight = batch_pq_loss / (batch_ivf_loss + 1e-6)
                            if world_size > 1:
                                weight = dist_gather_tensor(weight.unsqueeze(0), world_size=world_size)
                                weight = torch.mean(weight)
                            loss_weight['ivf_weight'] = weight.detach().float().item()
                            logging.info(f"ivf_weight = {loss_weight['ivf_weight']}")
                        else:
                            loss_weight['ivf_weight'] = 0.0

                    batch_loss = loss_weight['encoder_weight'] * batch_dense_loss + \
                                 loss_weight['pq_weight'] * batch_pq_loss + \
                                 loss_weight['ivf_weight'] * batch_ivf_loss

                    batch_loss.backward()
                    loss += batch_loss.item()
                    if not isinstance(batch_dense_loss, float):
                        dense_loss += loss_weight['encoder_weight'] * batch_dense_loss.item()
                    if not isinstance(batch_ivf_loss, float):
                        ivf_loss += loss_weight['ivf_weight'] * batch_ivf_loss.item()
                    if not isinstance(batch_pq_loss, float):
                        pq_loss += loss_weight['pq_weight'] * batch_pq_loss.item()

                    if max_grad_norm != -1:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                    optimizer.step()
                    scheduler.step()

                    if use_ivf:
                        if isinstance(model, DDP):
                            model.module.ivf.grad_accumulate(world_size=world_size)
                            model.module.ivf.update_centers(lr=lr_params['ivf_lr'])
                        else:
                            model.ivf.grad_accumulate(world_size=world_size)
                            model.ivf.update_centers(lr=lr_params['ivf_lr'])

                    optimizer.zero_grad()
                    if use_ivf:
                        if isinstance(model, DDP):
                            model.module.ivf.zero_grad()
                        else:
                            model.ivf.zero_grad()
                    global_step += 1

                    if global_step % logging_steps == 0:
                        step_num = logging_steps
                        logging.info(
                            '[{}] step:{}, train_loss: {:.5f} = '
                            'dense:{:.5f} + ivf:{:.5f} + pq:{:.5f}'.
                                format(local_rank,
                                       global_step,
                                       loss / step_num,
                                       dense_loss / step_num,
                                       ivf_loss / step_num,
                                       pq_loss / step_num))
                        loss, dense_loss, ivf_loss, pq_loss = 0., 0., 0., 0.

                    if checkpoint_save_steps:
                        if global_step % checkpoint_save_steps == 0 and (local_rank == 0 or local_rank == -1):
                            ckpt_path = os.path.join(checkpoint_path, f'epoch_{epoch}_step_{global_step}')
                            Path(ckpt_path).mkdir(parents=True, exist_ok=True)
                            if isinstance(model, DDP):
                                model.module.save(ckpt_path)
                            else:
                                model.save(ckpt_path)
                            logging.info(f"model saved to {ckpt_path}")


                if local_rank == 0 or local_rank == -1:
                    ckpt_path = os.path.join(checkpoint_path, f'epoch_{epoch}_step_{global_step}')
                    Path(ckpt_path).mkdir(parents=True, exist_ok=True)
                    if isinstance(model, DDP):
                        model.module.save(ckpt_path)
                    else:
                        model.save(ckpt_path)
                    logging.info(f"model saved to {ckpt_path}")
        except:
            error_type, error_value, error_trace = sys.exc_info()
            traceback.print_tb(error_trace)
            logging.info(error_value)
