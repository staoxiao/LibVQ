import sys
sys.path.append('./src')
import os
import logging
from tqdm.autonotebook import trange
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional
from pathlib import Path
import faiss
import numpy as np
import argparse
import traceback

import torch
from torch.optim import Optimizer
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AdamW, get_linear_schedule_with_warmup, AutoConfig

from models.encoder import Encoder
from learnable_vq import LearnableVQ
from index.FaissIndex import FaissIndex
from dataset.dataset import DatasetForVQ, DataCollatorForVQ
from dataset.dataset import load_rel
from inference import inference
from utils import setup_worker, setuplogging

class LearnableIndex(FaissIndex):
    def __init__(self,
                 pretrained_model_for_encoder: str = 'bert-base-uncased',
                 trained_encoder_ckpt: str = None,
                 use_two_encoder: bool = True,
                 sentence_pooling_method: str = 'first',
                 index_file : str = None
                 ):
        super(LearnableIndex).__init__()
        if trained_encoder_ckpt is not None:
            config = AutoConfig.from_pretrained(os.path.join(trained_encoder_ckpt, 'config.json'))
            encoder = Encoder(config)
            encoder.load_state_dict(torch.load(os.path.join(trained_encoder_ckpt, 'encoder.bin'), map_location='cpu'))
        else:
            config = AutoConfig.from_pretrained(pretrained_model_for_encoder)
            config.pretrained_model_name = pretrained_model_for_encoder
            config.use_two_encoder = use_two_encoder
            config.sentence_pooling_method = sentence_pooling_method
            encoder = None
        config.index_file = index_file

        self.learnable_vq = LearnableVQ(config, encoder=encoder)
        self.index = faiss.read_index(config.index_file)

    def update_encoder(self, encoder_file):
        self.learnable_vq.encoder.load_state_dict(torch.load(encoder_file, map_location='cpu'))

    def update_index_with_ckpt(self, ckpt_path, doc_embeddings=None):
        if os.path.exists(os.path.join(ckpt_path, 'ivf_centers')):
            logging.info(f"loading ivf centers from {os.path.join(ckpt_path, 'ivf_centers')}")
            center_vecs = np.load(os.path.join(ckpt_path, 'ivf_centers'))

            if isinstance(self.index, faiss.IndexPreTransform):
                ivf_index = faiss.downcast_index(self.index.index)
                coarse_quantizer = faiss.downcast_index(ivf_index.quantizer)
            else:
                coarse_quantizer = faiss.downcast_index(self.index.quantizer)

            faiss.copy_array_to_vector(
                center_vecs.ravel(),
                coarse_quantizer.xb)

        if os.path.exists(os.path.join(ckpt_path, 'codebook')):
            logging.info(f"loading codebook from {os.path.join(ckpt_path, 'codebook')}")
            codebook = np.load(os.path.join(ckpt_path, 'codebook'))

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

    def encode(self, data_dir, prefix, max_length, output_dir, batch_size, is_query):
        inference(data_dir=data_dir,
                  is_query=is_query,
                  encoder=self.learnable_vq.encoder,
                  prefix=prefix,
                  max_length=max_length,
                  output_dir=output_dir,
                  batch_size=batch_size)

    def fit_with_multi_gpus(
            self,
            data_dir: str = None,
            max_query_length: int = None,
            max_doc_length: int = None,
            epochs: int = 5,
            per_device_train_batch_size: int = 128,
            per_query_neg_num: int = 1,
            neg_file: str = None,
            query_embeddings_file: str = None,
            doc_embeddings_file: str = None,
            warmup_steps: int = 1000,
            optimizer_class: Type[Optimizer] = AdamW,
            lr_params: Dict[str, float] = {'encoder_lr': 1e-5, 'pq_lr':1e-4, 'ivf_lr':1e-3},
            loss_weight: Dict[str, float] = {'encoder_weight': 1.0, 'pq_weight': 9e-3, 'ivf_weight': 1.0},
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
                       data_dir,
                       max_query_length,
                       max_doc_length,
                       world_size,
                       epochs,
                       per_device_train_batch_size,
                       per_query_neg_num,
                       neg_file,
                       query_embeddings_file,
                       doc_embeddings_file,
                       warmup_steps,
                       optimizer_class,
                       lr_params,
                       loss_weight,
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
            data_dir: str = None,
            max_query_length: int = None,
            max_doc_length: int = None,
            world_size: int = 1,
            epochs: int = 5,
            per_device_train_batch_size: int = 128,
            per_query_neg_num: int = 1,
            neg_file: str = None,
            query_embeddings_file: str = None,
            doc_embeddings_file: str = None,
            warmup_steps_ratio: float = 0.1,
            optimizer_class: Type[Optimizer] = AdamW,
            lr_params: Dict[str, float] = {'encoder_lr': 1e-5, 'pq_lr':1e-4, 'ivf_lr':1e-3},
            loss_weight: Dict[str, float] = {'encoder_weight': 1.0, 'pq_weight': 9e-3, 'ivf_weight': 1.0},
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
                dataset = DatasetForVQ(data_dir=data_dir,
                                       max_query_length=max_query_length,
                                       max_doc_length=max_doc_length,
                                       rel_file=os.path.join(data_dir, 'train-qrel.tsv'),
                                       per_query_neg_num=per_query_neg_num,
                                       neg_file=neg_file,
                                       doc_embeddings_file=doc_embeddings_file,
                                       query_embeddings_file=query_embeddings_file)

            if world_size > 1:
                setup_worker(local_rank, world_size)
                assert data_dir is not None
                sampler = DistributedSampler(dataset=dataset)
            else:
                sampler = RandomSampler(dataset)
            dataloader = DataLoader(dataset, sampler=sampler,
                                batch_size=per_device_train_batch_size,
                                collate_fn=DataCollatorForVQ())
            device = torch.device("cuda" if torch.cuda.is_available() else 'cpu',
                                  local_rank if local_rank >= 0 else 0)

            model = model.to(device)
            if world_size > 1:
                model = DDP(model,
                                device_ids=[local_rank],
                                output_device=local_rank,
                                find_unused_parameters=True)

            # Prepare optimizers
            num_train_steps = len(dataloader)*epochs
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            pq_parameter = ['rotate', 'codebook']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and n not in pq_parameter],
                 'weight_decay': weight_decay,
                 "lr": lr_params['encoder_lr']},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and n not in pq_parameter],
                 'weight_decay': 0.0,
                 "lr": lr_params['encoder_lr']},
                {
                    "params": [p for n, p in param_optimizer if n in pq_parameter],
                    "weight_decay": 0.0,
                    "lr": lr_params['pq_lr']
                },
            ]
            optimizer = optimizer_class(optimizer_grouped_parameters)
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps_ratio*num_train_steps, num_training_steps=num_train_steps
            )

            # train
            loss, dense_loss, ivf_loss, pq_loss = 0., 0., 0., 0.
            global_step = 0
            for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
                optimizer.zero_grad()
                model.train()

                for step, sample in enumerate(dataloader):
                    sample = {k:v.to(device) if isinstance(v, torch.Tensor) else v for k,v in sample.items()}

                    batch_dense_loss, batch_ivf_loss, batch_pq_loss = model(**sample)
                    batch_loss = loss_weight['encoder_weight'] * batch_dense_loss + \
                                 loss_weight['pq_weight'] * batch_pq_loss + \
                                 loss_weight['ivf_weight'] * batch_ivf_loss
                    batch_loss.backward()
                    loss += batch_loss.item()
                    if not isinstance(batch_dense_loss, float):
                        dense_loss += batch_dense_loss.item()
                    if not isinstance(batch_ivf_loss, float):
                        ivf_loss += loss_weight['ivf_weight'] * batch_ivf_loss.item()
                    if not isinstance(batch_pq_loss, float):
                        pq_loss += loss_weight['pq_weight'] * batch_pq_loss.item()

                    if max_grad_norm != -1:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                    optimizer.step()
                    if isinstance(model, DDP):
                        model.module.ivf.update_centers(lr=lr_params['ivf_lr'])
                    else:
                        model.ivf.update_centers(lr=lr_params['ivf_lr'])

                    scheduler.step()
                    optimizer.zero_grad()
                    if isinstance(model, DDP):
                        model.module.ivf.zero_grad()
                    else:
                        model.ivf.zero_grad()
                    global_step += 1

                    if global_step % logging_steps == 0:
                        step_num = logging_steps
                        logging.info(
                            '[{}] step:{}, train_loss: {:.5f} = {:.5f} + {:.5f} + {:.5f}'.
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
                            model.module.save(ckpt_path)
                if local_rank == 0 or local_rank == -1:
                    ckpt_path = os.path.join(checkpoint_path, f'epoch_{epoch}_step_{global_step}')
                    Path(ckpt_path).mkdir(parents=True, exist_ok=True)
                    model.module.save(ckpt_path)
        except:
            error_type, error_value, error_trace = sys.exc_info()
            traceback.print_tb(error_trace)
            logging.info(error_value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", choices=["passage", 'doc'], type=str, required=True)
    parser.add_argument("--preprocess_dir", type=str, required=True)
    parser.add_argument("--max_query_length", type=int, default=32)
    parser.add_argument("--max_doc_length", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--mode", type=str, choices=["train", "dev", "test", "test2019","test2020"], default='dev')
    parser.add_argument("--root_output_dir", type=str, required=False, default='./data')
    parser.add_argument("--gpu_rank", type=str, required=False, default=None)

    parser.add_argument("--rank_file", type=str, required=False, default=None)
    parser.add_argument("--mink", type=int, required=False, default=0)
    parser.add_argument("--maxk", type=int, required=False, default=200)
    parser.add_argument("--per_query_neg_num", type=int, required=False, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, required=False, default=128)

    args = parser.parse_args()

    learnable_index = LearnableIndex(trained_encoder_ckpt='./saved_ckpts/AR_G/0/',
                                     index_file='./data/passage/evaluate/AR_G_0/ivf_opq.index')

    # learnable_index.fit(model=learnable_index.learnable_vq,
    #                     data_dir=args.preprocess_dir,
    #                     max_query_length=32,
    #                     max_doc_length=128,
    #                     query_embeddings_file='./data/passage/evaluate/AR_G_0/train-query.memmap',
    #                     doc_embeddings_file='./data/passage/evaluate/AR_G_0/passages.memmap',
    #                     checkpoint_path='./saved_ckpts/try',
    #                     per_device_train_batch_size=100,
    #                     logging_steps=1)

    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    learnable_index.fit_with_multi_gpus(data_dir=args.preprocess_dir,
                                        max_query_length=32,
                                        max_doc_length=128,
                                        query_embeddings_file = './data/passage/evaluate/AR_G_0/train-query.memmap',
                                        doc_embeddings_file = './data/passage/evaluate/AR_G_0/passages.memmap',
                                        checkpoint_path='./saved_ckpts/try',
                                        per_device_train_batch_size=256,
                                        epochs=10)

    learnable_index.update_encoder('./saved_ckpts/try/epoch_0_step_393/encoder.bin')
    learnable_index.encode(data_dir=args.preprocess_dir,
                           prefix='dev-query',
                           max_length=32,
                           output_dir='./data/passage/evaluate/try_epoch_0_step_393',
                           batch_size=8196,
                           is_query=True)
    ground_truths = load_rel(os.path.join(args.preprocess_dir, 'train-qrel.tsv'))

    query_embeddings = np.memmap('./data/passage/evaluate/try_epoch_0_step_393/dev-query.memmap', dtype=np.float32, mode="r")
    query_embeddings = query_embeddings.reshape(-1, 768)

    doc_embeddings = np.memmap('./data/passage/evaluate/AR_G_0/passages.memmap', dtype=np.float32, mode="r")
    doc_embeddings = doc_embeddings.reshape(-1, 768)

    learnable_index.update_index_with_ckpt(ckpt_path='./saved_ckpts/try/epoch_0_step_393/', doc_embeddings=doc_embeddings)
    learnable_index.set_nprobe(100)

    qids = list(range(len(query_embeddings)))
    learnable_index.test(query_embeddings, qids, ground_truths, topk=100, batch_size=64,
               MRR_cutoffs=args.MRR_cutoffs, Recall_cutoffs=args.Recall_cutoffs)


if __name__ == '__main__':
    main()

