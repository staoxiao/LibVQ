import sys
sys.path.append('./src')

import os
import logging
from tqdm.autonotebook import trange
import torch
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional
from torch.optim import Optimizer
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup, AutoConfig
import argparse
import torch.multiprocessing as mp

from learnable_vq import LearnableVQ
from index.FaissIndex import FaissIndex
from dataset.dataloader import DataloaderForSubGraphHard
from dataset.dataset import TextTokenIdsCache

class LearnableIndex(FaissIndex):
    def __init__(self):
        super(LearnableIndex).__init__()
        config = AutoConfig.from_pretrained('bert-base-uncased')
        config.pretrained_model_name = 'bert-base-uncased'
        config.use_two_encoder = True
        config.sentence_pooling_method = 'first'
        config.index_file = './data/passage/evaluate/AR_G_0/ivf_opq.index'
        self.model = LearnableVQ(config)

    def fit_multi_gpu(self,
            dataloader,
            world_size: int = 1,
            epochs: int = 1,
            warmup_steps: int = 1000,
            optimizer_class: Type[Optimizer] = AdamW,
            lr_params: Dict[str, object] = {'encoder_lr': 2e-5, 'pq_lr':1e-6, 'ivf_lr':1e-4},
            weight_decay: float = 0.01,
            output_path: str = None,
            max_grad_norm: float = -1,
            use_amp: bool = False,
            show_progress_bar: bool = True,
            checkpoint_path: str = None,
            checkpoint_save_steps: int = 500,
            logging_steps: int = 100
            ):
        mp.spawn(self.fit,
                 args=(dataloader),
                 nprocs=world_size,
                 join=True)

    def fit(self,
            local_rank,
            dataloader,
            epochs: int = 1,
            warmup_steps: int = 1000,
            optimizer_class: Type[Optimizer] = AdamW,
            lr_params: Dict[str, object] = {'encoder_lr': 2e-5, 'pq_lr':1e-6, 'ivf_lr':1e-4},
            weight_decay: float = 0.01,
            output_path: str = None,
            max_grad_norm: float = -1,
            use_amp: bool = False,
            show_progress_bar: bool = True,
            checkpoint_path: str = None,
            checkpoint_save_steps: int = 500,
            logging_steps: int = 100
            ):

        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu',
                              local_rank)
        model = self.model.to(device)


        num_train_steps = len(dataloader)*epochs
        # Prepare optimizers
        param_optimizer = list(self.model.named_parameters())
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
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps
        )

        loss, dense_loss, ivf_loss, pq_loss = 0., 0., 0., 0.
        global_step = 0
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):

            optimizer.zero_grad()
            self.model.train()

            for step, sample in enumerate(dataloader):
                query_token_ids, query_attention_mask, \
                doc_token_ids, doc_attention_mask, \
                neg_token_ids, neg_doc_attention_mask, \
                origin_q_emb, origin_d_emb, origin_n_emb, \
                doc_ids, neg_ids = sample

                batch_loss, batch_dense_loss, batch_ivf_loss, batch_pq_loss = model(query_token_ids, query_attention_mask,
                                                                                        doc_token_ids, doc_attention_mask,
                                                                                        neg_token_ids, neg_doc_attention_mask,
                                                                                        origin_q_emb, origin_d_emb, origin_n_emb,
                                                                                        doc_ids, neg_ids)
                batch_loss.backward()
                loss += batch_loss.item()
                if not isinstance(batch_dense_loss, float):
                    dense_loss += batch_dense_loss.item()
                if not isinstance(batch_ivf_loss, float):
                    ivf_loss += batch_ivf_loss.item()
                if not isinstance(batch_pq_loss, float):
                    pq_loss += batch_pq_loss.item()

                if max_grad_norm != -1:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                optimizer.step()
                self.model.ivf.update_centers(lr=lr_params['ivf_lr'])

                scheduler.step()
                optimizer.zero_grad()
                self.model.ivf.zero_grad()
                global_step += 1

                print(
                    '[{}] step:{}, train_loss: {:.5f} = {:.5f} + {:.5f} + {:.5f}'.
                        format(local_rank,
                               global_step,
                               loss / global_step,
                               dense_loss / global_step,
                               ivf_loss / global_step,
                               pq_loss / global_step))

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

            #TODO: save


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

    learnable_index = LearnableIndex()

    dataloader = DataloaderForSubGraphHard(os.path.join(args.preprocess_dir, 'train-qrel.tsv'),
                 rank_file=args.rank_file,
                 mink=args.mink,
                 maxk=args.maxk,
                 per_query_neg_num=args.per_query_neg_num,
                 per_device_train_batch_size=args.per_device_train_batch_size,
                                           generate_batch_method='random',
                queryids_cache=TextTokenIdsCache(data_dir=args.preprocess_dir, prefix="train-query"),
                docids_cache=TextTokenIdsCache(data_dir=args.preprocess_dir, prefix="passages"),
                 max_query_length=24,
                 max_doc_length=128,
                 local_rank=0,
                 world_size=1,
                 fix_emb='doc, score',
                 doc_file='./data/passage/evaluate/AR_G_0/passages.memmap',
                 query_file='./data/passage/evaluate/AR_G_0/train-query.memmap',
                 query_length=None,
                 doc_length=None,
                 enable_prefetch=True,
                 random_seed=42,
                 enable_gpu=True)

    learnable_index.fit(local_rank=0, dataloader=dataloader)


if __name__ == '__main__':
    main()

