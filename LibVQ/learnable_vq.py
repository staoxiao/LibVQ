import os
from typing import List, Type

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from LibVQ.baseindex import IndexConfig
from LibVQ.models import IVF_CPU
from LibVQ.models import Quantization
from LibVQ.utils import dist_gather_tensor


class LearnableVQ(nn.Module):
    def __init__(self,
                 config: Type[IndexConfig] = None,
                 index_file: str = None,
                 index_method: str = 'ivf_opq',
                 encoder=None):
        nn.Module.__init__(self)

        self.config = config
        self.encoder = encoder
        if index_file is not None:
            self.pq = Quantization.from_faiss_index(index_file)
            if 'ivf' in index_method:
                self.ivf = IVF_CPU.from_faiss_index(index_file)
            else:
                self.ivf = None
        else:
            self.pq = Quantization(emb_size=config.emb_size,
                                   subvector_num=config.subvector_num,
                                   subvector_bits=config.subvector_bits)
            self.ivf = None

    def compute_score(self, query_vecs, doc_vecs, neg_vecs, temperature=1.0):
        if any(v is None for v in (query_vecs, doc_vecs, neg_vecs)): return None

        score = torch.matmul(query_vecs, doc_vecs.T)
        n_score = torch.matmul(query_vecs, neg_vecs.T)
        score = torch.cat([score, n_score], dim=-1)
        return score / temperature

    def contras_loss(self, score):
        if score is None: return 0.
        labels = torch.arange(start=0, end=score.shape[0],
                              dtype=torch.long, device=score.device)
        loss = F.cross_entropy(score, labels)
        return loss

    def distill_loss(self, teacher_score, student_score):
        if teacher_score is None or student_score is None: return 0.
        preds_smax = F.softmax(student_score, dim=1)
        true_smax = F.softmax(teacher_score, dim=1)
        preds_smax = preds_smax + 1e-6
        preds_log = torch.log(preds_smax)
        loss = torch.mean(-torch.sum(true_smax * preds_log, dim=1))
        return loss

    def compute_loss(self, origin_score, dense_score, ivf_score, pq_score, loss_method='contras'):
        if loss_method == 'contras':
            dense_loss = self.contras_loss(dense_score)
            ivf_loss = self.contras_loss(ivf_score)
            pq_loss = self.contras_loss(pq_score)
        elif loss_method == 'distill':
            dense_loss = self.distill_loss(origin_score, dense_score)
            ivf_loss = self.distill_loss(origin_score, ivf_score)
            pq_loss = self.distill_loss(origin_score, pq_score)
        return dense_loss, ivf_loss, pq_loss

    def forward(self,
                query_token_ids: torch.LongTensor,
                query_attention_mask: torch.LongTensor,
                doc_token_ids: torch.LongTensor,
                doc_attention_mask: torch.LongTensor,
                neg_token_ids: torch.LongTensor,
                neg_attention_mask: torch.LongTensor,
                origin_q_emb: torch.FloatTensor,
                origin_d_emb: torch.FloatTensor,
                origin_n_emb: torch.FloatTensor,
                doc_ids, neg_ids: List[int],
                temperature: float = 1.0,
                loss_method: str = 'distill',
                fix_emb: str = 'doc',
                world_size: int = 1,
                cross_device_sample: bool = True):

        if fix_emb is not None and 'query' in fix_emb:
            query_vecs = origin_q_emb
        else:
            query_vecs = self.encoder.query_emb(query_token_ids, query_attention_mask)
        rotate_query_vecs = self.pq.rotate_vec(query_vecs)

        if fix_emb is not None and 'doc' in fix_emb:
            doc_vecs, neg_vecs = origin_d_emb, origin_n_emb
        else:
            doc_vecs = self.encoder.doc_emb(doc_token_ids, doc_attention_mask)
            neg_vecs = self.encoder.doc_emb(neg_token_ids, neg_attention_mask)
        rotate_doc_vecs = self.pq.rotate_vec(doc_vecs)
        rotate_neg_vecs = self.pq.rotate_vec(neg_vecs)

        if self.ivf is not None:
            dc_emb, nc_emb = self.ivf.select_centers(doc_ids, neg_ids, query_vecs.device, world_size=world_size)
            residual_doc_vecs = rotate_doc_vecs - dc_emb
            residual_neg_vecs = rotate_neg_vecs - nc_emb
            quantized_doc = self.pq.quantization(residual_doc_vecs) + dc_emb
            quantized_neg = self.pq.quantization(residual_neg_vecs) + nc_emb
        else:
            dc_emb, nc_emb = None, None
            quantized_doc = self.pq.quantization(rotate_doc_vecs)
            quantized_neg = self.pq.quantization(rotate_neg_vecs)

        if world_size > 1 and cross_device_sample:
            origin_q_emb = self.dist_gather_tensor(origin_q_emb)
            origin_d_emb = self.dist_gather_tensor(origin_d_emb)
            origin_n_emb = self.dist_gather_tensor(origin_n_emb)

            rotate_query_vecs = self.dist_gather_tensor(rotate_query_vecs)
            rotate_doc_vecs = self.dist_gather_tensor(rotate_doc_vecs)
            rotate_neg_vecs = self.dist_gather_tensor(rotate_neg_vecs)

            if dc_emb is not None:
                dc_emb = self.dist_gather_tensor(dc_emb)
                nc_emb = self.dist_gather_tensor(nc_emb)

            quantized_doc = self.dist_gather_tensor(quantized_doc)
            quantized_neg = self.dist_gather_tensor(quantized_neg)

        origin_score = self.compute_score(origin_q_emb, origin_d_emb, origin_n_emb, temperature)
        dense_score = self.compute_score(rotate_query_vecs, rotate_doc_vecs, rotate_neg_vecs, temperature)
        ivf_score = self.compute_score(rotate_query_vecs, dc_emb, nc_emb, temperature)
        pq_score = self.compute_score(rotate_query_vecs, quantized_doc, quantized_neg, temperature)

        dense_loss, ivf_loss, pq_loss = self.compute_loss(origin_score, dense_score, ivf_score, pq_score,
                                                          loss_method=loss_method)
        return dense_loss, ivf_loss, pq_loss

    def dist_gather_tensor(self, vecs):
        if vecs is None:
            return vecs
        return dist_gather_tensor(vecs, world_size=dist.get_world_size(), local_rank=dist.get_rank(), detach=False)

    def save(self, save_path):
        self.pq.save(save_path)

        if self.encoder is not None: self.encoder.save(os.path.join(save_path, 'encoder.bin'))
        if self.ivf is not None: self.ivf.save(os.path.join(save_path, 'ivf_centers'))
        if self.config is not None: self.config.to_json_file(os.path.join(save_path, 'config.json'))
