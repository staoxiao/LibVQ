import os
from typing import List, Type

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from LibVQ.base_index import IndexConfig
from LibVQ.models import IVFCPU
from LibVQ.models import Quantization
from LibVQ.utils import dist_gather_tensor


class LearnableVQ(nn.Module):
    """
    Learnable VQ model, supports both IVF and PQ.
    """
    def __init__(self,
                 config: Type[IndexConfig] = None,
                 init_index_file: str = None,
                 init_index_type: str = 'faiss',
                 index_method: str = 'ivf_opq',
                 dist_mode: str = 'ip',
                 encoder=None):
        nn.Module.__init__(self)

        self.config = config
        self.encoder = encoder
        self.dist_mode = dist_mode
        if init_index_file is not None:
            if 'ivf' not in index_method and 'pq' not in index_method:
                self.ivf = None
                self.pq = None
                print(f'There in no learnable parameters in this index: {init_index_file}')
                # raise ValueError(f"The index type:{index_method} is not supported!")

            if 'ivf' in index_method:
                if init_index_type == 'SPANN':
                    self.ivf = IVFCPU.from_spann_index(init_index_file)
                else:
                    self.ivf = IVFCPU.from_faiss_index(init_index_file)
            else:
                self.ivf = None

            if 'pq' in index_method:
                self.pq = Quantization.from_faiss_index(init_index_file)
            else:
                self.pq = None
        else:
            self.pq = Quantization(emb_size=config.emb_size,
                                   subvector_num=config.subvector_num,
                                   subvector_bits=config.subvector_bits)
            self.ivf = None

    def compute_score(self,
                      query_vecs: torch.Tensor,
                      doc_vecs: torch.Tensor,
                      neg_vecs: torch.Tensor,
                      dist_mode: str = 'ip',
                      temperature: float = 1.0):
        if any(v is None for v in (query_vecs, doc_vecs, neg_vecs)): return None

        if dist_mode == 'ip':
            score = self.inner_product(query_vecs, doc_vecs)
            n_score = self.inner_product(query_vecs, neg_vecs)
        elif dist_mode == 'l2':
            score = - self.euclidean_distance(query_vecs, doc_vecs)
            n_score = - self.euclidean_distance(query_vecs, neg_vecs)
        else:
            raise ValueError("The dist_mode must be ip or l2")
        score = torch.cat([score, n_score], dim=-1)
        return score / temperature

    def inner_product(self, vec_1, vec_2):
        return torch.matmul(vec_1, vec_2.T)

    def euclidean_distance(self, vec_1, vec_2):
        ip = torch.matmul(vec_1, vec_2.T)  # B D
        norm_1 = torch.sum(vec_1 * vec_1, dim=-1, keepdim=False).unsqueeze(1).expand(-1, vec_2.size(0))  # B D
        norm_2 = torch.sum(vec_2 * vec_2, dim=-1, keepdim=False).unsqueeze(0).expand(vec_1.size(0), -1)  # B D
        return norm_1 + norm_2 - 2 * ip

    def contras_loss(self,
                     score: torch.Tensor):
        if score is None: return 0.
        labels = torch.arange(start=0, end=score.shape[0],
                              dtype=torch.long, device=score.device)
        loss = F.cross_entropy(score, labels)
        return loss

    def distill_loss(self,
                     teacher_score: torch.Tensor,
                     student_score: torch.Tensor):
        if teacher_score is None or student_score is None: return 0.
        preds_smax = F.softmax(student_score, dim=1)
        true_smax = F.softmax(teacher_score, dim=1)
        preds_smax = preds_smax + 1e-6
        preds_log = torch.log(preds_smax)
        loss = torch.mean(-torch.sum(true_smax * preds_log, dim=1))
        return loss

    def compute_loss(self,
                     origin_score: torch.Tensor,
                     dense_score: torch.Tensor,
                     ivf_score: torch.Tensor,
                     pq_score: torch.Tensor,
                     loss_method: str = 'distill'):
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

        if fix_emb is not None and 'doc' in fix_emb:
            doc_vecs, neg_vecs = origin_d_emb, origin_n_emb
        else:
            doc_vecs = self.encoder.doc_emb(doc_token_ids, doc_attention_mask)
            neg_vecs = self.encoder.doc_emb(neg_token_ids, neg_attention_mask)

        if self.pq is not None:
            rotate_query_vecs = self.pq.rotate_vec(query_vecs)
            rotate_doc_vecs = self.pq.rotate_vec(doc_vecs)
            rotate_neg_vecs = self.pq.rotate_vec(neg_vecs)
        else:
            rotate_query_vecs = query_vecs
            rotate_doc_vecs = doc_vecs
            rotate_neg_vecs = neg_vecs

        if self.ivf is not None:
            dc_emb, nc_emb = self.ivf.select_centers(doc_ids, neg_ids, query_vecs.device, world_size=world_size)
        else:
            dc_emb, nc_emb = None, None

        if self.pq is not None:
            if self.ivf is not None:
                residual_doc_vecs = rotate_doc_vecs - dc_emb
                residual_neg_vecs = rotate_neg_vecs - nc_emb
                quantized_doc = self.pq.quantization(residual_doc_vecs) + dc_emb
                quantized_neg = self.pq.quantization(residual_neg_vecs) + nc_emb
            else:
                quantized_doc = self.pq.quantization(rotate_doc_vecs)
                quantized_neg = self.pq.quantization(rotate_neg_vecs)
        else:
            quantized_doc, quantized_neg = None, None

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

            if quantized_doc is not None:
                quantized_doc = self.dist_gather_tensor(quantized_doc)
                quantized_neg = self.dist_gather_tensor(quantized_neg)

        origin_score = self.compute_score(origin_q_emb, origin_d_emb, origin_n_emb,
                                          dist_mode=self.dist_mode,
                                          temperature=temperature)
        dense_score = self.compute_score(rotate_query_vecs, rotate_doc_vecs, rotate_neg_vecs,
                                         dist_mode=self.dist_mode,
                                         temperature=temperature)
        ivf_score = self.compute_score(rotate_query_vecs, dc_emb, nc_emb,
                                       dist_mode=self.dist_mode,
                                       temperature=temperature)
        pq_score = self.compute_score(rotate_query_vecs, quantized_doc, quantized_neg,
                                      dist_mode=self.dist_mode,
                                      temperature=temperature)

        dense_loss, ivf_loss, pq_loss = self.compute_loss(origin_score, dense_score, ivf_score, pq_score,
                                                          loss_method=loss_method)
        return dense_loss, ivf_loss, pq_loss

    def dist_gather_tensor(self, vecs):
        if vecs is None:
            return vecs
        return dist_gather_tensor(vecs, world_size=dist.get_world_size(), local_rank=dist.get_rank(), detach=False)

    def save(self, save_path):
        if self.pq is not None: self.pq.save(save_path)
        if self.encoder is not None: self.encoder.save(os.path.join(save_path, 'encoder.bin'))
        if self.ivf is not None: self.ivf.save(os.path.join(save_path, 'ivf_centers'))
        if self.config is not None: self.config.to_json_file(os.path.join(save_path, 'config.json'))
