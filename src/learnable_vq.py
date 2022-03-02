import os
import torch
from torch import nn
import torch.nn.functional as F

from src.encoder import Encoder
from src.ivf_quantizer import IVF_CPU
from src.pq_quantizer import Quantization

class LearnableVQ(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)

        self.encoder = Encoder(config)
        self.pq = Quantization.from_faiss_index(config.index_file)
        self.ivf = IVF_CPU.from_faiss_index(config.index_file)

    def compute_score(self, query_vecs, doc_vecs, neg_vecs, temperature=1.0):
        score = torch.matmul(query_vecs, doc_vecs.T)
        n_score = torch.matmul(query_vecs, neg_vecs.T)
        score = torch.cat([score, n_score], dim=-1)
        return score/temperature

    def contras_loss(self, score):
        labels = torch.arange(start=0, end=score.shape[0],
                              dtype=torch.long, device=score.device)
        loss = F.cross_entropy(score, labels)
        return loss

    def distill_loss(self, teacher_score, student_score):
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
                    query_token_ids, query_attention_mask,
                    doc_token_ids, doc_attention_mask,
                    neg_token_ids, neg_doc_attention_mask,
                    rel_pair_mask, hard_pair_mask,
                    origin_q_emb, origin_d_emb, origin_n_emb,
                    doc_ids, neg_ids,
                temperature,
                loss_method,
                fix_emb='doc'):

        if 'query' in fix_emb:
            query_vecs = origin_q_emb
        else:
            query_vecs = self.encoder.query_encoder(query_token_ids, query_attention_mask)
        rotate_query_vecs = self.pq.rotate_vec(query_vecs)

        if 'doc' in fix_emb:
            doc_vecs, neg_vecs = origin_d_emb, origin_n_emb
        else:
            doc_vecs = self.encoder.query_encoder(doc_token_ids, doc_attention_mask)
            neg_vecs = self.encoder.query_encoder(neg_token_ids, neg_doc_attention_mask)

        dc_emb, nc_emb = self.ivf.select_centers(doc_ids, neg_ids, query_vecs.device)

        quantized_doc = self.pq.quantization(doc_vecs)
        quantized_neg = self.pq.quantization(neg_vecs)

        origin_score = self.compute_score(origin_q_emb, origin_d_emb, origin_n_emb, temperature)
        dense_score = self.compute_score(query_vecs, doc_vecs, neg_vecs, temperature)
        ivf_score = self.compute_score(rotate_query_vecs, dc_emb, nc_emb, temperature)
        pq_score = self.compute_score(rotate_query_vecs, quantized_doc, quantized_neg, temperature)

        dense_loss, ivf_loss, pq_loss = self.compute_loss(origin_score, dense_score, ivf_score, pq_score, loss_method=loss_method)
        loss = dense_loss + ivf_loss + pq_loss
        return loss. dense_loss, ivf_loss, pq_loss

    def save(self, save_path):
        torch.save(self.encoder.state_dict(), os.path.join(save_path, 'encoder.bin'))
        self.ivf.save(os.path.join(save_path, 'ivf_centers'))
        self.pq.save(save_path)