import math

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import os
from LibVQ.base_index import FaissIndex

class Quantization(nn.Module):
    def __init__(self,
                 rotate: np.ndarray = None,
                 codebook: np.ndarray = None):
        """
        :param rotate:  The rotate Matrix. Used for OPQ
        :param codebook: The parameter for codebooks. If set None, it will randomly initialize the codebooks.
        """
        super(Quantization, self).__init__()
        self.codebook = nn.Parameter(torch.FloatTensor(codebook), requires_grad=False)
        self.subvector_num = self.codebook.size(0)
        self.subvector_bits = int(math.log2(self.codebook.size(1)))

        if rotate is not None:
            self.rotate = nn.Parameter(torch.FloatTensor(rotate), requires_grad=False)
        else:
            self.rotate = None

    @classmethod
    def from_faiss_index(cls, index_file: str):
        print(f'loading PQ from Faiss index: {index_file}')
        index = faiss.read_index(index_file)

        if isinstance(index, faiss.IndexPreTransform):
            vt = faiss.downcast_VectorTransform(index.chain.at(0))
            assert isinstance(vt, faiss.LinearTransform)
            rotate = faiss.vector_to_array(vt.A).reshape(vt.d_out, vt.d_in)
            pq_index = faiss.downcast_index(index.index)
        else:
            pq_index = index
            rotate = None

        centroid_embeds = faiss.vector_to_array(pq_index.pq.centroids)
        codebook = centroid_embeds.reshape(pq_index.pq.M, pq_index.pq.ksub, pq_index.pq.dsub)

        pq = cls(rotate=rotate, codebook=codebook)
        return pq

    def rotate_vec(self,
                   vecs):
        if self.rotate is None:
            return vecs
        return torch.matmul(vecs, self.rotate.T)

    def code_selection(self,
                       vecs):
        # 将向量拆分成subvector_num×d的形式
        vecs = vecs.view(vecs.size(0), self.subvector_num, -1)
        # 改变codebook的形状，让和vecs的数目一致，方便运算
        codebook = self.codebook.unsqueeze(0).expand(vecs.size(0), -1, -1, -1)
        # 计算每个维度每个码本中的向量与vecs的差值的和
        proba = - torch.sum((vecs.unsqueeze(-2) - codebook) ** 2, -1)
        # 最后一维进行softmax 结果为大小(num, subvector_num, sub_center_num)
        assign = F.softmax(proba, -1)
        return assign

    def STEstimator(self,
                    index):
        index = index.unsqueeze(-1)
        assign_size = torch.rand((index.size(0), self.subvector_num, 2 ** self.subvector_bits))
        assign_hard = torch.zeros_like(assign_size, device=index.device, dtype=self.codebook.dtype)
        assign_hard.scatter_(-1, index, 1.0)
        return assign_hard.detach()

    def quantized_vecs(self,
                       index):
        assign = self.STEstimator(index)
        assign = assign.unsqueeze(2)
        codebook = self.codebook.unsqueeze(0).expand(assign.size(0), -1, -1, -1)
        quantized_vecs = torch.matmul(assign, codebook).squeeze(2)
        quantized_vecs = quantized_vecs.view(assign.size(0), -1)
        return quantized_vecs

    def quantization(self,
                     vecs):
        assign = self.code_selection(vecs)
        quantized_vecs = self.quantized_vecs(assign)
        return quantized_vecs

    def code_selection(self,
                       vecs):
        # 将向量拆分成subvector_num×d的形式
        vecs = vecs.view(vecs.size(0), self.subvector_num, -1)
        # 改变codebook的形状，让和vecs的数目一致，方便运算
        codebook = self.codebook.unsqueeze(0).expand(vecs.size(0), -1, -1, -1)
        # 计算每个维度每个码本中的向量与vecs的差值的和
        proba = - torch.sum((vecs.unsqueeze(-2) - codebook) ** 2, -1)
        # 最后一维进行softmax 结果为大小(num, subvector_num, sub_center_num)
        assign = F.softmax(proba, -1)
        return assign

    def get_index(self, assign):
        index = assign.max(dim=-1, keepdim=True)[1]
        index = index.squeeze(-1)
        return index

    def embedding_compression(self, vecs):
        assign = self.code_selection(vecs)
        compression_index = self.get_index(assign)
        return compression_index

    def get_quantization_vecs(self, index):
        vecs = self.quantized_vecs(index)
        if self.rotate:
            vecs = self.rotate_vec(vecs)
        return vecs

def get_emb(query, parameters_path):
    encoder_config_file = os.path.join(parameters_path, 'encoder_config.json')
    if not os.path.exists(encoder_config_file):
        print('You should provide ' + encoder_config_file)
        return None
    index = FaissIndex.load_all(parameters_path)
    if isinstance(query, str):
        query = [query]
    input_data = index.model.text_tokenizer(query, padding=True)
    input_ids = torch.LongTensor(input_data['input_ids'])
    attention_mask = torch.LongTensor(input_data['attention_mask'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    index.model.to(device)
    query_emb = index.model.encoder.query_emb(input_ids, attention_mask).detach().cpu().numpy()
    if index.pooler:
        with torch.no_grad():
            query_emb = torch.Tensor(query_emb.copy()).to(device)
            index.pooler.to(device)
            query_emb = index.pooler(query_emb).detach().cpu().numpy()
    return query_emb