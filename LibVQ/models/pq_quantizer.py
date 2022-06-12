import math
import os

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Quantization(nn.Module):
    """
    End-to-end Product Quantization
    """
    def __init__(self,
                 emb_size: int = 768,
                 subvector_num: int = 96,
                 subvector_bits: int = 8,
                 rotate: np.ndarray = None,
                 codebook: np.ndarray = None):
        """
        :param emb_size: Dim of embeddings
        :param subvector_num: The number of codebooks
        :param subvector_bits: The number of codewords in each codebook
        :param rotate:  The rotate Matrix. Used for OPQ
        :param codebook: The parameter for codebooks. If set None, it will randomly initialize the codebooks.
        """
        super(Quantization, self).__init__()

        if codebook is not None:
            self.codebook = nn.Parameter(torch.FloatTensor(codebook), requires_grad=True)
        else:
            self.codebook = nn.Parameter(
                torch.empty(subvector_num, 2 ** subvector_bits,
                            emb_size // subvector_num).uniform_(-0.1, 0.1)).type(
                torch.FloatTensor)
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
        subvector_num = pq_index.pq.M

        pq = cls(subvector_num=subvector_num, rotate=rotate, codebook=codebook)
        return pq

    def rotate_vec(self,
                   vecs):
        if self.rotate is None:
            return vecs
        return torch.matmul(vecs, self.rotate.T)

    def code_selection(self,
                       vecs):
        vecs = vecs.view(vecs.size(0), self.subvector_num, -1)
        codebook = self.codebook.unsqueeze(0).expand(vecs.size(0), -1, -1, -1)
        proba = - torch.sum((vecs.unsqueeze(-2) - codebook) ** 2, -1)
        assign = F.softmax(proba, -1)
        return assign

    def STEstimator(self,
                    assign):
        index = assign.max(dim=-1, keepdim=True)[1]
        assign_hard = torch.zeros_like(assign, device=assign.device, dtype=assign.dtype).scatter_(-1, index, 1.0)
        return assign_hard.detach() - assign.detach() + assign

    def quantized_vecs(self,
                       assign):
        assign = self.STEstimator(assign)
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

    def quantization_loss(self,
                          vec,
                          quantized_vecs):
        return torch.mean(torch.sum((vec - quantized_vecs) ** 2, dim=-1))

    def save(self,
             save_path):
        if self.rotate is not None:
            np.save(os.path.join(save_path, 'rotate_matrix'), self.rotate.detach().cpu().numpy())
        np.save(os.path.join(save_path, 'codebook'), self.codebook.detach().cpu().numpy())
