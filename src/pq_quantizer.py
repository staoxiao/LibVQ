import torch
from torch import nn
import torch.nn.functional as F
import faiss
import numpy as np
import os

class Quantization(nn.Module):
    def __int__(self, embedding_size=768, partition=96, centroids=256, rotate=None, codebook=None):
        super(Quantization, self).__init__()

        self.partition = partition

        if codebook:
            self.codebook = nn.Parameter(torch.FloatTensor(codebook), requires_grad=True)
        else:
            self.codebook = nn.Parameter(
                torch.empty(partition, centroids,
                            embedding_size // partition).uniform_(-0.1, 0.1)).type(
                torch.FloatTensor)

        if rotate:
            self.rotate = nn.Parameter(torch.FloatTensor(rotate), requires_grad=False)
        else:
            self.rotate = None

    @classmethod
    def from_faiss_index(cls, index_file):
        print(f'loading PQ from Faiss index: {index_file}')
        opq_index = faiss.read_index(index_file)
        vt = faiss.downcast_VectorTransform(opq_index.chain.at(0))
        assert isinstance(vt, faiss.LinearTransform)
        rotate = faiss.vector_to_array(vt.A).reshape(vt.d_out, vt.d_in)

        ivf_index = faiss.downcast_index(opq_index.index)
        centroid_embeds = faiss.vector_to_array(ivf_index.pq.centroids)
        codebook = centroid_embeds.reshape(ivf_index.pq.M, ivf_index.pq.ksub, ivf_index.pq.dsub)
        partition = ivf_index.pq.M

        pq = cls(partition=partition, rotate=rotate, codebook=codebook)
        return pq

    def rotate_vec(self, vecs):
        if self.rotate is None:
            return vecs
        return torch.matmul(vecs, self.rotate.T)

    def code_selection(self, vecs):
        vecs = vecs.view(vecs.size(0), self.partition, -1)
        codebook = self.codebook.unsqueeze(0).expand(vecs.size(0), -1, -1, -1)
        proba = - torch.sum((vecs.unsqueeze(-2) - codebook) ** 2, -1)
        assign = F.softmax(proba, -1)
        return assign

    def STEstimator(self, assign):
        index = assign.max(dim=-1, keepdim=True)[1]
        assign_hard = torch.zeros_like(assign, device=assign.device, dtype=assign.dtype).scatter_(-1, index, 1.0)
        return assign_hard.detach() - assign.detach() + assign

    def quantized_vecs(self, assign):
        assign = self.STEstimator(assign)
        assign = assign.unsqueeze(2)
        codebook = self.codebook.unsqueeze(0).expand(assign.size(0), -1, -1, -1)
        quantized_vecs = torch.matmul(assign, codebook).squeeze(2)
        quantized_vecs = quantized_vecs.view(assign.size(0), -1)
        return quantized_vecs

    def quantization(self, vecs):
        vecs = self.rotate_vec(vecs)
        assign = self.code_selection(vecs)
        quantized_vecs = self.quantized_vecs(assign)
        return quantized_vecs

    def quantization_loss(self, vec, quantized_vecs):
        return torch.mean(torch.sum((vec - quantized_vecs) ** 2, dim=-1))

    def save(self, save_path):
        if self.rotate:
            np.save(os.path.join(save_path, 'rotate_matrix'), self.rotate.cpu().numpy())
        np.save(os.path.join(save_path, 'codebook'), self.codebook.cpu().numpy())
