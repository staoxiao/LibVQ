import torch
from torch import nn
import faiss
import numpy as np

from LibVQ.utils import dist_gather_tensor

class IVF_CPU(nn.Module):
    def __init__(self, center_vecs, id2center):
        super(IVF_CPU, self).__init__()
        self.center_vecs = center_vecs
        self.id2center = id2center
        self.center_grad = np.zeros_like(center_vecs)

    @classmethod
    def from_faiss_index(cls, index_file):
        print(f'loading IVF from Faiss index: {index_file}')
        index = faiss.read_index(index_file)
        ivf_index = faiss.downcast_index(index.index)
        coarse_quantizer = faiss.downcast_index(ivf_index.quantizer)
        coarse_embeds = faiss.vector_to_array(coarse_quantizer.xb)
        center_vecs = coarse_embeds.reshape((-1, ivf_index.d))

        invlists = ivf_index.invlists
        id2center = {}
        for i in range(len(center_vecs)):
            ls = invlists.list_size(i)
            list_ids = faiss.rev_swig_ptr(invlists.get_ids(i), ls)

            for docid in list_ids:
                id2center[docid] = i

        ivf = cls(center_vecs, id2center)
        return ivf

    def get_batch_centers(self, batch_centers_index, device):
        c_embs = self.center_vecs[batch_centers_index]
        self.batch_centers_index = batch_centers_index
        self.batch_center_vecs = torch.FloatTensor(c_embs).to(device)
        self.batch_center_vecs.requires_grad = True

    def merge_and_dispatch(self, doc_ids, neg_ids):
        dc_ids = [self.id2center[i] for i in doc_ids]
        nc_ids = [self.id2center[i] for i in neg_ids]
        batch_cids = sorted(list(set(dc_ids+nc_ids)))

        cid2bid = {}
        for i, c in enumerate(batch_cids):
            cid2bid[c] = i

        batch_dc_ids = torch.LongTensor([cid2bid[x] for x in dc_ids])
        batch_nc_ids = torch.LongTensor([cid2bid[x] for x in nc_ids])
        return batch_cids, batch_dc_ids, batch_nc_ids

    def select_centers(self, doc_ids, neg_ids, device):
        batch_cids, batch_dc_ids, batch_nc_ids = self.merge_and_dispatch(doc_ids, neg_ids)

        self.get_batch_centers(batch_cids, device)
        batch_dc_ids, batch_nc_ids = batch_dc_ids.to(device), batch_nc_ids.to(device)
        dc_emb = self.batch_center_vecs.index_select(dim=0, index=batch_dc_ids)
        nc_emb = self.batch_center_vecs.index_select(dim=0, index=batch_nc_ids)
        return dc_emb, nc_emb

    def update_centers(self, lr):
        grad = self.batch_center_vecs.grad
        # grad = dist_gather_tensor(grad.unsqueeze(0), dist.get_world_size(), detach=True)
        # grad = torch.mean(grad, dim=0)
        self.center_grad[self.batch_centers_index] += grad.detach().cpu().numpy()

        self.center_vecs = self.center_vecs - lr * self.center_grad

    def zero_grad(self):
        self.center_grad = np.zeros_like(self.center_grad)

    def save(self, save_file):
        np.save(save_file, self.center_vecs)
