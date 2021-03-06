from collections import defaultdict
import faiss
import numpy as np
import struct
import torch
from torch import nn
from typing import Dict, List

from LibVQ.utils import dist_gather_tensor


class IVFCPU(nn.Module):
    """
    Efficiently training IVF on CPU. In this way you can use a large-scale IVF index.
    """
    def __init__(self,
                 center_vecs: np.ndarray,
                 id2center: Dict[int, int]):
        """
        :param center_vecs: Vectors for all centers
        :param id2center: Mapping Doc id to center id
        """
        super(IVFCPU, self).__init__()
        self.center_vecs = center_vecs
        self.id2center = id2center
        self.center_grad = np.zeros_like(center_vecs)
        self.ivf_centers_num = len(center_vecs)

    @classmethod
    def from_faiss_index(cls, index_file):
        print(f'loading IVF from Faiss index: {index_file}')

        index = faiss.read_index(index_file)
        if isinstance(index, faiss.IndexPreTransform):
            ivf_index = faiss.downcast_index(index.index)
        else:
            ivf_index = index

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

    @classmethod
    def from_spann_index(cls, index_file):
        print(f'loading IVF from spann index: {index_file}')

        with open(f'{index_file}/SPTAGHeadVectors.bin', 'rb') as fq:
            count = struct.unpack('i', fq.read(4))[0]
            dimension = struct.unpack('i', fq.read(4))[0]
            center_vecs = np.frombuffer(fq.read(count * dimension * 4), dtype=np.float32).reshape((count, dimension))

        id2center = defaultdict(int)
        with open(f'{index_file}/output.txt', 'r') as f:
            cid = 0
            for line in f:
                line = line.strip().split(',')
                for id in line:
                    id2center[int(id)] = cid
                cid += 1

        assert cid == len(center_vecs)
        ivf = cls(center_vecs, id2center)
        return ivf

    def get_batch_centers(self,
                          batch_centers_index: List[int],
                          device: torch.device):
        c_embs = self.center_vecs[batch_centers_index]
        self.batch_centers_index = batch_centers_index
        self.batch_center_vecs = torch.FloatTensor(c_embs).to(device)
        self.batch_center_vecs.requires_grad = True

    def merge_and_dispatch(self,
                           doc_ids: List[int],
                           neg_ids: List[int],
                           world_size: int):
        dc_ids = [self.id2center[i] for i in doc_ids]
        nc_ids = [self.id2center[i] for i in neg_ids]

        if world_size > 1:
            all_dc_ids = dist_gather_tensor(torch.LongTensor(dc_ids).cuda(), world_size=world_size, detach=True)
            all_dc_ids = list(all_dc_ids.detach().cpu().numpy())
            all_nc_ids = dist_gather_tensor(torch.LongTensor(nc_ids).cuda(), world_size=world_size, detach=True)
            all_nc_ids = list(all_nc_ids.detach().cpu().numpy())
        else:
            all_dc_ids, all_nc_ids = dc_ids, nc_ids

        batch_cids = sorted(list(set(all_dc_ids + all_nc_ids)))
        cid2bid = {}
        for i, c in enumerate(batch_cids):
            cid2bid[c] = i

        batch_dc_ids = torch.LongTensor([cid2bid[x] for x in dc_ids])
        batch_nc_ids = torch.LongTensor([cid2bid[x] for x in nc_ids])
        return batch_cids, batch_dc_ids, batch_nc_ids

    def select_centers(self,
                       doc_ids: List[int],
                       neg_ids: List[int],
                       device: torch.device,
                       world_size: int):
        batch_cids, batch_dc_ids, batch_nc_ids = self.merge_and_dispatch(doc_ids, neg_ids, world_size=world_size)

        self.get_batch_centers(batch_cids, device)
        batch_dc_ids, batch_nc_ids = batch_dc_ids.to(device), batch_nc_ids.to(device)
        dc_emb = self.batch_center_vecs.index_select(dim=0, index=batch_dc_ids)
        nc_emb = self.batch_center_vecs.index_select(dim=0, index=batch_nc_ids)
        return dc_emb, nc_emb

    def grad_accumulate(self,
                        world_size: int):
        if world_size > 1:
            grad = dist_gather_tensor(self.batch_center_vecs.grad.unsqueeze(0), world_size=world_size, detach=True)
            grad = torch.mean(grad, dim=0).detach().cpu().numpy()
        else:
            grad = self.batch_center_vecs.grad.detach().cpu().numpy()

        self.center_grad[self.batch_centers_index] += grad

    def update_centers(self,
                       lr: float):
        self.center_vecs = self.center_vecs - lr * self.center_grad

    def zero_grad(self):
        self.center_grad = np.zeros_like(self.center_grad)

    def save(self,
             save_file: str):
        np.save(save_file, self.center_vecs)
