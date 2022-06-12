import logging
import os
import random
import struct
import sys

import faiss
import numpy as np
import torch
import torch.distributed as dist

from LibVQ.base_index import FaissIndex


def is_main_process(local_rank):
    return local_rank in [-1, 0]


def setuplogging(level=logging.INFO):
    # silent transformers
    logging.getLogger("transformers").setLevel(logging.ERROR)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    if (root.hasHandlers()):
        root.handlers.clear()  # otherwise logging have multi output
    root.addHandler(handler)


def setup_worker(rank, world_size):
    # initialize the process group
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)


def dist_gather_tensor(vecs, world_size, local_rank=0, detach=True):
    all_tensors = [torch.empty_like(vecs) for _ in range(world_size)]
    dist.all_gather(all_tensors, vecs)
    if not detach:
        all_tensors[local_rank] = vecs
    all_tensors = torch.cat(all_tensors, dim=0)
    return all_tensors


def save_to_STAG_binart_file(index: FaissIndex,
                             save_dir: str):
    # The function only supports OPQ currently.
    rotate_matrix = index.get_rotate_matrix()
    codebooks = index.get_codebook()
    with open(os.path.join(save_dir, 'index_parameters.bin'), 'wb') as f:
        f.write(struct.pack('B', 2))
        f.write(struct.pack('B', 3))
        f.write(struct.pack('i', codebooks.shape[0]))
        f.write(struct.pack('i', codebooks.shape[1]))
        f.write(struct.pack('i', codebooks.shape[2]))
        f.write(codebooks.tobytes())
        f.write(rotate_matrix.tobytes())

    pq_index = faiss.downcast_index(index.index.index)
    codes = faiss.vector_to_array(pq_index.codes).reshape(pq_index.ntotal, -1)
    with open(os.path.join(save_dir, 'quantized_vectors.bin'), 'wb') as f:
        f.write(struct.pack('ii', codes.shape[0], codes.shape[1]))
        f.write(codes.tobytes())
