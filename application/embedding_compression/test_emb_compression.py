import sys
sys.path.append('./')

import numpy as np
import torch
from embedding_compression.quantization import Quantization

from transformers import HfArgumentParser
from arguments import IndexArguments, DataArguments

if __name__ == '__main__':
    parser = HfArgumentParser((IndexArguments, DataArguments))
    index_args, data_args = parser.parse_args_into_dataclasses()
    pq = Quantization.from_faiss_index(index_args.index_path)
    sample_embedding = np.memmap(data_args.sample_emb_path, dtype=np.float32, mode="r")
    sample_embedding = sample_embedding.reshape(-1, data_args.data_emb_size)
    print('The compressed results are as follows:')
    tensor_emb = torch.Tensor(sample_embedding)
    result = pq.embedding_compression(tensor_emb)
    print(result)
    print('The nearest centroid of the vector is as follows:')
    centroid = pq.get_quantization_vecs(result)
    print(centroid)