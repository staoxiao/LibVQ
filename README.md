# LibVQ
LibVQ


## Install
```
git clone https://github.com/staoxiao/LibVQ.git
cd LibVQ
pip install .
```

## Getting data
LibVQ needs two sets of vectors: one for the entire corpus, containing all embeddings that you are going to search in;
and the other is the queries' embeddings. LibVQ also supports passing the two filenames (`.npy`, `.memmap`) which save the vector matrices.
```python
import numpy as np
np.random.seed(42)  
d = 64                                                    # dimension
doc_embeddings = np.random.random((100000, d)).astype('float32')    # corpus embeddings
queries_embeddings = np.random.random((1000, d)).astype('float32')   # queries' embeddings
```
### Jointly training Index and Encoder
We recommend to jointly train the encoder and index, which can get the best performance. 
For this method, you should prepare a trained encoder and the text data:
- The encoder should inherit the class [Encoder](./LibVQ/models/encoder.py).
- Please refer to [dataset.README](./LibVQ/dataset/README.md)
for the data format, or you can give a child class of `torch.utils.dataet` for your data.


## Index
IndexConfig
```
emb_size = d
ivf_centers = 10000
subvector_num = 32
subvector_bits = 8
nprobe = 1
```

- [ScaNN](https://github.com/google-research/google-research/tree/master/scann)
```python
from LibVQ.baseindex import ScaNNIndex
scann_index = ScaNNIndex(doc_embeddings = doc_embeddings,
                         ivf_centers = ivf_centers,
                         subvector_num = subvector_num)
scores, ann_items = scann_index.search(query_embeddings = query_embeddings, 
                                       topk = 100,
                                       nprobe = nprobe) 
```

- [Faiss](https://github.com/facebookresearch/faiss)
```python
from LibVQ.baseindex import FaissIndex
faiss_index = FaissIndex(doc_embeddings = doc_embeddings,
                         ivf_centers = ivf_centers_num,
                         subvector_num = subvector_num,
                         subvector_bits = subvector_bits,
                         index_method = 'ivf_opq',
                         dist_mode = 'ip')
scores, ann_items = faiss_index.search(query_embeddings = query_embeddings, 
                                       topk = 100,
                                       nprobe = nprobe)
```


- [ours](Reference)

```python
from LibVQ.learnable_index import LearnableIndex
learnable_index = LearnableIndex(doc_embeddings = doc_embeddings,
                         ivf_centers = ivf_centers_num,
                         subvector_num = subvector_num,
                         subvector_bits = subvector_bits,
                         index_method = 'ivf_opq',
                         dist_mode = 'ip')

# 
query2pos, query2neg = learnable_index.generate_virtual_traindata(query_embeddings, nprobe = ivf_centers_num)


# training the index
learnable_index.fit(query2pos, query2neg, query_embeddings, doc_embeddings, save_ckpts_path)

learnable_index.update_index_with_ckpt(saved_ckpts_path)

scores, ann_items = faiss_index.search(query_embeddings = query_embeddings, 
                                       topk = 100,
                                       nprobe = nprobe)
```

jointly training index and query encoder
```python
from LibVQ.learnable_index import LearnableIndex
learnable_index = LearnableIndex(encoder = text_encoder,
                                 index_file = faiss_index_file)

learnable_index.fit(rel_file, query_data_dir, max_query_length, query_embeddings, doc_embeddings, save_ckpts_path)

learnable_index.update_encoder(save_ckpt_path)
learnable_index.encode(query_data_dir, max_query_length)

learnable_index.update_index_with_ckpt(saved_ckpts_path)

scores, ann_items = faiss_index.search(query_embeddings = query_embeddings, 
                                       topk = 100,
                                       nprobe = nprobe)
```

## Examples
- [MSMARCO](./examples/MSMARCO/README.md)  
- More examples are comming soon


## Reference


