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
d = 32                                                             # dimension
doc_embeddings = np.random.random((10000, d)).astype('float32')    # corpus embeddings
query_embeddings = np.random.random((1000, d)).astype('float32')   # queries' embeddings
```
### Jointly training Index and Encoder
We recommend to jointly train the encoder and index, which can get the best performance. 
For this method, you should prepare a trained encoder and the text data:
- The encoder should inherit the class [Encoder](./LibVQ/models/encoder.py).
- Please refer to [dataset.README](./LibVQ/dataset/README.md)
for the data format, or you can give a child class of `torch.utils.data.dataset` for your data.


## Index
IndexConfig
```
emb_size = d
ivf_centers_num = 100
subvector_num = 8
subvector_bits = 8
nprobe = 10
```

- [ScaNN](https://github.com/google-research/google-research/tree/master/scann)
```python
from LibVQ.baseindex import ScaNNIndex
scann_index = ScaNNIndex(doc_embeddings = doc_embeddings,
                         ivf_centers_num = ivf_centers_num,
                         subvector_num = subvector_num)
scores, ann_items = scann_index.search(query_embeddings = query_embeddings,
                                       topk = 100,
                                       nprobe = nprobe) 
```

- [Faiss](https://github.com/facebookresearch/faiss)
```python
from LibVQ.baseindex import FaissIndex
faiss_index = FaissIndex(doc_embeddings = doc_embeddings,
                         ivf_centers_num = ivf_centers_num,
                         subvector_num = subvector_num,
                         subvector_bits = subvector_bits,
                         index_method = 'ivf_pq',
                         dist_mode = 'ip')
scores, ann_items = faiss_index.search(query_embeddings = query_embeddings,
                                       topk = 100,
                                       nprobe = nprobe)
```


- [ours](Reference)

```python
from LibVQ.learnable_index import LearnableIndex
learnable_index = LearnableIndex(doc_embeddings = doc_embeddings,
                         ivf_centers_num = ivf_centers_num,
                         subvector_num = subvector_num,
                         subvector_bits = subvector_bits,
                         index_method = 'ivf_opq',
                         dist_mode = 'ip')

# generate virtual train data if there is no labeled data
brute_index = FaissIndex(doc_embeddings = doc_embeddings,
                         index_method = 'flat',
                         dist_mode = 'ip')
query2pos, query2neg = brute_index.generate_virtual_traindata(query_embeddings, nprobe = ivf_centers_num)

# training the index
learnable_index.fit(model=learnable_index.learnable_vq,
                    rel_data=query2pos,
                    neg_data=query2neg,
                    query_embeddings=query_embeddings,
                    doc_embeddings=doc_embeddings,
                    checkpoint_path='./saved_ckpts/test_model/',
                    epochs=5,
                    lr_params={'encoder_lr': 0.0, 'pq_lr': 1e-6, 'ivf_lr': 1e-6},
                    loss_method='distill',
                    fix_emb='query, doc')

learnable_index.update_index_with_ckpt('./saved_ckpts/test_model/')
scores, ann_items = learnable_index.search(query_embeddings = query_embeddings,
                                       topk = 100,
                                       nprobe = nprobe)
```

Given a text encoder and jointly training index and query encoder:
```python
from LibVQ.learnable_index import LearnableIndex
learnable_index = LearnableIndex(encoder = text_encoder,
                                 index_file = faiss_index_file)

learnable_index.fit(model=learnable_index.learnable_vq,
                    rel_data=query2pos,
                    neg_data=query2neg,
                    query_data_dir=train_query_data_dir,
                    max_query_length=max_query_length,
                    query_embeddings=query_embeddings,
                    doc_embeddings=doc_embeddings,
                    checkpoint_path='./saved_ckpts/test_model/',
                    epochs=5,
                    lr_params={'encoder_lr': 1e-5, 'pq_lr': 1e-4, 'ivf_lr': 1e-3},
                    loss_method='distill',
                    fix_emb='doc')

# update encoder and generate queries' embeddings
learnable_index.update_encoder('./saved_ckpts/test_model/')
query_embeddings = learnable_index.encode(data_dir, max_query_length)

# update index
learnable_index.update_index_with_ckpt('./saved_ckpts/test_model/')

scores, ann_items = learnable_index.search(query_embeddings = query_embeddings, 
                                       topk = 100,
                                       nprobe = nprobe)
```

**Besides, learnable_index.index is a faiss index, which supports all operations in [faiss](https://github.com/facebookresearch/faiss) (e.g., GPU acceleration).**

## Examples
- [MSMARCO](./examples/MSMARCO/)  
- More examples are comming soon





