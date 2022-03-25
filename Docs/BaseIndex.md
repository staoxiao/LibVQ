# Basic Index

Settings:
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
                         index_method = 'ivf_pq',  # flat, ivf, ivf_pq, ivf_opq, pq, opq
                         dist_mode = 'ip')
scores, ann_items = faiss_index.search(query_embeddings = query_embeddings,
                                       topk = 100,
                                       nprobe = nprobe)
```


