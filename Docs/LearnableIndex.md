# [LearnableIndex](../LibVQ/learnable_index/learnable_index.py)
```python
from LibVQ.learnable_index import LearnableIndex
```
This class can finetune the index and get better performance than transitional index (e.g., scann, faiss).
For efficiency, it is implemented base on [faiss](https://github.com/facebookresearch/faiss), so it supports all operations in faiss (e.g., GPU acceleration). Specifically,
you can get a origin faiss index by `index = LearnableIndex.index`, and apply faiss operations to it.


**1.  Prepare embedding**  
LibVQ needs two sets of vectors: one is docs' embeddings, containing all embeddings that you are going to search in;
and the other is the queries' embeddings. 

**2.  Construct index**   

Currently it supports five index methods: `ivf`, `ivf_pq`, `ivf_opq`, `pq`, and `opq`.
```python
learnable_index = LearnableIndex(doc_embeddings = doc_embeddings,
                                 ivf_centers_num = 10000, # the number of centers in ivf
                                 subvector_num = 32,      # the number of codebooks in pq
                                 subvector_bits = 8,      # the number of codewords (2^subvector_bits) in each codebook
                                 index_method = 'ivf_opq',# ivf, ivf_opq, ivf_pq, pq, opq
                                 dist_mode = 'ip'         # distance metric
                                 )
```
**3.  Train index**  
`fit` function will train the parameters of pq and ivf (if has) based on the relevance relationship between query and doc.
```python
learnable_index.fit(query_embeddings = query_embeddings,
                    doc_embeddings = doc_embeddings,
                    rel_data = None,   # relevance relationship between query and doc; if set None, it will automatically generate the data for training
                    loss_method = 'distill',
                    epochs=5)
```
For distributed training on multi GPUs, you cans use `fit_with_multi_gpus`.  

**4. Search**  
```python
scores, ann_items = learnable_index.search(query_embeddings = query_embeddings,
                                       topk = 100,
                                       nprobe = 10)
```


**5. Save and Load**

Save the index:
```python
learnable_index.save_index(saved_index_file)
```

Load the index:
```python
learnable_index = LearnableIndex(index_method='ivf_opq',
                                 init_index_file=saved_index_file)
```

Besides, you also can load it as a faiss index:
```python
import faiss
index = faiss.read_index(saved_index_file)
```

**Please refer to example: [MSMARCO/train_index](../examples/MSMARCO/learnable_index/train_index.py) or [NQ/train_index](../examples/NQ/learnable_index/train_index.py) for more information**

