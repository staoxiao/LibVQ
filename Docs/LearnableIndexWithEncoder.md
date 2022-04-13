#  [LearnableIndexWithEncoder](../LibVQ/learnable_index/learnable_index_with_encoder.py)   
This class can jointly train the index and encoder, which often can get the best performance. For efficiency, it is also implemented base on [faiss](https://github.com/facebookresearch/faiss), so it supports all operations in faiss (e.g., GPU acceleration). Specifically,
you can get a origin faiss index by `index = LearnableIndex.index`, and apply faiss operations to it.

**1. Prepare encoder and data**  
For this method, you should prepare a trained encoder and the text data:
- You can create the Enocder by passing the query encoder and doc encoder: `Encoder(query_encoder, doc_encoder)`, 
or you can create a new encoder class by inherit the class [BaseEncoder](./LibVQ/models/encoder.py).
- Please refer to [dataset.README](./LibVQ/dataset/README.md)
for the data format, or you can overwrite the [DatasetForVQ](./LibVQ/dataset/dataset.py) for your data.

Save the data to `data_dir`, you need to preprocess it:
```python
from LibVQ.dataset import preprocess_data
preprocess_data(data_dir=data_dir,
                output_dir=preprocess_dir,
                tokenizer_name=pretrained_model_name,
                max_doc_length=max_doc_length,
                max_query_length=max_query_length,
                workers_num=64)
```

If you has no embeddings, you can generate it by:
```python
from LibVQ.inference import inference
inference(data_dir = preprocess_dir,
          is_query = is_query,
          encoder = encoder,
          prefix = f'docs', # 'docs', 'train-queries', 'dev-queries', 'test-queries'
          max_length = max_length,
          output_dir = output_dir,
          batch_size = 10240)
```


**2.  Construct index**   
You should give a encoder when create the index:
```python
from LibVQ.learnable_index import LearnableIndex
learnable_index = LearnableIndex(encoder=encoder,
                                 doc_embeddings = doc_embeddings,
                                 ivf_centers_num = 10000, # the number of centers in ivf
                                 subvector_num = 32,      # the number of codebooks in pq
                                 subvector_bits = 8,      # the number of codewords (2^subvector_bits) in each codebook
                                 index_method = 'ivf_opq',# ivf, ivf_opq, ivf_pq, pq, opq
                                 dist_mode = 'ip'         # distance metric
                                 )
```
Currently it supports five index methods: `ivf`, `ivf_pq`, `ivf_opq`, `pq`, and `opq`.


**3.  Train index and encoder**  
```python
learnable_index.fit(query_embeddings = query_embeddings,
                    doc_embeddings = doc_embeddings,
                    query_data_dir = preprocess_dir, # give the query data when train query encoder
                    max_query_length = max_query_length,
                    rel_data = None,   # relevance relationship between query and doc; if set None, it will automatically generate the data for training
                    loss_method = 'distill',
                    fix_emb = 'doc'                  # you can select to fix the embeddings of query/doc or set None.
                    )
```


**4. Update query embeddings and test**  
```python
query_embeddings = learnable_index.encode(data_dir = preprocess_dir,            # update query embeddings
                       prefix = 'dev-queries',
                       max_length = max_query_length,
                       batch_size = 8196,
                       is_query = True,
                       return_vecs = True
                       )
scores, ann_items = learnable_index.search(query_embeddings = query_embeddings,
                                       topk = 100,
                                       nprobe = 10)
```

**5. Save and Load**  
Same usage as [LearnableIndex](./LearnableIndex.md).

**Please refer to example: [MSMARCO/train_index_and_encoder](../examples/MSMARCO/learnable_index/train_index_and_encoder.py) or 
[NQ/train_index_and_encoder](../examples/NQ/learnable_index/train_index_and_encoder.py) for more information**
