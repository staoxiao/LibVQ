# Basic Index

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


# Learnable Index
```python
from LibVQ.learnable_index import LearnableIndex
```
We provide two modes to train the index: only training index and jointly training index and encoder. 
You can use them to update the index and get better performance than transitional index (e.g., scann, faiss).  
**Noted that The LearnableIndex is implemented base on [faiss](https://github.com/facebookresearch/faiss), so it supports all operations in faiss (e.g., GPU acceleration). Specifically,
you can get a origin faiss index by `index = LearnableIndex.index`, and apply faiss operations to it.**


## A. Training
### - Only train index
**1.  Prepare embedding**  
LibVQ needs two sets of vectors: one is docs' embeddings, containing all embeddings that you are going to search in;
and the other is the queries' embeddings. 

**2.  Construct index**   

Currently it supports four index methods: `ivf_pq`, `ivf_opq`, `pq`, and `opq`.
```python
learnable_index = LearnableIndex(doc_embeddings = doc_embeddings,
                                 ivf_centers_num = 10000, # the number of centers in ivf
                                 subvector_num = 32,      # the number of codebooks in pq
                                 subvector_bits = 8,      # the number of codewords (2^subvector_bits) in each codebook
                                 index_method = 'ivf_opq',
                                 dist_mode = 'ip'         # distance metric
                                 )
```
**3.  Train index**  
`fit` function will train the parameters of pq and ivf (if has) based on the relevance relationship between query and doc.
```python
# if there is no relevance label, you can generate by "generate_virtual_traindata" function
# query2pos, query2neg = learnable_index.generate_virtual_traindata(query_embeddings, nprobe = 10000)
learnable_index.fit(model = learnable_index.learnable_vq,
                    rel_data = query2pos,
                    query_embeddings = query_embeddings,
                    doc_embeddings = doc_embeddings,
                    checkpoint_path = './temp/',   # the parameters of index will saved to this path
                    loss_method = 'distill',
                    fix_emb = 'query, doc')
```
For distributed training on multi GPUs, you cans use `fit_with_multi_gpus`.  


**Please refer to example: [MSMARCO/train_index](examples/MSMARCO/train_index.py) for more information**


### - Jointly train Index and Encoder
**We recommend to jointly train the encoder and index, which can get the best performance.**  
**1. Prepare encoder and data**  
For this method, you should prepare a trained encoder and the text data:
- The encoder should inherit the class [BaseEncoder](./LibVQ/models/encoder.py) or have the same functions.
- Please refer to [dataset.README](./LibVQ/dataset/README.md)
for the data format, or you can overwrite the [DatasetForVQ](./LibVQ/dataset/dataset.py) for your data.

Save the data to `data_dir`, you need to preprocess it:
```python
preprocess_data(data_dir=data_dir,
                output_dir=preprocess_dir,
                tokenizer_name=pretrained_model_name,
                max_doc_length=max_doc_length,
                max_query_length=max_query_length,
                workers_num=64)
```

If you has no embeddings, you can generate it by:
```python
inference(data_dir = preprocess_dir,
          is_query = is_query,
          encoder = encoder,
          prefix = f'docs',
          max_length = max_length,
          output_dir = output_dir,
          batch_size = 10240)
```


**2.  Construct index**   
You can give a encoder when create the index:
```python
from LibVQ.learnable_index import LearnableIndex
learnable_index = LearnableIndex(encoder=encoder,
                                 doc_embeddings = doc_embeddings,
                                 ivf_centers_num = 10000, # the number of centers in ivf
                                 subvector_num = 32,      # the number of codebooks in pq
                                 subvector_bits = 8,      # the number of codewords (2^subvector_bits) in each codebook
                                 index_method = 'ivf_opq',
                                 dist_mode = 'ip'         # distance metric
                                 )
```
**3.  Train index and encoder**  
```python
# if there is no relevance label, you can generate by "generate_virtual_traindata" function
# query2pos, query2neg = learnable_index.generate_virtual_traindata(query_embeddings, nprobe = 10000)
learnable_index.fit(model = learnable_index.learnable_vq,
                    rel_data = query2pos,
                    query_embeddings = query_embeddings,
                    doc_embeddings = doc_embeddings,
                    query_data_dir = preprocess_dir, # give the query data when train query encoder
                    max_query_length = max_query_length,
                    checkpoint_path = './temp/',     # the parameters of index will saved to this path
                    loss_method = 'distill',
                    fix_emb = 'doc'                  # you can select to train the query encoder or train both query and doc encoder.
                    )
```
**Please refer to example: [MSMARCO/train_index_and_encoder](examples/MSMARCO/train_index_and_encoder.py) for more information**



## B. Update Index and Test
**1. Update embeddings**  
This step is needed only when you jointly the index and encoder.
```python
learnable_index.update_encoder(saved_ckpts_path = './temp/') # update encoder
learnable_index.encode(data_dir = preprocess_dir,            # update query embeddings
                       prefix = 'dev-queries',
                       max_length = max_query_length,
                       output_dir = output_dir,
                       batch_size = 8196,
                       is_query = True,
                       )
query_embeddings = np.memmap(f'{output_dir}/dev-queries.memmap', dtype=np.float32, mode="r")
query_embeddings = new_query_embeddings.reshape(-1, emb_size)

# if you re-train the doc encoder, generate the new doc embeddings 
# doc_embeddings = ... 
```

**2. Update index**  
After training, you can select a set of trained parameters to update the index:
```python
learnable_index.update_index_with_ckpt(saved_ckpts_path = './temp/', doc_embeddings = doc_embeddings)
```

**3. Search**  
```python
scores, ann_items = faiss_index.search(query_embeddings = query_embeddings,
                                       topk = 100,
                                       nprobe = 10)
```




