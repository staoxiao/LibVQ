# MSMARCO Passage Ranking
We take the Passage dataset as the example.   

## Preparing Data
Download data and convert to our format:
```
bash ./prepare_data/download_data.sh
```
The data will be saved into `./data/passage/dataset`.


## Preprocess and Generate Embeddings 
We use the [co-codenser](https://github.com/luyug/Condenser) as the text encoder:
```
python ./prepare_data/get_embeddings.py  \
--data_dir ./data/passage/dataset \
--preprocess_dir ./data/passage/preprocess \
--pretrained_model_name Luyu/co-condenser-marco-retriever \
--max_doc_length 256 \
--max_query_length 32 \
--output_dir ./data/passage/evaluate/co-condenser 
```
The code will preprocess the data into `preprocess_dir` (for training encoder)
and generate embeddings into `output_dir` (for training index). More information about data format 
pleaser refer to [dataset.README.md](../../LibVQ/dataset/README.md)



## IVFPQ
+ ### Faiss Index
```
python ./basic_index/faiss_index.py  \
--preprocess_dir ./data/passage/preprocess \
--embeddings_dir ./data/passage/evaluate/co-condenser \
--index_method ivf_pq \
--ivf_centers_num 10000 \
--subvector_num 32 \
--subvector_bits 8 \
--nprobe 100
```

+ ### ScaNN Index
```
python ./basic_index/scann_index.py  \
--preprocess_dir ./data/passage/preprocess \
--embeddings_dir ./data/passage/evaluate/co-condenser \
--ivf_centers_num 10000 \
--subvector_num 32 \
--nprobe 100
```


+ ### Learnable Index
**Finetune the index with fixed embeddings:**  
(need the embeddings of queries and docs)
```
python ./learnable_index/train_index.py  \
--preprocess_dir ./data/passage/preprocess \
--embeddings_dir ./data/passage/evaluate/co-condenser \
--index_method ivf_opq \
--ivf_centers_num 10000 \
--subvector_num 32 \
--subvector_bits 8 \
--nprobe 100 \
--training_mode {distill_index, distill_index_nolabel, contrastive_index} \
--per_device_train_batch_size 512


python ./learnable_index/train_index.py  \
--preprocess_dir ./data/passage/preprocess \
--embeddings_dir ./data/passage/evaluate/co-condenser \
--index_method ivf_opq \
--ivf_centers_num 10000 \
--subvector_num 32 \
--subvector_bits 8 \
--nprobe 100 \
--training_mode distill_index_nolabel \
--per_device_train_batch_size 512
```

**Jointly train index and query encoder (always has a better performance):**  
(need embeddings and a query encoder)
```
python ./learnable_index/train_index_and_encoder.py  \
--data_dir ./data/passage/dataset \
--preprocess_dir ./data/passage/preprocess \
--pretrained_model_name Luyu/co-condenser-marco-retriever \
--max_doc_length 256 \
--max_query_length 32 \
--embeddings_dir ./data/passage/evaluate/co-condenser \
--index_method ivf_opq \
--ivf_centers_num 10000 \
--subvector_num 32 \
--subvector_bits 8 \
--nprobe 100 \
--training_mode {distill_index-and-query-encoder, distill_index-and-query-encoder_nolabel, contrastive_index-and-query-encoder} \
--per_device_train_batch_size 512
```
We provide several different training modes:
1. **contrastive_()**: contrastive learning;
2. **distill_()**: knowledge distillation; transfer knowledge (i.e., the order of docs) from the dense vector to the IVF and PQ
3. **distill_()_nolabel**: knowledge distillation for non-label data; in this way, 
first to find the top-k docs for each train queries by brute-force search (or a index with high performance), 
then use these results to form a new train data.    

More details of implementation please refer to [train_index.py](train_index.py) and [train_index_and_encoder](train_index_and_encoder.py).


+ ### Results

Methods | MRR@10 | Recall@10 | Recall@100 | 
------- | ------- | ------- |  ------- |
[Faiss-IVFPQ](./examples/MSMARCO/basic_index/faiss_index.py) | 0.1380 | 0.2820 | 0.5617 |  
[Faiss-IVFOPQ](./examples/MSMARCO/basic_index/faiss_index.py) | 0.3102 | 0.5593 | 0.8148 |  
[Scann](./examples/MSMARCO/basic_index/scann_index.py) | 0.1791 | 0.3499 | 0.6345 | 
[LibVQ(contrastive_index)](./examples/MSMARCO/learnable_index/train_index.py) | 0.3179 | 0.5724 | 0.8214 | 
[LibVQ(distill_index)](./examples/MSMARCO/learnable_index/train_index.py) | 0.3253 | 0.5765 | 0.8256 | 
[LibVQ(distill_index)](./examples/MSMARCO/learnable_index/train_index.py) | 0.3253 | 0.5765 | 0.8256 | 
[LibVQ(contrastive_index-and-query-encoder)](./examples/MSMARCO/learnable_index/train_index_and_encoder.py) | 0.3192 | 0.5799 | 0.8427 |  
[LibVQ(distill_index-and-query-encoder)](./examples/MSMARCO/learnable_index/train_index_and_encoder.py) | **0.3311** | **0.5907** | **0.8429** |  
[LibVQ(distill_index-and-query-encoder_nolabel)](./examples/MSMARCO/learnable_index/train_index_and_encoder.py) | 0.3285 | 0.5875 | 0.8401 | 




## PQ
+ ### Index      
For PQ, you can reuse above commands and only change the `--index_method` to `pq` or `opq`.
For example:
```
python ./basic_index/faiss_index.py  \
--preprocess_dir ./data/passage/preprocess \
--embeddings_dir ./data/passage/evaluate/co-condenser \
--index_method pq \
--subvector_num 32 \
--subvector_bits 8 
```

Besides, you can train both doc encoder and query encoder when only train PQ (`training_mode = distill_jointly_v2`).
```
python ./learnable_index/train_index_and_encoder.py  \
--data_dir ./data/passage/dataset \
--preprocess_dir ./data/passage/preprocess \
--pretrained_model_name Luyu/co-condenser-marco-retriever \
--max_doc_length 256 \
--max_query_length 32 \
--embeddings_dir ./data/passage/evaluate/co-condenser \
--index_method opq \
--subvector_num 32 \
--subvector_bits 8 \
--training_mode distill_index-and-two-encoders \
--per_device_train_batch_size 128

python ./learnable_index/train_index_and_encoder.py  \
--data_dir ./data/passage/dataset \
--preprocess_dir ./data/passage/preprocess \
--pretrained_model_name Luyu/co-condenser-marco-retriever \
--max_doc_length 256 \
--max_query_length 32 \
--embeddings_dir ./data/passage/evaluate/co-condenser \
--index_method opq \
--subvector_num 32 \
--subvector_bits 8 \
--training_mode distill_index \
--per_device_train_batch_size 128
```

+ ### Results

Methods | MRR@10 | Recall@10 | Recall@100 | 
------- | ------- | ------- |  ------- | 
[Faiss-PQ](./examples/MSMARCO/basic_index/faiss_index.py) | 0.1145 | 0.2369 | 0.5046 |  
[Faiss-OPQ](./examples/MSMARCO/basic_index/faiss_index.py) | 0.3268 | 0.5939 | 0.8651 |    
[LibVQ(distill_index-and-query-encoder)](./examples/MSMARCO/learnable_index/train_index_and_encoder.py) | 0.3437 | 0.6201 | 0.8819 | 
[LibVQ(distill_index-and-two-encoders)](./examples/MSMARCO/learnable_index/train_index_and_encoder.py) | **0.3475** | **0.6223** | **0.8901** |  


