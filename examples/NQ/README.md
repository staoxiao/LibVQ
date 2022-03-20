# NQ

## Preparing Data
Dataload data:
```
bash ./prepare_data/download_data.sh
```
Then convert and preprocess them to the format which is need for our dataset class: 
```
python ./prepare_data/convert_data_format.py
```
The data will be saved into `./data/NQ/dataset`.


## Generate Embeddings
We use the [co-codenser](https://github.com/luyug/Condenser) as the text encoder:
```
python ./prepare_data/get_embeddings.py  \
--data_dir ./data/NQ/dataset \
--preprocess_dir ./data/NQ/preprocess \
--max_doc_length 256 \
--max_query_length  32 \
--output_dir ./data/NQ/evaluate/dpr 
```
The code will preprocess the data into `preprocess_dir` (for training encoder)
and generate embeddings into `output_dir` (for training index). More information about data format 
pleaser refer to [dataset.README.md](../../LibVQ/dataset/README.md)


## IVFPQ
+ ### Faiss Index
```
python ./basic_index/faiss_index.py  \
--preprocess_dir ./data/NQ/preprocess \
--output_dir ./data/NQ/evaluate/dpr \
--index_method ivf_pq \
--ivf_centers_num 10000 \
--subvector_num 64 \
--subvector_bits 8 \
--nprobe 100

python ./basic_index/faiss_index.py  \
--preprocess_dir ./data/NQ/preprocess \
--output_dir ./data/NQ/evaluate/dpr \
--index_method ivf_opq \
--ivf_centers_num 10000 \
--subvector_num 64 \
--subvector_bits 8 \
--nprobe 1
```

+ ### ScaNN Index
```
python ./basic_index/scann_index.py  \
--preprocess_dir ./data/passage/preprocess \
--pretrained_model_name Luyu/co-condenser-marco-retriever \
--output_dir ./data/passage/evaluate/co-condenser \
--ivf_centers_num 10000 \
--subvector_num 64 \
--nprobe 100
```


+ ### Learnable Index
**Finetune the index with fixed embeddings:**  
(need the embeddings of queries and docs)
```
python ./learnable_index/train_index.py  \
--preprocess_dir ./data/passage/preprocess \
--output_dir ./data/passage/evaluate/co-condenser \
--query_embeddings_file ./data/passage/evaluate/co-condenser/train-queries.memmap \
--doc_embeddings_file  ./data/passage/evaluate/co-condenser/docs.memmap \
--index_method ivf_opq \
--ivf_centers_num 10000 \
--subvector_num 64 \
--subvector_bits 8 \
--nprobe 100 \
--training_mode {distill_index, distill_virtual-data_index, contrastive_index} \
--per_device_train_batch_size 512


python ./learnable_index/train_index.py  \
--preprocess_dir ./data/passage/preprocess \
--output_dir ./data/passage/evaluate/co-condenser \
--query_embeddings_file ./data/passage/evaluate/co-condenser/train-queries.memmap \
--doc_embeddings_file  ./data/passage/evaluate/co-condenser/docs.memmap \
--index_method ivf_opq \
--ivf_centers_num 10000 \
--subvector_num 64 \
--subvector_bits 8 \
--nprobe 100 \
--training_mode {distill_index, distill_virtual-data_index, contrastive_index} \
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
--output_dir ./data/passage/evaluate/co-condenser \
--query_embeddings_file ./data/passage/evaluate/co-condenser/train-queries.memmap \
--doc_embeddings_file  ./data/passage/evaluate/co-condenser/docs.memmap \
--index_method ivf_opq \
--ivf_centers_num 10000 \
--subvector_num 32 \
--subvector_bits 8 \
--nprobe 100 \
--training_mode {distill_jointly, distill_virtual-data_jointly, contrastive_jointly} \
--per_device_train_batch_size 512


python ./learnable_index/train_index_and_encoder.py  \
--data_dir ./data/NQ/dataset \
--preprocess_dir ./data/NQ/preprocess \
--max_doc_length 256 \
--max_query_length 32 \
--output_dir ./data/NQ/evaluate/dpr \
--query_embeddings_file ./data/NQ/evaluate/dpr/train-queries.memmap \
--doc_embeddings_file  ./data/NQ/evaluate/dpr/docs.memmap \
--index_method ivf_opq \
--ivf_centers_num 10000 \
--subvector_num 64 \
--subvector_bits 8 \
--nprobe 1 \
--training_mode contrastive_jointly \
--per_device_train_batch_size 512
```


+ ### Results

nprobe=1  

Methods | MRR@5 | Recall@10 | Recall@20 | Recall@100 | 
------- | ------- | ------- |   ------- | ------- |  
Faiss-IVFPQ | 0.5725 | 0.6351 | 0.6861 |  0.7656 |  
Faiss-IVFOPQ | 0. | 0. | 0. |  0. |  
Scann | 0. | 0. | 0. | 0. |  
LearnableIndex(contrastive_index) | 0. | 0. | 0. | 0. |  
LearnableIndex(distill_index) | 0. | 0. | 0. | 0. | 
LearnableIndex(contrastive_jointly) | 0. | 0. | 0. |  0. |  
LearnableIndex(distill_jointly) | **0.** | **0.** | **0.** |  0. |  
LearnableIndex(distill_virtual-data_jointly) | 0. | 0. | 0. | 0. |   


nprobe=100

Methods | MRR@5 | Recall@10 | Recall@20 | Recall@100 | 
------- | ------- | ------- |   ------- | ------- |  
Faiss-IVFPQ | 0.6551 | 0.7199 | 0.7673 |  0.8409 |  
Faiss-IVFOPQ | 0. | 0. | 0. |  0. |  
Scann | 0. | 0. | 0. | 0. |  
LearnableIndex(contrastive_index) | 0. | 0. | 0. | 0. |  
LearnableIndex(distill_index) | 0. | 0. | 0. | 0. | 
LearnableIndex(contrastive_jointly) | 0. | 0. | 0. |  0. |  
LearnableIndex(distill_jointly) | **0.** | **0.** | **0.** |  0. |  
LearnableIndex(distill_virtual-data_jointly) | 0. | 0. | 0. | 0. |

