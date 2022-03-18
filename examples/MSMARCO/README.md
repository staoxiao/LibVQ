# MSMARCO Passage Ranking
We take the Passage dataset as the example.   

## Preparing Data
Dataload data:
```
bash download_data.sh
```
Then convert and preprocess them to the format which is need for our dataset class: 
```
python convert_data_format.py
```
The data will be saved into `./data/passage/dataset`.

## Generate Embeddings
We use the [co-codenser](https://github.com/luyug/Condenser) as the text encoder:
```
python get_embeddings.py  \
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
python faiss_index.py  \
--preprocess_dir ./data/passage/preprocess \
--pretrained_model_name Luyu/co-condenser-marco-retriever \
--output_dir ./data/passage/evaluate/co-condenser \
--index_method ivf_pq \
--ivf_centers_num 10000 \
--subvector_num 32 \
--subvector_bits 8 \
--nprobe 100
```

+ ### ScaNN Index
```
python scann_index.py  \
--preprocess_dir ./data/passage/preprocess \
--pretrained_model_name Luyu/co-condenser-marco-retriever \
--output_dir ./data/passage/evaluate/co-condenser \
--ivf_centers_num 10000 \
--subvector_num 32 \
--nprobe 100
```


+ ### Learnable Index
**Finetune the index with fixed embeddings:**  
(need the embeddings of queries and docs)
```
python train_index.py  \
--preprocess_dir ./data/passage/preprocess \
--output_dir ./data/passage/evaluate/co-condenser \
--query_embeddings_file ./data/passage/evaluate/co-condenser/train-queries.memmap \
--doc_embeddings_file  ./data/passage/evaluate/co-condenser/docs.memmap \
--index_method ivf_opq \
--ivf_centers_num 10000 \
--subvector_num 32 \
--subvector_bits 8 \
--nprobe 100 \
--training_mode {distill_index, distill_virtual-data_index, contrastive_index} \
--per_device_train_batch_size 512
```

**Jointly train index and query encoder (always has a better performance):**  
(need embeddings and a query encoder)
```
python train_index_and_encoder.py  \
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


python train_index_and_encoder.py  \
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
--training_mode contrastive_jointly \
--per_device_train_batch_size 512
```
We provide several different training modes:
1. **contrastive**: contrastive learning;
2. **distill**: knowledge distillation; transfer knowledge (i.e., the order of docs) from the dense vector to the IVF and PQ
3. **distill_virtual-data**: knowledge distillation for non-label data; in this way, 
first to find the top-k docs for each train queries by brute-force search (or a index with high performance), 
then use these results to form a new train data.    

More details of implementation please refer to [train_index.py](train_index.py) and [train_index_and_encoder](train_index_and_encoder.py).


+ ### Results

Methods | MRR@10 | Recall@10 | Recall@100 | 
------- | ------- | ------- |  ------- |
Faiss-IVFPQ | 0.1380 | 0.2820 | 0.5617 |  
Faiss-IVFOPQ | 0.3102 | 0.5593 | 0.8148 |  
Scann | 0.1791 | 0.3499 | 0.6345 | 
LearnableIndex(contrastive_index) | 0.3179 | 0.5724 | 0.8214 | 
LearnableIndex(distill_index) | 0.3253 | 0.5765 | 0.8256 | 
LearnableIndex(contrastive_jointly) | 0.3192 | 0.5799 | 0.8427 |  
LearnableIndex(distill_jointly) | **0.3311** | **0.5907** | **0.8429** |  
LearnableIndex(distill_virtual-data_jointly) | 0.3285 | 0.5875 | 0.8401 | 





## PQ
+ ### Index      
For PQ, you can reuse above commands and only change the `--index_method` to `pq` or `opq`.
For example:
```
python faiss_index.py  \
--preprocess_dir ./data/passage/preprocess \
--pretrained_model_name Luyu/co-condenser-marco-retriever \
--output_dir ./data/passage/evaluate/co-condenser \
--index_method pq \
--subvector_num 32 \
--subvector_bits 8 
```

Besides, you can train both doc encoder and query encoder when only train PQ (`training_mode = distill_jointly_v2`).
```
python train_index_and_encoder.py  \
--data_dir ./data/passage/dataset \
--preprocess_dir ./data/passage/preprocess \
--pretrained_model_name Luyu/co-condenser-marco-retriever \
--max_doc_length 256 \
--max_query_length 32 \
--output_dir ./data/passage/evaluate/co-condenser \
--query_embeddings_file ./data/passage/evaluate/co-condenser/train-queries.memmap \
--doc_embeddings_file  ./data/passage/evaluate/co-condenser/docs.memmap \
--index_method opq \
--subvector_num 32 \
--subvector_bits 8 \
--training_mode distill_jointly_v2 \
--per_device_train_batch_size 128
```

+ ### Results
Methods | MRR@10 | Recall@10 | Recall@100 | 
------- | ------- | ------- |  ------- | 
Faiss-PQ | 0.1145 | 0.2369 | 0.5046 |  
Faiss-OPQ | 0.3268 | 0.5939 | 0.8651 |    
LearnableIndex(distill_jointly) | 0.3437 | 0.6201 | 0.8819 | 
LearnableIndex(distill_jointly_v2) | **0.3475** | **0.6223** | **0.8901** |  
 

