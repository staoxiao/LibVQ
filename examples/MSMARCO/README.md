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
 or you can rewrite the `torch.nn.utils.Dataset` for your data.


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
and generate embeddings into `output_dir` (for training index).

## Faiss Index
```
python faiss_ivfopq.py  \
--preprocess_dir ./data/passage/preprocess \
--pretrained_model_name Luyu/co-condenser-marco-retriever \
--output_dir ./data/passage/evaluate/co-condenser \
--index_method ivf_pq \
--ivf_centers_num 10000 \
--subvector_num 32 \
--subvector_bits 8 \
--nprobe 1
```

# ScaNN Index
```
python scann_pq.py  \
--preprocess_dir ./data/passage/preprocess \
--pretrained_model_name Luyu/co-condenser-marco-retriever \
--output_dir ./data/passage/evaluate/co-condenser \
--ivf_centers_num 10000 \
--subvector_num 32 \
--nprobe 1
```


## Learnable Index
- Finetune the faiss index:
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
--training_mode distill_index \
--per_device_train_batch_size 64
```


- jointly train index and encoder
```
python learnable_ivfopq.py  \
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
--training_mode contrastive \
--per_device_train_batch_size 512


python learnable_ivfopq.py  \
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
--training_mode distill \
--per_device_train_batch_size 512


python learnable_ivfopq.py  \
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
--training_mode distill_nolabel \
--per_device_train_batch_size 64
```
Besides, we give some examples for different methods (contrastive learning, distillation) to train the index in `learnable_ivfopq.py`.




## Results
Settings:
```
ivf_centers_num=10000
subvector_num=32
subvector_bits=8
```

Methods (nprobe=1) | MRR@10 | Recall@10 | Recall@100 | Time(ms per query) |
------- | ------- | ------- |  ------- | ------- |
Faiss-IVFPQ | 0.0976 | 0.1867 | 0.3679 | |
Faiss-IVFOPQ | 0.1684 | 0.2968 | 0.4341 |  |
Scann | 0.1212 | 0.2290 | 0.3994 | |
LearnableIndex-contras | 0.1728 | 0.3010 | 0.4353 | |
LearnableIndex-Distill_nolabel |  | | | |
jointly_LearnableIndex-Contras | 0.1829 | 0.3198 | 0.4661 | |
jointly_LearnableIndex-Distill | 0.1976 | 0.3379 | 0.4790 | |
jointly_LearnableIndex-Distill_nolabel | | | | |



Methods (nprobe=100)| MRR@10 | Recall@10 | Recall@100 | Time(ms per query) |
------- | ------- | ------- |  ------- | ------- |
Faiss-IVFPQ | 0.1380 | 0. 2820 | 0.5617 |  |
Faiss-IVFOPQ | 0.3102 | 0.5593 | 0.8148 |  |
Scann | 0.1791 | 0.3499 | 0.6345 | |
LearnableIndex-contras | 0.3179 | 0.5724 | 0.8214 | |
LearnableIndex-Distill | | | | |
jointly_LearnableIndex-Contras | 0.3192 | 0.5799 | 0.8427 |  |
jointly_LearnableIndex-Distill | 0.3311 | 0.5907 | 0.8429 |  |
jointly_LearnableIndex-Distill_nolabel | | | | |





