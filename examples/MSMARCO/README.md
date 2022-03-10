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


## Faiss Index
We use [co-codenser](https://github.com/luyug/Condenser) as the text encoder to show the running workflow.

```
python faiss_ivfopq.py  \
--data_dir ./data/passage/dataset \
--preprocess_dir ./data/passage/preprocess \
--pretrained_model_name Luyu/co-condenser-marco-retriever \
--max_doc_length 256 \
--max_query_length 32 \
--output_dir ./data/passage/evaluate/co-condenser \
--index_method ivf_pq \
--ivf_centers_num 10000 \
--subvector_num 32 \
--subvector_bits 8 \
--nprobe 1 
```

# ScaNN Index
```
python scann.py  \
--data_dir ./data/passage/dataset \
--preprocess_dir ./data/passage/preprocess \
--pretrained_model_name Luyu/co-condenser-marco-retriever \
--max_doc_length 256 \
--max_query_length 32 \
--output_dir ./data/passage/evaluate/co-condenser \
--ivf_centers_num 10000 \
--subvector_num 32 \
--nprobe 1 
```


## Learnable Index
Finetune the faiss index:
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
--nprobe 1 \
--loss_method contras \
--per_device_train_batch_size 160


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
--nprobe 1 \
--loss_method distill \
--per_device_train_batch_size 640
```
Besides, we give some examples for different methods (contrastive learning, distillation) to train the index in `learnable_ivfopq.py`.


## Results
Settings:
```
nprobe=1
ivf_centers_num=10000
subvector_num=32
subvector_bits=8
```

Methods | MRR@10 | Recall@10 | Recall@100 | Time(ms per query) |
------- | ------- | ------- |  ------- | ------- |
Faiss-IVFPQ | 0.0976 | 0.1867 | 0.3679 | 0.37 |
Faiss-IVFOPQ | 0.1684 | 0.2968 | 0.4341 | 0.44 |
Scann | | | | |
LearnableIndex-Contras | | | | |
LearnableIndex-Distill | | | | |
LearnableIndex-Distill_nolabel | | | | |
LearnableIndex-Distill_fix-emb | | | | |