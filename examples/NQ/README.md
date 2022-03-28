# NQ

## Preparing Data
Download data and convert to our format:
```
bash ./prepare_data/download_data.sh
```
The data will be saved into `./data/NQ/dataset`.


## Preprocess and Generate Embeddings 
We use the [DPR](https://github.com/facebookresearch/DPR) as the text encoder:
```
python ./prepare_data/get_embeddings.py  \
--data_dir ./data/NQ/dataset \
--preprocess_dir ./data/NQ/preprocess \
--tokenizer_name bert-base-uncased \
--max_doc_length 256 \
--max_query_length 32 \
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
--embeddings_dir ./data/NQ/evaluate/dpr \
--index_method ivf_opq \
--ivf_centers_num 1000 \
--subvector_num 8 \
--subvector_bits 8 \
--nprobe 10
```

top5:0.2886426592797784, top10:0.3664819944598338, top20:0.45013850415512463, top30:0.4955678670360111, top50:0.5421052631578948, top100:0.6013850415512465


+ ### ScaNN Index
```
python ./basic_index/scann_index.py  \
--preprocess_dir ./data/NQ/preprocess \
--embeddings_dir ./data/NQ/evaluate/dpr \
--ivf_centers_num 1000 \
--subvector_num 8 \
--nprobe 10
```


+ ### Learnable Index
**Finetune the index with fixed embeddings:**  
(need the embeddings of queries and docs)
```
python ./learnable_index/train_index.py  \
--preprocess_dir ./data/NQ/preprocess \
--embeddings_dir ./data/NQ/evaluate/dpr \
--index_method ivf_opq \
--ivf_centers_num 1000 \
--subvector_num 8 \
--subvector_bits 8 \
--nprobe 10 \
--training_mode {distill_index, distill_index_nolabel, contrastive_index} 
```

**Jointly train index and query encoder (always has a better performance):**  
(need embeddings and a query encoder)
```
python ./learnable_index/train_index_and_encoder.py  \
--data_dir ./data/NQ/dataset \
--preprocess_dir ./data/NQ/preprocess \
--max_doc_length 256 \
--max_query_length 32 \
--embeddings_dir ./data/NQ/evaluate/dpr \
--index_method ivf_opq \
--ivf_centers_num 1000 \
--subvector_num 8 \
--subvector_bits 8 \
--nprobe 10 \
--training_mode {distill_index-and-query-encoder, distill_index-and-query-encoder_nolabel, contrastive_index-and-query-encoder} 
```
We provide several different training modes:
1. **contrastive_()**: contrastive learning;
2. **distill_()**: knowledge distillation; transfer knowledge (i.e., the order of docs) from the dense vector to the IVF and PQ
3. **distill_()_nolabel**: knowledge distillation for non-label data; in this way, 
first to find the top-k docs for each train queries by brute-force search (or a index with high performance), 
then use these results to form a new train data.    

More details of implementation please refer to [train_index.py](./learnable_index/train_index.py) and [train_index_and_encoder](./learnable_index/train_index_and_encoder.py).


+ ### Results

Methods | Recall@5 | Recall@10 | Recall@20 | Recall@100 | 
------- | ------- | ------- |  ------- |  ------- |
[Faiss-IVFPQ](./examples/NQ/basic_index/faiss_index.py) | 0.1504 | 0.2052 | 0.2722 | 0.4523 |  
[Faiss-IVFOPQ](./examples/NQ/basic_index/faiss_index.py) | 0.3332 | 0.4279 | 0.5110 | 0.6817 |  
[Scann](./examples/NQ/basic_index/scann_index.py) | 0.2526 | 0.3351 | 0.4144 | 0.6016 |
[LibVQ(contrastive_index)](./examples/NQ/learnable_index/train_index.py) | 0.3398 | 0.4415 | 0.5232 | 0.6911 
[LibVQ(distill_index)](./examples/NQ/learnable_index/train_index.py) | 0.3952 | 0.4900 | 0.5667 | 0.7232
[LibVQ(distill_index_nolabel)](./examples/NQ/learnable_index/train_index.py) | 0.4066 | 0.4936 | 0.5759 | 0.7301
[LibVQ(contrastive_index-and-query-encoder)](./examples/NQ/learnable_index/train_index_and_encoder.py) | 0.3548 | 0.4470 | 0.5390 | 0.7120 
[LibVQ(distill_index-and-query-encoder)](./examples/NQ/learnable_index/train_index_and_encoder.py) | 0.4725 | 0.5681 | 0.6429 | 0.7739 
[LibVQ(distill_index-and-query-encoder_nolabel)](./examples/NQ/learnable_index/train_index_and_encoder.py) | **0.4977** | **0.5822** | **0.6484** | **0.7764**




## PQ
+ ### Index      
For PQ, you can reuse above commands and only change the `--index_method` to `pq` or `opq`.
For example:
```
python ./learnable_index/train_index.py  \
--data_dir ./data/NQ/dataset \
--preprocess_dir ./data/NQ/preprocess \
--max_doc_length 256 \
--max_query_length 32 \
--embeddings_dir ./data/NQ/evaluate/dpr \
--index_method opq \
--subvector_num 8 \
--subvector_bits 8 \
--training_mode distill_index_nolabel
```

Besides, you can train both doc encoder and query encoder when only train PQ (`training_mode = distill_index-and-two-encoders`).
```
python ./learnable_index/train_index_and_encoder.py  \
--data_dir ./data/NQ/dataset \
--preprocess_dir ./data/NQ/preprocess \
--max_doc_length 256 \
--max_query_length 32 \
--embeddings_dir ./data/NQ/evaluate/dpr \
--index_method opq \
--subvector_num 8 \
--subvector_bits 8 \
--training_mode distill_index-and-two-encoders
```


+ ### Results

Methods | Recall@5 | Recall@10 | Recall@20 | Recall@100 |
------- | ------- | ------- |  ------- | ------- | 
[Faiss-PQ](./examples/NQ/basic_index/faiss_index.py) | 0.1301 | 0.1861 | 0.2495 | 0.4188  
[Faiss-OPQ](./examples/NQ/basic_index/faiss_index.py) | 0.3166 | 0.4105 | 0.4961 | 0.6836  
[Scann](./examples/NQ/basic_index/scann_index.py) | 0.2526 | 0.3351 | 0.4144 | 0.6013 |
[LibVQ(distill_index)](./examples/NQ/learnable_index/train_index.py) | 0.3817 | 0.4806 | 0.5681 | 0.7357  
[LibVQ(distill_index_nolabel)](./examples/NQ/learnable_index/train_index.py) | 0.3880 | 0.4858 | 0.5819 | 0.7423    
[LibVQ(distill_index-and-query-encoder)](./examples/NQ/learnable_index/train_index_and_encoder.py) | 0.4709 | 0.5689 | 0.6481 | 0.7930   
[LibVQ(distill_index-and-query-encoder_nolabel)](./examples/NQ/learnable_index/train_index_and_encoder.py) | 0.4883 | 0.5903 | 0.6678 | 0.7914   
[LibVQ(distill_index-and-two-encoders)](./examples/NQ/learnable_index/train_index_and_encoder.py) | **0.5637** | **0.6515** | **0.7171** | **0.8257**  
[LibVQ(distill_index-and-two-encoders_nolabel)](./examples/NQ/learnable_index/train_index_and_encoder.py) | 0.5285 | 0.6144 | 0.7296 | 0.8096  


