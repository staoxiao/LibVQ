# MSMARCO Ranking

## Preparing Data
Dataload data:
```
bash download_data.sh
```
Then convert and preprocess them to the format which is need for our class `DatasetForVQ`: 
```
python convert_data_format.py
python preprocess.py
```
 or you can rewrite the `torch.nn.utils.Dataset` for your data.

##### *TODO: upload our neg.json and models (huggingface website)*

## Faiss Index
Generate the embeddings of query and doc
```
data_type=passage
savename=AR_G
epoch=0
python ./inference.py \
--data_type ${data_type} \
--preprocess_dir ./data/${data_type}/preprocess/ \
--max_doc_length 256 --max_query_length 32 \
--eval_batch_size 8192 \
--ckpt_path ./saved_ckpts/${savename}/${epoch}/ \
--output_dir  evaluate/${savename}_${epoch} \
--mode train
```

Creat Faiss index:
```
data_type=passage
savename=AR_G
epoch=0
python ./src/index/FaissIndex.py \
--output_dir ./data/${data_type}/evaluate/${savename}_${epoch} \
--index ivf_opq \
--subvector_num 64 \
--subvector_bits 8 \
--ivf_centers 10000 \
--dist_mode ip \
--emb_size 768 \
--topk 1000 \
--data_type ${data_type} \
--rel_file ./data/${data_type}/preprocess/dev-qrel.tsv \
--nprobe 1
```

## Learnable Index

```
data_type=passage
python ./LibVQ/learnable_index.py \
--data_type ${data_type} \
--preprocess_dir ./data/${data_type}/preprocess/ 
```