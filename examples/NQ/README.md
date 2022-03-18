# NQ

## Preparing Data
Dataload data:
```
bash download_data.sh
```
Then convert and preprocess them to the format which is need for our dataset class: 
```
python convert_data_format.py
```
The data will be saved into `./data/NQ/dataset`.


## Generate Embeddings
We use the [co-codenser](https://github.com/luyug/Condenser) as the text encoder:
```
python get_embeddings.py  \
--data_dir ./data/NQ/dataset \
--preprocess_dir ./data/NQ/preprocess \
--pretrained_model_name Luyu/co-condenser-wiki \
--max_doc_length 256 \
--max_query_length  32 \
--output_dir ./data/NQ/evaluate/ 
```
The code will preprocess the data into `preprocess_dir` (for training encoder)
and generate embeddings into `output_dir` (for training index). More information about data format 
pleaser refer to [dataset.README.md](../../LibVQ/dataset/README.md)



