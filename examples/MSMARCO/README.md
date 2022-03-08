# MSMARCO Passage Ranking
We take the Passage dataset as the example.   
*For Doc dataset, just set `--data_type` to `doc`.*

## Preparing Data
Dataload data:
```
bash download_data.sh
```
Then convert and preprocess them to the format which is need for our dataset class: 
```
python convert_data_format.py
python preprocess.py 
```
 or you can rewrite the `torch.nn.utils.Dataset` for your data.


##### *TODO: upload our neg.json and models (huggingface website)*

## Faiss Index and Learnable Index
```
python ivfopq.py
```