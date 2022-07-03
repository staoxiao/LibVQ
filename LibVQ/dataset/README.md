# Dataset
```python
from LibVQ.dataset import DatasetForVQ
```
params:
- **rel_data**: positive doc ids for each query: {query_id:[doc1_id, doc2_id,...]}, or a tsv file.
- **query_data_dir**: path to the preprocessed tokens data (needed for jointly training query encoder).
- **max_query_length**: max length of query tokens sequence.
- **doc_data_dir**: path to the preprocessed tokens data (needed for jointly training doc encoder).
- **max_doc_length**: max length of doc tokens sequence.
- **per_query_neg_num**: the number of negatives for each query.
- **neg_data**: negative doc ids for each query: {query_id:[doc_id1, doc_id2,...]}, or a pickle file which save the query2neg.
                        If set None, it will randomly sample negative.
- **query_embeddings**: embeddings for each query, also support pass a filename('.npy', '.memmap').
- **doc_embeddings**: embeddigns for each doc, also support pass a filename('.npy', '.memmap').
- **emb_size**: dim of embeddings.

To use our dataset module, you should organize the data in the following format:

## Embeddings and Relevance label
Two matrices are needed: query_embeddings, doc_embeddings, where query/doc_embeddings[i] 
is the embeddings of i'th query/doc. 
They also can be saved as `.npy` or `.memmap` file by `numpy`, then input the filename to the DatasetForVQ.

For training the index, the relevance relationship between query and doc should be provided. 
The format of rel_data should be: `{query_id: [doc1_id, doc2_id,...]`, and you can save it as a pickle file.
If there is no relevance lable, you can create a flat index and generate the relevance label by 
searching the nearest docs:
```python
from LibVQ.baseindex import FaissIndex
index = FaissIndex(doc_embeddings=doc_embeddings, index_method='flat')
query2pos, query2neg = index.generate_virtual_traindata(query_embeddings)
```


## Text Data (needed only when jointly encoder and index)
### 1. Dataset Format
- collection.tsv  
`doc_id \t text_1 \t text_2,... \n`  
For example:
```
0    https://answers.yahoo.com/question/index?qid=20080718121858AAmfk0V      I have trouble swallowing due to MS, can I crush valium & other meds to be easier to swallowll?
1    http://vanrcook.tripod.com/presidentroosevelt.htm       President Roosevelt Led US To Victory In World War 2    "In World War 2, the three great Allied leaders against 
```

- {mode}-queries.tsv  
`query_id, text_1, text_2,... \n`  
For example:
```
0    what does physical medicine do
1    what is a flail chest
```

- {mode}-rels.tsv  
`query_id \t doc_id`  
For example:
```
0    3
0    2022
1    666
```

### 2. Preprocess

```python
from LibVQ.dataset.preprocess import preprocess_data
preprocess_data(data_dir={path to dataset},
                output_dir={path to preprocess_dir},
                text_tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased'),
                add_cls_tokens=True,
                max_doc_length=32,
                max_query_length=256,
                workers_num=64)

```
The above code will preprocess the dataset in `data_dir` and save the results to `output_dir`.
It will tokenize the docs and queries, and save the tokens and the number of tokens 
into `xx.memmap` and `xx_length.npy`, respectively.
Besides, the docs and queries will be assgned to a new id, and a new rel file will be generated.


