# LibVQ
LibVQ is a library for efficient and effective approximate nearest neighbor search.


## Introduction
LibVQ can create a learnable ANN index, and provide several strategies to fintune the index.
The other important function it provieded is integrating the representation learning and index construction,
which can improve the quality of ANN significantly.  

LibVQ is implemented based on [Faiss](https://github.com/facebookresearch/faiss) library, 
and can get a better performance than Faiss with the same efficiency. 
Now, the type of index it supported including IVF and PQ.



## Install
- From source
```
git clone https://github.com/staoxiao/LibVQ.git
cd LibVQ
pip install .
```

## Workflow
In LibVQ, users can construct a index and train it by a simple way.
Please refer to our [docs](Docs.md) for more details.
Besides, we provide some examples below to illustrate the usage of LibVQ.

## Examples
### [MSMARCO](./examples/MSMARCO/)  
- IVFPQ    

Methods | MRR@10 | Recall@10 | Recall@100 | 
------- | ------- | ------- |  ------- |
Faiss-IVFPQ | 0.1380 | 0.2820 | 0.5617 |  
Faiss-IVFOPQ | 0.3102 | 0.5593 | 0.8148 |  
Scann | 0.1791 | 0.3499 | 0.6345 | 
LibVQ(contrastive_index) | 0.3179 | 0.5724 | 0.8214 | 
LibVQ(distill_index) | 0.3253 | 0.5765 | 0.8256 | 
LibVQ(contrastive_index-and-query-encoder) | 0.3192 | 0.5799 | 0.8427 |  
LibVQ(distill_index-and-query-encoder) | **0.3311** | **0.5907** | **0.8429** |  
LibVQ(distill_index-and-query-encoder_nolabel) | 0.3285 | 0.5875 | 0.8401 | 

- PQ

Methods | MRR@10 | Recall@10 | Recall@100 | 
------- | ------- | ------- |  ------- | 
Faiss-PQ | 0.1145 | 0.2369 | 0.5046 |  
Faiss-OPQ | 0.3268 | 0.5939 | 0.8651 |    
LibVQ(distill_index-and-query-encoder) | 0.3437 | 0.6201 | 0.8819 | 
LibVQ(distill_index-and-two-encoders) | **0.3475** | **0.6223** | **0.8901** |  

More details please refer to [examples/MSMARCO](./examples/MSMARCO/).  

### More examples are comming soon





