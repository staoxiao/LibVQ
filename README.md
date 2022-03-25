# LibVQ
A Library For Dense Retrieval Oriented Vector Quantization


## Introduction
Vector quantization (VQ) is widely applied to many ANN libraries, like [FAISS](https://github.com/facebookresearch/faiss), [ScaNN](https://github.com/google-research/google-research/tree/master/scann), [SPTAG](https://github.com/microsoft/SPTAG), [DiskANN](https://github.com/microsoft/DiskANN) to facilitate real-time and memory-efficient dense retrieval. However, conventional vector quantization methods, like [IVF](https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_searching_with_quantization.pdf), [PQ](https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_searching_with_quantization.pdf), [OPQ](http://kaiminghe.com/cvpr13/index.html), are not optimized for the retrieval quality. In this place, We present **LibVQ**, the first library developed for dense retrieval oriented vector quantization. LibVQ is highlighted for the following features:

- **Knowledge Distillation.** The knowledge distillation  based learning process can be directly applied to the off-the-shelf embeddings. It gives rise to the strongest retrieval performance in comparison with any existing VQ based ANN indexes. 

- **Flexible usage and input conditions.** LibVQ may flexibly support different usages, e.g., training VQ parameters only, or joint adaptation of query encoder. LibVQ is designed to handle a wide range of input conditions: it may work only with off-the-shelf embeddings; it may also leverage extra data, e.g., relevance labels, and source queries, for further enhancement. 

- **Learning and Deployment.** The learning is backended by **PyTorch**, which can be easily configured for the efficient training based on different computation resources. The well-trained VQ parameters are wrapped up with **FAISS** backend ANN indexes, e.g., IndexPQ, IndexIVFPQ, etc., which are directly deployable for large-scale dense retrieval applications. 



## Install
- From source
```
git clone https://github.com/staoxiao/LibVQ.git
cd LibVQ
pip install .
```

## Quickstart
In LibVQ, users can construct a index and train it by a simple way.
Please refer to our [docs](./Docs) for more details.
Besides, we provide some examples below to illustrate the usage of LibVQ.

## Examples
### [MSMARCO](./examples/MSMARCO/)  
- IVFPQ (Compression Ratio = 96)   

Methods | MRR@10 | Recall@10 | Recall@100 | 
------- | ------- | ------- |  ------- |
[Faiss-IVFPQ](./examples/MSMARCO/basic_index/faiss_index.py) | 0.1380 | 0.2820 | 0.5617 |  
[Faiss-IVFOPQ](./examples/MSMARCO/basic_index/faiss_index.py) | 0.3102 | 0.5593 | 0.8148 |  
[Scann](./examples/MSMARCO/basic_index/scann_index.py) | 0.1791 | 0.3499 | 0.6345 | 
[LibVQ(contrastive_index)](./examples/MSMARCO/learnable_index/train_index.py) | 0.3179 | 0.5724 | 0.8214 | 
[LibVQ(distill_index)](./examples/MSMARCO/learnable_index/train_index.py) | 0.3253 | 0.5765 | 0.8256 | 
[LibVQ(distill_index_nolabel)](./examples/MSMARCO/learnable_index/train_index.py) | 0.3234 | 0.5813 | 0.8269 | 
[LibVQ(contrastive_index-and-query-encoder)](./examples/MSMARCO/learnable_index/train_index_and_encoder.py) | 0.3192 | 0.5799 | 0.8427 |  
[LibVQ(distill_index-and-query-encoder)](./examples/MSMARCO/learnable_index/train_index_and_encoder.py) | **0.3311** | **0.5907** | **0.8429** |  
[LibVQ(distill_index-and-query-encoder_nolabel)](./examples/MSMARCO/learnable_index/train_index_and_encoder.py) | 0.3285 | 0.5875 | 0.8401 | 


- PQ (Compression Ratio = 96) 

Methods | MRR@10 | Recall@10 | Recall@100 | 
------- | ------- | ------- |  ------- | 
[Faiss-PQ](./examples/MSMARCO/basic_index/faiss_index.py) | 0.1145 | 0.2369 | 0.5046 |  
[Faiss-OPQ](./examples/MSMARCO/basic_index/faiss_index.py) | 0.3268 | 0.5939 | 0.8651 |   
[Scann](./examples/MSMARCO/basic_index/scann_index.py) | 0.1795 | 0.3516 | 0.6409 |  
[LibVQ(distill_index)](./examples/MSMARCO/learnable_index/train_index.py) | 0.3435 | 0.6203 | 0.8825 | 
[LibVQ(distill_index_nolabel)](./examples/MSMARCO/learnable_index/train_index.py) | 0.3467 | 0.6180 | 0.8849 | 
[LibVQ(distill_index-and-query-encoder)](./examples/MSMARCO/learnable_index/train_index_and_encoder.py) | 0.3446 | 0.6201 | 0.8837 | 
[LibVQ(distill_index-and-two-encoders)](./examples/MSMARCO/learnable_index/train_index_and_encoder.py) | **0.3475** | **0.6223** | **0.8901** |  


### [NQ](./examples/NQ/)  
- IVFPQ  (Compression Ratio = 384) 

Methods | Recall@5 | Recall@10 | Recall@20 | Recall@100 | 
------- | ------- | ------- |  ------- |  ------- |
[Faiss-IVFPQ](./examples/NQ/basic_index/faiss_index.py) | 0.1504 | 0.2052 | 0.2722 | 0.4523 |  
[Faiss-IVFOPQ](./examples/NQ/basic_index/faiss_index.py) | 0.3332 | 0.4279 | 0.5110 | 0.6817 |  
[Scann](./examples/NQ/basic_index/scann_index.py) | 0.2526 | 0.3351 | 0.4144 | 0.6016 |
[LibVQ(contrastive_index)](./examples/NQ/learnable_index/train_index.py) | 0.3398 | 0.4415 | 0.5232 | 0.6911 
[LibVQ(distill_index)](./examples/NQ/learnable_index/train_index.py) | 0.3952 | 0.4900 | 0.5667 | 0.7232
[LibVQ(distill_index_nolabel)](./examples/NQ/learnable_index/train_index.py) | **0.4066** | **0.4936** | **0.5759** | **0.7301**
[LibVQ(contrastive_index-and-query-encoder)](./examples/NQ/learnable_index/train_index_and_encoder.py) | 0.3548 | 0.4470 | 0.5390 | 0.7120 
[LibVQ(distill_index-and-query-encoder)](./examples/NQ/learnable_index/train_index_and_encoder.py) | 0.3759 | 0.4698 | 0.5515 | 0.7121 
[LibVQ(distill_index-and-query-encoder_nolabel)](./examples/NQ/learnable_index/train_index_and_encoder.py) | 0.3878 | 0.4789 | 0.5523 | 0.7152



- PQ  (Compression Ratio = 384) 

Methods | Recall@5 | Recall@10 | Recall@20 | Recall@100 |
------- | ------- | ------- |  ------- | ------- | 
[Faiss-PQ](./examples/NQ/basic_index/faiss_index.py) | 0.1301 | 0.1861 | 0.2495 | 0.4188  
[Faiss-OPQ](./examples/NQ/basic_index/faiss_index.py) | 0.3166 | 0.4105 | 0.4961 | 0.6836  
[Scann](./examples/NQ/basic_index/scann_index.py) | 0.2526 | 0.3351 | 0.4144 | 0.6013 |
[LibVQ(distill_index)](./examples/NQ/learnable_index/train_index.py) | 0.3817 | 0.4806 | 0.5681 | 0.7357  
[LibVQ(distill_index_nolabel)](./examples/NQ/learnable_index/train_index.py) | 0.3880 | 0.4858 | 0.5819 | 0.7423    
[LibVQ(distill_index-and-query-encoder)](./examples/NQ/learnable_index/train_index_and_encoder.py) | 0.3590 | 0.4570 | 0.5465 | 0.7202   
[LibVQ(distill_index-and-two-encoders)](./examples/NQ/learnable_index/train_index_and_encoder.py) | **0.4426** | **0.5376** | **0.6191** | **0.7709**  
[LibVQ(distill_index-and-two-encoders_nolabel)](./examples/NQ/learnable_index/train_index_and_encoder.py) | 0.4265 | 0.5210 | 0.6038 | 0.7540  

