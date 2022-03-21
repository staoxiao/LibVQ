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

## Workflow
In LibVQ, users can construct a index and train it by a simple way.
Please refer to our [docs](Docs.md) for more details.
Besides, we provide some examples below to illustrate the usage of LibVQ.

## Examples
### [MSMARCO](./examples/MSMARCO/)  
- IVFPQ    

Methods | MRR@10 | Recall@10 | Recall@100 | 
------- | ------- | ------- |  ------- |
[Faiss-IVFPQ](./examples/MSMARCO/basic_index/faiss_index.py) | 0.1380 | 0.2820 | 0.5617 |  
[Faiss-IVFOPQ](./examples/MSMARCO/basic_index/faiss_index.py) | 0.3102 | 0.5593 | 0.8148 |  
[Scann](./examples/MSMARCO/basic_index/scann_index.py) | 0.1791 | 0.3499 | 0.6345 | 
[LibVQ(contrastive_index)](./examples/MSMARCO/learnable_index/train_index.py) | 0.3179 | 0.5724 | 0.8214 | 
[LibVQ(distill_index)](./examples/MSMARCO/learnable_index/train_index.py) | 0.3253 | 0.5765 | 0.8256 | 
[LibVQ(contrastive_index-and-query-encoder)](./examples/MSMARCO/learnable_index/train_index_and_encoder.py) | 0.3192 | 0.5799 | 0.8427 |  
[LibVQ(distill_index-and-query-encoder)](./examples/MSMARCO/learnable_index/train_index_and_encoder.py) | **0.3311** | **0.5907** | **0.8429** |  
[LibVQ(distill_index-and-query-encoder_nolabel)](./examples/MSMARCO/learnable_index/train_index_and_encoder.py) | 0.3285 | 0.5875 | 0.8401 | 

- PQ

Methods | MRR@10 | Recall@10 | Recall@100 | 
------- | ------- | ------- |  ------- | 
[Faiss-PQ](./examples/MSMARCO/basic_index/faiss_index.py) | 0.1145 | 0.2369 | 0.5046 |  
[Faiss-OPQ](./examples/MSMARCO/basic_index/faiss_index.py) | 0.3268 | 0.5939 | 0.8651 |    
[LibVQ(distill_index-and-query-encoder)](./examples/MSMARCO/learnable_index/train_index_and_encoder.py) | 0.3437 | 0.6201 | 0.8819 | 
[LibVQ(distill_index-and-two-encoders)](./examples/MSMARCO/learnable_index/train_index_and_encoder.py) | **0.3475** | **0.6223** | **0.8901** |  
