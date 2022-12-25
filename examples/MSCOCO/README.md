# MSCOCO Dataset using
We take the capture retrieval image dataset as the example.   

## Getting Started

### Preparing MSCOCO Data
Because our code is main for text retrieval text dataset, so if you want to use other dataset such as text retrieval images dataset MSCOCO, you must use the off-the-shelf embeddings and implement the search function yourself.(At the same time, you can't train encoder.)

Here we provide a sample of the CLIP model and MSCOCO dataset.

Download data, convert to our format, and infer embedding:
(if you want to use mini-MSCOCO, you can remove the comments in the file download_data.sh)
```
pip install pycocotools
pip install Pillow
bash ./prepare_data/download_data.sh
```
The data will be saved into `./data/MSCOCO`.

**Here the he content in collection.tsv is d_id \t image_url**


### Building index for searching
We build index by the following method:

distill learnable index

```
python ./performance/test_distillLearnableIndex.py \
--data_dir ./data/MSCOCO/embedding \
--data_emb_size 512 \
--index_method opq \
--subvector_num 64 \
--subvector_bits 8 \
--emb_size 512 \
--save_path ./data/MSCOCO/embedding/parameters \
--per_device_train_batch_size 512
```
contrastive learnable index

```
python ./performance/test_contrastiveLearnableIndex.py \
--data_dir ./data/MSCOCO/embedding \
--data_emb_size 512 \
--index_method opq \
--subvector_num 64 \
--subvector_bits 8 \
--emb_size 512 \
--save_path ./data/MSCOCO/embedding/parameters \
--per_device_train_batch_size 512
```

auto index

```
python ./performance/test_autoIndex.py \
--data_dir ./data/MSCOCO/embedding \
--data_emb_size 512 \
--index_method opq \
--subvector_num 64 \
--subvector_bits 8 \
--emb_size 512 \
--save_path ./data/MSCOCO/embedding/parameters \
--per_device_train_batch_size 512
```

### search
To use the search function here, you must have built the index, and then pass in the save path of the parameters of index and query statement
```
python ./search/test_search.py \
--load_path ./data/MSCOCO/embedding/parameters \
--collection_path ./data/MSCOCO/embedding/collection.tsv \
--query 'a man falls off his skateboard in a skate park.' 'A woman standing next to a  brown and white dog.'
```

## Quick Start

### Preparing MSCOCO Data

Because our code is main for text retrieval text dataset, so if you want to use other dataset such as text retrieval images dataset MSCOCO, you must use the off-the-shelf embeddings and implement the search function yourself.(At the same time, you can't train encoder.)

Here we provide a sample of the CLIP model and MSCOCO dataset.

Download data, convert to our format, and infer embedding:

```
pip install pycocotools
pip install Pillow
bash ./prepare_data/download_data.sh
```

The data will be saved into `./data/MSCOCO`.


### Building index for searching

We build index by the following code:

```python
from LibVQ.dataset import Datasets
from LibVQ.base_index import IndexConfig
from LibVQ.learnable_index import AutoIndex
from LibVQ.dataset.dataset import load_rel

data = Datasets('./data/MSCOCO/embedding', emb_size=512)
index_config = IndexConfig(index_method='opq',
                           subvector_num=64,
                           subvector_bits=8,
                           emb_size=512)

index = AutoIndex.get_index(index_config, data=data)
index.train(data=data,
            per_query_neg_num=1,
            per_device_train_batch_size=512,
            epochs=16)

index.save_all('./data/MSCOCO/embedding/parameters')
```

### search

To use the search function here, you must have built the index, and then pass in the save path of the parameters of index and query statement

```python
from search.img_search import img_search
from LibVQ.learnable_index import LearnableIndex
from transformers import CLIPProcessor, CLIPModel

index = LearnableIndex.load_all('./data/MSCOCO/embedding/parameters')

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
query = ['a man falls off his skateboard in a skate park.', 'A woman standing next to a  brown and white dog.']
answer, answer_id = img_search(query,
                               index, './data/MSCOCO/embedding/collection.tsv',
                               model, processor)
print(answer)
print(answer_id)
```

The result maybe follow:
```bash
[['http://images.cocodataset.org/val2017/000000145665.jpg', 'http://images.cocodataset.org/train2017/000000398087.jpg', 'http://images.cocodataset.org/train2017/000000143666.jpg', 'http://images.cocodataset.org/train2017/000000568872.jpg', 'http://images.cocodataset.org/train2017/000000135733.jpg', 'http://images.cocodataset.org/train2017/000000158924.jpg', 'http://images.cocodataset.org/train2017/000000432908.jpg', 'http://images.cocodataset.org/train2017/000000422689.jpg', 'http://images.cocodataset.org/train2017/000000455536.jpg', 'http://images.cocodataset.org/train2017/000000287434.jpg', 'http://images.cocodataset.org/train2017/000000535871.jpg', 'http://images.cocodataset.org/train2017/000000166205.jpg', 'http://images.cocodataset.org/train2017/000000219486.jpg', 'http://images.cocodataset.org/train2017/000000124569.jpg', 'http://images.cocodataset.org/train2017/000000099308.jpg', 'http://images.cocodataset.org/train2017/000000178771.jpg', 'http://images.cocodataset.org/train2017/000000146757.jpg', 'http://images.cocodataset.org/train2017/000000247360.jpg', 'http://images.cocodataset.org/train2017/000000195568.jpg', 'http://images.cocodataset.org/train2017/000000216075.jpg'], ['http://images.cocodataset.org/train2017/000000164910.jpg', 'http://images.cocodataset.org/train2017/000000286302.jpg', 'http://images.cocodataset.org/train2017/000000491660.jpg', 'http://images.cocodataset.org/train2017/000000373193.jpg', 'http://images.cocodataset.org/train2017/000000460222.jpg', 'http://images.cocodataset.org/train2017/000000557543.jpg', 'http://images.cocodataset.org/train2017/000000329001.jpg', 'http://images.cocodataset.org/train2017/000000059943.jpg', 'http://images.cocodataset.org/train2017/000000415026.jpg', 'http://images.cocodataset.org/train2017/000000146163.jpg', 'http://images.cocodataset.org/train2017/000000136132.jpg', 'http://images.cocodataset.org/train2017/000000170406.jpg', 'http://images.cocodataset.org/val2017/000000512836.jpg', 'http://images.cocodataset.org/train2017/000000414421.jpg', 'http://images.cocodataset.org/train2017/000000344921.jpg', 'http://images.cocodataset.org/train2017/000000401758.jpg', 'http://images.cocodataset.org/train2017/000000468487.jpg', 'http://images.cocodataset.org/train2017/000000332434.jpg', 'http://images.cocodataset.org/train2017/000000362567.jpg', 'http://images.cocodataset.org/train2017/000000174718.jpg']]
[['145665', '398087', '143666', '568872', '135733', '158924', '432908', '422689', '455536', '287434', '535871', '166205', '219486', '124569', '99308', '178771', '146757', '247360', '195568', '216075'], ['164910', '286302', '491660', '373193', '460222', '557543', '329001', '59943', '415026', '146163', '136132', '170406', '512836', '414421', '344921', '401758', '468487', '332434', '362567', '174718']]
```

