# quora Dataset using
We take the quora dataset to show the duplication detection as the example.   

## Getting Started

### Preparing quora Data
You must make sure that you have installed kaggle, you have verified your account and you have accepted the rules for quora competition.(https://www.kaggle.com/competitions/quora-question-pairs/rules)

Download data, convert to our format, we ignored the question pairs of non repetition:
```
bash ./prepare_data/download_data.sh
```
The data will be saved into `./data/quora`.
### Building index
We build index by the following method:

distill learnable index

```
python ./performance/test_distillLearnableIndex.py  \
--data_dir ./data/quora \
--index_method pq \
--subvector_num 64 \
--subvector_bits 8 \
--emb_size 768 \
--is_finetune True \
--doc_encoder_name_or_path Shitao/RetroMAE_MSMARCO_distill \
--query_encoder_name_or_path Shitao/RetroMAE_MSMARCO_distill \
--save_path ./data/quora/parameters \
--per_device_train_batch_size 512
```
contrastive learnable index

```
python ./performance/test_contrastiveLearnableIndex.py \
--data_dir ./data/quora \
--index_method pq \
--subvector_num 64 \
--subvector_bits 8 \
--emb_size 768 \
--is_finetune True \
--doc_encoder_name_or_path Shitao/RetroMAE_MSMARCO_distill \
--query_encoder_name_or_path Shitao/RetroMAE_MSMARCO_distill \
--save_path ./data/quora/parameters \
--per_device_train_batch_size 512
```

auto index

```
python ./performance/test_autoIndex.py \
--data_dir ./data/quora \
--index_method pq \
--subvector_num 64 \
--subvector_bits 8 \
--emb_size 768 \
--is_finetune True \
--doc_encoder_name_or_path Shitao/RetroMAE_MSMARCO_distill \
--query_encoder_name_or_path Shitao/RetroMAE_MSMARCO_distill \
--save_path ./data/quora/parameters \
--per_device_train_batch_size 512
```

### Duplication detection
To use the duplication detection here, you must have built the index, and then pass in the save path of the parameters of index and query statement, and then you can check the returned results to see if there are duplicates

```
python ./duplication_detection/test_search.py \
--load_path ./data/quora/parameters \
--data_emb_size 768 \
--query 'Which food not emulsifiers?' 'Is it gouging and price fixing?'
```

## Quick Start

### Preparing quora Data

You must make sure that you have installed kaggle, you have verified your account and you have accepted the rules for quora competition.(https://www.kaggle.com/competitions/quora-question-pairs/rules)

Download data, convert to our format, we ignored the question pairs of non repetition:

```
bash ./prepare_data/download_data.sh
```

The data will be saved into `./data/quora`.

### Building index

We build index by the following method:

```python
from LibVQ.dataset import Datasets
from LibVQ.base_index import IndexConfig
from LibVQ.models import EncoderConfig
from LibVQ.learnable_index import AutoIndex

data = Datasets('./data/quora')
index_config = IndexConfig(index_method='ivf_pq',
                           ivf_centers_num=5000,
                           subvector_num=64,
                           subvector_bits=8,
                           nprobe=100,
                           emb_size=768)
encoder_config = EncoderConfig(is_finetune=True,
                              doc_encoder_name_or_path='Shitao/RetroMAE_MSMARCO_distill',
                            query_encoder_name_or_path='Shitao/RetroMAE_MSMARCO_distill')

index = AutoIndex.get_index(index_config, encoder_config, data)
index.train(data=data,
            per_query_neg_num=1,
            per_device_train_batch_size=512,
            logging_steps=100,
            epochs=16)

index.save_all('./data/quora/parameters')
```

### Duplication detection

To use the duplication detection here, you must have built the index, and then pass in the save path of the parameters of index and query statement, and then you can check the returned results to see if there are duplicates

```python
from LibVQ.learnable_index import LearnableIndex
from LibVQ.dataset import Datasets

data = Datasets('./data/quora', emb_size=768)
index = LearnableIndex.load_all('./data/quora/parameters')
if data.docs_path is not None:
    query = ['Which food not emulsifiers?', 'Is it gouging and price fixing?']
    answer, answer_id = index.search_query(query, data)
    print(answer)
    print(answer_id)
```

The result maybe follow:
```bash
[['What food the children should not buy for eat?', "What are some types of food that aren't sold in a fast food practice restaurant but should be?", 'What foods or ingredients should I never eat?', 'What can hamsters eat besides hamster food?', 'What can hamsters eat besides hamster food?', 'What can hamsters eat besides hamster food?', 'What can hamsters eat besides hamster food?', 'What can hamsters eat besides hamster food?', 'What can hamsters eat besides hamster food?', 'Why should we not eat eggs?', 'What is lowest calorie food?', "What are didn't eat and only drank water?", 'What are the foods one should stop eating?', 'What If not, specifically why not?', 'What foods fibre?', 'What modern foods are considered to be aphrodisiacs?', 'What modern foods are considered to be aphrodisiacs?', 'What some food vendors accept EBT cards?', 'Which are the lowest calorie foods?', 'Which are the lowest calorie foods?'], ['Will oil prices go back up?', 'Will oil prices go back up?', 'Will oil prices go back up?', 'Who sets the price for my game on Steam?', 'Why does the price of oil keep down?', 'What determines the price of drugs?', 'How quality is Goku?', 'What are some ways of fixing a crooked smile?', "What's unit and supplying parts to OEMS . OEMS have fixed the price till the product life . and no increase?", 'What are some costs of owning a ferret?', 'price of something', 'How are goods and services rationed if there is a price ceiling?', 'How do you fix a bad circuit breaker?', 'Is Kubo and the two strings Good?', 'How much does it cost to repair a laptop screen?', 'What is the expectations from share market? Will it tank down further below and if it sinks then by what time?', 'What are some of the Pokemon go hacks?', 'What are some of the Pokemon go hacks?', 'How price and output will be determined in a monopoly? Is it true that monopoly price will always be higher than the perfect competitive price?', 'What is difference between grounding and neutral?']]
[['38394', '166837', '136841', '107264', '108400', '145812', '95847', '51954', '39032', '81271', '27671', '163190', '140432', '159576', '147099', '108246', '19886', '151382', '75010', '31267'], ['100873', '5050', '60569', '141252', '169333', '68712', '164900', '107557', '156660', '109377', '155371', '72691', '94350', '72885', '89658', '134313', '113954', '109285', '11545', '85991']]
```
