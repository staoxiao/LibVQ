# MSMARCO Dataset using
We take the Passage dataset as the example.   

## Preparing MSMARCO Data
Download data and convert to our format:
```
bash ./prepare_data/download_data.sh
```
The data will be saved into `./MSMARCO`.


## Using index 
We use index by the following method:

distill learnable index

```
python ./test_distillLearnableIndex.py 
```
constrative learnable index

```
python ./test_constrativeLearnableIndex.py 
```

distill learnable index with encoder

```
python ./test_distillLearnableIndexWithEncoder.py 
```

constrative learnable index with encoder

```
python ./test_constrativeLearnableIndexWithEncoder.py 
```

auto index

```
python ./test_autoIndex.py 
```