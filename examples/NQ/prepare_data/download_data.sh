mkdir data
mkdir data/NQ
mkdir data/NQ/raw_dataset
cd data/NQ/raw_dataset

wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
gzip -d psgs_w100.tsv.gz

wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz
gzip -d biencoder-nq-train.json.gz

wget https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-dev.qa.csv


wget https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-test.qa.csv

cd ../../../
python ./prepare_data/convert_data_format.py