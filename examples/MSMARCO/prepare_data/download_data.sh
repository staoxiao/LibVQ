mkdir data
mkdir data/passage
mkdir data/passage/raw_dataset
cd data/passage/raw_dataset

wget --no-check-certificate https://rocketqa.bj.bcebos.com/corpus/marco.tar.gz
tar -zxf marco.tar.gz
rm -rf marco.tar.gz
cd marco

join  -t "$(echo -en '\t')"  -e '' -a 1  -o 1.1 2.2 1.2  <(sort -k1,1 para.txt) <(sort -k1,1 para.title.txt) | sort -k1,1 -n > corpus.tsv
mv corpus.tsv ../collection_with_title.tsv
cd ..
rm -r marco

wget https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz
tar xvfz collectionandqueries.tar.gz -C ./
rm -r collectionandqueries.tar.gz

cd ../../../
python ./prepare_data/convert_data_format.py

mkdir data/MSMARCO
cp -r ./data/passage/dataset ./data/MSMARCO
rm -r ./data/passage