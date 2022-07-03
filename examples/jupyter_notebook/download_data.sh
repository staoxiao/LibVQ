mkdir data
mkdir data/dataset

wget --no-check-certificate https://rocketqa.bj.bcebos.com/corpus/marco.tar.gz
tar -zxf marco.tar.gz
rm -rf marco.tar.gz

join  -t "$(echo -en '\t')"  -e '' -a 1  -o 1.1 2.2 1.2  <(sort -k1,1 marco/para.txt) <(sort -k1,1 marco/para.title.txt) | sort -k1,1 -n > corpus.tsv
mv corpus.tsv ./data/dataset/collection_with_title.tsv
rm -r marco
 
wget https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz
tar xvfz collectionandqueries.tar.gz -C ./data/dataset/
rm -r collectionandqueries.tar.gz
rm -r ./data/dataset/collection.tsv
mv ./data/dataset/collection_with_title.tsv ./data/dataset/collection.tsv

cut -f1,3 ./data/dataset/qrels.dev.small.tsv >./data/dataset/dev-rels.tsv
sed -i 's/\s\+/\t/g' ./data/dataset/dev-rels.tsv
rm ./data/dataset/qrels.dev.small.tsv
cut -f1,3 ./data/dataset/qrels.train.tsv >./data/dataset/train-rels.tsv
sed -i 's/\s\+/\t/g' ./data/dataset/train-rels.tsv
rm ./data/dataset/qrels.train.tsv
rm ./data/dataset/queries.dev.tsv
rm ./data/dataset/queries.eval*
mv ./data/dataset/queries.train.tsv ./data/dataset/train-queries.tsv
mv ./data/dataset/queries.dev.small.tsv ./data/dataset/dev-queries.tsv