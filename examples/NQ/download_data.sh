mkdir raw_data
mkdir raw_data/passage
mkdir raw_data/passage/dataset
cd raw_data/passage/dataset

# download MSMARCO passage data
wget https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz
tar xvfz collectionandqueries.tar.gz -C ./

wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz
gunzip ./msmarco-test2019-queries.tsv.gz

wget https://trec.nist.gov/data/deep/2019qrels-pass.txt


cd ../../../
mkdir raw_data/doc
mkdir raw_data/doc/dataset
cd raw_data/doc/dataset

# download MSMARCO doc data
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz
gunzip msmarco-docs.tsv.gz

wget https://msmarco.blob.core.windows.net/msmarcoranking/docleaderboard-queries.tsv.gz
gunzip docleaderboard-queries.tsv

wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-queries.tsv.gz
gunzip msmarco-doctrain-queries.tsv.gz

wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-qrels.tsv.gz
gunzip msmarco-doctrain-qrels.tsv.gz

wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz
gunzip msmarco-test2019-queries.tsv.gz

wget https://trec.nist.gov/data/deep/2019qrels-docs.txt

wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docdev-queries.tsv.gz
gunzip msmarco-docdev-queries.tsv.gz

wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docdev-qrels.tsv.gz
gunzip msmarco-docdev-qrels.tsv.gz

wget https://msmarco.blob.core.windows.net/msmarcoranking/docleaderboard-queries.tsv.gz

wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-top100.gz
gunzip ./msmarco-doctrain-top100.gz
