mkdir data
mkdir data/quora
cd data/quora

kaggle competitions download -c quora-question-pairs
unzip quora-question-pairs.zip
unzip sample_submission.csv.zip
unzip train.csv.zip
rm sample_submission.csv.zip
rm test.csv.zip
rm train.csv.zip
rm quora-question-pairs.zip

cd ../..
python ./prepare_data/convert_quora_data_format.py
rm ./datasample_submission.csv
rm ./data/quora/test.csv
rm ./data/quora/train.csv