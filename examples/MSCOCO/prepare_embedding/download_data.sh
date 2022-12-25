mkdir data
mkdir data/MSCOCO
mkdir data/MSCOCO/raw_data
mkdir data/MSCOCO/annotations_trainval2017
mkdir data/MSCOCO/train2017
mkdir data/MSCOCO/val2017

cd data/MSCOCO/annotations_trainval2017
echo "Prepare to download train-val2017 anotation zip file..."
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
rm -r annotations_trainval2017.zip

cd ../train2017
echo "Prepare to download train2017 image zip file..."
wget -c http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
rm -f train2017.zip

cd ../val2017
echo "Prepare to download test2017 image zip file..."
wget -c http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
rm -f val2017.zip

cd ../../..

# if you want to use mini-MSCOCO, please use the following code
#python ./prepare_data/mini_mscoco.py
#rm -r data/MSCOCO/annotations_trainval2017
#rm -r data/MSCOCO/train2017
#rm -r data/MSCOCO/val2017
#mkdir data/MSCOCO/annotations_trainval2017
#mkdir data/MSCOCO/annotations_trainval2017/annotations
#mv data/MSCOCO/mini_train2017/instances_train2017.json data/MSCOCO/annotations_trainval2017/annotations
#mv data/MSCOCO/mini_val2017/instances_val2017.json data/MSCOCO/annotations_trainval2017/annotations
#mv data/MSCOCO/mini_train2017 data/MSCOCO/train2017
#mv data/MSCOCO/mini_val2017 data/MSCOCO/val2017

python ./prepare_data/convert_data_format.py
python ./prepare_data/prepare_embedding.py

cd data/MSCOCO
rm -r annotations_trainval2017
rm -r train2017
rm -r val2017
cp ./raw_data/collection.tsv ./embedding
#rm -r ./raw_data
cd ../..