from pycocotools.coco import COCO
from PIL import Image
import PIL
import requests

def load_train():
    docs = open('./data/MSCOCO/raw_data/collection.tsv', 'w+', encoding='utf-8')
    train_queries = open('./data/MSCOCO/raw_data/train-queries.tsv', 'w+', encoding='utf-8')
    train_rels = open('./data/MSCOCO/raw_data/train-rels.tsv', 'w+', encoding='utf-8')

    dataDir = './data/MSCOCO'
    dataType_train = 'train2017'
    annFile_train = '{}/annotations_trainval2017/annotations/instances_{}.json'.format(dataDir, dataType_train)
    coco_train = COCO(annFile_train)

    imgIds_train = coco_train.getImgIds()

    img_train = coco_train.loadImgs(imgIds_train)

    annFile_caption_train = '{}/annotations_trainval2017/annotations/captions_{}.json'.format(dataDir, dataType_train)
    coco_caps_train = COCO(annFile_caption_train)
    for i in range(len(img_train)):
        try:
            # Image.open(requests.get(img_train[i]['coco_url'], stream=True).raw)
            print(i)
            docs.write(str(img_train[i]['id']) + '\t' + img_train[i]['coco_url'] + '\n')
            annIds = coco_caps_train.getAnnIds(imgIds=img_train[i]['id'])
            anns = coco_caps_train.loadAnns(annIds)
            for j in range(len(anns)):
                train_queries.write(str(anns[j]['id']) + '\t' + anns[j]['caption'].replace('\n', '.') + '\n')
                train_rels.write(str(anns[j]['id']) + '\t' + str(anns[j]['image_id']) + '\n')
        except (PIL.UnidentifiedImageError, requests.exceptions.ReadTimeout):
            print('Image address is damaged: ' + img_train[i]['coco_url'])

    docs.close()
    train_queries.close()
    train_rels.close()

def load_val():
    docs = open('./data/MSCOCO/raw_data/collection.tsv', 'a+', encoding='utf-8')
    dev_queries = open('./data/MSCOCO/raw_data/dev-queries.tsv', 'w+', encoding='utf-8')
    dev_rels = open('./data/MSCOCO/raw_data/dev-rels.tsv', 'w+', encoding='utf-8')

    dataDir = './data/MSCOCO'
    dataType_val = 'val2017'
    annFile_val = '{}/annotations_trainval2017/annotations/instances_{}.json'.format(dataDir, dataType_val)
    coco_val = COCO(annFile_val)

    imgIds_val = coco_val.getImgIds()

    img_val = coco_val.loadImgs(imgIds_val)

    annFile_caption_val = '{}/annotations_trainval2017/annotations/captions_{}.json'.format(dataDir, dataType_val)
    coco_caps_val = COCO(annFile_caption_val)
    for i in range(len(img_val)):
        try:
            # Image.open(requests.get(img_val[i]['coco_url'], stream=True).raw)
            print(i)
            docs.write(str(img_val[i]['id']) + '\t' + img_val[i]['coco_url'] + '\n')
            annIds = coco_caps_val.getAnnIds(imgIds=img_val[i]['id'])
            anns = coco_caps_val.loadAnns(annIds)
            for j in range(len(anns)):
                dev_queries.write(str(anns[j]['id']) + '\t' + anns[j]['caption'].replace('\n', '.') + '\n')
                dev_rels.write(str(anns[j]['id']) + '\t' + str(anns[j]['image_id']) + '\n')
        except (PIL.UnidentifiedImageError, requests.exceptions.ReadTimeout):
            print('Image address is damaged: ' + img_val[i]['coco_url'])

    docs.close()
    dev_queries.close()
    dev_rels.close()

if __name__ == '__main__':
    load_train()
    load_val()