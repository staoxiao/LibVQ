from PIL import Image
import PIL
import requests
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
import os
import numpy as np
import time
from tqdm import tqdm

def infer_text(input_file, output_file, model, processor, batch_size = 512):
    files = open(input_file, 'r', encoding='utf-8')
    lines = files.readlines()

    output_memmap = np.memmap(output_file, dtype=np.float32, mode="w+", shape=(len(lines), model.config.projection_dim))
    text_id = {}

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    texts = []
    num = 0
    last = 0
    for i in tqdm(range(len(lines))):
        text_id[lines[i].split('\t')[0]] = str(num)
        texts.append(lines[i].strip('\n').split('\t')[1])
        num += 1
        if num % batch_size == 0:
            inputs = processor(
                text=texts, images=image, return_tensors="pt", padding=True
            )
            outputs = model(**inputs)
            output_memmap[last:num] = outputs.text_embeds.detach().numpy()
            last = num
            texts = []
    if num % batch_size != 0:
        inputs = processor(
            text=texts, images=image, return_tensors="pt", padding=True
        )
        outputs = model(**inputs)
        output_memmap[last:num] = outputs.text_embeds.detach().numpy()
    files.close()
    return text_id

def infer_image(input_file, output_file, model, processor, batch_size = 512):
    files = open(input_file, 'r', encoding='utf-8')
    lines = files.readlines()

    output_memmap = np.memmap(output_file, dtype=np.float32, mode="w+", shape=(len(lines), model.config.projection_dim))
    image_id = {}

    texts = ["a photo of a cat"]

    image = []
    num = 0
    last = 0
    headers = {
        'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) " +
                      "Chrome/103.0.0.0 Safari/537.36"
    }
    for i in tqdm(range(len(lines))):
        image_id[lines[i].split('\t')[0]] = str(num)
        url = lines[i].strip('\n').split('\t')[1]
        image_path = os.path.join('./data/MSCOCO', url.split('/')[-2], url.split('/')[-1])
        imageIO = Image.open(image_path)
        image.append(imageIO)
        num += 1
        if num % batch_size == 0:
            inputs = processor(
                text=texts, images=image, return_tensors="pt", padding=True
            )
            outputs = model(**inputs)
            output_memmap[last:num] = outputs.image_embeds.detach().numpy()
            last = num
            image = []
        # except PIL.UnidentifiedImageError:
        #     image_id[lines[i].split('\t')[0]] = str(num)
        #     imageIO = Image.open(requests.get(lines[i].strip('\n').split('\t')[1], stream=True, timeout=10000,
        #                                       headers=headers).raw)
        #     image.append(imageIO)
        #     num += 1
        #     time.sleep(0.2)
    if num % batch_size != 0:
        inputs = processor(
            text=texts, images=image, return_tensors="pt", padding=True
        )
        outputs = model(**inputs)
        output_memmap[last:num] = outputs.image_embeds.detach().numpy()
    files.close()
    return image_id

def generate_rel(input_file, output_file, doc_id, query_id):
    rel_file = open(input_file, 'r', encoding='utf-8')
    new_rel_file = open(output_file, 'w+', encoding='utf-8')
    lines = rel_file.readlines()
    for line in lines:
        q_id = query_id[line.split('\t')[0]]
        d_id = doc_id[line.strip('\n').split('\t')[1]]
        new_rel_file.write(q_id + '\t' + d_id + '\n')
    rel_file.close()
    new_rel_file.close()

if __name__ == '__main__':
    os.makedirs('./data/MSCOCO/embedding', exist_ok=True)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    doc_id = infer_image('./data/MSCOCO/raw_data/collection.tsv',
                         './data/MSCOCO/embedding/docs.memmap', model, processor, batch_size=64)
    train_query_id = infer_text('./data/MSCOCO/raw_data/train-queries.tsv',
                                './data/MSCOCO/embedding/train-queries.memmap', model, processor, batch_size=128)
    dev_query_id = infer_text('./data/MSCOCO/raw_data/dev-queries.tsv',
                              './data/MSCOCO/embedding/dev-queries.memmap', model, processor, batch_size=128)
    generate_rel('./data/MSCOCO/raw_data/train-rels.tsv', './data/MSCOCO/embedding/train-rels.tsv',
                 doc_id, train_query_id)
    generate_rel('./data/MSCOCO/raw_data/dev-rels.tsv', './data/MSCOCO/embedding/dev-rels.tsv',
                 doc_id, dev_query_id)
