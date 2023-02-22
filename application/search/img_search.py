import requests
from PIL import Image


def img_search(queries, index, model, processor, docs_path: None, kValue=20):
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(
        text=queries, images=image, return_tensors="pt", padding=True
    )
    outputs = model(**inputs)
    query_embeddings = outputs.text_embeds.detach().numpy()

    if docs_path is not None:
        if isinstance(docs_path, str):
            id2img = dict()
            img_id = dict()
            imgsFile = open(docs_path, 'r', encoding='UTF-8')
            count = 0
            for line in imgsFile:
                img_id[count] = line.split('\t')[0]
                id2img[count] = ' '.join(line.strip('\n').split('\t')[1:])
                count += 1
            imgsFile.close()
        else:
            id2img = docs_path[0]
            img_id = docs_path[1]
    else:
        if index.id2text is None or index.doc_id is None:
            raise ValueError("You should provide your docs path")
        id2img = index.id2text
        img_id = index.doc_id

    _, topk_ids = index.search(query_embeddings, kValue, nprobe=index.index_config.nprobe)
    output_imgs = []
    output_ids = []
    for ids in topk_ids:
        temp_img = []
        temp_id = []
        for id in ids:
            if id != -1:
                temp_img.append(id2img[id])
                temp_id.append(img_id[id])
        output_imgs.append(temp_img)
        output_ids.append(temp_id)
    return output_imgs, output_ids
