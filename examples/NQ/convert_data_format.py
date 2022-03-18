import csv
import json
import os

from tqdm import tqdm


def get_collection(file):
    with open('./data/NQ/dataset/collection.tsv', 'w', encoding='utf-8') as fout:
        with open(file, encoding='utf-8') as fin:
            reader = csv.reader(fin, delimiter='\t')
            for k, row in enumerate(reader):
                if not row[0] == 'id':
                    try:
                        fout.write(str(int(row[0]) - 1) + '\t' + row[2] + '\t' + row[1] + '\n')
                    except:
                        print(f'The following input line has not been correctly loaded: {row}')


def get_train(file):
    fquery = open('./data/NQ/dataset/train-queries.tsv', 'w', encoding='utf-8')
    frel = open('./data/NQ/dataset/train-rels.tsv', 'w', encoding='utf-8')

    qid = 0
    data = json.load(open(file))
    for idx, item in enumerate(tqdm(data)):
        query = item['question']
        query = query.replace("’", "'")
        positives = item['positive_ctxs']

        fquery.write(str(qid) + '\t' + query + '\n')

        for p in positives:
            pid = p['passage_id']
            pid = str(int(pid) - 1)
            frel.write(str(qid) + '\t' + pid + '\n')

        qid += 1


def get_dev(file):
    fquery = open('./data/NQ/dataset/dev-queries.tsv', 'w', encoding='utf-8')
    frel = open('./data/NQ/dataset/dev-rels.tsv', 'w', encoding='utf-8')

    data = open(file, encoding='utf-8')
    for idx, item in enumerate(tqdm(data)):
        query, _ = item.split('\t')
        query = query.replace("’", "'")

        fquery.write(str(idx) + '\t' + query + '\n')
        frel.write(str(idx) + '\t' + '0' + '\n')


def get_test(file):
    fquery = open('./data/NQ/dataset/test-queries.tsv', 'w', encoding='utf-8')
    frel = open('./data/NQ/dataset/test-rels.tsv', 'w', encoding='utf-8')

    data = open(file, encoding='utf-8')
    for idx, item in enumerate(tqdm(data)):
        query, _ = item.split('\t')
        query = query.replace("’", "'")

        fquery.write(str(idx) + '\t' + query + '\n')
        frel.write(str(idx) + '\t' + '0' + '\n')


if __name__ == '__main__':
    os.makedirs('./data/NQ/dataset', exist_ok=True)
    get_collection('./data/NQ/raw_dataset/psgs_w100.tsv')
    # get_train('./data/NQ/raw_dataset/biencoder-nq-train.json')
    get_dev('./data/NQ/raw_dataset/nq-dev.qa.csv')
    get_test('./data/NQ/raw_dataset/nq-test.qa.csv')
