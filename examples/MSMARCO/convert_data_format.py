import os


if __name__ == '__main__':
    data_type = 'passage'
    output_dir = f'./data/{data_type}/dataset/'
    raw_data_dir = f'./data/{data_type}/raw_dataset/'
    os.makedirs(output_dir, exist_ok=True)

    # collection.tsv
    collection_file = os.path.join(raw_data_dir, 'collection_with_title.tsv') if data_type == 'passage' else os.path.join(raw_data_dir, 'msmarco-docs.tsv')
    output_collection_file = open(os.path.join(output_dir, 'collection.tsv'), 'w', encoding='utf-8')
    for line in open(collection_file, 'r', encoding='utf-8'):
        if data_type == 'passage':
            output_collection_file.write(line)
        else:
            line = line.strip('\n').split('\t')
            id = line[0][1:]
            output_collection_file.write(id+'\t'+'\t'.join(line[1:])+'\n')


    # train data
    os.system(f'cp {raw_data_dir}/queries.train.tsv {output_dir}/train-queries.tsv')

    rel_file = os.path.join(raw_data_dir, 'qrels.train.tsv')
    output_rel_file = open(os.path.join(output_dir, 'train-rels.tsv'), 'w', encoding='utf-8')
    for line in open(rel_file, 'r', encoding='utf-8'):
        qid, _, did, _ = line.strip('\n').split()
        output_rel_file.write(qid + '\t' + did +'\n')

    # dev data
    os.system(f'cp {raw_data_dir}/queries.dev.small.tsv {output_dir}/dev-queries.tsv')

    rel_file = os.path.join(raw_data_dir, 'qrels.dev.small.tsv')
    output_rel_file = open(os.path.join(output_dir, 'dev-rels.tsv'), 'w', encoding='utf-8')
    for line in open(rel_file, 'r', encoding='utf-8'):
        qid, _, did, _ = line.strip('\n').split()
        output_rel_file.write(qid + '\t' + did + '\n')
