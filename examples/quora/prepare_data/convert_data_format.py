def modify(file):
    fR = open(file, 'r', encoding='utf-8')
    lines = fR.readlines()
    fR.close()
    fW = open(file, 'w+', encoding='utf-8')
    for line in lines:
        if line.strip('\n')[-1] != '"':
            fW.write(line.strip('\n'))
        else:
            fW.write(line)
    fW.close()

def load_test():
    query_read = open('./data/quora/test.csv', 'r', encoding='utf-8')
    rel_read = open('./data/quora/sample_submission.csv', 'r', encoding='utf-8')
    query_line = query_read.readlines()
    rel_line = rel_read.readlines()
    query_read.close()
    rel_read.close()

    docs = open('./data/quora/collection.tsv', 'r', encoding='utf-8')
    num_doc = len(docs.readlines())
    docs.close()

    query_write = open('./data/quora/dev-queries.tsv', 'w+', encoding='utf-8')
    doc_write = open('./data/quora/collection.tsv', 'a+', encoding='utf-8')
    rel_write = open('./data/quora/dev-rels.tsv', 'w+', encoding='utf-8')
    num_query = 0
    for i in range(1, len(query_line)//10): # if your memory is large enough, you can use len(query_line)
        if query_line[i].split(',')[0] == rel_line[i].split(',')[0]:
            if rel_line[i].strip('\n').split(',')[1] == '1':
                query = query_line[i].split('"')[1]
                doc = query_line[i].split('"')[3]
                query_write.write(str(num_query) + '\t' + query + '\n')
                doc_write.write(str(num_doc) + '\t' + doc + '\n')
                rel_write.write(str(num_query) + '\t' + str(num_doc) + '\n')
                num_query += 1
                num_doc += 1
    query_write.close()
    doc_write.close()
    rel_write.close()

def load_train():
    query_read = open('./data/quora/train.csv', 'r', encoding='utf-8')
    query_line = query_read.readlines()
    query_read.close()

    query_write = open('./data/quora/train-queries.tsv', 'w+', encoding='utf-8')
    doc_write = open('./data/quora/collection.tsv', 'w+', encoding='utf-8')
    rel_write = open('./data/quora/train-rels.tsv', 'w+', encoding='utf-8')
    num_query = 0
    num_doc = 0
    for i in range(1, len(query_line)):
        if query_line[i].strip('\n').split('"')[11] == '1':
            query = query_line[i].split('"')[7]
            doc = query_line[i].split('"')[9]
            query_write.write(str(num_query) + '\t' + query + '\n')
            doc_write.write(str(num_doc) + '\t' + doc + '\n')
            rel_write.write(str(num_query) + '\t' + str(num_doc) + '\n')
            num_query += 1
            num_doc += 1
    query_write.close()
    doc_write.close()
    rel_write.close()

if __name__ == '__main__':
    modify('./data/quora/test.csv')
    modify('./data/quora/train.csv')
    load_train()
    load_test()