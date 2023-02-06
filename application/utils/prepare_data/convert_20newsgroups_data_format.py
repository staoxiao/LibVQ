from sklearn.datasets import fetch_20newsgroups

docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']

f = open('data/20newsgroups/collection.tsv', 'w+', encoding='utf-8')
s = 0
for i in range(len(docs)):
    content = docs[i].replace('\n', ' ').replace('\r', ' ').replace('\x0c', ' ')
    if content.replace(' ', '') != '':
        f.write(str(s) + '\t' + content + '\n')
        s += 1
f.close()