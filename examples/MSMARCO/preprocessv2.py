import os
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional
from transformers import AutoTokenizer

tokenizer = None
max_seq_length = 512

def count_line(path):
    return sum(1 for _ in open(path))

def data_generator(path):
    with open(path, 'rt') as f:
        line = f.readline()
        while line:
            yield line[:-1]
            line = f.readline()

class MpTokenizer:
    def __init__(self):
        self.total = None

    def _set_total(self, total):
        self.total = total

    def __call__(self, input_file, output_file, func, worker=None, initializer=None, initargs=None):
        id2offset = {}
        token_ids_array = np.memmap(output_file + ".memmap", shape=(all_linecnt, max_seq_length), mode='w+', dtype=np.int32)
        token_length_array = []

        dataset = data_generator(input_file)
        with mp.Pool(worker, initializer=initializer, initargs=initargs) as pool:
            pool.apply_async(count_line, (input_file,), callback=self._set_total)
            with tqdm(pool.imap(func, dataset)) as pbar:
                for res in pbar:
                    id, tokens, tokens_num = res
                    print(res)
                    if self.total:
                        pbar.total = self.total
        return id2offset

def job(line):
    line = line.split('\t')
    id, text = int(line[0]), '[SEP]'.join(line[1:])

    tokens = tokenizer.encode(
        text,
        add_special_tokens=True,
        max_length=max_seq_length,
        truncation=True
    )
    tokens_num = len(tokens)
    return id, tokens, tokens_num


def init(tokenizer_name, max_length):
    global tokenizer
    global max_seq_length
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    max_seq_length = max_length


def tokenize_data(input_file: str,
                  output_file: str,
                  tokenizer_name: str,
                  max_length:int):
    MpTokenizer()(input_file, output_file, job, initializer=init,
             initargs=(tokenizer_name, max_length, ))


def preprocess_data(data_fir: str,
                    output_dir: str,
                    tokenizer_name: str,
                    max_doc_length: int,
                    max_query_length: int
                    ):
    docs_file = os.path.join(data_fir, 'collection.tsv')
    tokenize_data(docs_file)

if __name__ == '__main__':
    pass