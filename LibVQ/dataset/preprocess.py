import gc
import json
import multiprocessing as mp
import os
import pickle
from typing import Dict

import faiss
import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from LibVQ.base_index import FaissIndex
from LibVQ.dataset import write_rel

tokenizer = None
max_seq_length = None
add_special_tokens = True


def count_line(path: str):
    return sum(1 for _ in open(path))


def data_generator(path: str):
    with open(path, 'rt') as f:
        line = f.readline()
        while line:
            yield line[:-1]
            line = f.readline()


class MpTokenizer:
    def __init__(self):
        self.total = None

    def _set_total(self, total: int):
        self.total = total

    def __call__(self,
                 input_file: str,
                 output_file: str,
                 max_length: int,
                 func,
                 workers_num=None,
                 initializer=None,
                 initargs=None):
        total_line_num = count_line(input_file)
        self._set_total(total_line_num)

        global max_seq_length
        max_seq_length = max_length

        id2offset = {}
        token_array = np.memmap(output_file + ".memmap", shape=(total_line_num, max_seq_length), mode='w+',
                                dtype=np.int32)
        token_length_array = []

        dataset = data_generator(input_file)
        with mp.Pool(workers_num, initializer=initializer, initargs=initargs) as pool:
            with tqdm(pool.imap(func, dataset)) as pbar:
                for res in pbar:
                    id, tokens, tokens_num = res
                    offset = len(id2offset)
                    id2offset[id] = offset
                    token_array[offset, :] = tokens
                    token_length_array.append(tokens_num)
                    if self.total:
                        pbar.total = self.total

        assert len(token_length_array) == total_line_num
        pickle.dump(id2offset, open(output_file + "_id2offset.pickle", 'wb'))
        np.save(output_file + '_length', np.array(token_length_array))
        meta = {'type': 'int32', 'total_number': total_line_num,
                'max_seq_length': max_seq_length}
        with open(output_file + "_meta", 'w') as f:
            json.dump(meta, f)

        return id2offset


def job(line):
    line = line.split('\t')
    if hasattr(tokenizer, 'sep_token'):
        id, text = int(line[0]), f' {tokenizer.sep_token} '.join(line[1:])
    else:
        id, text = int(line[0]), ' '.join(line[1:])

    tokens = tokenizer.encode(
        text,
        add_special_tokens=add_special_tokens,
        max_length=max_seq_length,
        truncation=True
    )
    tokens_num = len(tokens)

    tokens = tokens + [0] * (max_seq_length - tokens_num)
    return id, tokens, tokens_num


def init():
    return


def tokenize_data(input_file: str,
                  output_file: str,
                  max_length: int,
                  workers_num: int = None):
    id2offset = MpTokenizer()(input_file,
                              output_file,
                              max_length,
                              job,
                              workers_num=workers_num)
    return id2offset


def offset_rel(rel_file: str,
               output_offset_rel: str,
               did2offset: Dict,
               qid2offset: Dict):
    with open(output_offset_rel, 'w') as f:
        for line in open(rel_file):
            qid, did = line.strip('\n').split('\t')
            new_qid, new_did = qid2offset[int(qid)], did2offset[int(did)]
            f.write(str(new_qid) + '\t' + str(new_did) + '\n')


def preprocess_data(data_dir: str,
                    output_dir: str,
                    text_tokenizer: PreTrainedTokenizer,
                    max_doc_length: int,
                    max_query_length: int,
                    add_cls_tokens: bool = True,
                    workers_num: int = None
                    ):
    os.makedirs(output_dir, exist_ok=True)

    global tokenizer, add_special_tokens
    tokenizer = text_tokenizer
    add_special_tokens = add_cls_tokens

    docs_file = os.path.join(data_dir, 'collection.tsv')
    output_docs_file = os.path.join(output_dir, 'docs')
    did2offset = tokenize_data(docs_file, output_docs_file, workers_num=workers_num,
                               max_length=max_doc_length)

    for file in os.listdir(data_dir):
        if 'queries' in file:
            prefix = file[:-12]
            query_file = os.path.join(data_dir, file)
            rel_file = os.path.join(data_dir, f'{prefix}-rels.tsv')
            if not os.path.exists(rel_file):
                print(f'There is no {rel_file} for {query_file}')
                raise

            output_query_file = os.path.join(output_dir, f'{prefix}-queries')
            qid2offset = tokenize_data(query_file, output_query_file, workers_num=workers_num,
                                       max_length=max_query_length)

            output_offset_rel = os.path.join(output_dir, f'{prefix}-rels.tsv')
            offset_rel(rel_file=rel_file,
                       output_offset_rel=output_offset_rel,
                       qid2offset=qid2offset,
                       did2offset=did2offset)




def generate_virtual_traindata(
        doc_embeddings,
        train_query,
        output_dir: str,
        use_gpu: bool,
        topk: int = 400,
        index_method: str = 'flat',
        ivf_centers_num=-1,
        subvector_num=-1,
        subvector_bits=8,
        dist_mode='ip'):
    faiss.omp_set_num_threads(32)
    index = FaissIndex(index_method=index_method,
                       emb_size=len(doc_embeddings[0]),
                       ivf_centers_num=ivf_centers_num,
                       subvector_num=subvector_num,
                       subvector_bits=subvector_bits,
                       dist_mode=dist_mode,
                       doc_embeddings=doc_embeddings)

    if use_gpu:
        faiss.index_cpu_to_all_gpus(index.index)

    query2pos, query2neg = index.generate_virtual_traindata(train_query,
                                                            topk=topk,
                                                            batch_size=64
                                                            )

    write_rel(os.path.join(output_dir, 'train-virtual_rel.tsv'), query2pos)
    pickle.dump(query2neg, open(os.path.join(output_dir, f"train-queries-virtual_hardneg.pickle"), 'wb'))

    del query2neg, query2pos
    gc.collect()
