import sys
sys.path.append('./src')

import argparse
import logging
import os
import numpy as np

import torch
from transformers import AutoConfig
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler

from encoder import Encoder
from dataset.dataset import (
    TextTokenIdsCache, SubsetSeqDataset, SequenceDataset, single_get_collate_function)

logger = logging.Logger(__name__)


def prediction(model, data_collator, args, test_dataset, embedding_memmap, ids_memmap, is_query):
    os.makedirs(args.output_dir, exist_ok=True)
    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.eval_batch_size,
        collate_fn=data_collator,
        drop_last=False,
    )
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    batch_size = test_dataloader.batch_size
    num_examples = len(test_dataloader.dataset)
    logger.info("***** Running *****")
    logger.info("  Num examples = %d", num_examples)
    logger.info("  Batch size = %d", batch_size)

    model.eval()
    write_index = 0
    for step, (inputs, ids) in enumerate(tqdm(test_dataloader)):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(args.device)
        with torch.no_grad():
            if not is_query:
                logits = model(is_query=is_query, **inputs)
                logits = logits.detach().cpu().numpy()
            else:
                logits = model(is_query=is_query, **inputs)
                logits = logits.detach().cpu().numpy()

        write_size = len(logits)
        assert write_size == len(ids)
        embedding_memmap[write_index:write_index+write_size] = logits
        ids_memmap[write_index:write_index+write_size] = ids
        write_index += write_size
    assert write_index == len(embedding_memmap) == len(ids_memmap)

def query_inference(model, args, embedding_size):
    if os.path.exists(args.query_memmap_path+'_finished.flag'):
        print(f"{args.query_memmap_path} exists, skip inference")
        return
    query_collator = single_get_collate_function(args.max_query_length)
    query_dataset = SequenceDataset(
        ids_cache=TextTokenIdsCache(data_dir=args.preprocess_dir, prefix=f"{args.mode}-query"),
        max_seq_length=args.max_query_length
    )

    if os.path.exists(args.query_memmap_path): os.remove(args.query_memmap_path)
    query_memmap = np.memmap(args.query_memmap_path,
        dtype=np.float32, mode="w+", shape=(len(query_dataset), embedding_size))
    queryids_memmap = np.memmap(args.queryids_memmap_path,
        dtype=np.int32, mode="w+", shape=(len(query_dataset), ))

    prediction(model, query_collator, args,
            query_dataset, query_memmap, queryids_memmap, is_query=True)

    open(args.query_memmap_path+'_finished.flag','w')

def doc_inference(model, args, embedding_size):
    if os.path.exists(args.doc_memmap_path+'_finished.flag'):
        print(f"{args.doc_memmap_path} exists, skip inference")
        return
    doc_collator = single_get_collate_function(args.max_doc_length)
    ids_cache = TextTokenIdsCache(data_dir=args.preprocess_dir, prefix="passages")
    subset=list(range(len(ids_cache)))
    doc_dataset = SubsetSeqDataset(
        subset=subset,
        ids_cache=ids_cache,
        max_seq_length=args.max_doc_length
    )

    if os.path.exists(args.doc_memmap_path):os.remove(args.doc_memmap_path)

    doc_memmap = np.memmap(args.doc_memmap_path,
        dtype=np.float32, mode="w+", shape=(len(doc_dataset), embedding_size))
    docid_memmap = np.memmap(args.docid_memmap_path,
        dtype=np.int32, mode="w+", shape=(len(doc_dataset), ))
    prediction(model, doc_collator, args,
        doc_dataset, doc_memmap, docid_memmap, is_query=False
    )
    open(args.doc_memmap_path+'_finished.flag','w')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", choices=["passage", 'doc'], type=str, required=True)
    parser.add_argument("--preprocess_dir", type=str, required=True)
    parser.add_argument("--max_query_length", type=int, default=32)
    parser.add_argument("--max_doc_length", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--mode", type=str, choices=["train", "dev", "test", "test2019","test2020"], default='dev')
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True, default='evaluate/')
    parser.add_argument("--root_output_dir", type=str, required=False, default='./data')
    parser.add_argument("--gpu_rank", type=str, required=False, default=None)

    args = parser.parse_args()

    if args.gpu_rank is not None:
        gpus = ','.join(args.gpu_rank.split('_'))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    args.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()


    print(f'------------------------use dataset: {args.preprocess_dir}------------------------')
    args.output_dir = os.path.join(f"{args.root_output_dir}/{args.data_type}/", args.output_dir)

    args.doc_memmap_path = os.path.join(args.output_dir, "passages.memmap")
    args.docid_memmap_path = os.path.join(args.output_dir, "passages-id.memmap")
    args.query_memmap_path = os.path.join(args.output_dir, f"{args.mode}-query.memmap")
    args.queryids_memmap_path = os.path.join(args.output_dir, f"{args.mode}-query-id.memmap")

    logger.info(args)
    os.makedirs(args.output_dir, exist_ok=True)
    config = AutoConfig.from_pretrained(os.path.join(args.ckpt_path, 'config.json'))
    model = Encoder(config)
    model.load_state_dict(torch.load(os.path.join(args.ckpt_path, 'encoder.bin')))

    output_embedding_size = config.hidden_size
    model = model.to(args.device)

    doc_inference(model, args, output_embedding_size)
    query_inference(model, args, output_embedding_size)


if __name__ == "__main__":
    main()

