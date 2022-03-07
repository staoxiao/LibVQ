import sys

sys.path.append('./LibVQ')

import argparse
import logging
import os
import numpy as np

import torch
from transformers import AutoConfig
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler

from models.encoder import Encoder
from dataset.dataset import DatasetForEncoding

logger = logging.Logger(__name__)


def inference_dataset(encoder, dataset, is_query, output_file, batch_size, dataparallel=True):
    if os.path.exists(output_file + '_finished.flag'):
        print(f"{output_file}_finished.flag exists, skip inference")
        return
    if os.path.exists(output_file): os.remove(output_file)
    output_memmap = np.memmap(output_file, dtype=np.float32, mode="w+", shape=(len(dataset), encoder.output_embedding_size))

    dataloader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=batch_size,
        drop_last=False,
    )

    encoder = encoder.cuda()
    if dataparallel:
        encoder = torch.nn.DataParallel(encoder)
    encoder.eval()
    write_index = 0
    for step, data in enumerate(tqdm(dataloader, total=len(dataloader))):
        input_ids, attention_mask = data
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()

        with torch.no_grad():
            logits = encoder(input_ids=input_ids, attention_mask=attention_mask,
                             is_query=is_query).detach().cpu().numpy()

        write_size = len(logits)
        output_memmap[write_index:write_index + write_size] = logits
        write_index += write_size

    open(output_file + '_finished.flag', 'w')


def inference(data_dir,
              is_query,
              encoder,
              prefix,
              max_length,
              output_dir,
              batch_size
              ):
    dataset = DatasetForEncoding(data_dir=data_dir, prefix=prefix, max_length=max_length)
    inference_dataset(encoder=encoder,
                      dataset=dataset,
                      is_query=is_query,
                      output_file=os.path.join(output_dir, f"{prefix}.memmap"),
                      batch_size=batch_size,
                      dataparallel=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", choices=["passage", 'doc'], type=str, required=True)
    parser.add_argument("--preprocess_dir", type=str, required=True)
    parser.add_argument("--max_query_length", type=int, default=32)
    parser.add_argument("--max_doc_length", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--mode", type=str, choices=["train", "dev", "test", "test2019", "test2020"], default='dev')
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True, default='evaluate/')
    parser.add_argument("--root_output_dir", type=str, required=False, default='./data')
    parser.add_argument("--gpu_rank", type=str, required=False, default=None)

    args = parser.parse_args()

    if args.gpu_rank is not None:
        gpus = ','.join(args.gpu_rank.split('_'))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    args.n_gpu = torch.cuda.device_count()

    print(f'------------------------use dataset: {args.preprocess_dir}------------------------')
    args.output_dir = os.path.join(f"{args.root_output_dir}/{args.data_type}/", args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    config = AutoConfig.from_pretrained(os.path.join(args.ckpt_path, 'config.json'))
    model = Encoder(config)
    model.load_state_dict(torch.load(os.path.join(args.ckpt_path, 'encoder.bin')))
    model = model.cuda()

    inference(data_dir=args.preprocess_dir,
              is_query=True,
              encoder=model,
              prefix=f'{args.mode}-query',
              max_length=args.max_query_length,
              output_dir=args.output_dir,
              batch_size=args.batch_size)

    inference(data_dir=args.preprocess_dir,
              is_query=False,
              encoder=model,
              prefix=f'passages',
              max_length=args.max_doclength,
              output_dir=args.output_dir,
              batch_size=args.batch_size)


if __name__ == "__main__":
    main()

