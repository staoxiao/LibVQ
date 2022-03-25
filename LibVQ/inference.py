import logging
import os

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm

from LibVQ.dataset.dataset import DatasetForEncoding
from LibVQ.models import Encoder


def inference_dataset(encoder: Encoder,
                      dataset: DatasetForEncoding,
                      is_query: bool,
                      output_file: str,
                      batch_size: int,
                      enable_rewrite: bool = True,
                      dataparallel: bool = True,
                      return_vecs: bool = False,
                      save_to_memmap: bool = True):
    if output_file is not None:
        if os.path.exists(output_file + '_finished.flag') and not enable_rewrite:
            print(f"{output_file}_finished.flag exists, skip inference")
            return
        if os.path.exists(output_file): os.remove(output_file)

        output_memmap = None

    if return_vecs or not save_to_memmap: vecs = []

    dataloader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=batch_size,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    encoder = encoder.to(device)
    if dataparallel and torch.cuda.is_available():
        encoder = torch.nn.DataParallel(encoder)
    encoder.eval()
    write_index = 0
    for step, data in enumerate(tqdm(dataloader, total=len(dataloader))):
        input_ids, attention_mask = data
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            logits = encoder(input_ids=input_ids, attention_mask=attention_mask,
                             is_query=is_query).detach().cpu().numpy()

        if output_file is not None:
            if save_to_memmap:
                if output_memmap is None: output_memmap = np.memmap(output_file, dtype=np.float32, mode="w+",
                                                                    shape=(len(dataset), np.shape(logits)[-1]))
                write_size = len(logits)
                output_memmap[write_index:write_index + write_size] = logits
                write_index += write_size

        if return_vecs or not save_to_memmap: vecs.extend(logits)

    if output_file is not None:
        if save_to_memmap:
            assert write_index == len(output_memmap)
        else:
            np.save(output_file, np.array(vecs))
        open(output_file + '_finished.flag', 'w')

    if return_vecs: return np.array(vecs)


def inference(data_dir: str,
              is_query: bool,
              encoder: Encoder,
              prefix: str,
              max_length: int,
              output_dir: str = None,
              batch_size: int = 1024,
              enable_rewrite: bool = True,
              dataparallel: bool = True,
              return_vecs: bool = False,
              save_to_memmap: bool = True
              ):
    dataset = DatasetForEncoding(data_dir=data_dir, prefix=prefix, max_length=max_length)

    if output_dir is not None:
        if save_to_memmap:
            output_file = os.path.join(output_dir, f"{prefix}.memmap")
        else:
            output_file = os.path.join(output_dir, f"{prefix}")
    else:
        output_file = None

    return inference_dataset(encoder=encoder,
                             dataset=dataset,
                             is_query=is_query,
                             output_file=output_file,
                             batch_size=batch_size,
                             enable_rewrite=enable_rewrite,
                             dataparallel=dataparallel,
                             return_vecs=return_vecs,
                             save_to_memmap=save_to_memmap)
