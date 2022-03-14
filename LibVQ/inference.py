import argparse
import logging
import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler

from LibVQ.dataset.dataset import DatasetForEncoding

logger = logging.Logger(__name__)


def inference_dataset(encoder,
                      dataset,
                      is_query,
                      output_file,
                      batch_size,
                      enable_rewrite=True,
                      dataparallel=True):
    if os.path.exists(output_file + '_finished.flag') and not enable_rewrite:
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
    assert write_index == len(output_memmap)

def inference(data_dir,
              is_query,
              encoder,
              prefix,
              max_length,
              output_dir,
              batch_size,
              enable_rewrite=True,
              dataparallel=True
              ):
    dataset = DatasetForEncoding(data_dir=data_dir, prefix=prefix, max_length=max_length)
    inference_dataset(encoder=encoder,
                      dataset=dataset,
                      is_query=is_query,
                      output_file=os.path.join(output_dir, f"{prefix}.memmap"),
                      batch_size=batch_size,
                      enable_rewrite=enable_rewrite,
                      dataparallel=dataparallel)

