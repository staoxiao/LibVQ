import sys
sys.path.append('./')
import os

import numpy as np
import torch
from transformers import HfArgumentParser, AutoModel, AutoConfig, AutoTokenizer

from LibVQ.dataset.preprocess import preprocess_data
from LibVQ.inference import inference
from LibVQ.models import Encoder

from arguments import DataArguments, ModelArguments


class MS_Encoder(torch.nn.Module):
    def __init__(self, pretrained_model_name):
        torch.nn.Module.__init__(self, )

        self.ms_encoder = AutoModel.from_pretrained(pretrained_model_name)
        config = AutoConfig.from_pretrained(pretrained_model_name)
        self.output_embedding_size = config.hidden_size

    def pooling(self, emb_all):
        return emb_all[0][:, 0]

    def forward(self, input_ids, attention_mask):
        outputs = self.ms_encoder(input_ids, attention_mask)
        return self.pooling(outputs)


if __name__ == '__main__':
    parser = HfArgumentParser((DataArguments, ModelArguments))
    data_args, model_args = parser.parse_args_into_dataclasses()

    os.makedirs(data_args.output_dir, exist_ok=True)

    # preprocess
    text_tokenizer = AutoTokenizer.from_pretrained(data_args.tokenizer_name)
    if not os.path.exists(data_args.preprocess_dir):
        preprocess_data(data_dir=data_args.data_dir,
                        output_dir=data_args.preprocess_dir,
                        text_tokenizer=text_tokenizer,
                        add_cls_tokens=True,
                        max_doc_length=data_args.max_doc_length,
                        max_query_length=data_args.max_query_length,
                        workers_num=64)

    # Load encoder
    query_encoder = MS_Encoder(pretrained_model_name='Luyu/co-condenser-marco-retriever')
    doc_encoder = MS_Encoder(pretrained_model_name='Luyu/co-condenser-marco-retriever')
    emb_size = doc_encoder.output_embedding_size

    text_encoder = Encoder(query_encoder=query_encoder,
                           doc_encoder=doc_encoder)


    # Generate embeddings of queries and docs
    inference(data_dir=data_args.preprocess_dir,
              is_query=False,
              encoder=text_encoder,
              prefix=f'docs',
              max_length=data_args.max_doc_length,
              output_dir=data_args.output_dir,
              batch_size=256,
              enable_rewrite=False,
              return_vecs=False)
    inference(data_dir=data_args.preprocess_dir,
              is_query=True,
              encoder=text_encoder,
              prefix=f'dev-queries',
              max_length=data_args.max_query_length,
              output_dir=data_args.output_dir,
              batch_size=256,
              enable_rewrite=False,
              return_vecs=False)
    inference(data_dir=data_args.preprocess_dir,
              is_query=True,
              encoder=text_encoder,
              prefix=f'train-queries',
              max_length=data_args.max_query_length,
              output_dir=data_args.output_dir,
              batch_size=256,
              enable_rewrite=False,
              return_vecs=False)

    # you can load the generated embeddings as following:
    doc_embeddings = np.memmap(os.path.join(data_args.output_dir, 'docs.memmap'),
                               dtype=np.float32, mode="r")
    doc_embeddings = doc_embeddings.reshape(-1, emb_size)
    test_query_embeddings = np.memmap(os.path.join(data_args.output_dir, 'dev-queries.memmap'),
                                 dtype=np.float32, mode="r")
    test_query_embeddings = test_query_embeddings.reshape(-1, emb_size)
