import sys
sys.path.append('./')
import os

import numpy as np
import torch
from arguments import DataArguments, ModelArguments
from transformers import DPRContextEncoder, DPRQuestionEncoder, AutoConfig, AutoTokenizer
from transformers import HfArgumentParser

from LibVQ.dataset.preprocess import preprocess_data
from LibVQ.inference import inference
from LibVQ.models import Encoder


class DPR_Encoder(torch.nn.Module):
    def __init__(self, encoder):
        torch.nn.Module.__init__(self, )
        self.nq_encoder = encoder

    def forward(self, input_ids, attention_mask):
        outputs = self.nq_encoder(input_ids, attention_mask)
        return outputs.pooler_output


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
    doc_encoder = DPR_Encoder(DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base"))
    query_encoder = DPR_Encoder(DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base'))
    config = AutoConfig.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    emb_size = config.hidden_size

    text_encoder = Encoder(query_encoder=query_encoder,
                           doc_encoder=doc_encoder)

    # Generate embeddings of queries and docs
    inference(data_dir=data_args.preprocess_dir,
              is_query=False,
              encoder=text_encoder,
              prefix=f'docs',
              max_length=data_args.max_doc_length,
              output_dir=data_args.output_dir,
              batch_size=10240,
              enable_rewrite=False)
    inference(data_dir=data_args.preprocess_dir,
              is_query=True,
              encoder=text_encoder,
              prefix=f'dev-queries',
              max_length=data_args.max_query_length,
              output_dir=data_args.output_dir,
              batch_size=10240,
              enable_rewrite=False)
    inference(data_dir=data_args.preprocess_dir,
              is_query=True,
              encoder=text_encoder,
              prefix=f'train-queries',
              max_length=data_args.max_query_length,
              output_dir=data_args.output_dir,
              batch_size=10240,
              enable_rewrite=False)
    inference(data_dir=data_args.preprocess_dir,
              is_query=True,
              encoder=text_encoder,
              prefix=f'test-queries',
              max_length=data_args.max_query_length,
              output_dir=data_args.output_dir,
              batch_size=10240,
              enable_rewrite=False)

    # you can load the generated embeddings as following:
    doc_embeddings = np.memmap(os.path.join(data_args.output_dir, 'docs.memmap'),
                               dtype=np.float32, mode="r")
    doc_embeddings = doc_embeddings.reshape(-1, emb_size)
    query_embeddings = np.memmap(os.path.join(data_args.output_dir, 'test-queries.memmap'),
                                 dtype=np.float32, mode="r")
    query_embeddings = query_embeddings.reshape(-1, emb_size)