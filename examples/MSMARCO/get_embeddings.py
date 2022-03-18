import os
import numpy as np
import faiss
import torch
from transformers import HfArgumentParser, AutoModel, AutoConfig

from LibVQ.inference import inference
from LibVQ.models import Encoder, EncoderConfig
from LibVQ.dataset.preprocess import preprocess_data

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
    if not os.path.exists(data_args.preprocess_dir):
        preprocess_data(data_dir=data_args.data_dir,
                        output_dir=data_args.preprocess_dir,
                        tokenizer_name=model_args.pretrained_model_name,
                        max_doc_length=data_args.max_doc_length,
                        max_query_length=data_args.max_query_length,
                        workers_num=64)

    # Load encoder
    # config = EncoderConfig.from_pretrained(model_args.pretrained_model_name)
    # config.pretrained_model_name = model_args.pretrained_model_name
    # config.use_two_encoder = model_args.use_two_encoder
    # config.sentence_pooling_method = model_args.sentence_pooling_method
    # text_encoder = Encoder(config)
    # emb_size = text_encoder.output_embedding_size


    query_encoder = MS_Encoder(model_args.pretrained_model_name)
    doc_encoder = MS_Encoder(model_args.pretrained_model_name)
    emb_size = doc_encoder.output_embedding_size

    text_encoder = Encoder(query_encoder = query_encoder,
                           doc_encoder = doc_encoder)

    # Generate embeddings of queries and docs
    # inference(data_dir=data_args.preprocess_dir,
    #           is_query=False,
    #           encoder=text_encoder,
    #           prefix=f'docs',
    #           max_length=data_args.max_doc_length,
    #           output_dir=data_args.output_dir,
    #           batch_size=10240,
    #           enable_rewrite=False,
    #           return_vecs=False)
    inference(data_dir=data_args.preprocess_dir,
              is_query=True,
              encoder=text_encoder,
              prefix=f'dev-queries',
              max_length=data_args.max_query_length,
              output_dir=data_args.output_dir,
              batch_size=10240,
              enable_rewrite=False,
              return_vecs=False)
    # inference(data_dir=data_args.preprocess_dir,
    #           is_query=True,
    #           encoder=text_encoder,
    #           prefix=f'train-queries',
    #           max_length=data_args.max_query_length,
    #           output_dir=data_args.output_dir,
    #           batch_size=10240,
    #           enable_rewrite=False,
    #           return_vecs=False)


    # you can load the generated embeddings as following:
    doc_embeddings = np.memmap(os.path.join(data_args.output_dir, 'docs.memmap'),
                               dtype=np.float32, mode="r")
    doc_embeddings = doc_embeddings.reshape(-1, emb_size)
    query_embeddings = np.memmap(os.path.join(data_args.output_dir, 'dev-queries.memmap'),
                                 dtype=np.float32, mode="r")
    query_embeddings = query_embeddings.reshape(-1, emb_size)



