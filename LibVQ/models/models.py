from LibVQ.dataset import Datasets, preprocess_data
from transformers import AutoTokenizer
from LibVQ.inference import inference
from LibVQ.models import TransformerModel, Encoder, Pooler
from torch import nn
import os


class Model(nn.Module):
    def __init__(self,
                 query_encoder_path_or_name=None,
                 doc_encoder_path_or_name=None
                 ):
        super(Model, self).__init__()
        query_encoder = TransformerModel.from_pretrained(query_encoder_path_or_name)
        doc_encoder = TransformerModel.from_pretrained(doc_encoder_path_or_name)
        self.encoder = Encoder(query_encoder, doc_encoder)

        self.text_tokenizer = AutoTokenizer.from_pretrained(query_encoder_path_or_name)

    def encode(self,
               datasets: Datasets = None,
               add_cls_tokens: bool = True,
               works_num: int = 64,
               batch_size: int = 512,
               save_to_memmap: bool = True):
        # preprocess
        preprocess_flag = True
        if os.path.exists(datasets.preprocess_dir):
            files = os.listdir(datasets.preprocess_dir)
            if len(files) > 0:
                preprocess_flag = False
        if preprocess_flag:
            preprocess_data(data_dir=datasets.file_path,
                            output_dir=datasets.preprocess_dir,
                            text_tokenizer=self.text_tokenizer,
                            add_cls_tokens=add_cls_tokens,
                            max_doc_length=datasets.max_doc_length,
                            max_query_length=datasets.max_query_length,
                            workers_num=works_num)
        if datasets.emb_size is None:
            params = self.encoder.query_encoder.parameters()
            for last in params:
                continue
            datasets.emb_size = last.size()[-1]
        # generate embeddings
        if datasets.doc_embeddings_dir is None and datasets.docs_path is not None:
            os.makedirs(datasets.embedding_dir, exist_ok=True)

            if datasets.train_rels_path is not None:
                datasets.train_rels_path = os.path.join(datasets.preprocess_dir, 'train-rels.tsv')
            if datasets.dev_rels_path is not None:
                datasets.dev_rels_path = os.path.join(datasets.preprocess_dir, 'dev-rels.tsv')
            inference(data_dir=datasets.preprocess_dir,
                      is_query=False,
                      encoder=self.encoder,
                      prefix=f'docs',
                      max_length=datasets.max_doc_length,
                      output_dir=datasets.embedding_dir,
                      batch_size=batch_size,
                      enable_rewrite=False,
                      return_vecs=False,
                      save_to_memmap=save_to_memmap)
            datasets.doc_embeddings_dir = os.path.join(datasets.embedding_dir, 'docs.memmap')
        if datasets.train_queries_embedding_dir is None and datasets.train_queries_path is not None:
            inference(data_dir=datasets.preprocess_dir,
                      is_query=True,
                      encoder=self.encoder,
                      prefix=f'train-queries',
                      max_length=datasets.max_query_length,
                      output_dir=datasets.embedding_dir,
                      batch_size=batch_size,
                      enable_rewrite=False,
                      return_vecs=False,
                      save_to_memmap=save_to_memmap)
            datasets.train_queries_embedding_dir = os.path.join(datasets.embedding_dir, 'train-queries.memmap')
        if datasets.dev_queries_embedding_dir is None and datasets.dev_queries_path is not None:
            inference(data_dir=datasets.preprocess_dir,
                      is_query=True,
                      encoder=self.encoder,
                      prefix=f'dev-queries',
                      max_length=datasets.max_query_length,
                      output_dir=datasets.embedding_dir,
                      batch_size=batch_size,
                      enable_rewrite=False,
                      return_vecs=False,
                      save_to_memmap=save_to_memmap)
            datasets.dev_queries_embedding_dir = os.path.join(datasets.embedding_dir, 'dev-queries.memmap')