from typing import Dict

import torch
from LibVQ.learnable_index import LearnableIndexWithEncoder
from LibVQ.base_index import IndexConfig
from LibVQ.models import EncoderConfig
from LibVQ.dataset import Datasets
import os


class ConstrativeLearnableIndexWithEncoder(LearnableIndexWithEncoder):
    def __init__(self,
                 index_config: IndexConfig = None,
                 encoder_config: EncoderConfig = None,
                 init_index_file: str = None
                 ):
        """
        finetune the distill index

        :param index_config: Config of index. Default is None.
        :param encoder_config: Config of Encoder. Default is None.
        :param init_index_file: Create the learnable idex from the faiss index file; if is None, it will create a faiss index and save it
        """
        LearnableIndexWithEncoder.__init__(self, index_config,
                                            encoder_config,
                                            init_index_file)

    def train(self,
              data: Datasets = None,
              per_query_neg_num: int = 1,
              save_ckpt_dir: str = None,
              logging_steps: int = 100,
              per_device_train_batch_size: int = 512,
              loss_weight: Dict[str, object] = {'encoder_weight': 1.0, 'pq_weight': 1.0,
                                                'ivf_weight': 'scaled_to_pqloss'},
              lr_params: Dict[str, object] = {'encoder_lr': 1e-5, 'pq_lr': 1e-4, 'ivf_lr': 1e-3},
              epoch: int = 10
              ):
        if self.index_config.emb_size is not None and data.emb_size is not None:
            if self.index_config.emb_size != data.emb_size:
                raise ValueError("Your index config emb_size is not equal to your dataset emb_size")

        if not self.model:
            raise ValueError("Due to the lack of encoder, you can't train encoder")

        if data.docs_path is None:
            raise ValueError("Due to the lack of docs file, you can't train encoder")

        if data.train_rels_path is None:
            raise ValueError("Your dataset don't include train-rels.tsv, so you can't train index with constrative")

        self.model.encode(datasets=data)
        if self.is_trained is False:
            self.faiss_train(data)
        if self.index_config.index_method != 'flat':
            world_size = torch.cuda.device_count()
            if world_size == 1:
                self.fit(rel_data=data.train_rels_path,
                         query_embeddings=data.train_queries_embedding_dir,
                         doc_embeddings=data.doc_embeddings_dir,
                         query_data_dir=data.preprocess_dir,
                         max_query_length=data.max_query_length,
                         doc_data_dir=data.preprocess_dir,
                         max_doc_length=data.max_doc_length,
                         emb_size=self.index_config.emb_size,
                         per_query_neg_num=per_query_neg_num,
                         checkpoint_path=save_ckpt_dir,
                         logging_steps=logging_steps,
                         per_device_train_batch_size=per_device_train_batch_size,
                         loss_weight=loss_weight,
                         lr_params=lr_params,
                         loss_method='constrative',
                         epochs=epoch)
            else:
                self.fit_with_multi_gpus(rel_file=data.train_rels_path,
                                         query_embeddings_file=data.train_queries_embedding_dir,
                                         doc_embeddings_file=data.doc_embeddings_dir,
                                         query_data_dir=data.preprocess_dir,
                                         max_query_length=data.max_query_length,
                                         doc_data_dir=data.preprocess_dir,
                                         max_doc_length=data.max_doc_length,
                                         emb_size=self.index_config.emb_size,
                                         per_query_neg_num=per_query_neg_num,
                                         checkpoint_path=save_ckpt_dir,
                                         logging_steps=logging_steps,
                                         per_device_train_batch_size=per_device_train_batch_size,
                                         loss_weight=loss_weight,
                                         lr_params=lr_params,
                                         loss_method='constrative',
                                         epochs=epoch)
        if data.dev_queries_path is not None:
            os.remove(data.dev_queries_embedding_dir)
            self.encode(data_dir=data.preprocess_dir,
                        prefix='dev-queries',
                        max_length=data.max_query_length,
                        output_dir=data.embedding_dir,
                        batch_size=2048,
                        is_query=True,
                        return_vecs=True
                        )
