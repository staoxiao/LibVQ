import os
import torch

from LibVQ.inference import inference
from LibVQ.learnable_index import LearnableIndex
from LibVQ.base_index import IndexConfig
from LibVQ.models import EncoderConfig


class LearnableIndexWithEncoder(LearnableIndex):
    def __init__(self,
                 index_config: IndexConfig = None,
                 encoder_config: EncoderConfig = None,
                 init_index_file: str = None,
                 init_pooler_file: str = None
                 ):
        """
        finetune the distill index

        :param index_config: Config of index. Default is None.
        :param encoder_config: Config of Encoder. Default is None.
        :param init_index_file: Create the learnable idex from the faiss index file; if is None, it will create a faiss index and save it
        :param init_pooler_file: Create the pooler layer from the faiss index file
        """
        LearnableIndex.__init__(self, index_config,
                                encoder_config,
                                init_index_file,
                                init_pooler_file)

    def update_encoder(self,
                       encoder_file: str = None,
                       saved_ckpts_path: str = None):
        '''
        update the encoder

        :param encoder_file: Ckpt of encoder. If set None, it will select the latest ckpt in saved_ckpts_path
        :param saved_ckpts_path: The path to save the ckpts
        '''
        if encoder_file is None:
            assert saved_ckpts_path is not None
            ckpt_path = self.get_latest_ckpt(saved_ckpts_path)
            encoder_file = os.path.join(ckpt_path, 'encoder.bin')

        self.learnable_vq.encoder.load_state_dict(torch.load(encoder_file, map_location='cpu'))

    def encode(self,
               data_dir: str,
               prefix: str,
               max_length: int,
               output_dir: str,
               batch_size: int,
               is_query: bool,
               return_vecs: bool = False):
        '''
        encode text into embedding

        :param data_dir: Path to preprocessed data
        :param prefix: Prefix of data. e.g., docs, train-queries, test-queries
        :param max_length: The max length of tokens
        :param output_dir: Path to save embeddings
        :param batch_size: Batch size
        :param is_query: Set True when infer the embeddigns of query
        :param return_vecs: Whether return vectors
        :return: None or embeddigns
        '''
        os.makedirs(output_dir, exist_ok=True)
        vecs = inference(data_dir=data_dir,
                         is_query=is_query,
                         encoder=self.learnable_vq.encoder,
                         prefix=prefix,
                         max_length=max_length,
                         output_dir=output_dir,
                         batch_size=batch_size,
                         return_vecs=return_vecs)
        return vecs


