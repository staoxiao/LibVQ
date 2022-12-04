import logging
import os
import time
from typing import Dict, Union

import faiss
import numpy
import numpy as np
import torch

from LibVQ.base_index import FaissIndex, IndexConfig
from LibVQ.models import LearnableVQ, Model, EncoderConfig, Pooler
from LibVQ.dataset import Datasets


class LearnableIndex(FaissIndex):
    def __init__(self,
                 index_config: IndexConfig = None,
                 encoder_config: EncoderConfig = None,
                 init_index_file: str = None,
                 init_pooler_file: str = None
                 ):
        """
        finetune the index

        :param index_config: Config of index. Default is None.
        :param encoder_config: Config of Encoder. Default is None.
        :param init_index_file: Create the learnable idex from the faiss index file; if is None, it will create a faiss index and save it
        :param init_pooler_file: Create the pooler layer from the faiss index file
        """
        super(LearnableIndex).__init__()
        assert index_config is not None
        self.index_config = index_config
        self.encoder_config = encoder_config
        self.init_index_file = init_index_file

        self.learnable_vq = None
        self.model = None
        self.pooler = None

        if init_index_file is None or not os.path.exists(init_index_file):
            self.is_trained = False
            logging.info(f"generating the init index by faiss")
            faiss_index = FaissIndex(emb_size=index_config.emb_size,
                                    ivf_centers_num=index_config.ivf_centers_num,
                                    subvector_num=index_config.subvector_num,
                                    subvector_bits=index_config.subvector_bits,
                                    index_method=index_config.index_method,
                                    dist_mode=index_config.dist_mode)
            self.faiss = faiss_index
            self.index = faiss_index.index
        else:
            self.is_trained = True
            if index_config.index_backend == 'SPANN':
                logging.info(f"loading the init SPANN index from {init_index_file}")
                self.index = None
                self.learnable_vq = LearnableVQ(index_config, init_index_file=init_index_file, init_index_type='SPANN',
                                                index_method=index_config.index_method, dist_mode=index_config.dist_mode)
            else:
                logging.info(f"loading the init faiss index from {init_index_file}")
                self.index = faiss.read_index(init_index_file)

                self.learnable_vq = LearnableVQ(index_config, init_index_file=init_index_file,
                                                index_method=index_config.index_method, dist_mode=index_config.dist_mode)

            self.check_index_parameters(self.learnable_vq, index_config.ivf_centers_num, index_config.subvector_num,
                                        index_config.subvector_bits, init_index_file,
                                        index_config.index_method)
            if self.learnable_vq.ivf:
                self.ivf_centers_num = self.learnable_vq.ivf.ivf_centers_num
            else:
                self.ivf_centers_num = None

        if init_pooler_file is not None:
            dicts = torch.load(init_pooler_file)
            self.pooler = Pooler(dicts['A'], dicts['b']).load_state_dict(dicts, strict=True)
            if self.learnable_vq:
                self.learnable_vq.encoder = self.model.encoder

        if encoder_config is not None and encoder_config.query_encoder_name_or_path is not None and encoder_config.doc_encoder_name_or_path is not None:
            self.model = Model(encoder_config.query_encoder_name_or_path,
                               encoder_config.doc_encoder_name_or_path)
            if self.learnable_vq:
                self.learnable_vq.pooler = self.pooler


    def faiss_train(self, data:Datasets = None):
        if self.is_trained is False:
            doc_embeddings = np.memmap(data.doc_embeddings_dir,
                                       dtype=np.float32, mode="r")
            doc_embeddings = doc_embeddings.reshape(-1, data.emb_size)
            print(self.index_config.emb_size)
            print(data.emb_size)
            if self.index_config.emb_size != data.emb_size:
                if not self.pooler:
                    # training data
                    mat = faiss.PCAMatrix(data.emb_size, self.index_config.emb_size)
                    mat.train(doc_embeddings)
                    assert mat.is_trained
                    # doc_embeddings = mat.apply(len(doc_emb), doc_emb)
                    # print(doc_embeddings.shape)
                    # print this to show that the magnitude of tr's columns is decreasing
                    b = faiss.vector_to_array(mat.b)
                    A = faiss.vector_to_array(mat.A).reshape(mat.d_out, mat.d_in)
                    self.pooler = Pooler(A, b)
                    doc_embeddings = self.pooler(torch.Tensor(doc_embeddings.copy())).detach().cpu().numpy()
                else:
                    doc_embeddings = self.pooler(torch.Tensor(doc_embeddings.copy())).detach().cpu().numpy()

            self.faiss.fit(doc_embeddings)
            self.faiss.add(doc_embeddings)
            self.is_trained = True

            if self.init_index_file is None:
                init_index_file = f'./temp/{self.index_config.index_method}_ivf{self.index_config.ivf_centers_num}_pq{self.index_config.subvector_num}x{self.index_config.subvector_bits}.index'
                os.makedirs('./temp', exist_ok=True)

            logging.info(f"save the init index to {init_index_file}")
            self.faiss.save_index(init_index_file)
            self.index = self.faiss.index
            # del self.faiss
            self.learnable_vq = LearnableVQ(self.index_config, init_index_file=init_index_file,
                                            index_method=self.index_config.index_method, dist_mode=self.index_config.dist_mode)

            self.check_index_parameters(self.learnable_vq, self.index_config.ivf_centers_num, self.index_config.subvector_num,
                                        self.index_config.subvector_bits, init_index_file,
                                        self.index_config.index_method)

            if self.learnable_vq.ivf:
                self.ivf_centers_num = self.learnable_vq.ivf.ivf_centers_num
            else:
                self.ivf_centers_num = None

            if self.model:
                self.learnable_vq.encoder = self.model.encoder
            if self.pooler:
                self.learnable_vq.pooler = self.pooler

    def check_index_parameters(self,
                               vq_model: LearnableVQ,
                               ivf_centers_num: int,
                               subvector_num: int,
                               subvector_bits: int,
                               init_index_file: str,
                               index_method: str):
        if 'ivf' in index_method:
            if ivf_centers_num is not None and vq_model.ivf.ivf_centers_num != ivf_centers_num:
                raise ValueError(
                    f"The ivf_centers_num :{vq_model.ivf.ivf_centers_num} of index from {init_index_file} is not equal to you set: {ivf_centers_num}. "
                    f"please use the correct saved index or set it None to create a new faiss index")

        if 'pq' in index_method:
            if subvector_num is not None and vq_model.pq.subvector_num != subvector_num:
                raise ValueError(
                    f"The subvector_num :{vq_model.pq.subvector_num} of index from {init_index_file} is not equal to you set: {subvector_num}. "
                    f"please use the correct saved index or set it None to create a new faiss index")
            if subvector_bits is not None and vq_model.pq.subvector_bits != subvector_bits:
                raise ValueError(
                    f"The subvector_bits :{vq_model.pq.subvector_bits} of index from {init_index_file} is not equal to you set: {subvector_bits}. "
                    f"please use the correct saved index or set it None to create a new faiss index")

    def update_index_with_ckpt(self,
                               ckpt_file: str = None,
                               saved_ckpts_path: str = None,
                               doc_embeddings: numpy.ndarray = None):
        '''
        Update the index based on the saved ckpt
        :param ckpt_path: The trained ckpt file. If set None, it will select the lateset ckpt in saved_ckpts_path.
        :param saved_ckpts_path: The path to save the ckpts
        :param doc_embeddings: embeddings of docs
        :return:
        '''
        if ckpt_file is None:
            assert saved_ckpts_path is not None
            ckpt_file = self.get_latest_ckpt(saved_ckpts_path)

        logging.info(f"updating index based on {ckpt_file}")

        ivf_file = os.path.join(ckpt_file, 'ivf_centers.npy')
        if os.path.exists(ivf_file):
            logging.info(f"loading ivf centers from {ivf_file}")
            center_vecs = np.load(ivf_file)
            self.update_ivf(center_vecs)

        codebook_file = os.path.join(ckpt_file, 'codebook.npy')
        if os.path.exists(codebook_file):
            logging.info(f"loading codebook from {codebook_file}")
            codebook = np.load(codebook_file)
            self.update_pq(codebook=codebook, doc_embeddings=doc_embeddings)


    def update_ivf(self,
                   center_vecs: numpy.ndarray):
        if isinstance(self.index, faiss.IndexPreTransform):
            ivf_index = faiss.downcast_index(self.index.index)
            coarse_quantizer = faiss.downcast_index(ivf_index.quantizer)
        else:
            coarse_quantizer = faiss.downcast_index(self.index.quantizer)

        faiss.copy_array_to_vector(
            center_vecs.ravel(),
            coarse_quantizer.xb)

    def update_pq(self,
                  codebook: numpy.ndarray,
                  doc_embeddings: numpy.ndarray):
        if isinstance(self.index, faiss.IndexPreTransform):
            ivf_index = faiss.downcast_index(self.index.index)
            faiss.copy_array_to_vector(
                codebook.ravel(),
                ivf_index.pq.centroids)
        else:
            faiss.copy_array_to_vector(
                codebook.ravel(),
                self.index.pq.centroids)

        logging.info(f"updating the quantized results of docs' embeddings")

        if self.pooler:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.pooler.to(device)
            doc_embeddings = torch.Tensor(doc_embeddings.copy()).to(device)
            doc_embeddings = self.pooler(doc_embeddings).detach().cpu().numpy()
        self.index.remove_ids(faiss.IDSelectorRange(0, len(doc_embeddings)))
        self.index.add(doc_embeddings)

    def get_latest_ckpt(self, saved_ckpts_path: str):
        if len(os.listdir(saved_ckpts_path)) == 0: raise IOError(f"There is no ckpt in path: {saved_ckpts_path}")

        latest_epoch, latest_step = 0, 0
        for ckpt in os.listdir(saved_ckpts_path):
            if 'epoch' in ckpt and 'step' in ckpt:
                name = ckpt.split('_')
                epoch, step = int(name[1]), int(name[3])
                if epoch > latest_epoch:
                    latest_epoch, latest_step = epoch, step
                elif epoch == latest_epoch:
                    latest_step = max(latest_step, step)
        assert latest_epoch > 0 or latest_step > 0
        return os.path.join(saved_ckpts_path, f"epoch_{latest_epoch}_step_{latest_step}")

    def get_temp_checkpoint_save_path(self):
        time_str = time.strftime('%m_%d-%H-%M-%S', time.localtime(time.time()))
        return f'./temp/{time_str}'

    def load_embedding(self,
                       emb: Union[str, numpy.ndarray],
                       emb_size: int):
        if isinstance(emb, str):
            assert 'npy' in emb or 'memmap' in emb
            if 'memmap' in emb:
                embeddings = np.memmap(emb, dtype=np.float32, mode="r")
                return embeddings.reshape(-1, emb_size)
            elif 'npy' in emb:
                return np.load(emb)
        else:
            return emb

    def train(self,
              data: Datasets = None,
              per_query_neg_num: int = 1,
              save_ckpt_dir: str = None,
              logging_steps: int = 100,
              per_device_train_batch_size: int = 512,
              loss_weight: Dict[str, object] = {'encoder_weight': 0.0, 'pq_weight': 1.0,
                                                'ivf_weight': 'scaled_to_pqloss'},
              lr_params: Dict[str, object] = {'encoder_lr': 0.0, 'pq_lr': 1e-4, 'ivf_lr': 1e-3},
              epoch: int = 30
              ):
        raise NotImplementedError

    def search_query(self, queries, data, kValue: int=20):
        id2text = dict()
        docsFile = open(data.docs_path, 'r', encoding='UTF-8')
        count = 0
        for line in docsFile:
            id2text[count] = line.replace('\n', '').replace(line.split('\t')[0] + '\t', '')
            count += 1
        docsFile.close()

        input_data = self.model.text_tokenizer(queries, padding=True)
        input_ids = torch.LongTensor(input_data['input_ids'])
        attention_mask = torch.LongTensor(input_data['attention_mask'])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        self.model.to(device)
        query_embeddings = self.model.encoder.query_emb(input_ids, attention_mask).detach().cpu().numpy()
        if self.pooler:
            query_embeddings = torch.Tensor(query_embeddings.copy()).to(device)
            self.pooler.to(device)
            query_embeddings = self.pooler(query_embeddings).detach().cpu().numpy()
        _, topk_ids = self.search(query_embeddings, kValue, nprobe=self.index_config.nprobe)
        output_texts = []
        for ids in topk_ids:
            temp = []
            for id in ids:
                if id != -1:
                    temp.append(id2text[id])
            output_texts.append(temp)
        return output_texts, topk_ids

    def get(self, data, file):
        emb = np.memmap(os.path.join(data.embedding_dir, file), dtype=np.float32, mode="r")
        emb = emb.reshape(-1, data.emb_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        emb = torch.Tensor(emb.copy()).to(device)
        self.pooler.to(device)
        emb = self.pooler(emb).detach().cpu().numpy()
        return emb



