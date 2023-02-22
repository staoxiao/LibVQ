import math
import time
from typing import Dict, List

import os
import faiss
import numpy
import torch
import numpy as np
from tqdm import tqdm

from LibVQ.base_index import BaseIndex, IndexConfig
from LibVQ.models import Model, EncoderConfig, Pooler
from LibVQ.dataset import Datasets

class FaissIndex(BaseIndex):
    def __init__(self,
                 index_config: IndexConfig = None,
                 encoder_config: EncoderConfig = None,
                 init_index_file: str = None,
                 init_pooler_file: str = None,
                 doc_embeddings: np.ndarray = None
                 ):
        """
        finetune the index

        :param index_config: Config of index. Default is None.
        :param encoder_config: Config of Encoder. Default is None.
        :param init_index_file: Create the learnable idex from the faiss index file; if is None, it will create a faiss index and save it
        :param init_pooler_file: Create the pooler layer from the faiss index file
        """
        BaseIndex.__init__(self, )
        assert index_config is not None
        self.index_config = index_config
        self.encoder_config = encoder_config
        self.init_index_file = init_index_file
        self.init_pooler_file = init_pooler_file

        self.model = None
        self.pooler = None

        self.id2text = None
        self.doc_id = None

        assert index_config.dist_mode in ('ip', 'l2')
        self.index_metric = faiss.METRIC_INNER_PRODUCT if index_config.dist_mode == 'ip' else faiss.METRIC_L2

        if index_config.emb_size is not None:
            emb_size = index_config.emb_size
        if doc_embeddings is not None:
            emb_size = np.shape(doc_embeddings)[-1]

        if init_pooler_file is not None and os.path.exists(init_pooler_file):
            dicts = torch.load(init_pooler_file)
            self.pooler = Pooler(dicts['A'], dicts['b']).load_state_dict(dicts, strict=True)

        if encoder_config is not None and encoder_config.query_encoder_name_or_path is not None and encoder_config.doc_encoder_name_or_path is not None:
            self.model = Model(encoder_config.query_encoder_name_or_path,
                               encoder_config.doc_encoder_name_or_path)

        if init_index_file is None or not os.path.exists(init_index_file):
            if index_config.index_method == 'flat':
                self.index = faiss.IndexFlatIP(emb_size) if index_config.dist_mode == 'ip' else faiss.IndexFlatL2(emb_size)
            elif index_config.index_method == 'ivf':
                quantizer = faiss.IndexFlatIP(emb_size) if index_config.dist_mode == 'ip' else faiss.IndexFlatL2(emb_size)
                self.index = faiss.IndexIVFFlat(quantizer, emb_size, index_config.ivf_centers_num, self.index_metric)
            elif index_config.index_method == 'ivf_opq':
                self.index = faiss.index_factory(emb_size,
                                                 f"OPQ{index_config.subvector_num},IVF{index_config.ivf_centers_num}," +
                                                 f"PQ{index_config.subvector_num}x{index_config.subvector_bits}",
                                                 self.index_metric)
            elif index_config.index_method == 'ivf_pq':
                self.index = faiss.index_factory(emb_size, f"IVF{index_config.ivf_centers_num}," +
                                                           f"PQ{index_config.subvector_num}x{index_config.subvector_bits}",
                                                 self.index_metric)
            elif index_config.index_method == 'opq':
                self.index = faiss.index_factory(emb_size, f"OPQ{index_config.subvector_num}," +
                                                           f"PQ{index_config.subvector_num}x{index_config.subvector_bits}",
                                                 self.index_metric)
            elif index_config.index_method == 'pq':
                self.index = faiss.index_factory(emb_size, f"PQ{index_config.subvector_num}x{index_config.subvector_bits}",
                                                 self.index_metric)
            self.is_trained = False
        else:
            self.index = faiss.read_index(init_index_file)
            self.is_trained = True

        self.index_method = index_config.index_method
        self.ivf_centers_num = index_config.ivf_centers_num
        self.subvector_num = index_config.subvector_num

        if doc_embeddings is not None:
            self.fit(doc_embeddings)
            self.add(doc_embeddings)
            self.is_trained = True

    def train(self,
              data: Datasets = None):
        if data.doc_embeddings_dir is None:
            if not self.model:
                raise ValueError("Due to the lack of encoder, you cannot infer embedding")
            elif self.encoder_config.is_finetune is False:
                raise ValueError("Due to your encoder is not finetune, you can't use distill")
            self.model.encode(datasets=data)

        if self.is_trained is False:
            doc_embeddings = np.memmap(data.doc_embeddings_dir,
                                       dtype=np.float32, mode="r")
            doc_embeddings = doc_embeddings.reshape(-1, data.emb_size)
            if self.index_config.emb_size != data.emb_size:
                with torch.no_grad():
                    if not self.pooler:
                        # training data
                        mat = faiss.PCAMatrix(data.emb_size, self.index_config.emb_size)
                        mat.train(doc_embeddings)
                        assert mat.is_trained
                        # print this to show that the magnitude of tr'search2 columns is decreasing
                        b = faiss.vector_to_array(mat.b)
                        A = faiss.vector_to_array(mat.A).reshape(mat.d_out, mat.d_in)
                        self.pooler = Pooler(A, b)
                        doc_embeddings = self.pooler(torch.Tensor(doc_embeddings.copy())).detach().cpu().numpy()
                    else:
                        doc_embeddings = self.pooler(torch.Tensor(doc_embeddings.copy())).detach().cpu().numpy()

            self.fit(doc_embeddings)
            self.add(doc_embeddings)
            self.is_trained = True

    def build(self, docs_path):
        if isinstance(docs_path, str):
            id2text = dict()
            doc_id = dict()
            docsFile = open(docs_path, 'r', encoding='UTF-8')
            count = 0
            for line in docsFile:
                self.doc_id[count] = line.split('\t')[0]
                self.id2text[count] = ' '.join(line.strip('\n').split('\t')[1:])
                count += 1
            docsFile.close()
        else:
            self.id2text = docs_path[0]
            self.doc_id = docs_path[1]

    def search_query(self, queries, docs_path: None, kValue: int = 20):
        if docs_path is not None:
            if isinstance(docs_path, str):
                id2text = dict()
                doc_id = dict()
                docsFile = open(docs_path, 'r', encoding='UTF-8')
                count = 0
                for line in docsFile:
                    doc_id[count] = line.split('\t')[0]
                    id2text[count] = ' '.join(line.strip('\n').split('\t')[1:])
                    count += 1
                docsFile.close()
            else:
                id2text = docs_path[0]
                doc_id = docs_path[1]
        else:
            if self.id2text is None or self.doc_id is None:
                raise ValueError("You should provide your docs path")
            id2text = self.id2text
            doc_id = self.doc_id

        input_data = self.model.text_tokenizer(queries, padding=True, truncation=True)
        input_ids = torch.LongTensor(input_data['input_ids'])
        attention_mask = torch.LongTensor(input_data['attention_mask'])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        self.model.to(device)
        with torch.no_grad():
            query_embeddings = self.model.encoder.query_emb(input_ids, attention_mask).detach().cpu().numpy()
        if self.pooler:
            with torch.no_grad():
                query_embeddings = torch.Tensor(query_embeddings.copy()).to(device)
                self.pooler.to(device)
                query_embeddings = self.pooler(query_embeddings).detach().cpu().numpy()
        _, topk_ids = self.search(query_embeddings, kValue, nprobe=self.index_config.nprobe)
        output_texts = []
        output_ids = []
        for ids in topk_ids:
            temp_text = []
            temp_id = []
            for id in ids:
                if id != -1:
                    temp_text.append(id2text[id])
                    temp_id.append(doc_id[id])
            output_texts.append(temp_text)
            output_ids.append(temp_id)
        return output_texts, output_ids

    @classmethod
    def load_all(cls, load_path):
        """

        :param load_path: load all parameters from this path
        :return:
        """
        index_config_file = os.path.join(load_path, 'index_config.json')
        index_config = IndexConfig.load(index_config_file) if os.path.exists(index_config_file) else None
        encoder_config_file = os.path.join(load_path, 'encoder_config.json')
        encoder_config = EncoderConfig.load(encoder_config_file) if os.path.exists(encoder_config_file) else None
        index_file = os.path.join(load_path, 'index.index')
        pooler_file = os.path.join(load_path, 'pooler.pth')

        index = cls(index_config, encoder_config, index_file, pooler_file)

        if index.model:
            encoder_file = os.path.join(load_path, 'encoder.bin')
            if os.path.exists(encoder_file):
                index.model.encoder.load_state_dict(torch.load(encoder_file, map_location='cpu'))
        return index

    def save_all(self, save_path):
        """

        :param save_path: save all parameters to this path
        :return:
        """
        os.makedirs(save_path, exist_ok=True)
        if self.index_config is not None: self.index_config.save(os.path.join(save_path, 'index_config.json'))
        if self.encoder_config is not None: self.encoder_config.save(os.path.join(save_path, 'encoder_config.json'))
        if self.index is not None: self.save_index(os.path.join(save_path, 'index.index'))
        if self.model is not None: self.model.encoder.save(os.path.join(save_path, 'encoder.bin'))
        if self.pooler is not None: self.pooler.save(os.path.join(save_path, 'pooler.pth'))

    def get_emb(self, data, file):
        emb = np.memmap(file, dtype=np.float32, mode="r")
        emb = emb.reshape(-1, data.emb_size)
        if self.pooler:
            with torch.no_grad():
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                emb = torch.Tensor(emb.copy()).to(device)
                self.pooler.to(device)
                emb = self.pooler(emb).detach().cpu().numpy()
        return emb

    def fit(self, embeddings):
        if self.index_method != 'flat':
            self.index.train(embeddings)
        self.is_trained = True

    def add(self, embeddings):
        if self.is_trained:
            self.index.add(embeddings)
        else:
            raise RuntimeError("The index need to be trained")

    def load_index(self, index_file):
        self.index = faiss.read_index(index_file)

    def save_index(self, index_file):
        faiss.write_index(self.index, index_file)

    def CPU_to_GPU(self, gpu_index=0):
        res = faiss.StandardGpuResources()
        res.noTempMemory()
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        self.index = faiss.index_cpu_to_gpu(res, gpu_index, self.index, co)

    def GPU_to_CPU(self):
        self.index = faiss.index_gpu_to_cpu(self.index)


    def set_nprobe(self, nprobe):
        if isinstance(self.index, faiss.IndexPreTransform):
            ivf_index = faiss.downcast_index(self.index.index)
            ivf_index.nprobe = nprobe
        else:
            self.index.nprobe = nprobe

    def search(self,
               query_embeddings: numpy.ndarray,
               topk: int = 1000,
               nprobe: int = None,
               batch_size: int = 64):
        if nprobe is not None:
            self.set_nprobe(nprobe)

        start_time = time.time()
        if batch_size:
            batch_num = math.ceil(len(query_embeddings) / batch_size)
            all_scores = []
            all_search_results = []
            for step in tqdm(range(batch_num)):
                start = batch_size * step
                end = min(batch_size * (step + 1), len(query_embeddings))
                batch_emb = np.array(query_embeddings[start:end])
                score, batch_results = self.index.search(batch_emb, topk)
                all_search_results.extend([list(x) for x in batch_results])
                all_scores.extend([list(x) for x in score])
        else:
            all_scores, all_search_results = self.index.search(query_embeddings, topk)
        search_time = time.time() - start_time
        print(
            f'number of query:{len(query_embeddings)},  searching time per query: {search_time / len(query_embeddings)}')
        return all_scores, all_search_results

    def test(self,
             query_embeddings: numpy.ndarray,
             ground_truths: Dict[int, List[int]],
             topk: int,
             MRR_cutoffs: List[int],
             Recall_cutoffs: List[int],
             nprobe: int = 1,
             qids: List[int] = None,
             batch_size: int = 64):
        assert max(max(MRR_cutoffs), max(Recall_cutoffs)) <= topk
        scores, retrieve_results = self.search(query_embeddings, topk, nprobe, batch_size)
        return self.evaluate(retrieve_results, ground_truths, MRR_cutoffs, Recall_cutoffs, qids)

    def get_rotate_matrix(self):
        assert isinstance(self.index, faiss.IndexPreTransform)
        vt = faiss.downcast_VectorTransform(self.index.chain.at(0))
        rotate = faiss.vector_to_array(vt.A).reshape(vt.d_out, vt.d_in)
        return rotate

    def get_codebook(self):
        if isinstance(self.index, faiss.IndexPreTransform):
            pq_index = faiss.downcast_index(self.index.index)
        else:
            pq_index = self.index

        centroid_embeds = faiss.vector_to_array(pq_index.pq.centroids)
        codebook = centroid_embeds.reshape(pq_index.pq.M, pq_index.pq.ksub, pq_index.pq.dsub)
        return codebook
