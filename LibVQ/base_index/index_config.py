import json

class IndexConfig():
    def __init__(self,
                 index_backend: str = 'FAISS',
                 index_method: str = 'ivfopq',
                 emb_size: int = 768,
                 ivf_centers_num: int = 10000,
                 subvector_num: int = 32,
                 subvector_bits: int = 8,
                 nprobe: int = 100,
                 dist_mode: str = 'ip',
                 **kwargs):
        """

        :param index_backend: The library of index, e.g., faiss, SPANN
        :param index_method: The type of index, e.g., ivf_pq, ivf_opq, pq, opq
        :param emb_size: Dim of embeddings
        :param ivf_centers_num: The number of post lists
        :param subvector_num: The number of codebooks
        :param subvector_bits: The number of codewords for each codebook
        :param nprobe: The number of lists returned in retrieval
        :param dist_mode: Metric to calculate the distance between query and doc
        :param kwargs:
        """
        self.index_backend = index_backend
        self.index_method = index_method
        self.emb_size = emb_size
        self.ivf_centers_num = ivf_centers_num
        self.subvector_num = subvector_num
        self.subvector_bits = subvector_bits
        self.nprobe = nprobe
        self.dist_mode = dist_mode

    def save(self, path: str = 'encoder_configuration.json'):
        config_dict = dict()
        config_dict['index_backend'] = self.index_backend
        config_dict['index_method'] = self.index_method
        config_dict['emb_size'] = self.emb_size
        config_dict['ivf_centers_num'] = self.ivf_centers_num
        config_dict['subvector_num'] = self.subvector_num
        config_dict['subvector_bits'] = self.subvector_bits
        config_dict['nprobe'] = self.nprobe
        config_dict['dist_mode'] = self.dist_mode
        json.dump(config_dict, open(path, 'w+'))
        print(f'save index_configuration to: {path}')

    @classmethod
    def load(cls, path: str):
        print(f'loading index_configuration from: {path}')
        config_dict = json.load(open(path, 'r'))
        return cls(**config_dict)
