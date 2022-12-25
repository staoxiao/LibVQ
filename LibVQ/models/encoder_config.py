import json

class EncoderConfig():
    def __init__(self,
                 is_finetune: bool = False,
                 doc_encoder_name_or_path: str = None,
                 query_encoder_name_or_path: str = None,
                 **kwargs):
        """

        :param is_finetune: Whether encoder has been pre trained, including: True, False
        :param doc_encoder_name_or_path: The name of the doc_encoder in huggingface, or the path of doc_encoder in your
                own folder, e.g. Shitao/msmarco_doc_encoder
        :param query_encoder_name_or_path: The name of the query_encoder in huggingface, or the path of query_encoder in
                your own folder, e.g. Shitao/msmarco_query_encoder
        :param kwargs:
        """
        self.is_finetune = is_finetune
        self.doc_encoder_name_or_path = doc_encoder_name_or_path
        self.query_encoder_name_or_path = query_encoder_name_or_path

    def save(self, path: str = 'encoder_configuration.json'):
        config_dict = dict()
        config_dict['is_finetune'] = self.is_finetune
        config_dict['doc_encoder_name_or_path'] = self.doc_encoder_name_or_path
        config_dict['query_encoder_name_or_path'] = self.query_encoder_name_or_path
        json.dump(config_dict, open(path, 'w+'))
        print(f'save encoder_configuration to: {path}')

    @classmethod
    def load(cls, path: str):
        print(f'loading encoder_configuration from: {path}')
        config_dict = json.load(open(path, 'r'))
        return cls(**config_dict)