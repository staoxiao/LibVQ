import json

class EncoderConfig():
    def __init__(self,
                 is_finetune: bool = False,
                 doc_encoder_name_or_path: str = None,
                 query_encoder_name_or_path: str = None,
                 max_doc_length: int = 256,
                 max_query_length: int = 32,
                 **kwargs):
        """

        :param is_finetune: Whether encoder has been pre trained
        :param doc_encoder_name_or_path: The name of the doc_encoder in huggingface, or the path of doc_encoder in your own folder
        :param query_encoder_name_or_path: The name of the query_encoder in huggingface, or the path of query_encoder in your own folder
        :param max_doc_length: The maximum doc length. If it is greater than this length, it will be truncated
        :param max_query_length: The maximum query length. If it is greater than this length, it will be truncated
        :param kwargs:
        """
        self.is_finetune = is_finetune
        self.doc_encoder_name_or_path = doc_encoder_name_or_path
        self.query_encoder_name_or_path = query_encoder_name_or_path
        self.max_doc_length = max_doc_length
        self.max_query_length = max_query_length

    def save(self, path: str = 'encoder_configuration.json'):
        config_dict = dict()
        config_dict['encoder_pretrained'] = self.encoder_pretrained
        config_dict['doc_encoder_name_or_path'] = self.doc_encoder_name_or_path
        config_dict['query_encoder_name_or_path'] = self.query_encoder_name_or_path
        config_dict['max_doc_length'] = self.max_doc_length
        config_dict['max_query_length'] = self.max_query_length
        json.dump(config_dict, open(path, 'w+'))
        print(f'save encoder_configuration to: {path}')

    @classmethod
    def load(cls, path: str):
        print(f'loading encoder_configuration from: {path}')
        config_dict = json.load(open(path, 'r'))
        return cls(**config_dict)