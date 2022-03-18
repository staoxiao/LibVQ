import torch
from torch import nn
from transformers import AutoConfig


class EncoderConfig(AutoConfig):
    def __init__(self):
        super(EncoderConfig).__init__()


class BaseEncoder(nn.Module):
    def query_emb(self, input_ids, attention_mask):
        raise NotImplementedError

    def doc_emb(self, input_ids, attention_mask):
        raise NotImplementedError

    def forward(self, input_ids, attention_mask, is_query):
        if is_query:
            return self.query_emb(input_ids, attention_mask)
        else:
            return self.doc_emb(input_ids, attention_mask)

    def save(self, save_file):
        torch.save(self.state_dict(), save_file)


class Encoder(BaseEncoder):
    def __init__(self,
                 query_encoder=None,
                 doc_encoder=None):
        nn.Module.__init__(self)
        self.query_encoder = query_encoder
        self.doc_encoder = doc_encoder

    def query_emb(self, input_ids, attention_mask):
        outputs = self.query_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.contiguous()

    def doc_emb(self, input_ids, attention_mask):
        outputs = self.doc_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.contiguous()

    def forward(self, input_ids, attention_mask, is_query):
        if is_query:
            return self.query_emb(input_ids, attention_mask)
        else:
            return self.doc_emb(input_ids, attention_mask)
