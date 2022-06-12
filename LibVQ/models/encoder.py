import torch
from torch import nn
from transformers import AutoModel


class TransformerModel(nn.Module):
    def __init__(self, model):
        super(TransformerModel, self).__init__()
        self.encoder = model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = AutoModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return cls(model)

    def sentence_pooling(self, hidden_states, attention_mask):
        return hidden_states[:, 0]

    def forward(self, input_ids, attention_mask):
        last_hidden_states = self.encoder(input_ids, attention_mask)[0]
        return self.sentence_pooling(last_hidden_states, attention_mask)


class Encoder(nn.Module):
    def __init__(self,
                 query_encoder=None,
                 doc_encoder=None):
        super(Encoder, self).__init__()
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

    def save(self, save_file):
        torch.save(self.state_dict(), save_file)