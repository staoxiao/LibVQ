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
                 doc_encoder=None
                 ):
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
            outputs = self.query_emb(input_ids, attention_mask)
        else:
            outputs = self.doc_emb(input_ids, attention_mask)
        return outputs

    def save(self, save_file):
        torch.save(self.state_dict(), save_file)


class Pooler(nn.Module):
    def __init__(self, A, b):
        super(Pooler, self).__init__()
        if type(A) != torch.Tensor:
            self.A = nn.Parameter(torch.FloatTensor(A), requires_grad=False)
            self.b = nn.Parameter(torch.FloatTensor(b), requires_grad=False)
        else:
            self.A = nn.Parameter(A, requires_grad=False)
            self.b = nn.Parameter(b, requires_grad=False)

    def forward(self, input_emb):
        outputs = input_emb @ self.A.T + self.b
        return outputs

    def save(self, save_file):
        torch.save(self.state_dict(), save_file)