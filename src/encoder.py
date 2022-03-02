import torch
from torch import nn
from transformers import AutoModel

class Encoder(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.query_encoder = AutoModel.from_pretrained(config.pretrained_model_name)
        if config.use_two_encoder:
            self.doc_encoder = AutoModel.from_pretrained(config.pretrained_model_name)
        else:
            self.doc_encoder = self.query_encoder

        self.config = config

    def pooling(self, emb_all, mask):
        if self.config.pooling_method == 'mean':
            return self.masked_mean(emb_all[0], mask)
        elif self.config.pooling_method == 'first':
            return emb_all[0][:, 0]

    def masked_mean(self, t, mask):
        s = torch.sum(t * mask.unsqueeze(-1).float(), dim=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d

    def query_emb(self, input_ids, attention_mask):
        outputs = self.query_encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.pooling(outputs, attention_mask)
        return embeddings

    def doc_emb(self, input_ids, attention_mask):
        outputs = self.doc_encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.pooling(outputs, attention_mask)
        return embeddings

    def forward(self, input_ids, attention_mask, is_query):
        if is_query:
            return self.query_emb(input_ids, attention_mask)
        else:
            return self.doc_emb(input_ids, attention_mask)













