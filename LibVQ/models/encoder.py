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
        # print('input_ids is cuda ', input_ids.is_cuda)
        # print('attention_mask is cuda ', attention_mask.is_cuda)
        last_hidden_states = self.encoder(input_ids, attention_mask)[0]
        return self.sentence_pooling(last_hidden_states, attention_mask)


class Encoder(nn.Module):
    def __init__(self,
                 query_encoder=None,
                 doc_encoder=None,
                 output_size=768,
                 ):
        super(Encoder, self).__init__()
        self.query_encoder = query_encoder
        self.doc_encoder = doc_encoder
        self.output_size = output_size

        # params = query_encoder.parameters()
        # for last in params:
        #     continue
        # input_size = last.size()[-1]
        # input_size = query_encoder.encoder.config.hidden_size
        # self.query_linear = nn.Linear(in_features=input_size, out_features=output_size, bias=True)

        # params = doc_encoder.parameters()
        # for last in params:
        #     continue
        # input_size = last.size()[-1]
        # input_size = doc_encoder.encoder.config.hidden_size
        # self.doc_linear = nn.Linear(in_features=input_size, out_features=output_size, bias=True)

    def query_emb(self, input_ids, attention_mask):
        # print('input_ids is cuda ', input_ids.is_cuda)
        # print('attention_mask is cuda ', attention_mask.is_cuda)
        outputs = self.query_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # outputs = self.query_linear(outputs)
        return outputs.contiguous()

    def doc_emb(self, input_ids, attention_mask):
        outputs = self.doc_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # outputs = self.doc_linear(outputs)
        return outputs.contiguous()

    def forward(self, input_ids, attention_mask, is_query):
        if is_query:
            outputs = self.query_emb(input_ids, attention_mask)
            # return self.query_linear(outputs)
        else:
            outputs = self.doc_emb(input_ids, attention_mask)
            # return self.doc_linear(outputs)
        return outputs

    def save(self, save_file):
        torch.save(self.state_dict(), save_file)

