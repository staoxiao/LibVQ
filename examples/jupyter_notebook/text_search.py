from transformers import AutoTokenizer
import  torch
from LibVQ.inference import get_embeddings
from LibVQ.models import Encoder, TransformerModel
from LibVQ.base_index import FaissIndex

# Load tokenizer and encoder
tokenizer = AutoTokenizer.from_pretrained('Shitao/msmarco_query_encoder')
query_encoder = TransformerModel.from_pretrained('Shitao/msmarco_query_encoder')
doc_encoder = TransformerModel.from_pretrained('Shitao/msmarco_doc_encoder')
text_encoder = Encoder(query_encoder, doc_encoder)
# input_data = tokenizer(['what is apple', 'how much is....jjoi'], return_attention_mask=True, padding=True)
# print(tokenizer(['what is apple', 'how much is....jjoi'], return_attention_mask=True, padding=True))

def search(queries, index, id2text):
    input_data = tokenizer(queries)
    input_ids = torch.LongTensor(input_data['input_ids'])
    attention_mask = torch.LongTensor(input_data['attention_mask'])
    query_embeddings = text_encoder.query_emb(input_ids, attention_mask)
    _, topk_ids = index.search(query_embeddings)
    output_texts = []
    for ids in topk_ids:
        temp = []
        for id in ids:
            temp.append(id2text[id])
        output_texts.append(temp)
    return output_texts


search(['what is apple', 'how much is....jjoi'], FaissIndex(index_type='').load_index(index_file), {0:"fajdofj", 1:"hfado"})