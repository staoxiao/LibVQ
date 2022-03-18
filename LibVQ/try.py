from transformers import DPRContextEncoder, DPRContextEncoderTokenizer

tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
input_ids = tokenizer("Hello, is my dog cute ?", return_tensors="pt")["input_ids"]
embeddings = model(input_ids).pooler_output

from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
print(t)