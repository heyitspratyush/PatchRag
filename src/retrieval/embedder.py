import torch
'''from transformers import AutoTokenizer,AutoModel

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

print(type(tokenizer))
print(type(model))

text = "Proof of Work uses computational work."

encoded = tokenizer(text,return_tensors = "pt")
print(encoded)
print(encoded['input_ids'])
print(encoded['attention_mask'])
print(encoded['input_ids'].shape)
outputs = model(**encoded)

print(outputs.last_hidden_state)
print(outputs.last_hidden_state.shape)'''


def mean_pooling(last_hidden_state,attention_mask):
    expanded_attention_mask = attention_mask.unsqueeze(-1)
    masked_embeddings = last_hidden_state * expanded_attention_mask
    sum_embeddings  = masked_embeddings.sum(dim=1)  
    original_token_count = expanded_attention_mask.sum(dim=1).clamp(min=1)
    mean_embeddings = sum_embeddings / original_token_count
    return mean_embeddings