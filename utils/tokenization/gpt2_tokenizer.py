from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# model.resize_token_embeddings(len(tokenizer))

texts = ["Hello world", "how are you"]
input_ids = tokenizer(texts, return_tensors="pt", padding=True).input_ids
print(input_ids)

# токен -> текст
print(tokenizer.decode(input_ids[0]))
print(tokenizer.decode(input_ids[1]))