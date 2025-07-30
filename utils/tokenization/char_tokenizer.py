import torch


class CharTokenizer:
    def __init__(self, texts):
        chars = sorted(set("".join(texts))) # unique characters

        self.vocab = {c: i + 4 for i, c in enumerate(chars)}
        self.vocab["<pad>"] = 0
        self.vocab["<bos>"] = 1
        self.vocab["<eos>"] = 2
        self.vocab["<unk>"] = 3
        print(self.vocab)
        # {' ': 4, 'a': 5, 'e': 6, ..., '<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3}

        self.inv_vocab = {i: c for c, i in self.vocab.items()}
        print(self.inv_vocab)\
        # {4: ' ', 5: 'a', 6: 'e', ... , 0: '<pad>', 1: '<bos>', 2: '<eos>', 3: '<unk>'}

    def encode(self, text, add_special_tokens=True):
        ids = [self.vocab.get(c, self.vocab["<unk>"]) for c in text]
        if add_special_tokens:
            ids = [self.vocab["<bos>"]] + ids + [self.vocab["<eos>"]]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        tokens = [self.inv_vocab.get(i, "?") for i in ids]
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in ["<pad>", "<bos>", "<eos>"]]
        return "".join(tokens)

    def pad(self, sequences, pad_to_length=None):
        max_len = pad_to_length or max(len(seq) for seq in sequences)
        return [
            seq + [self.vocab["<pad>"]] * (max_len - len(seq))
            for seq in sequences
        ]

    @property
    def vocab_size(self):
        return len(self.vocab)


texts = ["hello", "hi", "how are you"]\

print("join:", "".join(texts)) # join: hellohihow are you
print("set:", set("".join(texts))) # set: {' ', 'w', 'r', 'y', 'l', 'e', 'h', 'a', 'u', 'i', 'o'}
print("sorted:", sorted(set("".join(texts)))) # set: {' ', 'w', 'r', 'y', 'l', 'e', 'h', 'a', 'u', 'i', 'o'}

tok = CharTokenizer(texts)
ids = [tok.encode(t) for t in texts]
padded = tok.pad(ids)

input_ids = torch.tensor(padded)
attention_mask = (input_ids != tok.vocab["<pad>"]).long()
