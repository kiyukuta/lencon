
class Vocabulary:
    def __init__(self, words, tags=["<unk>", "<s>", "</s>"]):
        self.words = words
        if tags:
            self.words = tags + self.words

        self.wids = {v: k for k, v in enumerate(self.words)}

    def __len__(self):
        return len(self.words)

    def itos(self, index):
        return self.words[index]

    def stoi(self, word):
        return self.wids.get(word, self.wids["<unk>"])

    def convert(self, words):
        return [self.stoi(w) for w in words]

    def revert(self, word_ids, with_tags=True):
        if with_tags:
            return [self.itos(w) for w in word_ids]

        bos = self.stoi('<s>')
        eos = self.stoi('</s>')
        try:
            first_eos = word_ids.index(eos)
            word_ids = word_ids[:first_eos]
        except:
            pass
        return [self.itos(w) for w in word_ids if w not in [bos, eos]]
