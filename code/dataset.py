import sys
import types
import random


class Seq2seqDataset():

    def __init__(self, source_file, target_file, 
            no_shuffle=False, reverse_output=False):
        self.source_file = source_file
        self.target_file = target_file
        self.src = None
        self.trg = None
        self.no_shuffle = no_shuffle
        self.reverse_output = reverse_output

    def __len__(self):
        try:
            return len(self.src)
        except:
            raise "Run `reset` first."

    def __iter__(self):
        assert self.src and self.trg, "Run `reset` first."
        for x, y in zip(self.src, self.trg):
            x = self._preprocess(x)
            y = self._preprocess(y)
            yield x, y

        self.src = None
        self.trg = None

    def __getitem__(self, index):
        try:
            x = self._preprocess(self.src[index])
            if self.trg:
                y = self._preprocess(self.trg[index])
            else:
                y = None 
            return x, y
        except:
            raise "Run `reset` first."

    def _preprocess(self, sentence, reverse=False):
        return sentence.strip().split(' ')
    
    def _none_generator(self):
        yield None

    def reset(self, read_all=False):
        self.src = (d for d in open(self.source_file))
        if self.target_file:
            _trg = (d for d in open(self.target_file))
        else:
            _trg = self._none_generator()
        self.trg = _trg

        if read_all:
            self.src = list(self.src)
            if self.target_file:
                _trg = list(self.trg)
            else:
                _trg = [None for _ in range(len(self.src))]
            self.trg = _trg

    def generate(self, batch_size, batches_per_sort=10000):
        no_shuffle = self.no_shuffle
        self.reset()
        src, trg = self.src, self.trg

        if type(src) == types.GeneratorType:
            pass
        elif type(src) == list:
            data = list(zip(src, trg))
            if not no_shuffle:
                random.shuffle(data)
            src, trg = zip(*data)
        else:
            print(len(src))
            print(type(src))
            raise 

        data = []
        for x, y in zip(src, trg):
            x = x.strip().split(' ')
            if y:
                y = y.strip().split(' ')
            if self.reverse_output:
                y = y[::-1]

            data.append((x, y))
            if len(data) != batch_size * batches_per_sort:
                continue
            data = sorted(data, key=lambda x: len(x[1]))
            batches = [data[b * batch_size : (b + 1) * batch_size] 
                       for b in range(batches_per_sort)]
            if not no_shuffle:
                random.shuffle(batches)

            for b in batches:
                yield list(zip(*b))
            data = []

        if data:
            data = sorted(data, key=lambda x: len(x[-1]))
            batches = [data[b * batch_size : (b + 1) * batch_size] 
                       for b in range(int(len(data) / batch_size) + 1)]
            if not no_shuffle:
                random.shuffle(batches)

            for b in batches:
                yield list(zip(*b))
