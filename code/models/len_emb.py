import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F

from models.attention import EncDecEarlyAttn, EncDecLateAttn


class LenEmbEarlyAttn(EncDecEarlyAttn):

    def __init__(self,
                 src_vcb_num,
                 trg_vcb_num,
                 dim_emb,
                 dim_hid):

        super().__init__(src_vcb_num,
                         trg_vcb_num,
                         dim_emb,
                         dim_hid)
        max_len = 300
        self.add_link('lh', L.Linear(dim_emb, dim_hid * 4, nobias=True))
        self.add_link('len_emb', L.EmbedID(max_len, dim_emb, ignore_label=-1))

    def set_vocab(self, vocab):
        self.vocab = vocab

    def set_lengths(self, lengths, train=True):
        lengths = lengths.reshape((self.batchsize, 1))
        lengths = lengths.astype(np.int32)
        self.lengths = chainer.Variable(lengths, volatile=not train)
        self.zeros = chainer.Variable(self.xp.zeros((self.batchsize, 1),
                                                    dtype=np.int32),
                                      volatile=not train)

    def prepare_decoding(self, state, lengths, train=True):
        state = super().prepare_decoding(state, lengths, train=train)

        self.set_lengths(lengths, train=train)
        state['lengths'] = self.lengths
        return state

    def decode_once(self, x, state, train=True):
        l = state.get('lengths', self.lengths)
        h = state['h']
        c = state['c']

        emb = self.trg_emb(x)
        lemb = self.len_emb(l)
        a = self.attender(h, train=train)
        lstm_in = self.eh(emb) + self.hh(h) + self.ch(a) + self.lh(lemb)
        c, h = F.lstm(c, lstm_in)
        o = self.ho(h)
        state['h'] = h
        state['c'] = c

        return o, state

    def post_decode_once(self, output, state, train=True):
        lengths = state['lengths']
        if self.byte:
            itos = self.vocab.itos
            consumed = self.xp.array([[len(itos(oi)) + 1]
                                      for oi in output.tolist()])
            lengths -= consumed
        else:
            lengths -= 1
        flags = chainer.Variable(lengths.data >= 0, volatile=not train)
        lengths = F.where(flags, lengths, self.zeros)
        state['lengths'] = lengths
        return state


class LenEmbLateAttn(EncDecLateAttn):

    def __init__(self,
                 src_vcb_num,
                 trg_vcb_num,
                 dim_emb,
                 dim_hid):

        super().__init__(src_vcb_num,
                         trg_vcb_num,
                         dim_emb,
                         dim_hid)
        max_len = 300
        self.add_link('lh', L.Linear(dim_emb, dim_hid * 4, nobias=True))
        self.add_link('len_emb', L.EmbedID(max_len, dim_emb, ignore_label=-1))

    def set_vocab(self, vocab):
        self.vocab = vocab

    def set_lengths(self, lengths, train=True):
        lengths = lengths.reshape((self.batchsize, 1))
        lengths = lengths.astype(np.int32)
        self.lengths = chainer.Variable(lengths, volatile=not train)
        self.zeros = chainer.Variable(self.xp.zeros((self.batchsize, 1),
                                                    dtype=np.int32),
                                      volatile=not train)

    def prepare_decoding(self, state, lengths, train=True):
        state = super().prepare_decoding(state, lengths, train=train)

        self.set_lengths(lengths, train=train)
        state['lengths'] = self.lengths
        return state

    def decode_once(self, x, state, train=True):
        l = state.get('lengths', self.lengths)
        c = state['c']
        h = state['h']
        h_tilde = state.get('h_tilde', None)

        emb = self.trg_emb(x)
        lemb = self.len_emb(l)
        lstm_in = self.eh(emb) + self.hh(h) + self.lh(lemb)
        if h_tilde is not None:
            lstm_in += self.ch(h_tilde)
        c, h = F.lstm(c, lstm_in)
        a = self.attender(h, train=train)
        h_tilde = F.concat([a, h])

        h_tilde = F.tanh(self.w_c(h_tilde))
        o = self.ho(h_tilde)
        state['c'] = c
        state['h'] = h
        state['h_tilde'] = h_tilde
        return o, state

    def post_decode_once(self, output, state, train=True):
        lengths = state['lengths']
        if self.byte:
            itos = self.vocab.itos
            consumed = self.xp.array([[len(itos(oi)) + 1]
                                     for oi in output.tolist()])
            lengths -= consumed
        else:
            lengths -= 1
        flags = chainer.Variable(lengths.data >= 0, volatile=not train)
        lengths = F.where(flags, lengths, self.zeros)
        state['lengths'] = lengths
        return state
