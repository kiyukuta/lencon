import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F


def get_lstm_init_bias(dim_hid):
    lstm_init_bias = np.full((dim_hid * 4), 0., dtype=np.float32)
    lstm_init_bias[dim_hid * 2: dim_hid*3] = 1.0
    return lstm_init_bias


class LstmEncoder(chainer.Chain):

    def __init__(self, dim_emb, dim_hid):
        lstm_init_bias = get_lstm_init_bias(dim_hid)

        super().__init__(
            eh=L.Linear(dim_emb, dim_hid * 4, initial_bias=lstm_init_bias),
            hh=L.Linear(dim_hid, dim_hid * 4, nobias=True),
        )
        self.dim_emb = dim_emb
        self.dim_hid = dim_hid

    def __call__(self, e, c, h, cxt=None):
        lstm_in = self.eh(e) + self.hh(h)
        c, h = F.lstm(c, lstm_in)
        return c, h


class BiLstmEncoder(LstmEncoder):

    def __init__(self, dim_emb, dim_hid):
        lstm_init_bias = get_lstm_init_bias(dim_hid)

        super().__init__(dim_emb, dim_hid)
        self.add_link("b_eh", L.Linear(dim_emb, dim_hid * 4,
                                       initial_bias=lstm_init_bias))
        self.add_link("b_hh", L.Linear(dim_hid, dim_hid * 4, nobias=True))

    def step(self, e, c, h, backward):
        if backward:
            lstm_in = self.b_eh(e) + self.b_hh(h)
        else:
            lstm_in = self.eh(e) + self.hh(h)
        c, h = F.lstm(c, lstm_in)
        return c, h

    def __call__(self, x, train=True):
        self.batchsize = x[0].data.shape[0]

        zeros = self.xp.zeros((self.batchsize, self.dim_hid), dtype=np.float32)
        c = h = chainer.Variable(zeros, volatile=not train)
        bc = bh = chainer.Variable(zeros, volatile=not train)
        self.fh_list = []
        self.bh_list = []
        self.fc_list = []
        self.bc_list = []

        self.source_length = len(x)
        for i in range(self.source_length):
            c, h = self.step(x[i], c, h, backward=False)
            self.fh_list.append(h)
            self.fc_list.append(c)

            bxt = x[self.source_length - i - 1]
            bc, bh = self.step(bxt, bc, bh, backward=True)
            self.bh_list.insert(0, bh)
            self.bc_list.insert(0, bc)

    def get_final_states(self, reverse=True):
        if reverse:
            return self.bc_list[0], self.bh_list[0]
        return self.fc_list[-1], self.fh_list[-1]

    def get_source_states(self, concat=True):
        shape = (self.batchsize, self.source_length, self.dim_hid)
        fhs = F.concat(self.fh_list, axis=1)
        fhs = F.reshape(fhs, shape)
        bhs = F.concat(self.bh_list, axis=1)
        bhs = F.reshape(bhs, shape)
        if concat:
            fb_hs = F.concat([fhs, bhs], axis=2)
        else:
            fb_hs = fhs + bhs
        return fb_hs


class EncoderDecoder(chainer.Chain):

    def __init__(self,
                 src_vcb_num,
                 trg_vcb_num,
                 dim_emb,
                 dim_hid):

        lstm_init_bias = get_lstm_init_bias(dim_hid)

        super().__init__(
            src_emb=L.EmbedID(src_vcb_num, dim_emb, ignore_label=-1),
            encoder=BiLstmEncoder(dim_emb, dim_hid),
            # decoder (TODO: make Decoder class)
            trg_emb=L.EmbedID(trg_vcb_num, dim_emb, ignore_label=-1),
            eh=L.Linear(dim_emb, dim_hid * 4, initial_bias=lstm_init_bias),
            hh=L.Linear(dim_hid, dim_hid * 4, nobias=True),
            ho=L.Linear(dim_hid, trg_vcb_num),
        )

        self.dim_hid = dim_hid
        self.dim_emb = dim_emb
        self.src_vcb_num = src_vcb_num
        self.trg_vcb_num = trg_vcb_num

    def set_symbols(self, xbos, xeos, ybos, yeos):
        self.xbos = xbos
        self.xeos = xeos
        self.ybos = ybos
        self.yeos = yeos

    def embed(self, source, train=True):
        xp = self.xp
        mask = xp.expand_dims(source != -1, -1)
        self.mask = chainer.Variable(mask, volatile=not train)
        x = chainer.Variable(source, volatile=not train)
        embs = self.src_emb(x)
        embs = F.split_axis(embs, embs.data.shape[1], 1)
        return embs

    def encode(self, source, train=True):
        self.batchsize, self.source_length = source.shape

        embs = self.embed(source, train=train)
        self.encoder(embs, train=train)

        x0 = chainer.Variable(self.xp.full((self.batchsize, 1),
                                           self.ybos,
                                           dtype=np.int32),
                              volatile=not train)
        h0, c0 = self.encoder.get_final_states(reverse=True)

        return {'x': x0, 'c': c0, 'h': h0}

    def decode(self, state, y, train=True):

        max_len = 50 if y is None else len(y)  # maximum number of words

        loss = 0
        outs = []

        ended = np.full((self.batchsize,), False, dtype=bool)

        x = state['x']

        for i in range(max_len):
            o, state = self.decode_once(x, state, train=train)
            ot = chainer.cuda.to_cpu(o.data).argmax(1)
            outs.append(ot.tolist())

            state = self.post_decode_once(ot, state, train=train)

            if y is not None:
                # log likelihood
                t = F.reshape(y[i], (self.batchsize,))
                loss_n = F.softmax_cross_entropy(o, t)
                loss += loss_n
                x = t
            else:
                # greedy decoding
                ended = (ot == self.yeos) | ended
                if np.all(ended == True):
                    break
                x = chainer.Variable(ot.astype(np.int32), volatile=not train)

        return list(zip(*outs)), loss

    def decode_once(self, x, state, train=True):
        c = state['c']
        h = state['h']

        emb = self.trg_emb(x)
        lstm_in = self.eh(emb) + self.hh(h)
        c, h = F.lstm(c, lstm_in)
        o = self.ho(h)
        state['c'] = c
        state['h'] = h
        return o, state

    def post_decode_once(self, output, state, train=True):
        return state

    def prepare_decoding(self, state, lengths, train=True):
        return state

    def __call__(self, source, target, lengths=None, train=True):
        self.batchsize, self.source_length = source.shape

        state = self.encode(source, train=train)
        state = self.prepare_decoding(state, lengths, train=train)

        y = None
        if target is not None:
            y = chainer.Variable(target, volatile=not train)
            y = F.split_axis(y, y.data.shape[1], 1)

        outs, loss = self.decode(state, y, train=train)
        return outs, loss
