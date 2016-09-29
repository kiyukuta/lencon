import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F


class AttenderBase(chainer.Chain):

    def __init__(self, dim_hid):
        super().__init__()
        self.dim_hid = dim_hid

    def _prepare(self, source_hiddens):
        raise NotImplementedError()

    def set_source_info(self, source_hiddens, mask, train=True):
        self.batchsize, self.src_len, self.dim_out = source_hiddens.data.shape

        self.h = self._prepare(source_hiddens)

        minf = self.xp.full((self.batchsize, self.src_len, 1),
                            -1000.,
                            dtype=np.float32)
        self.minf = chainer.Variable(minf, volatile=not train)

        # for visualizing
        self.history = None
        if not train:
            self.history = []

        self.source_hiddens = source_hiddens
        self.mask = mask

    def _attend(self, p):
        raise NotImplementedError()

    def __call__(self, p, train=True):
        attention = self._attend(p)

        if self.history is not None:
            self.history.append(
                chainer.cuda.to_cpu(attention.data[0, :, 0]).tolist())

        ret = F.batch_matmul(F.swapaxes(self.source_hiddens, 2, 1), attention)
        return F.reshape(ret, (self.batchsize, self.dim_out))


class DotAttender(AttenderBase):

    def __init__(self, dim_hid):
        super().__init__(dim_hid)

    def _prepare(self, source_hiddens):
        return source_hiddens

    def _attend(self, p):
        weight = F.batch_matmul(self.source_hiddens, p)
        weight = F.where(self.mask, weight, self.minf)
        attention = F.softmax(weight)
        return attention


class MlpAttender(AttenderBase):

    def __init__(self, dim_hid):
        super().__init__(dim_hid)
        self.add_link('eh', L.Linear(dim_hid * 2, dim_hid),)
        self.add_link('xh', L.Linear(dim_hid, dim_hid))
        self.add_link('hw', L.Linear(dim_hid, 1))

    def _prepare(self, source_hiddens):
        self.shape1 = (self.batchsize * self.src_len, self.dim_hid * 2)
        src_hs = F.reshape(source_hiddens, self.shape1)
        h_reshaped = self.eh(src_hs)

        self.shape2 = (self.batchsize, self.src_len, self.dim_hid)
        h = F.reshape(h_reshaped, self.shape2)
        return h

    def _attend(self, p):
        p = self.xh(p)
        p = F.expand_dims(p, 1)
        p = F.broadcast_to(p, self.shape2)

        h = F.tanh(self.h + p)
        shape3 = (self.batchsize * self.src_len, self.dim_hid)
        h_reshaped = F.reshape(h, shape3)
        weight_reshaped = self.hw(h_reshaped)
        weight = F.reshape(weight_reshaped, (self.batchsize, self.src_len, 1))
        weight = F.where(self.mask, weight, self.minf)
        attention = F.softmax(weight)
        return attention
