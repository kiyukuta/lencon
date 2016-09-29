
import numpy as np

import chainer.functions as F

from models.attention import EncDecEarlyAttn, EncDecLateAttn


class LenInitEarlyAttn(EncDecEarlyAttn):

    def __init__(self,
                 src_vcb_num,
                 trg_vcb_num,
                 dim_emb,
                 dim_hid):

        super(LenInitEarlyAttn, self).__init__(src_vcb_num,
                                               trg_vcb_num,
                                               dim_emb,
                                               dim_hid)
        # length
        self.encoder.add_param("c0", (1, dim_hid))
        param = np.random.normal(0, np.sqrt(1. / dim_hid), (1, dim_hid))
        self.encoder.c0.data[...] = param

    def prepare_decoding(self, state, lengths, train=True):
        state = super().prepare_decoding(state, lengths, train=train)

        x = state['x']
        h = state['h']

        c = F.broadcast_to(self.encoder.c0, (self.batchsize, self.dim_hid))
        lengths = lengths.astype(np.float32)
        lengths = lengths.reshape((self.batchsize, 1))
        c = c * lengths
        return {'x': x, 'c': c, 'h': h}


class LenInitLateAttn(EncDecLateAttn):

    def __init__(self,
                 src_vcb_num,
                 trg_vcb_num,
                 dim_emb,
                 dim_hid):

        super(LenInitLateAttn, self).__init__(src_vcb_num,
                                              trg_vcb_num,
                                              dim_emb,
                                              dim_hid,
                                              attention_type='dot')
        # length
        self.encoder.add_param("c0", (1, dim_hid))
        param = np.random.normal(0, np.sqrt(1. / dim_hid), (1, dim_hid))
        self.encoder.c0.data[...] = param

    def prepare_decoding(self, state, lengths, train=True):
        state = super().prepare_decoding(state, lengths, train=train)

        x = state['x']
        h = state['h']

        c = F.broadcast_to(self.encoder.c0, (self.batchsize, self.dim_hid))
        lengths = lengths.astype(np.float32)
        lengths = lengths.reshape((self.batchsize, 1))
        c = c * lengths
        return {'x': x, 'c': c, 'h': h}
