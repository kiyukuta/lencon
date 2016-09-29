import chainer.links as L
import chainer.functions as F

from models.encoder_decoder import EncoderDecoder
from models.attenders import MlpAttender, DotAttender


def get_attention_components(attention_type, dim_hid):

    if attention_type == 'dot':
        attender = DotAttender(dim_hid)
        linear = L.Linear(dim_hid, dim_hid * 4, nobias=True)
    elif attention_type == 'concat':
        attender = MlpAttender(dim_hid)
        linear = L.Linear(2*dim_hid, dim_hid * 4, nobias=True)
    else:
        raise RuntimeError('invalid attention_type')

    return {'attender': attender,
            'ch': linear}


class EncDecEarlyAttn(EncoderDecoder):
    """Attention mechanism inspired by [Bahdanau+15]
    """

    def __init__(self,
                 src_vcb_num,
                 trg_vcb_num,
                 dim_emb,
                 dim_hid,
                 attention_type='concat'):

        super().__init__(src_vcb_num,
                         trg_vcb_num,
                         dim_emb,
                         dim_hid)

        atten_components = get_attention_components(attention_type, dim_hid)
        for k, v in atten_components.items():
            self.add_link(k, v)

        self.attention_type = attention_type

    def prepare_attention(self, train=True):
        c = False if self.attention_type == 'dot' else True
        source_hiddens = self.encoder.get_source_states(concat=c)
        self.attender.set_source_info(source_hiddens, self.mask, train=train)

    def prepare_decoding(self, state, lengths, train=True):
        self.prepare_attention(train=train)
        return state

    def decode_once(self, x, state, train=True):

        h = state['h']
        c = state['c']

        emb = self.trg_emb(x)

        a = self.attender(h, train=train)
        lstm_in = self.eh(emb) + self.hh(h) + self.ch(a)
        c, h = F.lstm(c, lstm_in)
        o = self.ho(h)
        state['h'] = h
        state['c'] = c

        return o, state


class EncDecLateAttn(EncoderDecoder):
    """Attention mechanism inspired by [Luong+15]
    """

    def __init__(self,
                 src_vcb_num,
                 trg_vcb_num,
                 dim_emb,
                 dim_hid,
                 attention_type='dot'):

        super().__init__(src_vcb_num,
                         trg_vcb_num,
                         dim_emb,
                         dim_hid)

        self.add_link('w_c', L.Linear(2*dim_hid, dim_hid))

        atten_components = get_attention_components(attention_type, dim_hid)
        for k, v in atten_components.items():
            self.add_link(k, v)

        self.attention_type = attention_type

    def prepare_attention(self, train=True):
        c = False if self.attention_type == 'dot' else True
        source_hiddens = self.encoder.get_source_states(concat=c)
        self.attender.set_source_info(source_hiddens, self.mask, train=train)

    def prepare_decoding(self, state, lengths, train=True):
        self.prepare_attention(train=train)
        return state

    def decode_once(self, x, state, train=True):

        c = state['c']
        h = state['h']
        h_tilde = state.get('h_tilde', None)

        emb = self.trg_emb(x)

        lstm_in = self.eh(emb) + self.hh(h)
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
