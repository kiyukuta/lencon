#coding: utf8
from models import encoder_decoder
from models import attention
from models import len_emb
from models import len_init

LenInit = len_init.LenInitLateAttn
LenEmb = len_emb.LenEmbLateAttn
EncoderDecoder = attention.EncDecLateAttn
