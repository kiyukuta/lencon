import configparser
import glob
import os
import pickle
import shutil

import chainer
import chainer.optimizers
import chainer.serializers

import dataset
import models
import vocabulary


class Builder:

    def __init__(self, config_file, save_dir):
        config = configparser.SafeConfigParser()
        with open(config_file) as f:
            config.readfp(f)
        self.save_dir = save_dir
        self.config = config

    def _is_true(self, val):
        if val.lower().strip() == 'true':
            return True
        return False

    def _load_binary_config(self, config, attribute):
        return self._is_true(config.get(attribute, 'False'))

    def generate(self):
        config = self.config

        v = self._build_vocabulary(config['Dataset'])
        config['Model']['src_vcb_num'] = str(len(v[0]))
        config['Model']['trg_vcb_num'] = str(len(v[1]))

        m = self._build_model(config, v[0], v[1])

        o = self._build_optimizer(config['Optimizer'])
        o.use_cleargrads(True)
        o.setup(m)
        o.add_hook(chainer.optimizer.GradientClipping(5.))

        d = self._build_dataset(config)

        return m, o, d, v

    def _build_model(self, config, src_vocab, trg_vocab):

        def convert(val):
            if val.isdigit():
                return int(val)
            try:
                return float(val)
            except:
                return val
        model_config = config['Model']

        kwargs = {k: convert(v) for k, v in model_config.items() if k != 'name'}
        m = getattr(models, model_config['name'])(**kwargs)

        model_path = os.path.join(self.save_dir, 'model.hdf')
        # load
        if os.path.exists(model_path):
            chainer.serializers.load_hdf5(model_path, m)

        xstoi = src_vocab.stoi
        ystoi = trg_vocab.stoi
        xbos = xstoi('<s>')
        xeos = xstoi('</s>')
        ybos = ystoi('<s>')
        yeos = ystoi('</s>')
        m.set_symbols(xbos, xeos, ybos, yeos)

        m.name = model_config['name']
        m.byte = self._load_binary_config(config['Training'], 'byte')
        m.reverse_output = self._load_binary_config(
            config['Training'], 'reverse_output')
        if m.byte:
            m.vocab = trg_vocab
        return m

    def _build_optimizer(self, config):
        kwargs = {k: float(v) for k, v in config.items() if k != 'name'}
        o = getattr(chainer.optimizers, config['name'])(**kwargs)
        return o

    def _build_dataset(self, config):
        reverse_output = self._load_binary_config(config['Training'],
                                                  'reverse_output')
        data_config = config['Dataset']

        td = getattr(dataset, 'Seq2seqDataset')(data_config['train_src_file'],
                                                data_config['train_trg_file'],
                                                reverse_output=reverse_output)
        vd = getattr(dataset, 'Seq2seqDataset')(data_config['valid_src_file'],
                                                data_config['valid_trg_file'],
                                                no_shuffle=True,
                                                reverse_output=reverse_output)
        return td, vd

    def _load_words(self, vocab_file):
        with open(vocab_file) as f:
            words = [l.split()[0] for l in f.readlines()]
        return words

    def _build_vocabulary(self, config):
        vocab_path = os.path.join(self.save_dir, 'vocab.pkl')
        if os.path.exists(vocab_path):
            with open(vocab_path, 'rb') as f:
                v = pickle.load(f)
            return v

        src_vcb = vocabulary.Vocabulary(
            self._load_words(config['src_vcb']), tags=None)
        trg_vcb = vocabulary.Vocabulary(
            self._load_words(config['trg_vcb']), tags=None)
        return src_vcb, trg_vcb


def remove_if_exists(filepath):
    try:
        shutil.rmtree(filepath)
    except:
        pass

    try:
        os.remove(filepath)
    except:
        pass

    assert not os.path.exists(filepath)


def reset(model_dir):
    paths = glob.glob(os.path.join(model_dir, '*'))
    for path in paths:
        if path.endswith('.ini'):
            continue
        remove_if_exists(path)


def initiate(model_dir):
    configs = glob.glob(os.path.join(model_dir, '*.ini'))
    assert len(configs) == 1, 'Put only one config file in the dierectory'
    config = configs[0]
    b = Builder(config, model_dir)

    try:
        # first time
        os.mkdir(os.path.join(model_dir, 'train_samples'))
        os.mkdir(os.path.join(model_dir, 'valid_samples'))
    except:
        pass

    m, o, d, v = b.generate()
    ret = {'builder': b,
           'model': m,
           'optimizer': o,
           'datasets': d,
           'vocabularies': v}

    return ret
