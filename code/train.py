import argparse
import datetime
import os
import logging
import numpy as np
import pickle

# os.environ["CHAINER_TYPE_CHECK"] = "0"
import chainer
import chainer.optimizers
import chainer.serializers
import cupy

import builder


def prepare(data, stoi, eos, bos=None):
    maxlen = max([len(d) for d in data])

    ret = [[stoi(w) for w in s] + [eos] + [-1] * (maxlen - len(s))
           for s in data]
    if bos:
        ret = [[bos] + r for r in ret]
    return ret


class Trainer():
    ''' seq2seq trainer
    '''

    def __init__(self, model, optimizer, vocab, save_dir):
        assert len(vocab) == 2

        self.model = model
        self.optimizer = optimizer
        self.src_vcb, self.trg_vcb = vocab
        self.save_dir = save_dir

        # logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        self.sh = logging.StreamHandler()
        self.sh.setLevel(logging.INFO)
        logger.addHandler(self.sh)

        self.fh = logging.FileHandler(os.path.join(save_dir, "log"))
        self.fh.setLevel(logging.DEBUG)
        logger.addHandler(self.fh)
        self.logger = logger

        self.xbos = self.src_vcb.stoi("<s>")
        self.xeos = self.src_vcb.stoi("</s>")
        self.ybos = self.trg_vcb.stoi("<s>")
        self.yeos = self.trg_vcb.stoi("</s>")
        self.model.set_symbols(self.xbos, self.xeos, self.ybos, self.yeos)

    def train(self, epochs, train_data, valid_data, batch_size,
              batches_for_sort=100):
        self.train_data = train_data
        self.valid_data = valid_data

        train_loss = 0
        prev_valid_loss = -1.

        logger = self.logger

        while True:
            self.optimizer.new_epoch()
            e = self.optimizer.epoch

            # update logger format
            fmt = logging.Formatter(
                '%(asctime)s E{}: ## %(message)s'.format(e))
            self.fh.setFormatter(fmt)
            self.sh.setFormatter(fmt)
            logger.info("start training")

            # train
            train_loss = self.run_epoch(self.model, self.train_data,
                                        batch_size, batches_for_sort)
            logger.info("training loss: %f", train_loss)

            # validate
            logger.info("start validation")
            valid_loss = self.run_epoch(self.model, self.valid_data,
                                        batch_size, batches_for_sort,
                                        train=False)
            logger.info("valid loss: %f", valid_loss)

            if prev_valid_loss > 0 and valid_loss > prev_valid_loss:
                logger.info("End training2 (vl: {}, prev_vl: {})".format(
                    valid_loss, prev_valid_loss))
                break

            logger.info("save")
            self.save()

            if self.optimizer.epoch >= epochs:
                logger.info("End training1 (vl: {}, prev_vl: {})".format(
                    valid_loss, prev_valid_loss))
                break
            prev_valid_loss = valid_loss


    def run_epoch(self, model, data, batch_size, batches_for_sort, train=True):
        opt = self.optimizer

        epoch_loss = 0
        cnt = 0
        words_cnt = 0

        start = datetime.datetime.now()

        for n, (source, target) in enumerate(data.generate(batch_size,
                                                           batches_for_sort)):

            try:
                # model.zerograds()
                model.cleargrads()
                outs, loss = self._one_batch(model, source, target, opt, train)
                if train:
                    loss.backward()
                    opt.update()
                loss = chainer.cuda.to_cpu(loss.data)

            except cupy.cuda.runtime.CUDARuntimeError as err:
                # sl = max([len(s) for s in source])
                # tl = max([len(t) for t in target])
                # self.logger.info("Out of Memory: {} {}".format(sl, tl))
                # continue
                raise err

            words_cnt += sum([len(s) for s in source])
            words_cnt += sum([len(t) for t in target])

            epoch_loss += loss * batch_size

            if train:
                if cnt >= 500000:
                    self.logger.info(
                        "%d trained. save some samples" % (n * batch_size))
                    log_file = os.path.join(
                        self.save_dir,
                        "train_samples",
                        "e{}.txt".format(opt.epoch))
                    with open(log_file, 'a') as f:
                        self.log_samples(source, target, outs, f,
                                         n*batch_size, 5, train=True)

                    self.save()
                    self.logger.info("model saved")
                    second = (datetime.datetime.now() - start).seconds
                    self.logger.info("through put: {}, seconds: {}".format(
                        words_cnt / float(second), second))

                    words_cnt = 0
                    start = datetime.datetime.now()
                    cnt = 0
            else:
                log_file = os.path.join(
                    self.save_dir,
                    "valid_samples",
                    "e{}.txt".format(opt.epoch))
                with open(log_file, 'a') as f:
                    self.log_samples(source, target, outs, f,
                                     n*batch_size, -1, train=False)
            cnt += batch_size

        return epoch_loss

    def _one_batch(self, model, source, target, opt, train=True):
        xp = model.xp

        source_ids = prepare(source, self.src_vcb.stoi, self.xeos, self.xbos)
        target_ids = prepare(target, self.trg_vcb.stoi, self.yeos)
        xt = xp.array(source_ids, dtype=np.int32)
        yt = xp.array(target_ids, dtype=np.int32)

        if model.name.startswith('Len'):
            if model.byte:
                l = [len(' '.join(yi)) for yi in target]
            else:
                l = [len(yi) for yi in target]
            l = xp.array(l, dtype=np.float32)
            outs, loss = model(xt, yt, l, train=train)

        else:
            outs, loss = model(xt, yt, train=train)

        return outs, loss

    def save(self):
        save_dir = self.save_dir
        m = self.model.copy()
        m.name = self.model.name
        m.to_cpu()

        model_path = os.path.join(save_dir, 'model.hdf')
        chainer.serializers.save_hdf5(model_path, m)
        with open(os.path.join(save_dir, "vocab.pkl"), "wb") as f:
            pickle.dump((self.src_vcb, self.trg_vcb), f)

    def log_samples(self, x, y, o, fp, num, max_num=5, train=True):
        yitos = self.trg_vcb.itos

        mode = "TR" if train else "VL"

        for n, (xi, yi, oi) in enumerate(zip(x, y, o)):
            if n == max_num:
                break
            sid = n + num

            s = " ".join([w for w in xi if w != -1])
            print("{}{} src: {}".format(mode, sid, s), file=fp)
            s = " ".join([w for w in yi if w != -1])
            print("{}{} trg: {}".format(mode, sid, s), file=fp)
            s = " ".join([yitos(w) for w in oi])
            print("{}{} hyp: {}".format(mode, sid, s), file=fp)
            print("", file=fp)
        fp.flush()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--reset", action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    if args.reset:
        builder.reset(args.model_dir)
    d = builder.initiate(args.model_dir)
    model = d['model']

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    trainer = Trainer(model, d['optimizer'], d['vocabularies'], args.model_dir)
    train_data, valid_data = d['datasets']

    b = d['builder']
    batch_size = int(b.config["Training"]["batch_size"])
    batches_for_sort = int(b.config["Training"]["batches_for_sort"])
    epochs = int(b.config["Training"]["epochs"])

    trainer.train(epochs, train_data, valid_data, batch_size, batches_for_sort)
