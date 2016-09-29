import argparse
import numpy as np
import queue

import chainer
import chainer.functions as F

import builder
import dataset

np.seterr('ignore')


class FixedQueue:

    def __init__(self, size):
        self.queue = queue.PriorityQueue()
        self.size = size

    def put(self, e):
        self.queue.put(e)
        while self.queue.qsize() > self.size:
            self.queue.get(False)

    def empty(self):
        return self.queue.empty()

    def get(self):
        return self.queue.get(False)

    def __len__(self):
        return self.queue.qsize()


class BeamSearchGenerator:

    name = 'default'

    def __init__(self, model, xvocab, yvocab, beam_width=10):
        self.model = model
        self.xvocab = xvocab
        self.yvocab = yvocab
        self.beam_width = beam_width
        self.eos_replacing = False

        self.EOS = self.yvocab.stoi('</s>')

    def _calc_length(self, seq, byte):
        if byte:
            return len(' '.join(self.yvocab.revert(seq, False)))
        else:
            return len(seq)

    def _judge(self, seq, byte, desired_length=None):

        if len(seq) > 1 and seq[-1] == self.EOS:
            return True, False
        return False, False

    def _edit_probs(self, probs):
        probs[self.yvocab.stoi('<unk>')] = - 10000.
        probs[self.yvocab.stoi('<s>')] = - 10000.

    def _calc_top_n(self, model, x, state, beam_width):
        o, state = model.decode_once(x, state, train=False)
        o = F.log_softmax(o, use_cudnn=False)
        o = chainer.cuda.to_cpu(o.data[0])

        eos_score = o[self.EOS]
        self._edit_probs(o)

        inds = np.argpartition(o, len(o) - beam_width)
        inds = inds[::-1][:beam_width]
        return inds, o, state, eos_score

    def _next(self, q, beam_width, byte, desired_length):
        next_queue = FixedQueue(beam_width)

        updated = False
        seen = set()

        while not q.empty():
            cur_entry = q.get()
            score, seq, state = cur_entry

            is_end, is_rejected = self._judge(seq, byte, desired_length)
            if is_rejected:
                continue
            if is_end:
                next_queue.put(cur_entry)
                continue

            x = chainer.Variable(np.array([[seq[-1]]], np.int32),
                                 volatile=True)

            inds, probs, state, eos_score = self._calc_top_n(
                self.model, x, state, beam_width)

            for i in range(len(inds)):
                y = probs[inds[i]]

                tmp_state = state.copy()

                if hasattr(self.model, 'len_emb'):
                    lengths = tmp_state['lengths']

                    if byte:
                        consumed = len(self.yvocab.itos(inds[i])) + 1
                        lengths -= consumed
                    else:
                        lengths -= 1
                    flags = chainer.Variable(lengths.data >= 0, volatile=True)
                    lengths = F.where(flags, lengths, self.model.zeros)
                    tmp_state['lengths'] = lengths

                new_seq = seq + [inds[i]]
                entry = (score + y, new_seq, tmp_state)

                if self.eos_replacing:
                    length = self._calc_length(new_seq, byte)
                    if length >= desired_length:
                        entry = (score + eos_score, seq + [self.EOS], tmp_state)

                seq_hash = hash(' '.join([str(wid) for wid in entry[1]]))
                if seq_hash in seen:
                    continue
                seen.add(seq_hash)

                next_queue.put(entry)
                updated = True

            if updated is False:
                next_queue.put(cur_entry)

        while len(next_queue) >= beam_width:
            next_queue.get()

        return updated, next_queue

    def generate(self, source, length, byte, allow_unk=False):

        beam_width = self.beam_width
        model = self.model
        state = model.encode(source, train=False)
        initial_state = model.prepare_decoding(state, length, train=False)

        x = initial_state['x']
        x0 = (chainer.cuda.to_cpu(x.data[0])[0])
        initial_state['x'] = x0

        q = FixedQueue(beam_width)
        q.put((0, [x0], initial_state))

        while True:
            updated, next_queue = self._next(q, beam_width, byte, length)
            q = next_queue
            if not updated:
                break

        beams = []
        while len(q) >= 2:
            score, seq, state = q.get()
            beams.append((score, seq))

        if len(q) == 0:
            return [], None, None, None
        score, seq, state = q.get()
        beams.append((score, seq))

        return seq, score, state, beams[::-1]


def get_agiga_example(source_vocab, data, index):
    data.reset()

    for i, (source, target) in enumerate(data):
        source = ' '.join(source)
        if i == index:
            break

    src = [source.strip()]
    src = ['<s> ' + s + ' </s>' for s in src]
    src = [source_vocab.convert(s.split()) for s in src]
    src = np.array(src, dtype=np.int32)
    return source, src


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=str)
    parser.add_argument('--beam_width', type=int, default=5)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--length', type=int, default=30)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    beam_width = args.beam_width

    d = builder.initiate(args.model_dir)
    m = d['model']
    src_vcb, trg_vcb = d['vocabularies']
    _, valid_data = d['datasets']

    source, source_variables = get_agiga_example(src_vcb, valid_data, args.index)

    generator = BeamSearchGenerator(m,
                                    src_vcb,
                                    trg_vcb,
                                    beam_width=beam_width)

    length = np.array([args.length], dtype=np.int32)
    ret = generator.generate(source_variables, length, byte=True)
    output, score, state, beams = ret
    out = trg_vcb.revert(output, with_tags=False)
    print('{} & {} & {} \\\\'.format(m.name, args.length, ' '.join(out)))
    print(' ', len(out), len(' '.join(out)))

    print('- others')
    for b in beams:
        sent = ' '.join(trg_vcb.revert(b[1], with_tags=False))
        print('  %0.2f' % b[0], len(sent), sent, sep=' & ')


if __name__ == '__main__':
    main()
