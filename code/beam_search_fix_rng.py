
import argparse

import builder
import beam_search as bs


class FixRng(bs.BeamSearchGenerator):

    name = 'fixrng'

    def __init__(self, model, xvocab, yvocab, beam_width=10):
        super(FixRng, self).__init__(model, xvocab, yvocab, beam_width)
        self.eos_replacing = True

    def _judge(self, seq, byte, max_length):

        EOS = self.yvocab.stoi('</s>')
        length = self._calc_length(seq, byte)

        if len(seq) > 1 and seq[-1] == EOS:
            if length < self.min_len:
                return None, True
            return True, False

        if length > max_length:
            raise RuntimeError('Unexpected Arrival')
            return None, True
        return False, False

    def generate(self, source, length_range, byte, allow_unk=False):
        assert type(length_range) == tuple
        assert len(length_range) == 2
        assert length_range[0] <= length_range[1]

        self.min_len, max_len = length_range

        return super(FixRng, self).generate(source, max_len, byte, allow_unk=allow_unk)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=str)
    parser.add_argument('--beam_width', type=int, default=5)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--length', type=int, default=30)
    parser.add_argument('--min_length', type=int, default=25)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    beam_width = args.beam_width

    d = builder.initiate(args.model_dir)
    m = d['model']
    src_vcb, trg_vcb = d['vocabularies']
    source, source_variables = bs.get_agiga_example(src_vcb, args.index)

    generator = FixRng(m, src_vcb, trg_vcb, beam_width=beam_width)

    length_range = (args.length - 5, args.length)
    ret = generator.generate(source_variables, length_range, byte=True)
    output, score, state, beams = ret
    out = trg_vcb.revert(output, with_tags=False)
    print('{} & {} & {} \\\\'.format(m.name, args.length, ' '.join(out)))
    print(' ', len(out), len(' '.join(out)))

    print('- others')
    for b in beams:
        sent = ' '.join(trg_vcb.revert(b[1], with_tags=False))
        print('  %0.2f' % b[0], len(sent), sent, sep=' & ')
