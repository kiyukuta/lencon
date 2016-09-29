
import builder
import beam_search as bs

INF = float('inf')


class FixLen(bs.BeamSearchGenerator):

    name = 'fixlen'

    def __init__(self, model, xvocab, yvocab, beam_width=10):
        super(FixLen, self).__init__(model, xvocab, yvocab, beam_width)
        self.eos_replacing = True

    def _judge(self, seq, byte, desired_length):
        EOS = self.yvocab.stoi('</s>')

        length = self._calc_length(seq, byte)
        if length >= desired_length:
            raise RuntimeError('Unexpected Arrival')
            return True, False

        if len(seq) > 1 and seq[-1] == EOS:
            return True, False
        return False, False

    def _edit_probs(self, probs):
        super(FixLen, self)._edit_probs(probs)
        EOS = self.yvocab.stoi('</s>')
        probs[EOS] = -INF


if __name__ == '__main__':
    args = bs.parse_args()
    beam_width = args.beam_width

    d = builder.initiate(args.model_dir)
    m = d['model']
    src_vcb, trg_vcb = d['vocabularies']
    source, source_variables = bs.get_agiga_example(src_vcb, args.index)

    generator = FixLen(m, src_vcb, trg_vcb, beam_width=beam_width)

    ret = generator.generate(source_variables, args.length, byte=True)
    output, score, state, beams = ret
    out = trg_vcb.revert(output, with_tags=False)
    print('{} & {} & {} \\\\'.format(m.name, args.length, ' '.join(out)))
    print(' ', len(out), len(' '.join(out)))

    print('- others')
    for b in beams:
        sent = ' '.join(trg_vcb.revert(b[1], with_tags=False))
        print('  %0.2f' % b[0], len(sent), sent, sep=' & ')
