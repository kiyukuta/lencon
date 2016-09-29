
class GreedyGenerator:

    def __init__(self, model, xvocab, yvocab, beam_width=10):
        self.model = model
        self.xvocab = xvocab

    def generate(self, source, length, byte, allow_unk=False):
        seq, score = self.model(source, None, length, train=False)
        return seq[0], score, None, None
