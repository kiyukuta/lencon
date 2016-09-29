
import argparse
import json
import numpy as np
import sys

import beam_search
import beam_search_fix_len
import beam_search_fix_rng
import greedy_generator
import builder


def get_decoder(decoding_method, model, src_vcb, trg_vcb, beam_width):
    if decoding_method == 'bs_normal':
        gen = beam_search.BeamSearchGenerator(
            model, src_vcb, trg_vcb, beam_width=beam_width)
    elif decoding_method == 'bs_fixlen':
        gen = beam_search_fix_len.FixLen(
            model, src_vcb, trg_vcb, beam_width=beam_width)
    elif decoding_method == 'bs_fixrng':
        gen = beam_search_fix_rng.FixRng(
            model, src_vcb, trg_vcb, beam_width=beam_width)
    elif decoding_method == 'greedy':
        gen = greedy_generator.GreedyGenerator(model, src_vcb, trg_vcb)
    return gen


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=str)
    parser.add_argument('test_file', type=str)
    parser.add_argument('length', type=int)
    parser.add_argument('--decoding_method', '-d', type=str, 
        choices=['greedy', 'bs_normal', 'bs_fixlen', 'bs_fixrng'],
        default='bs_normal')
    parser.add_argument('--min_length', type=int, 
        help='set if and only if decoding_method is bs_fixrng')
    parser.add_argument('--beam', '-b', type=int, default=10)
    parser.add_argument('--no_truncate', action='store_true')
    parser.add_argument('--log', type=str)
    args = parser.parse_args()
    return args


def predict(raw_sentence, generator, length):
    words = raw_sentence.split()
    if words[0] != '<s>':
        words = ['<s>'] + words
    if words[-1] != '</s>':
        words = words + ['</s>']

    word_ids = [generator.xvocab.convert(words)]
    x = np.array(word_ids, dtype=np.int32)

    out, _, _, _ = generator.generate(x, length, byte=True)

    sent = ' '.join(generator.yvocab.revert(out, with_tags=False)).strip()
    return sent

if __name__ == '__main__':
    args = parse_args()

    d = builder.initiate(args.model_dir)

    xvocab, yvocab = d['vocabularies']
    m = d['model']

    gen = get_decoder(args.decoding_method, m, xvocab, yvocab, args.beam)

    l = np.array([args.length], dtype=np.int32)

    #assert (args.decoding_method == 'bs_fixrng') == (args.min_length is not None)
    min_length = args.min_length
    if args.decoding_method == 'bs_fixrng':
        if min_length is None:
            min_length = args.length - 5
            print('min_length is set to {}.'.format(min_length), file=sys.stderr)

        l = (min_length, l)

    truncated = []
    with open(args.test_file) as fin:
        for line in fin:
            sent = predict(line.strip(), gen, l)
            org_length = len(sent)
            if args.no_truncate == False:
                sent = sent[:args.length]
            print(sent)
            truncated.append(max(0, org_length - len(sent)))

    ave = sum(truncated) / len(truncated)
    num = sum([1 if t > 0 else 0 for t in truncated])

    out_dict = args.__dict__
    out_dict['ave_truncated'] = '%0.3f' % ave
    out_dict['truncated_num'] = str(num)
    
    if args.log:
        with open('{}.json'.format(args.log), 'w') as f:
            json.dump(out_dict, f, indent=2, ensure_ascii=False)
