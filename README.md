# Controlling Output Length in Neural Encoder-Decoders

Implementation of "Controlling Output Length in Neural Encoder-Decoders".

The purpose of this project is to give the capability of controlling output length to neural encoder-decoder models.
It is crucial for applications such as text summarization, in which we have to generate concise summaries with a desired length.

There are three models (standard encdec, leninit, and lenemb) and
 four decoding methods (greedy, standard beamsearch, fixlen beamsearch, and fixrng
beamsearch).

Although any combination is possible in prediction phase, please see `run_duc04.sh` for the combination we used in the paper.

## Requirements

- [Chainer](https://github.com/pfnet/chainer)
- python 3.5.1+

## Data preparation

We followed [NAMAS](https://github.com/facebook/NAMAS)
which is an implementation of [\[Rush+15\]](http://www.aclweb.org/anthology/D15-1044)

Please see https://github.com/facebook/NAMAS#constructing-the-data-set for detail.

### Preparing your own dataset

You can modify the config files to run experiments in your own dataset.  
For example in `lencon/models/XXX/config.ini`, you can see the following lines:
```
[Dataset]
base_dir =  /path/to/agiga_work/
src_vcb = %(base_dir)s/train.article.dict
trg_vcb = %(base_dir)s/train.title.dict
train_src_file = %(base_dir)s/train.article.txt
train_trg_file = %(base_dir)s/train.title.txt
valid_src_file = %(base_dir)s/valid.article.filter.txt
valid_trg_file = %(base_dir)s/valid.title.filter.txt
```
Please replace each path to corresponding files of your dataset.


#### Vocabulary list: `src_vcb` and `trg_vcb`
Each file has one word per line.
```txt
<unk> 10000.0
<s> 10000.0
</s> 10000.0
the 1230
, 1120
. 1110
, 980
a 970
...
```

The first three words indicate the special tags for `unknown word`, `begin of sentence`, and `end of sentence`.
The second column indicates the frequency of the word on the first column. 
Note that we don't use this frequencies (second column) and you can remove it.


#### Source and Target file: `train_src_file`, `train_trg_file`, `valid_src_file` and `valid_trg_file`
Each file has one title or article per line.
N'th line of `train_trg_file` is a title of n'th line of `train_src_file` (corresponding article).

See also https://github.com/facebookarchive/NAMAS#format-of-the-data-files
- `#.#` in the above link indicates that every number is replaced to `#`




## Training

```zsh
# for fixlen and fixrng
python train.py ../model/encdec --gpu X 
# for lenemb
python train.py ../model/lenemb --gpu X 
# for leninit
python train.py ../model/leninit --gpu X 
```

## Predicting

```zsh
python predict.py ../models/[Model] [input text file] [length you need] \
-d [Decoding method] \
--min_length [minumum length]  # if and only if -d == fixrng
```


## Reference 

+ Yuta Kikuchi, Graham Neubig, Ryohei Sasano, Hiroya Takamura and Manabu Okumura  
Controlling Output Length in Neural Encoder-Decoders  
[EMNLP 2016](http://aclweb.org/anthology/D/D16/D16-1140.pdf), [bibtex](http://aclweb.org/anthology/D/D16/D16-1140.bib), ([arXiv](https://arxiv.org/abs/1609.09552))
