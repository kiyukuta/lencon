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
EMNLP 2016 (to appear)  ([arXiv](https://arxiv.org/abs/1609.09552))
