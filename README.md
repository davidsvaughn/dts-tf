# dts-tf
Deep Text Scoring in TensorFlow

- this code implements either a flat or a 2 level hierarchical RNN model (see paper below) 
- can operate either on _character_ sequences or _word_ sequences... pretrained embeddings inluded for both
- everything is controlled through configuration (*.conf) file, which is set at top of train.py: i.e. `config_file = 'config/han.conf'`

## Get data from spshare
- copy files from: `//spshare/users/dvaughn/dts-tf/data` to  `./data`
- copy files from: `//spshare/users/dvaughn/dts-tf/embeddings` to `./embeddings`


## Run
```
cd python
python -u train.py | tee log.txt
```

## References

### Code

- [BERT: PyTorch Conversion](https://github.com/huggingface/pytorch-pretrained-BERT) ....WordPieces implemened here
- [BERT: Original TensorFlow Code](https://github.com/google-research/bert)

### Papers

- [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/abs/1609.08144)