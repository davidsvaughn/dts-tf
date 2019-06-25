from __future__ import print_function

import os
import codecs
import numpy as np
import re
from future_builtins import map  # Only on Python 2
from collections import Counter
from itertools import chain
from autocorrect import spell

import utils as U
# from nlp.util import utils as U


def word_counts(filename, min_freq=1):
    with open(filename) as f:
        d = Counter(chain.from_iterable(map(str.split, f)))
    if min_freq>1:
        for k in list(d):
            if d[k] < min_freq:
                del d[k]
    return d

def word_counts_2(filename, min_freq=1):
    tokenize = lambda s: Vocab.tokenize(s, lower=True, flat=True)
    
    with open(filename) as f:
        d = Counter(chain.from_iterable(map(tokenize, f)))
    if min_freq>1:
        for k in list(d):
            if d[k] < min_freq:
                del d[k]
    return d

def check_header(filename):
    with open(filename) as f:
        line = f.readline().rstrip().split()
        if len(line)!=2:
            return False
        try:
            float(line[0])
            float(line[1])
            return True
        except ValueError:
            return False
        return True

def strip_non_ascii(string):
    ''' Returns the string without non ASCII characters'''
    stripped = (c for c in string if 0 < ord(c) < 127)
    return str(''.join(stripped))

def get_field(line, col=0):
    try:
        s = line.split(None, 1)[col]
    except:
        s = strip_non_ascii(line.split(None, 1)[col])
    return s

def clean_tags(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def corr(s):
    return re.sub(r'\.(?! )', ' . ', re.sub(r' +', ' ', s))
  
def load_word_list(file):
    has_header = check_header(file)
    words = []
    with codecs.open(file, "r", "utf-8") as f:
        for line in f:
            if has_header:
                has_header=False
                continue
            word = line.split(None, 1)[0]
            words.append(strip_non_ascii(word))
    return words

def load_embeddings(file, filter_words=None, word_counts=None, verbose=True, spell_corr=False):
    word2emb = {}
    word_fix = None
    has_header = check_header(file)
    with codecs.open(file, "r", "utf-8") as f:
        try:
            for line in f:
                if has_header:
                    has_header=False
                    continue
                word = get_field(line)
                if filter_words:
                    if not word in filter_words:
                        continue
                    filter_words.discard(word)
    
                v = line.split()[1:]
                try:
                    word2emb[word] = np.array(v, dtype='float32')
                except ValueError:
                    pass
        except ValueError:
            pass
    
   
    ####### spell correction  ###########################################################
    if spell_corr:
        print('\nSpell correcting {} words...'.format(len(filter_words)))
        word_fix = {}
        for word in filter_words:
            w2 = spell(word)
            if w2!=word:
                word_fix[word] = w2
        new_words = set(word_fix.values()) - set(word2emb.keys())
        print('Done.')
        
        ## get new word embeddings
        has_header = check_header(file)
        with codecs.open(file, "r", "utf-8") as f:
            try:
                for line in f:
                    if has_header:
                        has_header=False
                        continue
                    word = get_field(line)
                    if not word in new_words:
                        continue
        
                    v = line.split()[1:]
                    try:
                        word2emb[word] = np.array(v, dtype='float32')
                    except ValueError:
                        pass
            except ValueError:
                pass
        
        ## prune lookup table
        w2e_keys = set(word2emb.keys())
        wf_keys = word_fix.keys()
        for k in wf_keys:
            if not word_fix[k] in w2e_keys:
                del word_fix[k]
        
        if verbose:
            print('{}-->{} spell corrections'.format(len(word_fix), len(set(word_fix.values()))))
        
        ################################################################
        ## random embeddings for other (frequent) words
        filter_words = filter_words - set(word_fix.keys())
        filter_words = filter_words - w2e_keys
        rand_words = []
        for word in filter_words:
            if word_counts[word] > 50:
                rand_words.append(word)
    #     print('{} filter words...'.format(len(filter_words)))
    #     print('{} rand words...'.format(len(rand_words)))
        
        d = len(word2emb.itervalues().next())
        f = 0.1 * 1./float(d)
        for word in rand_words:
    #         word2emb[word] = f * np.random.rand(d)
            word2emb[word] = f * np.random.randn(d)
    
        if verbose:
            print('{} random embeddings'.format(len(rand_words)))

    ################################################################
    
    if verbose:
        print('TOTAL: {} word embeddings of dim {}'.format(len(word2emb), word2emb[next(iter(word2emb))].size))
        
    return word2emb, word_fix

class Vocab:
    def __init__(self, 
                 token2index=None, 
                 index2token=None,
                 unk_index=0):
        self._token2index = token2index or {}
        self._index2token = index2token or []
        self.unk_index = unk_index
        self.spell = None
        self.reset_counts()
        
    def reset_counts(self):
        self._unk = 0
        self._tot = 0
        
    @property
    def unk2tot(self):
        return float(self._unk)/float(self._tot)

    def feed(self, token):
        if token not in self._token2index:
            # allocate new index for this token
            index = len(self._token2index)
            self._token2index[token] = index
            self._index2token.append(token)

        return self._token2index[token]
    
    def map(self, t1, t2):
        if (t1 not in self._token2index) and (t2 in self._token2index):
            idx = self._token2index.get(t2)
            self._token2index[t1] = idx
            print('MAPPING: {}-->{}'.format(t1,t2))

    @property
    def size(self):
        return len(self._index2token)

    def token(self, index):
        return self._index2token[index]

    def __getitem__(self, token):
        index = self.get(token)
        if index is None:
            raise KeyError(token)
        return index

    ''' returns index 0 for unknown tokens! '''
    def get(self, token):
        if self.spell and token in self.spell:
            token = self.spell[token]
        
        idx = self._token2index.get(token, self.unk_index)
        self._tot += 1
        self._unk += (1 if idx==self.unk_index else 0)
        
        ''' uncomment to test! '''
        #print('{2}\t{3}\t{4:0.1f}\t{0}\t{1}'.format(idx, token, self._tot, self._unk, self.unk2tot*100.))
        
        return idx
    
    def get_index(self, token):
        return self.get(token)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self._token2index, self._index2token), f, pickle.HIGHEST_PROTOCOL)
            
    def get_tok_array(self, toks):
        tok_array = []
        for tok in '{' + toks + '}':
            t = self.get(tok)
            if t>0:
                tok_array.append(t)
        return tok_array

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            token2index, index2token = pickle.load(f)
        return cls(token2index, index2token)
    
    @staticmethod
    def get_char_aray(word, char_vocab, word_vocab=None, all_chars=True):
        if all_chars:
            char_array = char_vocab.get_tok_array(word)
        else:
            char_array = char_vocab.get_tok_array(word_vocab.token(word_vocab.get(word)))
        return char_array
    
    @staticmethod
    def clean(word, max_word_length=None, eos='+', lower=False):
        word = word.strip().replace('}', '').replace('{', '').replace('|', '')
        if lower:
            word = word.lower()
        if eos:
            word = word.replace(eos, '')
        if max_word_length and len(word) > max_word_length - 2:  # space for 'start' and 'end' chars
            word = word[:max_word_length-2]
        return word
    
    @staticmethod
    def clean_line(line, tags=True, cor=True):
        if tags: line = clean_tags(line)
        if cor: line = corr(line)
        return line
    
    ## CLEAN LINE ##
    @staticmethod
    def tokenize(line, lower=True, flat=False, clean=True):
        if clean: line = Vocab.clean_line(line)
        toks = U.tokenize(line, lower=lower, flat=flat)
        return toks  

    @staticmethod
    def load_vocab(vocab_file, max_word_length=60, eos='+'):
        char_vocab = Vocab()
        char_vocab.feed(' ')  # blank is at index 0 in char vocab
        char_vocab.feed('{')  # start is at index 1 in char vocab
        char_vocab.feed('}')  # end   is at index 2 in char vocab
        char_vocab.feed('|')
        if eos:
            char_vocab.feed(eos)
            
        word_vocab = Vocab()
        word_vocab.feed('|')  # <unk> is at index 0 in word vocab
        if eos:
            word_vocab.feed(eos)
    
        actual_max_word_length = 0
        has_header = check_header(vocab_file)
        with codecs.open(vocab_file, "r", "utf-8") as f:
            for line in f:
                if has_header:
                    has_header=False
                    continue
                word = line.split(None, 1)[0]
                word = Vocab.clean(word, max_word_length, eos)
                word_vocab.feed(word)
                
                for c in word:
                    char_vocab.feed(c)
                    
                actual_max_word_length = max(actual_max_word_length, len(word)+2)
        
        assert actual_max_word_length <= max_word_length
        
        char_vocab.map('[', '(')
        char_vocab.map(']', ')')
        for i in [48,52,53,54,55,56,57]:
            char_vocab.map(chr(i), '1')
        for i in range(65, 91):
            char_vocab.map(chr(i), chr(i+32))
        for i in range(32, 127):
            char_vocab.map(chr(i), '?')
        
        print()
        print('actual longest token length is:', actual_max_word_length)
        print('size of word vocabulary:', word_vocab.size)
        print('size of char vocabulary:', char_vocab.size)
        return word_vocab, char_vocab, actual_max_word_length

    @staticmethod
    def load_word_embeddings_ORIG(emb_path, emb_dim, data_file, min_freq=1, unk='<unk>', eos='+', verbose=True, spell_corr=False):
        #wc = word_counts(data_file, min_freq)
        wc = word_counts_2(data_file, min_freq)
        words = set(wc)
        words.discard(unk)
        
        emb_file = emb_path.format(emb_dim)
        word2emb = load_embeddings(emb_file, filter_words=words, word_counts=wc, verbose=verbose, spell_corr=spell_corr)
        #words = list(word2emb)
        
        word_vocab = Vocab()
        if isinstance(word2emb, tuple):
            word_vocab.spell = word2emb[1]
            word2emb = word2emb[0]
            
        word_vocab.feed(unk)    # <unk> is at index 0 in word vocab --> so idx=0 returned for unknown toks
        if eos: word_vocab.feed(eos)
        
#         word_vocab = Vocab(unk_index=1)
#         word_vocab.feed('<pad>')    # <pad> is at index 0 in word vocab
#         word_vocab.feed(unk)    # <unk> is at index 1 in word vocab --> so idx=1 returned for unknown toks

        #z = np.zeros_like(word2emb[next(iter(word2emb))])
        n = len(word2emb) + word_vocab.size
        d = word2emb[next(iter(word2emb))].size
        E = np.zeros([n, d], dtype=np.float32)# <unk> is given all-zero embedding... at E[0,:]
        
        for word in list(word2emb):
            idx = word_vocab.feed(word)
            #words.discard(word)
            try:
                E[idx,:] = word2emb[word]
            except ValueError:
                print('EMBEDDING ERROR....')
                print(word)
        
        
        # returns embedding matrix, word_vocab
        return E, word_vocab
    
    @staticmethod
    def load_word_embeddings_NEW(emb_path, emb_dim, data_file, min_freq=1, unk='<unk>', eos='+', verbose=True):
        ## pre-load emb words
        from deepats import ets_reader
        from deepats.w2vEmbReader import W2VEmbReader as EmbReader
        
        emb_reader = EmbReader(emb_path, emb_dim)
        emb_words = emb_reader.load_words()
        
        text = U.read_col(data_file, col=-1, type='string')
        vocab = ets_reader.create_vocab(text, tokenize_text=True, to_lower=True, min_word_freq=min_freq, emb_words=emb_words)
        
        #######################################################
        words = set(vocab)
        words.discard(unk)
        
        emb_file = emb_path.format(emb_dim)
        word2emb = load_embeddings(emb_file, filter_words=words, verbose=verbose)
        
        n = len(word2emb) + 3
        d = word2emb[next(iter(word2emb))].size
        E = np.zeros([n, d], dtype=np.float32)# <unk> is given all-zero embedding... at E[0,:]
        
        word_vocab = Vocab()
        word_vocab.feed(unk)
        if eos: word_vocab.feed(eos)
        
        for word in list(word2emb):
            idx = word_vocab.feed(word)
            E[idx,:] = word2emb[word]
            #print(word)
        
        return E, word_vocab
    
    @staticmethod
    def load_word_embeddings(emb_path, emb_dim, data_file, min_freq=1, verbose=True):
        ## pre-load emb words
        from deepats import ets_reader
        from deepats.w2vEmbReader import W2VEmbReader as EmbReader
        
        emb_reader = EmbReader(emb_path, emb_dim)
        emb_words = emb_reader.load_words()
        
        text = U.read_col(data_file, col=-1, type='string')
        vocab = ets_reader.create_vocab(text, tokenize_text=True, to_lower=True, min_word_freq=min_freq, emb_words=emb_words)
        #  vocab = {'<pad>':0, '<unk>':1, '<num>':2, .....}
        
        #######################################################
        pad='<pad>';unk='<unk>';num='<num>'
        words = set(vocab)
        words.discard(pad);words.discard(unk);words.discard(num)
        
        emb_file = emb_path.format(emb_dim)
        word2emb = load_embeddings(emb_file, filter_words=words, verbose=verbose)
        
        n = len(word2emb) + 3
        d = word2emb[next(iter(word2emb))].size
        E = np.zeros([n, d], dtype=np.float32)# <unk> is given all-zero embedding... at E[0,:]
        
        word_vocab = Vocab(unk_index=1)
        word_vocab.feed(pad)    # <pad> is at index 0 in word vocab
        word_vocab.feed(unk)    # <unk> is at index 1 in word vocab --> so idx=1 returned for unknown toks
        word_vocab.feed(num)    # <num> is at index 2 in word vocab
        for word in list(word2emb):
            idx = word_vocab.feed(word)
            E[idx,:] = word2emb[word]
            #print(word)
        
        return E, word_vocab
            
def test1():
    data_dir = '/home/david/data/ets1b/2016'
    vocab_file = os.path.join(data_dir, 'vocab_n250.txt')
    word_vocab, char_vocab, max_word_length = Vocab.load_vocab(vocab_file)
    
def test2():
    data_dir = '/home/david/data/embed'
    vocab_file = os.path.join(data_dir, 'glove.6B.50d.txt')
    word_vocab, char_vocab, max_word_length = Vocab.load_vocab(vocab_file)
        
def test3():
    data_dir = '/home/david/data/ets1b/2016'
    #vocab_file = os.path.join(data_dir, 'vocab_n250.txt')
    #word_vocab, char_vocab, max_word_length = Vocab.load_vocab(vocab_file)
    id = 63986; essay_file = os.path.join(data_dir, '{0}', '{0}.txt.clean.tok').format(id)
    d = word_counts(essay_file, 2)
    return d

def test4():
    emb_path = '/home/david/data/embed/glove.6B.{}d.txt'
    emb_dim = 100
    
#     data_dir = '/home/david/data/ets1b/2016'
#     id = 63986; data_file = os.path.join(data_dir, '{0}', '{0}.txt.clean.tok').format(id)
    
    data_dir = '/home/david/data/ats/ets'
    id = 61190; essay_file = os.path.join(data_dir, '{0}', 'text.txt').format(id)
    
    #words = set(word_counts(data_file, min_freq=1))
    #word2emb = load_embeddings(emb_file, filter_words=words)
    E, word_vocab = Vocab.load_word_embeddings(emb_path, emb_dim, essay_file, min_freq=2)
    print(E.shape)
    
if __name__ == '__main__':
    #test1()
    #test2()
    #d = test3()
    test4()
    print('done')    