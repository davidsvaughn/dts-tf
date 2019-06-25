
from __future__ import print_function
from __future__ import division

import os, sys
import codecs
import numpy as np
import pandas as pd
import random
import glob
import re
import pickle
import time
from tensorflow.python.util import nest

from vocab import Vocab
import utils as U

# from nlp.util.utils import adict, get_seed
# from nlp.util import utils as U

#REGEX_NUM = r'^[0-9]*\t[0-9]\t[0-9]\t[0-9]\t(?!\s*$).+'
#REGEX_MODE = r'^[0-9]*\tm\tm\tm\t(?!\s*$).+'

# REGEX_NUM = r'^[0-9b_]*\t([0-9]\t)+(?!\s*$).+'
# REGEX_MODE = r'^[0-9b]*\t(m\t)+(?!\s*$).+'

REGEX_NUM = r'^[0-9a-z_]*\t([0-9]\t)+(?!\s*$).+'

## for (possibly) nested lists
def isListEmpty(inList):
    if isinstance(inList, list): # Is a list
        return all( map(isListEmpty, inList) )
    return False # Not a list

'''
READ FILE IN CHUNKS
- reads chunk by chunk
- yields line by line
- filter by regex
- shuffles each chunk
'''
class ChunkReader(object):
    def __init__(self, file_name, chunk_size=1000, shuf=True, regex=None, seed=None, bucket=None, verbose=True):
        self.file_name = file_name
        self.chunk_size = chunk_size
        if chunk_size==None or chunk_size<=0: # read entire file as one chunk
            self.chunk_size=np.iinfo(np.int32).max
        self.shuf = shuf
        self.bucket = bucket
        self.verbose = verbose
        self.regex = regex
        if regex:
            self.regex = re.compile(regex)
        if seed==None or seed<=0:
            self.seed = U.get_seed()
        else:
            self.seed = seed
        self.rng = np.random.RandomState(self.seed + U.string2rand('ChunkReader'))
    
    def next_chunk(self, file_stream):
        lines = []
        try:
            for i in xrange(self.chunk_size):
                lines.append(next(file_stream).strip())
        except StopIteration:
            self.eof = True
            
        if self.regex:
            lines = filter(self.regex.search, lines)
            
        if self.bucket!=None:
            lens = [len(line) for line in lines]
            m = float(max(lens))
            lens = [float(ell)/m for ell in lens]
            lens = [ell + self.bucket * self.rng.rand() for ell in lens]
            pp = np.argsort(lens)
            lines = [lines[p] for p in pp]
            
        elif self.shuf:
            self.rng.shuffle(lines)
            
        return lines
        
    def chunk_stream(self, stop=True):
        while True:
            self.eof = False
            if self.shuf:
                if self.verbose:
                    print('\tSHUFFLING...\t', self.file_name)
                os.system('shuf {0} | sponge {0}'.format(self.file_name))
            if self.verbose:
                print('\tREADING...\t', self.file_name)
            with codecs.open(self.file_name, "r", "utf-8") as f:
                while not self.eof:
                    yield self.next_chunk(f)
            if stop:
                break
    
    def line_stream(self, stop=True):
        for chunk in self.chunk_stream(stop=stop):
            for line in chunk:
                yield line
                
    def sample(self, sample_every=100, stop=True):
        i=0
        for line in self.line_stream(stop=stop):
            i=i+1
            if i % 100 == 0:
                print('{} | {}'.format(i,line))

class GlobReader(object):
    def __init__(self, file_pattern, chunk_size=1000, shuf=True, regex=None, seed=None, bucket=None, verbose=True):
        self.file_pattern = file_pattern
        self.file_names = glob.glob(self.file_pattern)
        self.file_names.sort()
        self.chunk_size = chunk_size
        self.shuf = shuf
        self.bucket = bucket
        self.verbose = verbose
        self.regex = regex
        if seed==None or seed<=0:
            self.seed = U.get_seed()
        else:
            self.seed = seed
        self.rng = np.random.RandomState(self.seed + U.string2rand('GlobReader'))
        self.num_files = None
        self.bpf = None
        self.prev_file = ''
        
    def new_file(self):
        new = (self.cur_file != self.prev_file) and (len(self.prev_file)>0)
        self.prev_file = self.cur_file
        return new
        
    def file_stream(self):
        if self.shuf:
            self.rng.shuffle(self.file_names)
        if self.num_files is None:
            self.num_files = len(self.file_names)
        for file_name in self.file_names:
            yield file_name
    
    ''' reads files in sequence (NOT parallel) '''
    def chunk_stream(self, stop=True):
        while True:
            for file in self.file_stream():
                self.cur_file = file
                chunk_reader =  ChunkReader(file, 
                                            chunk_size=self.chunk_size, 
                                            shuf=self.shuf,
                                            bucket=self.bucket,
                                            verbose=self.verbose,
                                            regex=self.regex, 
                                            seed=self.seed)
                for chunk in chunk_reader.chunk_stream(stop=True):
                    yield chunk
            if stop:
                break
    
    def line_stream(self, stop=True):
        for chunk in self.chunk_stream(stop=stop):
            for line in chunk:
                yield line

num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')
def is_number(token):
    return bool(num_regex.match(token))

'''
returns dictionary: words->[word indices]
                    chars->[char indices]
set words=None, chars=None if not desired
'''

#punc = set(['-','--',',',':','.','...','\'','(',')','&','#','$'])
#punc = set(['-','--',':','...','\'','(',')','&','#','$'])
punc = set(['-','--',':','...','\'','&','#','$'])

from collections import Iterable
def flatten(coll):
    for i in coll:
        if isinstance(i, Iterable) and not isinstance(i, basestring):
            for subc in flatten(i):
                yield subc
        else:
            yield i

class TextParser(object):
    def __init__(self, word_vocab=None, char_vocab=None, max_word_length=None, reader=None, words='w', chars='c', text='t', eos='+', sep=' ', tokenize=True, keep_unk=True, FLAGS=None):
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.max_word_length = max_word_length
        self.reader = reader
        self.words = words
        self.chars = chars
        self.text = text
        self.ws = 'ws'
        self.cs = 'cs'
        #self.eos = eos
        self.eos = None
        self.sep = sep
        self._tokenize = tokenize
        self.keep_unk = keep_unk
        if tokenize: self._tokenize = U.tokenize
        self.FLAGS = FLAGS
    
    def tokenize(self, str):
        if self._tokenize:
            toks = self._tokenize(str)
        else:
            toks = str.split()
            
        ## filter out empty elements
        #toks = [t for t in toks if len(t)>0]

        return toks
    
    ''' parses line into word/char tokens, based on vocab(s) '''
    def _parse_line(self, line, word_tokens, char_tokens):
        ws, cs, text = [0],[0], []
        
        if self.char_vocab is None and self.word_vocab is None:
            text = line
            return word_tokens, char_tokens, ws, cs, text
        
        lower=(self.char_vocab==None)
        
        sentences = Vocab.tokenize(line, lower=lower, flat=False, clean=not self.FLAGS.no_clean)
        
        text = ' '.join(flatten(sentences)).strip()
        
        for toks in sentences:
            wi,ci = 0,0
            for word in toks:
                word = Vocab.clean(word, self.max_word_length, lower=lower, eos=self.eos)
                
                if self.char_vocab is None:
                    if word in punc:
                        continue
    #                 if is_number(word):
    #                     word='1'
                
                if self.word_vocab:
                    word_idx = self.word_vocab.get(word)
                    if word_idx>self.word_vocab.unk_index or self.keep_unk:
                        word_tokens.append(word_idx); wi+=1
                        #text.append(word)
                
                if self.char_vocab:
                    char_array = Vocab.get_char_aray(word, self.char_vocab, self.word_vocab)
                    char_tokens.append(char_array); ci+=1
                    
#                 if self.char_vocab is None and self.word_vocab is None:
#                     text.append(word)
                    
            if self.eos:
                if self.word_vocab: 
                    word_tokens.append(self.word_vocab.get(self.eos)); wi+=1
                    #text.append(self.eos)
                if self.char_vocab: 
                    char_tokens.append(self.char_vocab.get_tok_array(self.eos)); ci+=1
            
            if wi>0 or (self.char_vocab and ci>0):
                ws.append(len(word_tokens))
                cs.append(len(char_tokens))
        
        #ww = U.lindexsplit(word_tokens, ws)
        #cc = U.lindexsplit(char_tokens, cs)
            
        return word_tokens, char_tokens, ws, cs, text
    
    def package(self, word_tokens, char_tokens, ws, cs, text ):
        return U.adict( { self.words:word_tokens , self.chars:char_tokens, self.ws:ws , self.cs:cs, self.text:text } )
    
    def parse_line(self, line):
        return self.package(*self._parse_line(line, word_tokens=[], char_tokens=[]))
    
    def chunk_stream(self, reader=None, stop=True):
        if reader==None:
            reader=self.reader
        if reader!=None:
            for chunk in reader.chunk_stream(stop=stop):
                word_tokens, char_tokens = [], []
                for line in chunk:
                    self._parse_line(line, word_tokens, char_tokens)
                yield self.package(word_tokens, char_tokens)
    
    def line_stream(self, reader=None, stop=True):
        if reader==None:
            reader=self.reader
        if reader!=None:
            for line in reader.line_stream(stop=stop):
                d = self.parse_line(line)
                # SKIP ?
                if len(d.t)==0:
                    d = self.parse_line('e')
                    print('EMPTY!!!')
                yield d
    
    def new_file(self):
        return self.reader.new_file()
    
    @property
    def num_files(self):
        return self.reader.num_files
             
    def sample(self, sample_every=100, reader=None, stop=True):
        i=0
        for d in self.line_stream(reader=reader, stop=stop):
            i=i+1
            if i % 100 == 0:
                print('{} | {}'.format(i, d.w))
                
    #def tensor2list(self, T=None):
        

def sample_table(c, min_cut=0.5, r=0.98):
    n = float(c.sum())
    t = float(c.min())/c
    mc = abs(min_cut)
    
    if sum(t*c)/n < mc:
        while sum(t*c)/n < mc:
            t=1.-r*(1.-t)
        if min_cut<0:
            n = float(sum(t*c))
            while sum(t*c)/n > (1.0-mc):
                t=t*r
    else:
        while sum(t*c)/n > mc:
                t=t*r
    return t

def sample_dict(v, c, min_cut=0.5, r=0.95):
    t = sample_table(c, min_cut=min_cut, r=r)
    return dict(zip(v, t))
    
class FieldParser(object):
    def __init__(self, fields, reader=None, sep='\t', seed=None):
        self.fields = fields
        self.reader = reader# GlobReader!!
        self.sep = sep
        self.seed = seed
        if seed==None or seed<=0:
            self.seed = U.get_seed()
        else:
            self.seed = seed
        self.rng = np.random.RandomState(self.seed + U.string2rand('FieldParser'))
    
    def parse_line(self, line, t=None, keys=None):
        rec = line.strip().split(self.sep)
        d = {}
        for k,v in self.fields.items():
            if keys and v not in keys:
                continue
            if isinstance(v, basestring):
                d[v] = rec[k].strip()
            else:
                d.update(v.parse_line(rec[k].strip()))
            if t and v=='y':
                y = float(d[v])
                p = t[y]
                if self.rng.rand()>p:
                    #print('sample NO\t[{},{}]'.format(y,p))
                    return None
                #print('sample YES\t[{},{}]'.format(y,p))
        return U.adict(d)
    
    def line_stream(self, reader=None, stop=True, t=None, keys=None):
        if reader==None:
            reader=self.reader
        if reader!=None:
            for line in self.reader.line_stream(stop=stop):
                d = self.parse_line(line, t=t, keys=keys)
                if d: 
                    yield d
    
    def get_maxlen(self):
        n = 0
        for d in self.line_stream(stop=True):
            n = max(n,len(d.w))
        return n
    
    def compute_tstats(self, t):
        nw, ns = zip(*t)
        nw = np.array(nw, dtype=np.float32)
        ns = np.array(ns, dtype=np.float32)
        d = {}
        d['w'] = np.mean(nw)
        d['s'] = np.std(ns)
        return U.adict(d)
    
    def get_fields(self):
        y,t = 'y','t'
        data = {y:[], t:[]}
        
        for d in self.line_stream(stop=True):
            data[y].append(d[y])
            data[t].append((len(d.w), len(d.ws)))
            
        return data
    
    def get_stats(self):
        data = self.get_fields()
        ystats = self.compute_ystats(data['y'])
        tstats = self.compute_tstats(data['t'])
        return ystats, tstats
    
    def compute_ystats(self, y):
        y = np.array(y, dtype=np.float32)
        v, c = np.unique(y, return_counts=True)
        d = {}
        d['mean'] = np.mean(y)
        d['std'] = np.std(y)
        d['min'] = np.min(y)
        d['max'] = np.max(y)
        d['n'] = len(y)
        d['v'] = v
        d['c'] = c
        return U.adict(d)
    
    def get_keys(self, keys):
        data = {}
        for key in keys:
            data[key]=[]
        
        for d in self.line_stream(stop=True, keys=keys):
#         for d in self.line_stream(stop=True, keys=None):
            for key in keys:
                data[key].append(d[key])
            
        return data
    
    def get_field(self, key):
        x = []
        for d in self.line_stream(stop=True, keys=[key]):
            x.append(d[key])
        return x
    
    def get_ystats(self):
        #y = self.get_field(key='y')
        data = self.get_keys(['y','id'])
        y = data['y']
        id = data['id']
        ystats = self.compute_ystats(y)
        ystats.y = np.array(y, dtype=np.int32)
        ystats.id = np.array(id, dtype='unicode')
        return ystats
                       
    def sample(self, sample_every=100, reader=None, stop=True):
        i=0
        for d in self.line_stream(reader=reader, stop=stop):
            i=i+1
            if i % 100 == 0:
                print('{} | {}\t{}\t{}'.format(i, d.id, d.y, d.w))
                #print('{} | {}'.format(i, d.w))
        print('{} LINES'.format(i))
                
## reader=TextParser
class TextBatcher(object):
    def __init__(self, reader, batch_size, num_unroll_steps, batch_chunk=100, trim_chars=False):
        self.reader = reader
        self.batch_size = batch_size
        self.num_unroll_steps = num_unroll_steps
        self.max_word_length = reader.max_word_length# reader=TextParser
        self.batch_chunk = batch_chunk
        if batch_chunk==None or batch_chunk<=0:
            self.batch_chunk=np.iinfo(np.int32).max
        self.trim_chars = trim_chars
        self.wpb = self.batch_size * self.num_unroll_steps
    
    @property
    def bpf(self):
        return 3057
    @property
    def bps(self):
        return self.bpf
    
    def new_file(self):
        return self.reader.new_file()
    def new_shard(self):
        return self.new_file()
    
    @property
    def num_files(self):
        return self.reader.num_files
    
    def length(self):
        if not self.num_files is None and not self.bpf is None:
            return self.num_files*self.bpf
        return None
    
    def make_batches(self, tok_stream):
        word_toks, char_toks, N = [], [], 0
        for d in tok_stream:
            word_toks.extend(d.w)
            char_toks.extend(d.c)
            N = N + len(d.w)
            if N > self.batch_chunk * self.wpb:
                break
        
        word_tensor = np.array(word_toks, dtype=np.int32)
        char_tensor = np.zeros([len(char_toks), self.max_word_length], dtype=np.int32)
        for i, char_array in enumerate(char_toks):
            char_tensor [i,:len(char_array)] = char_array
        
        length = word_tensor.shape[0]
        assert char_tensor.shape[0] == length
        
        # round down length to whole number of slices
        reduced_length = (length // (self.batch_size * self.num_unroll_steps)) * self.batch_size * self.num_unroll_steps
        if reduced_length==0:
            return None
        
        word_tensor = word_tensor[:reduced_length]
        char_tensor = char_tensor[:reduced_length, :]
        
        ydata = np.zeros_like(word_tensor)
        ydata[:-1] = word_tensor[1:].copy()
        ydata[-1] = word_tensor[0].copy()

        x_batches = char_tensor.reshape([self.batch_size, -1, self.num_unroll_steps, self.max_word_length])
        y_batches = ydata.reshape([self.batch_size, -1, self.num_unroll_steps])

        x_batches = np.transpose(x_batches, axes=(1, 0, 2, 3))
        y_batches = np.transpose(y_batches, axes=(1, 0, 2))
        
        return list(x_batches), list(y_batches)
    
    ## trims zero-padding off 3rd (last) dimension (characters)
    def trim_batch(self, x):
        s = np.sum(np.sum(x,axis=1), axis=0)
        i = np.nonzero(s)[0][-1]+1
        return x[:,:,:i]
    
    ## x: char indices
    ## y: word indices
    def batch_stream(self, stop=False):
        tok_stream = self.reader.chunk_stream(stop=stop)
        
        while True:
            batches = self.make_batches(tok_stream)
            if batches is None:
                break
            for c, w in zip(batches[0], batches[1]):
                if self.trim_chars:
                    c = self.trim_batch(c)
                yield U.adict( { 'w':w , 'c':c } )

def nest_depth(x):
    #depth = lambda L: isinstance(L, list) and max(map(depth, L))+1
    #return depth(x)
    depths = []
    for item in x:
        if isinstance(item, list):
            depths.append(nest_depth(item))
    if len(depths) > 0:
        return 1 + max(depths)
    return 1

def _maxlens(x,d=0):
    n = len(x)
    ret = [(n,d)]
    if n>0 and nest.is_sequence(x[0]):
        ret.extend(map(lambda y: _maxlens(y,d+1), x))
    return nest.flatten(ret)

def maxlens(x):
    v = _maxlens(x)
    n = v[-1] + 1
    u = [[] for _ in range(n)]
    for i in range(int(len(v)/2)):
        u[v[2*i+1]].append(v[2*i])
    return map(max,u) 
        
def pad_sequences(sequences, trim_words=False, max_text_length=None, max_word_length=None, dtype='int32', wpad='post', cpad='post', value=0.):
    num_samples = len(sequences)
    seq_lengths = map(len, sequences)
    if trim_words:
        max_text_length = max(seq_lengths)
        #max_text_length+=100
    
    sample_shape = tuple()
    d = nest_depth(sequences)
    if d > 2:# <-- indicates char sequence
        if max_word_length is None:
            max_word_length = max(map(lambda x:max(map(len,x)), sequences))
        sample_shape = (max_word_length,)
        
    x = (np.ones((num_samples, max_text_length) + sample_shape) * value).astype(dtype)
    
    for i,s in enumerate(sequences):
        if d > 2:# <-- indicates char sequence
            y = (np.ones((max_text_length,) + sample_shape) * value).astype(dtype)
            k = (0 if wpad=='post' else max_text_length-len(s))
            for j,t in enumerate(s):
                if j>= max_text_length:
                    break
                if cpad == 'post':
                    y[j+k,:len(t)] = t
                else:
                    y[j+k,-len(t):] = t
            x[i,:] = y
        else:# <-- otherwise word sequence
            s = s[:max_text_length]
            if wpad == 'post':
                x[i,:len(s)] = s
            else:
                x[i,-len(s):] = s
    
#     if d<3 and wpad!='post':
#         seq_lengths = [max_text_length for i in seq_lengths]
        
    return x, seq_lengths


def print_ystats(y):
    #print('\nYSTATS (mean,std,min,max,#): {}\n'.format(y))
    print('\nSCORES:\t{}\nCOUNTS:\t{}\t= {}'.format(y.v, y.c, y.n))
    
def print_t(t, y):
    s = np.array(t.values(), np.float32)
    print('SAMPLE:\t{}'.format(s))
    c = y.c * s
    print('COUNTS:\t{}\t= {}'.format(c.astype(np.int32), int(c.sum())))

def print_tstats(t):
    print('#words:\t{0:0.1f}\n#sents:\t{1:0.1f}'.format(t.w, t.s))
    
## reader=FieldParser
class ResponseBatcher(object):
    def __init__(self, reader, FLAGS, ystats=None, verbose=True, normy=True, tstats=False, batch_size=None, pkl=False, name=''):
        self.reader = reader
        self.FLAGS = FLAGS
        self.pkl = pkl
        if len(name)>0:
            name = name + '_'
        self.name = name
        self.batch_size = batch_size if batch_size is not None else FLAGS.batch_size
        self.trim_chars = FLAGS.trim_chars
        self.trim_words = FLAGS.trim_words
        self.max_word_length = FLAGS.max_word_length
        self.max_text_length = FLAGS.max_text_length
        self.t = None
        self._word_count = 0
        
        self.epoch_ct = 0
        self.batch_ct = 0
        
        if self.trim_words:
            self.max_text_length = None
        elif self.max_text_length==None:
            self.max_text_length = self.reader.get_maxlen()# reader=FieldParser
            print('max essay length: {}'.format(self.max_text_length))
        if self.trim_chars:
            self.max_word_length = None
        
        if tstats and reader!=None:
            ystats, tstats = reader.get_stats()
            
        if ystats is None and reader!=None:
            ystats = reader.get_ystats()# for ATS: reader=field_parser
            
        if ystats:
            if FLAGS.min_cut<1.0:
                self.t = sample_dict(ystats.v, ystats.c, min_cut=FLAGS.min_cut)
                
            if verbose:
                print_ystats(ystats)
                if self.t:
                    print_t(self.t, ystats)
#                     sys.exit()
        
        if tstats:
            if verbose:
                print_tstats(tstats)
                
        self.ystats = ystats
        self.tstats = tstats
        self.normy = normy
    
    ## to interval [0,1]
    def normalize(self, y, min=None, max=None):
        if min==None:
            min = self.ystats.min
        if max==None:
            max = self.ystats.max
        y = (y-min) / (max-min)# --> [0,1]
        #y = y - (self.ystats.mean-self.ystats.min)/(self.ystats.max-self.ystats.min)
        return y
        #return (y - self.ystats.mean) / (self.ystats.max-self.ystats.min)
        
    @property
    def ymean(self):
        return self.normalize(self.ystats.mean)
    
    def word_count(self, reset=True):
        wc = self._word_count
        if reset:
            self._word_count = 0
        return wc
    
    def batch_file(self, ct=None):
        if ct is None: ct=self.batch_ct
        return os.path.join(self.FLAGS.chkpt_dir, '{}batch_{}.pkl'.format(self.name, ct))
        
    def batch(self, 
              ids=None, 
              labels=None, 
              words=None, 
              chars=None, 
              w=None, 
              c=None,
              trim_words=None,
              trim_chars=None,
              spad='pre', 
              wpad='post',
              cpad='post',
              split_sentences=False,
              batch_size=0,
              lines=None,
              is_test=False,
              ):
        
        if ids:
            self.last = (ids, labels, words, chars, w, c)
        else:
            (ids, labels, words, chars, w, c) = self.last
            
        if trim_words==None:
            trim_words=self.trim_words
        if not trim_words and self.max_text_length==None:
            self.max_text_length = self.reader.get_maxlen()
        if trim_chars==None:
            trim_chars=self.trim_chars
            
        n = len(ids)
        b = { 'n' : n }
        
        ''' partial ??? '''
#         # if not full batch.....just copy first item to fill
#         for i in range(batch_size-n):
#             ids.append(ids[0])
#             labels.append(labels[0])
#             words.append(words[0])
#             chars.append(chars[0])
        
        b['id'] = ids                              # <-- THIS key ('id') SHOULD COME FROM FIELD_PARSER.fields
                
        y = np.array(labels, dtype=np.float32)
        if self.normy:
            y = self.normalize(y)
        y = y[...,None]#y = np.expand_dims(y, 1)
        b['y'] = y                                  # <-- THIS key ('y') SHOULD COME FROM FIELD_PARSER.fields
        
        if w and not isListEmpty(words):
            m = (self.max_text_length,)
            if trim_words: m = (None,)
            if split_sentences: m = (None,) + m
            m = (None,) + m
            
            p = (wpad,)
            if split_sentences: p = (spad,) + p
            p = (None,) + p
            ## p = (None,spad,wpad) ???
            
            word_tensor, seq_lengths= U.pad_sequences(words, m=m, p=p)
            
            b['w'] = word_tensor
            b['s'] = seq_lengths
            b['p'] = p[1:] ## b.p = ( spad, wpad [,cpad] ) ???
            b['x'] = b['w']
        
        if c and not isListEmpty(chars):
            m = (self.max_word_length,)
            if trim_chars: m = (None,)
            if trim_words: m = (None,) + m
            else: m = (self.max_text_length,) + m
            if split_sentences: m = (None,) + m
            m = (None,) + m
            
            p = (wpad, cpad)
            if split_sentences: p = (spad,) + p
            p = (None,) + p
            ## p = (None,spad,wpad,cpad) ???
            
            try:
                char_tensor, seq_lengths = U.pad_sequences(chars, m=m, p=p)
            except (IndexError, ValueError):
                print('ERROR!')
                for i, cc in enumerate(chars):
                    if len(cc)==0:
                        print('')
                        print(ids[i])
                        print(cc)
                sys.exit()
            
            b['c'] = char_tensor
            b['s'] = seq_lengths
            b['p'] = p[1:] ## b.p = ( spad, wpad [,cpad] ) ???
            b['x'] = b['c']
            
        if not (c or w) and not isListEmpty(lines):
            b['t'] = lines
            #b['x'] = b['t']
        
        b['is_test']=is_test
        
        ##################
        # pickle b here...
        if self.pkl and self.epoch_ct==1:
            if self.batch_ct==0:
                # delete old pkl files
                cmd = 'rm {}'.format(self.batch_file(ct='*'))
                print(cmd)
                os.system(cmd)
            self.batch_ct +=1
            self.max_batch = self.batch_ct
            with open(self.batch_file(), 'wb') as handle:
                pickle.dump(b, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        ##################
        
        return U.adict(b)
    
    '''
    use batch padding!
    https://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html
    '''
    def batch_stream(self,
                     stop=False,
                     skip_ids=None, 
                     hit_ids=None, 
                     w=None,
                     c=None,
                     sample=False,
                     partial=False,
                     FLAGS=None,
                     SZ=0,
                     IS=None,
                     skip_test=None,
                     ):
        if FLAGS is None:
            FLAGS=self.FLAGS
        
        self.epoch_ct +=1
        self.batch_ct =0
        
        ##
        if self.pkl and self.epoch_ct>1:
            for i in range(self.max_batch):
                with open(self.batch_file(ct=i+1), 'rb') as handle:
                    b = pickle.load(handle)
                    yield U.adict(b)
            ## END FUNCTION HERE !!
            return
        
        #######################################################
        
        spad = FLAGS.spad
        wpad = FLAGS.wpad
        cpad = FLAGS.cpad
        
        if w is None: w=FLAGS.embed.word
        if c is None: c=FLAGS.embed.char
        
        trim_words = FLAGS.trim_words
        split_sentences = FLAGS.split_sentences
        if not split_sentences: SZ=0
        self._word_count = 0
        
        i, ids, labels, words, chars, ns, nw, lines = 0,[],[],[],[],0,0,[]
        is_test = FLAGS.is_test
        
        for d in self.reader.line_stream(stop=stop, t=self.t if sample else None):# reader=FieldParser!

            if skip_ids is not None:
                if d.id in skip_ids:
                    continue
            if is_test and skip_test is not None:
                if d.id in skip_test:
                    continue
            if hit_ids:
                if d.id not in hit_ids:
                    continue
                
            # SKIP ? 
            if w and len(d.w)==0:
                print('SKIP : len(d.w)==0!!!!')
                continue
            # SKIP ?
            if len(d.t)==0:
                print('SKIP : len(d.t)==0!!!!')
                continue
            
            dw = d.w
            dc = d.c
            if split_sentences:
                dw = U.lindexsplit(d.w, d.ws)
                dc = U.lindexsplit(d.c, d.cs)
                #############################
                if SZ>0:
                    _ns = max(ns, len(dw))
                    _nw = max(nw, max(map(len, dw)))
                    sz = (i+1) * _ns * _nw
                    if sz > 2*SZ:
                        #print('SKIP\tsz={0}\t[{1}x{2}x{3}] '.format(sz, i+1, _ns, _nw))
                        continue
                    ns, nw = _ns, _nw
                #############################
            
            ids.append(d.id)
            labels.append(d.y)
            lines.append(d.t)
            
            words.append(dw)
            chars.append(dc)
            
            if c or w:
                self._word_count+=len(d.w)
            else:
                self._word_count+= d.t.count(' ')
            i+=1
            
            #if i==self.batch_size:
            if (SZ>0 and (sz>SZ or i==self.batch_size*4)) or (SZ==0 and i==self.batch_size):
                #print('sz={0}\t[{1}x{2}x{3}] '.format(i*ns*nw, i, ns, nw))
                yield self.batch(ids, labels, words, chars, w, c, trim_words, spad=spad, wpad=wpad, cpad=cpad, split_sentences=split_sentences, lines=lines, is_test=is_test)
                i, ids, labels, words, chars, ns, nw, lines = 0,[],[],[],[],0,0,[]
                is_test = FLAGS.is_test    
        
        if i>0 and partial:
            yield self.batch(ids, labels, words, chars, w, c, trim_words, spad=spad, wpad=wpad, cpad=cpad, split_sentences=split_sentences, lines=lines, is_test=is_test)
            

def first_nonzero(a):
    di=[]
    for i in range(len(a)):
        idx=np.where(a[i]!=0)
        try:
            di.append(idx[0][0])
        except IndexError:
            di.append(len(a[i]))
    return di

def test_text_reader():
    data_dir = '/home/david/data/ets1b/2016'
    vocab_file = os.path.join(data_dir, 'vocab_n250.txt')
    shard_patt = os.path.join(data_dir, 'holdout', 'ets.2016.heldout-00001-of-00050')
    
    reader =  GlobReader(shard_patt, chunk_size=1000, shuf=False)
    text_parser = TextParser(vocab_file, reader=reader)
        
    for d in text_parser.chunk_stream(stop=True):
        print(len(d.w))

def test_ystats():
    emb_dir = '/home/david/data/embed'
    emb_file = os.path.join(emb_dir, 'glove.6B.100d.txt')
    
    data_dir = '/home/david/data/ets1b/2016'
    id = 70088; essay_file = os.path.join(data_dir, '{0}', '{0}.txt.clean.tok').format(id)
    
    reader =  GlobReader(essay_file, chunk_size=10000, regex=REGEX_NUM, shuf=True)
    
    E, word_vocab = Vocab.load_word_embeddings(emb_file, essay_file, min_freq=1)
    text_parser = TextParser(word_vocab=word_vocab)
    
    fields = {0:'id', 1:'y', -1:text_parser}
    field_parser = FieldParser(fields, reader=reader)
    
    #field_parser.sample()
    ystats = field_parser.get_ystats()
    print(ystats)
    
def test_essay_reader():
    data_dir = '/home/david/data/ets1b/2016'
    id = 63986; essay_file = os.path.join(data_dir, '{0}', '{0}.txt.clean.tok').format(id)
    
    data_dir = '/home/david/data/ats/ets'
    id = 55433; essay_file = os.path.join(data_dir, '{0}', 'text.txt').format(id)
    
    regex_num = r'^[0-9]*\t([0-9]\t)+(?!\s*$).+'
    
    reader =  GlobReader(essay_file, chunk_size=1000, regex=regex_num, shuf=False)
    for line in reader.line_stream(stop=True):
        print(line)
        break
    
def test_essay_parser():
    emb_dir = '/home/david/data/embed'
    emb_file = os.path.join(emb_dir, 'glove.6B.100d.txt')
    
#     data_dir = '/home/david/data/ets1b/2016'
#     id = 63986; essay_file = os.path.join(data_dir, '{0}', '{0}.txt.clean.tok').format(id)
    
    data_dir = '/home/david/data/ats/ets'
    id = 55433; essay_file = os.path.join(data_dir, '{0}', 'text.txt').format(id)
    
    regex_num = r'^[0-9]*\t([0-9]\t)+(?!\s*$).+'
    reader =  GlobReader(essay_file, chunk_size=1000, regex=regex_num, shuf=False)
    
    E, word_vocab = Vocab.load_word_embeddings(emb_file, essay_file, min_freq=10)
    text_parser = TextParser(word_vocab=word_vocab, tokenize=True)
    
    fields = {0:'id', 1:'y', -1:text_parser}
    field_parser = FieldParser(fields, reader=reader)
    
    for d in field_parser.line_stream(stop=True):
        print(d.w)
        #break

def test_text_batcher():
    data_dir = '/home/david/data/ets1b/2016'
    vocab_file = os.path.join(data_dir, 'vocab_n250.txt')
    #shard_patt = os.path.join(data_dir, 'train', 'ets.2016-00001-of-00100')
    shard_patt = os.path.join(data_dir, 'holdout', 'ets.2016.heldout-0000*-of-00050')
    
    reader =  GlobReader(shard_patt, chunk_size=1000, shuf=True)
    
    word_vocab, char_vocab, max_word_length = Vocab.load_vocab(vocab_file)
    text_parser = TextParser(word_vocab=word_vocab, char_vocab=char_vocab, max_word_length=max_word_length, reader=reader)
    
    batcher = TextBatcher(reader=text_parser, batch_size=128, num_unroll_steps=20, batch_chunk=50, trim_chars=True)
    
    #i=1
    for b in batcher.batch_stream(stop=True):
        #print(x)
        #print(y)
        print('{}\t{}'.format(b.w.shape, b.c.shape))
        #print(i);i=i+1

''' CHAR EMBEDDINGS '''
def test_essay_batcher_1():
    data_dir = '/home/david/data/ets1b/2016'
    vocab_file = os.path.join(data_dir, 'vocab_n250.txt')
    id = 62051 # 63986 62051 70088
    essay_file = os.path.join(data_dir, '{0}', '{0}.txt.clean.tok').format(id)
    
    reader =  GlobReader(essay_file, chunk_size=1000, regex=REGEX_NUM, shuf=True)
    
    word_vocab, char_vocab, max_word_length = Vocab.load_vocab(vocab_file)
    text_parser = TextParser(word_vocab=word_vocab, char_vocab=char_vocab, max_word_length=max_word_length)
    
    fields = {0:'id', 1:'y', -1:text_parser}
    field_parser = FieldParser(fields, reader=reader)
    
    batcher = EssayBatcher(reader=field_parser, batch_size=128, max_word_length=max_word_length, trim_words=True, trim_chars=False)
    for b in batcher.batch_stream(stop=True):
        print('{}\t{}\t{}'.format(b.w.shape, b.c.shape, b.y.shape))

''' GLOVE WORD EMBEDDINGS '''
def test_essay_batcher_2():
    char = True
#     char = False

    U.seed_random(1234)
    keep_unk = False
    
    emb_dim = 100
    emb_path = '/home/david/data/embed/glove.6B.{}d.txt'
    vocab_dir = '/home/david/data/ets1b/2016'
    vocab_file = os.path.join(vocab_dir, 'vocab_n250.txt')
    
    data_dir = '/home/david/data/ats/ets'
    id = 55433; essay_file = os.path.join(data_dir, '{0}', 'text.txt').format(id)
    #     id = 63986; essay_file = os.path.join(data_dir, '{0}', '{0}.txt.clean.tok').format(id)
    
    reader =  GlobReader(essay_file, chunk_size=1000, regex=REGEX_NUM, shuf=True)
    
    if char:
        word_vocab, char_vocab, max_word_length = Vocab.load_vocab(vocab_file)
        text_parser = TextParser(word_vocab=word_vocab, char_vocab=char_vocab, keep_unk=keep_unk)
    else:
        E, word_vocab = Vocab.load_word_embeddings_ORIG(emb_path, emb_dim, essay_file, min_freq=5)
        #E, word_vocab = Vocab.load_word_embeddings(emb_file, essay_file, min_freq=2)
        text_parser = TextParser(word_vocab=word_vocab, keep_unk=keep_unk)
    
    fields = {0:'id', 1:'y', -1:text_parser}
    field_parser = FieldParser(fields, reader=reader, seed=1234)
    
    batcher = EssayBatcher(reader=field_parser, batch_size=32, trim_words=True)
    for b in batcher.batch_stream(stop=True, split_sentences=True):
        print('{}\t{}'.format(b.w.shape, b.y.shape))

''' char and word embeddings '''
def test_response_batcher():
    char = True
    char = False

    U.seed_random(1234)
    keep_unk = False
    
    emb_dir = '/home/david/data/embed'; emb_file = os.path.join(emb_dir, 'glove.6B.100d.txt')
    emb_path = '/home/david/data/embed/glove.6B.{}d.txt'
    
    vocab_dir = '/home/david/data/ets1b/2016'
    vocab_file = os.path.join(vocab_dir, 'vocab_n250.txt')
    
    data_dir = '/home/david/data/ats/ets'
    id = 56375; 
    essay_file = os.path.join(data_dir, '{0}/text.txt').format(id)
#     essay_file = os.path.join(data_dir, '{0}/{0}.txt').format(id)
    
    reader =  GlobReader(essay_file, chunk_size=1000, regex=REGEX_NUM, shuf=False)
    
    if char:
        word_vocab, char_vocab, max_word_length = Vocab.load_vocab(vocab_file)
        text_parser = TextParser(word_vocab=word_vocab, char_vocab=char_vocab, keep_unk=keep_unk)
    else:
        E, word_vocab = Vocab.load_word_embeddings_ORIG(emb_path, 100, essay_file, min_freq=5)
        #E, word_vocab = Vocab.load_word_embeddings(emb_file, essay_file, min_freq=2)
        text_parser = TextParser(word_vocab=word_vocab, keep_unk=keep_unk)
    
    fields = {0:'id', 1:'y', -1:text_parser}
    field_parser = FieldParser(fields, reader=reader, seed=1234)
    
    tot_vol = 0
    batcher = ResponseBatcher(reader=field_parser, batch_size=64, trim_words=True)
    for b in batcher.batch_stream(stop=True,
                                  split_sentences=True,
                                  spad='post',
                                  wpad='post',
                                  ):
        vol = np.prod(b.x.shape); tot_vol+=vol
        print('{}\t{}\t{}'.format(vol, b.x.shape, b.y.shape))
    print('tot_vol:\t{}'.format(tot_vol))
                     
if __name__ == '__main__':
    #test_essay_reader()
#     test_essay_parser()
#     test_ystats()
#     test_text_reader()
#     test_text_batcher()
#     test_essay_batcher_1()
#     test_essay_batcher_2()
    test_response_batcher()
    print('done')