from __future__ import print_function
from __future__ import division

'''
sudo mount -t cifs -o username=dvaughn //spshare.miat.co/Users /media/david/spshare
'''

import sys
import os, errno
import logging
import numpy as np
import pandas as pd
import random
import codecs
import collections
import itertools
import glob
import shutil
import pickle
import json
import time
import re
from timeit import default_timer as timer
import tensorflow as tf

def any_in(a,b):
    return any(i in b for i in a)

def string2tf(s):
    return tf.py_func(lambda: s, [], tf.string)

def make_sequence_mask(lengths, pad='post', axis=[1]):
    mask = tf.sequence_mask(lengths, dtype=tf.float32)
    if pad=='pre': mask = tf.reverse(mask, axis)
    return mask

# dynamic_mean(inputs, self.seq_len)
def dynamic_mean(x, seq_len, pad='post', axis=1):
    mask = make_sequence_mask(seq_len, pad=pad, axis=[axis])
    #x = tf.multiply(x, mask)
    x = x * tf.expand_dims(mask, 2)
    m = tf.reduce_sum(x, axis=axis)
    return m / tf.expand_dims(tf.cast(seq_len, tf.float32), 1)
           
def softmask(x, axis=-1, mask=None, T=None):
    x_max = tf.reduce_max(x, axis=axis, keep_dims=True)
    x = x - x_max
    
    if T!=None:
#         if not tf.is_numeric_tensor(T):
#             T = tf.get_variable('T', shape=[1], initializer=tf.constant_initializer(T), dtype=tf.float32, trainable=True)
        x = x/T
    
    ex = tf.exp(x)
    
    if mask!=None:
        ex = tf.multiply(ex, mask)
        
    es = tf.reduce_sum(ex, axis=axis, keep_dims=True)
    
    ez = tf.cast( tf.equal( es, tf.constant( 0, dtype=tf.float32 ) ), tf.float32)
    es = es + ez
    ret = ex/es
    
    return ret#, ex, es, ez

#######################################################################
## NESTED SEQUENCES ##

from tensorflow.python.util import nest
######################################################################
def list2list(x,f):
    if isinstance(x, (list,)):
        return map(lambda y: list2list(y,f), x)
    return f(x)

def nested_lens(x, d=0):
    n = len(x)
    if d == None:
        ret = [n]
        dd = None
    else:
        ret = [(n,d)]
        dd = d+1
    if n>0 and nest.is_sequence(x[0]):
        ret.extend(map(lambda y: nested_lens(y,d=dd), x))
    return ret

def max_lens(x):
    v = nest.flatten(nested_lens(x))
    n = v[-1] + 1
    u = [[] for _ in range(n)]
    for i in range(int(len(v)/2)):
        u[v[2*i+1]].append(v[2*i])
    #return tuple(map(max,u))
    return map(max,u)

def _seq_lens(x, axis=1):
    if axis==0: return x[0]
    return map(lambda y: _seq_lens(y, axis-1), x[1:])
    
def seq_lens(x, axis=1, p=None):
    shape = max_lens(x)
    u = _seq_lens(nested_lens(x, d=None), axis=axis)
    v = pad(u, shape=shape[0:axis], p=p)
    return v
    
def pad(seq, shape, p=None, dtype='int32', value=0.):
    n = len(seq)
    d = len(shape)
    x = (np.ones(shape) * value).astype(dtype)
    if p==None: p=tuple([None]*n)
    if d==1:
#         if len(seq)==0:
#             print('len(seq)==0!')
#             q=3
        if p[0] and p[0]=='pre':
            x[-len(seq):]= seq
        else:
            x[:len(seq)] = seq
        return x
    j = (0 if (p[0]==None or p[0]=='post') else shape[0]-n)
    for i,s in enumerate(seq):
        y = pad(s, shape[1:], p[1:], dtype=dtype, value=value)
        x[i+j,:] = y
    return x

''' convert nested lists -> tensor/array '''      
def pad_sequences(seq, p=None, m=None, dtype='int32', value=0.):
    shape = max_lens(seq)
    ## necessary?? ##
    if m:
        ss = list(shape)
        for i in range(len(m)):
            if m[i]:
                ss[i] = m[i]
        shape = tuple(ss)
    #################
    seq_lengths = [seq_lens(seq, axis=i+1, p=p) for i in range(len(shape)-1)]
    return pad(seq, shape, p=p, dtype=dtype, value=value), tuple(seq_lengths)

def is_sequence(x):
    return nest.is_sequence(x)
        

#######################################################################
## NLP / TOKENIZATION ##

punc = '(),'# ?!
word_pattern = r'([0-9]+[0-9,.]*[0-9]+|[\w]+|[{}])'.format(punc)

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(word_pattern)

def tokenize_OLD(string):
    tokens = tokenizer.tokenize(string)
    for index, token in enumerate(tokens):
        if token == '@' and (index+1) < len(tokens):
            tokens[index+1] = '@' + re.sub('[0-9]+.*', '', tokens[index+1])
            tokens.pop(index)
    return tokens

from nltk.tokenize import sent_tokenize, word_tokenize

# import nltk
# sent_tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
# from sentence_tokenizer import split_into_sentences

def clean(s):
    s = re.sub("([0-9]+),([0-9]+)", "\\1COMMA\\2", s)
    for c in punc:
        a = '[{0}][\s{0}]*[{0}]*'.format(c)
        b = ' {0} '.format(c)
        s = re.sub('[{0}][\s{0}]*[{0}]*'.format(c), ' {0} '.format(c), s)
    s = re.sub("COMMA", ",", s)
    return s

def strip_non_ascii(string):
    ''' Returns the string without non ASCII characters'''
    stripped = (c for c in string if 0 < ord(c) < 127)
    return str(''.join(stripped))

def empty(inList):
    if isinstance(inList, list): # Is a list
        return all( map(empty, inList) )
    return False
   
def tokenize_NEW(string):
    string = re.sub("[.]\s*[.]\s*[.]"," . ", string)
    string = re.sub("([\w']+)-([\w']+)", "\\1 - \\2", string)
    string = re.sub("[+]","", string)
    
    try:
        sents = sent_tokenize(string)
    except UnicodeDecodeError:
        sents = sent_tokenize(strip_non_ascii(string))
    #sents2 = sent_tokenizer.tokenize(string)
    #sents3 = split_into_sentences(string)
    
    #words = [word_tokenize(clean(s)) for s in sents if len(s)>1]
    #tokens = [tokenizer.tokenize(clean(s)) + [u'.'] for s in sents if len(s)>1]
    
    #tokens = [tokenizer.tokenize(clean(s)) for s in sents if len(s)>1]
    tokens = [tokenizer.tokenize(clean(s)) for s in sents]
    
    #return list(itertools.chain(*tokens))
    return tokens if not empty(tokens) else [sents]

def flatten(l):
	return [item for sublist in l for item in sublist]

def tokenize(string, lower=True, flat=False):
    #toks = tokenize_OLD(string.lower())
    toks = tokenize_NEW(string.lower() if lower else string)
    
    ## filter out empty elements
    toks = [t for t in toks if len(t)>0]
    
    ## flatten ??
    if flat: 
    	toks = flatten(toks)
    
    return toks

def write_sequence(file, x, sep='\n'):
    with open(file, 'w') as output_file:
        for item in x:
            output_file.write(str(item) + sep)
    
class adict(dict):
    ''' Attribute dictionary - a convenience data structure, similar to SimpleNamespace in python 3.3
        One can use attributes to read/write dictionary content.
    '''
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self

def ps(x, s):
    print('{} : {}'.format(s, x.shape))
    #print('{} : {}'.format(s, tf.shape(x)))
    #print('{} : {}'.format(s, x.get_shape().as_list())
    
def softmask(x, axis=-1, mask=None):
    x_max = tf.reduce_max(x, axis=axis, keep_dims=True)
    x = x - x_max
    ex = tf.exp(x)
    
    if mask!=None:
        ex = tf.multiply(ex, mask)
        
    es = tf.reduce_sum(ex, axis=axis, keep_dims=True)
    return ex/es

def shuffle_lists(*ls):
    l =list(zip(*ls))
    random.shuffle(l, random=rng)
    return zip(*l)

def shuffle_arrays(*xx):
    n = xx[0].shape[0]
    p = rng.permutation(n)#p = np.random.permutation(n)
    yy=()
    for x in xx:
        yy += (x[p],)
    return yy

def lindexsplit(x, idx):
    return [x[start:end] for start, end in zip(idx, idx[1:])]

    # For a little more brevity, here is the list comprehension of the following
    # statements:
    #    return [some_list[start:end] for start, end in zip(args, args[1:])]
#     my_list = []
#     for start, end in zip(args, args[1:]):
#         my_list.append(some_list[start:end])
#     return my_list

def memoize(f):
    """ Memoization decorator for a function taking one or more arguments. """
    class memodict(dict):
        def __getitem__(self, *key):
            return dict.__getitem__(self, key)
        def __missing__(self, key):
            ret = self[key] = f(*key)
            return ret
    return memodict().__getitem__

T={}
def tic(K=0):
    global T
    if isinstance(K,list):
        for k in K:
            tic(k)
    else:
        T[K]=timer()
def toc(K=0, reset=True):
    global T
    t=timer()
    tt=t-T[K]
    if reset:
        T[K]=t
    return tt

def dot(x,y):
    x = tf.transpose(x)
    #y = tf.transpose(y)
    return tf.matmul(x,y)

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

def kappa_loss(labels, predictions, scope=None):
    with ops.name_scope(scope, "kappa_loss", (predictions, labels)) as scope:
        predictions = math_ops.to_float(predictions)
        labels = math_ops.to_float(labels)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        losses = math_ops.squared_difference(predictions, labels)
        u = math_ops.reduce_sum(losses)
        a = math_ops.reduce_mean(labels)
        b = math_ops.multiply(predictions, labels - a)
        v = math_ops.reduce_sum(b)
        k = u / (2*v + u)
        return k

## continuous kappa
def nkappa(t,x):
    t=np.array(t,np.float32)
    x=np.array(x,np.float32)
    u = 0.5 * np.sum(np.square(x - t))
    v = np.dot(np.transpose(x), t - np.mean(t))
    return v / (v + u)

import sklearn
def nacc(t,x):
    t=np.array(t,np.float32)
    x=np.round(np.array(x,np.float32))
    return sklearn.metrics.accuracy_score(t,x)

def save_roc(fn, t, p, id=None):
    if id is not None:
        id = np.array(id, np.int32)
    t = np.array(t, np.float64)
    p = np.array(p, np.float64)
    A = [t,p]
    if id is not None: A.append(np.array(id))
    x = sortrows(np.vstack(A).transpose(), 1, False)
    with open(fn, 'w') as f:
        if id is None:
            for i in range(x.shape[0]): f.write('{0}\t{1:0.12g}\n'.format(x[i,0], x[i,1]))
        else:
            for i in range(x.shape[0]): f.write('{0}\t{1:0.12g}\t{2}\n'.format(x[i,0], x[i,1], x[i,2]))    

# def qwok(t,y):
#     mu = t.mean()
#     t = t-mu
#     y = y-mu
#     return 2*t.dot(y)/(t.dot(t) + y.dot(y))

## int kappa
from quadratic_weighted_kappa import quadratic_weighted_kappa as qwk
def ikappa(t, x, minmax=None):
    if minmax is None:
        return qwk(t, x)
    minmax = np.array(minmax, np.int32)
    return qwk(t, x, minmax.min(), minmax.max())

def interleave(a,b):
    return list(itertools.chain.from_iterable(zip(a,b)))

def arrayfun(f,A):
    return list(map(f,A))

def isnum(a):
    try:
        float(repr(a))
        ans = True
    except:
        ans = False
    return ans

def get_seed():
    t = int( time.time() * 1000.0 )
    seed = ((t & 0xff000000) >> 24) + ((t & 0x00ff0000) >>  8) + ((t & 0x0000ff00) <<  8) + ((t & 0x000000ff) << 24)
    return seed % 2**30-1#    2**32-1

rng = 0
def seed_random(seed=None):
    global rng
    if seed==None or seed<=0:
        seed = get_seed()
    print(b_green('RAND_SEED == {}'.format(seed)))
    random.seed(seed)
    np.random.seed(seed=seed)
    rng = np.random.RandomState(seed)
    return seed

def string2rand(s):
    return abs(hash(s)) % (10 ** 8)

def get_max_norm(x, default=None):#FLAGS.max_grad_norm
    if len(x)<3 and default is not None:
        return default
    if len(x)<10:
        return 5.0*max(x)
    mu=np.mean(x[-20:])
    std=np.std(x[-20:])
    return mu + 2.0*std

def get_hostname():
    import socket
    return socket.gethostname()

def get_loc(check='home', default='work'):
    if check in get_hostname():
        return check
    return default

''' add root to path IF path not absolute'''
def try_abs(path, root):
    if os.path.isabs(path):
        return path
    return os.path.join(root,path)

def make_abs(path):
    return os.path.abspath(path)

# sort 2-D numpy array by col
def sortrows(x, col=0, asc=True):
    n=-1
    if asc:
        n=1
    x=x*n
    return n*x[x[:,col].argsort()]

def set_logger(out_dir=None):
    console_format = BColors.OKBLUE + '[%(levelname)s]' + BColors.ENDC + ' (%(name)s) %(message)s'
    #datefmt='%Y-%m-%d %Hh-%Mm-%Ss'
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(console_format))
    logger.addHandler(console)
    if out_dir:
        file_format = '[%(levelname)s] (%(name)s) %(message)s'
        log_file = logging.FileHandler(out_dir + '/log.txt', mode='w')
        log_file.setLevel(logging.DEBUG)
        log_file.setFormatter(logging.Formatter(file_format))
        logger.addHandler(log_file)


class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    WHITE = '\033[37m'
    YELLOW = '\033[33m'
    GREEN = '\033[32m'
    BLUE = '\033[34m'
    CYAN = '\033[36m'
    RED = '\033[31m'
    MAGENTA = '\033[35m'
    BLACK = '\033[30m'
    BHEADER = BOLD + '\033[95m'
    BOKBLUE = BOLD + '\033[94m'
    BOKGREEN = BOLD + '\033[92m'
    BWARNING = BOLD + '\033[93m'
    BFAIL = BOLD + '\033[91m'
    BUNDERLINE = BOLD + '\033[4m'
    BWHITE = BOLD + '\033[37m'
    BYELLOW = BOLD + '\033[33m'
    BGREEN = BOLD + '\033[32m'
    BBLUE = BOLD + '\033[34m'
    BCYAN = BOLD + '\033[36m'
    BRED = BOLD + '\033[31m'
    BMAGENTA = BOLD + '\033[35m'
    BBLACK = BOLD + '\033[30m'
    
    @staticmethod
    def cleared(s):
        return re.sub("\033\[[0-9][0-9]?m", "", s)

def red(message):
    return BColors.RED + str(message) + BColors.ENDC

def b_red(message):
    return BColors.BRED + str(message) + BColors.ENDC

def blue(message):
    return BColors.BLUE + str(message) + BColors.ENDC

def b_yellow(message):
    return BColors.BYELLOW + str(message) + BColors.ENDC

def green(message):
    return BColors.GREEN + str(message) + BColors.ENDC

def b_green(message):
    return BColors.BGREEN + str(message) + BColors.ENDC

#######################################################

def generate_json_docs(path, ids, targets, preds, words, w_alphas, s_alphas=None):
    n = len(ids)
    json_files = []
    for i in range(n):
        json_file = save_json_doc(path=path,
                                  id=ids[i],
                                  target=targets[i],
                                  pred=preds[i],
                                  words=words[i],
                                  w_alphas=w_alphas[i],
                                  s_alphas=s_alphas[i] if s_alphas else None)
        json_files.append(json_file)
    return json_files
    
def save_json_doc(path, id, target, pred, words, w_alphas, s_alphas=None):
    file_name = '{}.json'.format(id)
    json_data = generate_json_data(id, target, pred, words, w_alphas, s_alphas)
    with open(os.path.join(path, file_name), 'w') as f:
        f.write(json_data)
    return file_name

def generate_json_data(id, target, pred, words, w_alphas, s_alphas=None):
    data = {}
    data['id'] = id
    data['target'] = np.float64(target)
    data['pred'] = np.float64(pred)
    n = len(words)
    sentences = []
    for i in range(n):
        s = {}
        s['word'] = words[i]
        s['attn'] = np.float64(w_alphas[i]).tolist()
        sentences.append(s)
    data['sentence'] = sentences
    if s_alphas is not None:
        data['attn'] = np.float64(s_alphas).tolist()
    try:
        return json.dumps(data)
    except TypeError:
        print(data)
        return json.dumps(data)


def clear_path(path):
    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): 
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)
            
def purge_pattern(dir, pattern):
    for f in os.listdir(dir):
        if re.search(pattern, f):
            os.remove(os.path.join(dir, f))
            
def mkdirs(s):
    try:
        os.makedirs(s)
    except OSError as exc: 
        if exc.errno == errno.EEXIST and os.path.isdir(s):
            pass
        
def check_dir(s):
    return os.path.isdir(s)

def check_file(s):
    return os.path.isfile(s)

def rm_dir(s):
    #os.rmdir(s)
    shutil.rmtree(s)
    return not check_dir(s)

######################################################################

def slyce(x,i,j,pad='post'):
    if pad=='post':
        return x[i][:j]
    else:
        return x[i][-j:]
    
''' convert tensor/array -> nested lists '''      
def tensor2list(x, s, p):
    sl = s[0]
    wl = s[1]
    z = []
    (n,m,o) = x.shape
    for i in range(n):
        zz=[]
        s = slyce(wl,i,sl[i],p[0])
        k = 0 if p[0]=='post' else m-sl[i]
        for j in range(len(s)):
            t = list(slyce(x[i],j+k,s[j],p[1]))
            zz.append(t)
        z.append(zz)
    return z
    
def t2l(x, seq_len, pad):
    (n,m) = x.shape
    y = []
    for i in range(n):
        yy = list(slyce(x, i, seq_len[i], pad=pad))
        y.append(yy)
    return y

########################################

def compute_ystats(y):
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
    return adict(d)

##############################################################

def read_col(file, col, sep="\t", header=None, type='int32'):
    df = pd.read_csv(file, sep=sep, header=header)#.sort_values(by=col)
    vals = df[df.columns[col]].values.astype(type)
    return vals

def read_cols(file, cols=None, keycol=None, val=None):
    df = pd.read_csv(file, sep="\t", header=None)#.sort_values(by=0)
    if keycol is not None:
        df = df.loc[df[keycol]==val]
    return df.as_matrix(cols)