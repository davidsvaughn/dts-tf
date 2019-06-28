from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import time
import collections
import numpy as np
import tensorflow as tf
import sonnet as snt

from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.ops import variable_scope as vs
import tensorflow.contrib.layers as layers

import utils as U
from model import rnn_components as rc
from model import model_components as mc
from model.attention import Attention, Attention2D, task_specific_attention

import model.attention as attn

from config import config

class BaseModel(snt.AbstractModule):
    def __init__(self,
                 FLAGS=None,
                 embed_word=True,
                 embed_matrix=None,
                 char_vocab=None,
                 name="BaseModel"):
        super(BaseModel, self).__init__(name=name)
        self.embed_word = embed_word
        self.embed_matrix = embed_matrix
        self.char_vocab = char_vocab
        self.FLAGS = FLAGS
        self.max_word_length = FLAGS.max_word_length
        
        self.t = []
        self.ts = []
        self.aux_inputs = None
        
        with self._enter_variable_scope():
            #self._seq_len = tf.placeholder(tf.int32, [None])# [self.batch_size]
            self._seq_len = tf.placeholder(shape=(None,), dtype=tf.int32, name='seq_len')
            
            self._keep_prob = tf.placeholder_with_default(1.0-abs(self.FLAGS.dropout), shape=())
            self._penalty = tf.constant( 0, dtype=tf.float32 )#self._penalty = None
    
    @property
    def keep_prob(self):
        return self._keep_prob
    
    @property
    def seq_len(self):
        return self._seq_len
    
    @property
    def penalty(self):
        self._ensure_is_connected()
        return self._penalty
    
    @property
    def tensors(self):
        self._ensure_is_connected()
        return self.t
    
    @property
    def tensor_shapes(self):
        self._ensure_is_connected()
        return self.ts
    
    @property
    def alpha_word(self):
        self._ensure_is_connected()
        try:
            return self._alpha_word
        except AttributeError:
            return tf.constant([])
    
    @property
    def alpha_sent(self):
        self._ensure_is_connected()
        try:
            return self._alpha_sent
        except AttributeError:
            return tf.constant([])
    
    def log_tensor(self, x, name):
        self.t.append((U.string2tf(name), x))
        self.log_tensor_shape(x, name)
        
    def log_tensor_shape(self, x, name):
        self.ts.append((U.string2tf(name), tf.shape(x)))
        
    def update_tensors(self, mod):
        if mod.tensors is not None: 
            self.t.extend(mod.tensors)
        if mod.tensor_shapes is not None: 
            self.ts.extend(mod.tensor_shapes)
        
    def _get_input_shape(self, batch_size=None):
        return None
    
    def get_input_placeholder(self, batch_size=None):
        with self._enter_variable_scope():
            return tf.placeholder(tf.int32, shape=self._get_input_shape(batch_size), name="inputs")
    
    def get_output_placeholder(self, batch_size=None):
        with self._enter_variable_scope():
            return tf.placeholder(tf.float32, shape=[batch_size, 1], name="targets")

class Shrinker:
    def __init__(self, idx):
        self.idx = idx
        
    def reduce(self, x):
        self.n = tf.shape(x)[0]
        return tf.gather_nd(x, self.idx)
        
    def expand(self, x):
        output_shape = tf.cast( [ self.n, tf.shape(x)[-1]] , tf.int64 )
        return tf.scatter_nd(indices=self.idx,
                             updates=x,
                             shape=output_shape)
            
##############################################################################################
class FlatModel(BaseModel):
    def __init__(self,
                 FLAGS=None,
                 embed_word=True,
                 embed_matrix=None,
                 char_vocab=None,
                 name="FlatModel"):
        super(FlatModel, self).__init__(name=name,
                                        FLAGS=FLAGS,
                                        embed_word=embed_word,
                                        embed_matrix=embed_matrix,
                                        char_vocab=char_vocab
                                        )
    
    def _get_input_shape(self, batch_size=None):
        return [batch_size, None] if self.embed_word else [batch_size, None, self.max_word_length]
    
    def _build(self, inputs):
        
        if self.embed_word:
            word_embed_module = snt.Embed(existing_vocab=self.embed_matrix, trainable=True)
            outputs = word_embed_module(inputs)
            
        else:
            #output_shape = [self.FLAGS.batch_size, tf.shape(inputs)[1], sum(self.FLAGS.kernel_features)]
            outputs = mc.char_cnn_embedding(inputs,
                                            char_vocab_size=self.char_vocab.size,
                                            char_embed_size=self.FLAGS.char_embed_size,
                                            kernel_widths=self.FLAGS.kernel_widths,
                                            kernel_features=self.FLAGS.kernel_features,
                                            sparse=True,
                                            #stack_output_dims=2,
                                            #output_shape=output_shape,
                                            )

        ##################################################
        FLAGS = config.set_flags(self.FLAGS, 0)
        FLAGS.pad = FLAGS.wpad
        
        #rnn_word = rc.DeepBiRNN if FLAGS.bidirectional else rc.DeepRNN
        rnn_word = (rc.BiRNN if FLAGS.rnn_new else rc.DeepBiRNN) if FLAGS.bidirectional else rc.DeepRNN
        self._rnn_module = rnn_word(FLAGS=FLAGS, 
                                    keep_prob=self.keep_prob,
                                    seq_len=self.seq_len,
                                    )
        outputs = self._rnn_module(outputs)
        
        ##################################################
        
        self._pool = attn.Pool(FLAGS, 
                               seq_len=self.seq_len,#seq_len=self._rnn_module.seq_len,
                               final_rnn_state=self._rnn_module.final_rnn_state)
        outputs = self._pool(outputs)
        
        ##################################################
#         dim = FLAGS.rnn_size
#         #dim = tf.shape(outputs)[-1]
#         #dim = outputs.get_shape().as_list()[-1]
  
        w_init, b_init = mc.default_initializers(std=FLAGS.model_std, bias=FLAGS.model_b)
        lin_module = snt.Linear(output_size=1, initializers={ 'w':w_init, 'b':b_init })
        outputs = lin_module(outputs)
        self._score = outputs
        ##################################################
        
        ## tanh
        outputs = tf.nn.tanh(outputs)
        return outputs
    
    @property
    def rnn_module(self):
        self._ensure_is_connected()
        return self._rnn_module
    
    @property
    def final_rnn_state(self):
        return self.rnn_module.final_rnn_state
    
    @property
    def pool(self):
        self._ensure_is_connected()
        return self._pool
    
    @property
    def score(self):
        self._ensure_is_connected()
        return self._score
           
##############################################################################################
''' https://github.com/davidsvaughn/hierarchical-attention-networks/blob/master/HAN_model.py    '''
                 
class HANModel(BaseModel):
    def __init__(self,
                 FLAGS=None,
                 embed_word=True,
                 embed_matrix=None,
                 char_vocab=None,
                 name="HANModel"):
        super(HANModel, self).__init__(name=name,
                                       FLAGS=FLAGS,
                                       embed_word=embed_word,
                                       embed_matrix=embed_matrix,
                                       char_vocab=char_vocab
                                       )
        self.rnn_size = FLAGS.rnn_size
        self.unit = FLAGS.rnn_cell
        self.forget_bias = FLAGS.forget_bias
        self.train_initial_state = FLAGS.train_initial_state
    
    def _get_input_shape(self, batch_size=None):
        return [batch_size, None, None] if self.embed_word else [batch_size, None, None, self.max_word_length]
    
    @property
    def seq_len(self):
        return (self.sentence_lengths, self.word_lengths)
    
    def encoding_dim(self, FLAGS):
        return FLAGS.attn_depth * FLAGS.rnn_size * (2 if FLAGS.bidirectional else 1)
       
    def _build(self, inputs):
        self.log_tensor(inputs, name='inputs')
        
        ''' SETUP '''
        #self.keep_prob = tf.placeholder_with_default(1.0-abs(self.FLAGS.dropout), shape=())
        
        # 3D/4D [ document x sentence x word (x char) ]
        input_shape = tf.shape(inputs) #max_word_length = input_shape[-1]
        
        # 2D placeholder for SENTENCE LENGTHS (in # words)        [ document x sentence ]
        self.word_lengths = tf.placeholder(shape=(None, None), dtype=tf.int32, name='word_lengths')
        
        # 1D placeholder for DOCUMENT LENGTHS (in # sentences)    [ document ]
        self.sentence_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='sentence_lengths')
        
        self._seq_len = (self.sentence_lengths, self.word_lengths)
        
        # dynamic tensor dimensions of input
        (self.document_size,                # size of document dim [ #docs/batch, i.e. batch_size ]
        self.sentence_size,                 # size of sentence dim [ max(doc_lengths) i.e. max(sents/doc) ]
        self.word_size,                     # size of word dim [max(sent_lengths) i.e. max(words/sent) ]
        ) = tf.unstack(input_shape)[:3]
        
        ''' effective batch sizes (bs) '''
        #bs = self.FLAGS.batch_size
        bs = self.document_size
        word_bs = self.document_size * self.sentence_size
        char_bs = self.document_size * self.sentence_size * self.word_size
        
        word_level_lengths = tf.reshape(self.word_lengths, [word_bs]); self.log_tensor(word_level_lengths, name='word_level_lengths')
        
        ############################################################
        ''' EMBEDDING (word/char) '''
        if self.embed_word:
            word_embed_module = snt.Embed(existing_vocab=self.embed_matrix, trainable=True)
#             word_embed_module = snt.Embed(existing_vocab=self.embed_matrix, trainable=False)
            
            outputs = word_embed_module(inputs)
            self.log_tensor(outputs, name='word_embedded')
            # [docs x sentences x words x E]
            
            outputs = tf.reshape(outputs, [word_bs, self.word_size, self.FLAGS.embed_dim])
            
        else:
            outputs = mc.char_cnn_embedding(inputs,
                                            char_vocab_size=self.char_vocab.size,
                                            char_embed_size=self.FLAGS.char_embed_size,
                                            kernel_widths=self.FLAGS.kernel_widths,
                                            kernel_features=self.FLAGS.kernel_features,
                                            stack_output_dims=2,
                                            #output_shape=[word_bs, input_shape[2], sum(self.FLAGS.kernel_features)],
                                            )
        
        ####################################################################################################
        ''' word level : Level 1'''
        
        word_level_inputs = outputs
        self.log_tensor(word_level_inputs, name='word_level_inputs')
        # [sentences x words x E]
        
        ''' remove empty sentences '''
        sps_idx = tf.where(word_level_lengths>0); self.log_tensor(sps_idx, name='sps_idx')
        shrinker = Shrinker(sps_idx)
        word_level_inputs = shrinker.reduce(word_level_inputs); self.log_tensor(word_level_inputs, name='word_level_inputs_2')
        word_level_lengths = shrinker.reduce(word_level_lengths); self.log_tensor(word_level_lengths, name='word_level_lengths_2')
        
        FLAGS = config.set_flags(self.FLAGS, 0)
        with tf.variable_scope('word') as scope:
            ''' rnn '''
            rnn_word = (rc.BiRNN if FLAGS.rnn_new else rc.DeepBiRNN) if FLAGS.bidirectional else rc.DeepRNN
            rnn_module_word = rnn_word(FLAGS=FLAGS,
                                       keep_prob=self.keep_prob,
                                       seq_len=word_level_lengths,
                                       )
            word_encoder_output = rnn_module_word(word_level_inputs);self.log_tensor(word_encoder_output, name='word_encoder_output')
            ## [sentences x words x rnn_dim]
            
            ''' attention pooling (sum over words) '''
            pool_word = attn.Pool(FLAGS, 
                                  seq_len=word_level_lengths,
                                  final_rnn_state=rnn_module_word.final_rnn_state)
            word_level_output = pool_word(word_encoder_output)
            self.update_tensors(pool_word)
            self.log_tensor(word_level_output, name='word_level_output')
            self._alpha_word = pool_word.alpha
            self._penalty += pool_word.penalty
            
            word_output_dim = self.encoding_dim(FLAGS)
            
            ''' dropout '''
            with tf.variable_scope('dropout'):
                word_level_output = layers.dropout(
                    word_level_output, 
                    keep_prob=self.keep_prob)
        
            ''' restore empty sentences '''
            word_level_output = shrinker.expand(word_level_output)# n=word_bs
            self.log_tensor(word_level_output, name='word_level_output_2')
            if self.FLAGS.attn_vis:
                self._alpha_word = shrinker.expand(self._alpha_word)
                
        ####################################################################################################
        ''' sentence level : Level 2'''
        
        sentence_inputs_shape = tf.cast( [self.document_size, self.sentence_size, word_output_dim] , tf.int32 )
        sentence_inputs = tf.reshape( word_level_output, shape=sentence_inputs_shape )
        self.log_tensor(sentence_inputs, name='sentence_inputs')
        self.log_tensor(self.sentence_lengths, name='sentence_level_lengths')
        
        if self.FLAGS.attn_vis:
            alpha_shape = (self.document_size, self.sentence_size, self.word_size)
            self._alpha_word = tf.reshape( self._alpha_word, shape=alpha_shape )
            self.log_tensor(self._alpha_word, name='alpha_word')
        
        FLAGS = config.set_flags(self.FLAGS, 1)
        with tf.variable_scope('sentence') as scope:
            ''' rnn '''
            rnn_sent = (rc.BiRNN if FLAGS.rnn_new else rc.DeepBiRNN) if FLAGS.bidirectional else rc.DeepRNN
            rnn_module_sent = rnn_sent(FLAGS=FLAGS,
                                       keep_prob=self.keep_prob,
                                       seq_len=self.sentence_lengths,
                                       )
            sentence_encoder_output = rnn_module_sent(sentence_inputs); self.log_tensor(sentence_encoder_output, name='sentence_encoder_output')
            # [docs x sentences x rnn_dim]

            ''' attention pooling (sum over sentences) '''
            pool_sent = attn.Pool(FLAGS, 
                                  seq_len=self.sentence_lengths,
                                  final_rnn_state=rnn_module_sent.final_rnn_state)
            sentence_level_output = pool_sent(sentence_encoder_output)
            self.update_tensors(pool_sent)
            self.log_tensor(sentence_level_output, name='sentence_level_output')
            self._alpha_sent = pool_sent.alpha
            self._penalty += pool_sent.penalty
            
            ####################################################################################################
            ''' specify shape '''
            sent_output_dim = self.encoding_dim(FLAGS)
            sentence_output_shape = tf.cast( [self.document_size, sent_output_dim] , tf.int32 )
            #sentence_level_output.set_shape(tf.TensorShape([self.FLAGS.batch_size, sent_output_dim])) # OLD
            #sentence_level_output.set_shape(sentence_output_shape)# NOT WORKS!!!
            x = tf.reshape( sentence_level_output, shape=sentence_output_shape )# WORKS!!! ???
            
            ####################################################################################################
            
            ''' dropout '''
            with tf.variable_scope('dropout'):
                x = layers.dropout(x, keep_prob=self.keep_prob)
                
        ''' final dense layer '''
        w_init, b_init = mc.default_initializers(std=self.FLAGS.model_std, bias=self.FLAGS.model_b)
        
        lin_module = snt.Linear(output_size=1, initializers={ 'w':w_init, 'b':b_init })
        #lin_module = mc.Linear(output_size=1, initializers={ 'w':w_init, 'b':b_init }, act=None)
        
        outputs = lin_module(x)
        self.log_tensor(outputs, name='outputs')
        
        ##################################################
        
        ## tanh
        outputs = tf.nn.tanh(outputs)
        return outputs
    