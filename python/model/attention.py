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

from model import model_components as mc

import utils as U

def softmax_rescale(x, mask, axis=-1):
    u = tf.multiply(x, mask)
    v = tf.reduce_sum(u, axis=axis, keepdims=True)
    ## fix div by 0
    v = v + tf.cast( tf.equal( v, tf.constant( 0, dtype=tf.float32 ) ), tf.float32)
    ##
    return u/v

def softmax_T(x, axis=-1, T=1.0, trainable=False):
    x_max = tf.reduce_max(x, axis=axis, keepdims=True)
    x = x - x_max
    
    if T!=None:
        T = tf.get_variable('T', shape=[1], initializer=tf.constant_initializer(T), dtype=tf.float32, trainable=trainable)
        x = x/T
        
    ex = tf.exp(x)
    es = tf.reduce_sum(ex, axis=axis, keepdims=True)
    
    ## fix div by 0
    es = es + tf.cast( tf.equal( es, tf.constant( 0, dtype=tf.float32 ) ), tf.float32)

    return ex/es

''' softmax with 0-padding re-normalization '''  
def softmask(x, axis=-1, mask=None, T=None):
    x_max = tf.reduce_max(x, axis=axis, keepdims=True)
    x = x - x_max
#     if T!=None:
#         if not tf.is_numeric_tensor(T):
#             T = tf.get_variable('T', shape=[1], initializer=tf.constant_initializer(T), dtype=tf.float32, trainable=True)
#             T = tf.get_variable('T', shape=[1], initializer=tf.constant_initializer(1.0), dtype=tf.float32, trainable=True)
#         x = x/T
    ex = tf.exp(x)
    if mask!=None: ex = tf.multiply(ex, mask)
    es = tf.reduce_sum(ex, axis=axis, keepdims=True)
    ## fix div by 0
    es = es + tf.cast( tf.equal( es, tf.constant( 0, dtype=tf.float32 ) ), tf.float32)
    #     if mask!=None: es = es + tf.cast(tf.reduce_sum(mask, axis=-1, keep_dims=True)==0, tf.float32)
    return ex/es

def default_initializers(std=None, bias=None):
    if std != None:
        w_init = tf.truncated_normal_initializer(stddev=std)
    else:
        w_init = tf.glorot_uniform_initializer(dtype=tf.float32)
    
    if bias != None:
        b_init = tf.constant_initializer(bias)
    else:
        b_init = w_init
        
    return w_init, b_init

class AttnBase(snt.AbstractModule):
    def __init__(self, 
                 FLAGS,
                 seq_len=None,
                 renorm=True,
                 name="AttnBase"):
        super(AttnBase, self).__init__(name=name)
        self.FLAGS = FLAGS
        self.pad = FLAGS.pad
        self.seq_len = seq_len
        self.renorm = renorm
        self.t = []
        self.ts = []
        self._penalty = tf.constant( 0, dtype=tf.float32 )
    
    def log_tensor(self, x, name, prefix='attn_'):
        self.t.append((U.string2tf(prefix + name), x))
        self.log_tensor_shape(x, name)
        
    def log_tensor_shape(self, x, name, prefix='attn_'):
        self.ts.append((U.string2tf(prefix + name), tf.shape(x)))
        
    def get_mask(self):
        ##mask = tf.cast(tf.abs(tf.reduce_sum(inputs, axis=2))>0, tf.float32)
        mask = U.make_sequence_mask(self.seq_len, pad=self.pad, axis=[1])
        self.log_tensor(mask, name='mask')
        return mask
    
    @property
    def tensors(self):
        return self.t
    
    @property
    def tensor_shapes(self):
        return self.ts
    
    @property
    def penalty(self):
        #return self._penalty
        return tf.reduce_sum(self._penalty)
    
    @property
    def alpha(self):
        self._ensure_is_connected()
        try:
            if self.FLAGS.attn_vis:
                return self._alpha
            else:
                return tf.constant([])
        except AttributeError:
            print('AttnBase.alpha AttributeError !!!')
            return None

''' https://github.com/ilivans/tf-rnn-attention/blob/master/attention.py '''
class Attention(AttnBase):
    def __init__(self, FLAGS,
                 seq_len=None,
                 apply=True,
                 name="Attention"):
        super(Attention, self).__init__(name=name,
                                        FLAGS=FLAGS,
                                        seq_len=seq_len)
        self.apply = apply
    
    def _build(self, inputs):
        A = self.FLAGS.attn_size
        D = inputs.shape[-1].value # D value - hidden size of the RNN layer
        self.log_tensor(inputs, name='inputs')
        
        ''' Linear Projection Layer (Key, Ws1) '''
        w_init, b_init = default_initializers(std=self.FLAGS.attn_std, bias=self.FLAGS.attn_b)
        lin_module = mc.Linear(output_size=A, initializers={ 'w':w_init, 'b':b_init })#w_init=w_init, b_init=b_init)        
        K = lin_module(inputs)
        
        ''' Query '''
        q = tf.get_variable('q', shape=[A], initializer=w_init)
        beta = tf.einsum('ijk,k->ij', K, q)#beta = tf.tensordot(k, q, axes=1)
        self.log_tensor(K, name='K')
        self.log_tensor(beta, name='beta')
        
        ''' softmax '''
        with vs.variable_scope("alpha"):
            if self.FLAGS.attn_temp==1:
                alpha = tf.nn.softmax(beta, axis=1)
            else:
                alpha = softmax_T(beta, axis=1, T=self.FLAGS.attn_temp)
            
            ########################################
            
            if self.renorm: alpha = softmax_rescale(alpha, mask=self.get_mask(), axis=1)
            self.log_tensor(alpha, name='alpha')
            
        if not self.apply:
            return alpha
        
        ''' apply attn weights '''
        #w = inputs * tf.expand_dims(alpha, -1); output = tf.reduce_sum(w, 1)
        #output = tf.reduce_sum(inputs * tf.expand_dims(alpha, -1), 1)
        output = tf.einsum('ijk,ij->ik', inputs, alpha)
        
        ''' return '''
        self._alpha = alpha
        self.log_tensor(output, name='output')
        return output


''' https://github.com/flrngel/Self-Attentive-tensorflow/blob/master/model.py
    inputs == v
'''
class Attention2D(AttnBase):
    def __init__(self, FLAGS,
                 seq_len=None,
                 name="Attention2D"):
        super(Attention2D, self).__init__(name=name,
                                          FLAGS=FLAGS,
                                          seq_len=seq_len)
    
    def _build(self, inputs):
        self.log_tensor(inputs, name='inputs')
        self.batch_size = tf.shape(inputs)[0]
        A = self.FLAGS.attn_size
        R = self.FLAGS.attn_depth
        D = inputs.shape[-1].value # D value - hidden size of the RNN layer
        
        ''' Linear Projection Layer (Key, Ws1) '''
        w_init, b_init = default_initializers(std=self.FLAGS.attn_std, bias=self.FLAGS.attn_b)
        lin_module = mc.Linear(output_size=A, initializers={ 'w':w_init, 'b':b_init })#w_init=w_init, b_init=b_init)        
        K = lin_module(inputs)
        self.log_tensor(K, name='K')
        
        ''' Query '''
        Q = tf.get_variable('Q', shape=[A,R], initializer=w_init)
        Beta = tf.einsum('bij,jk->bik', K, Q)
        self.log_tensor(Q, name='Q')
        self.log_tensor(Beta, name='Beta')
#         ## old
#         q = tf.get_variable('q', shape=[A], initializer=w_init)
#         beta = tf.einsum('ijk,k->ij', K, q)#beta = tf.tensordot(k, q, axes=1)
#         self.log_tensor_shape(beta, name='beta')
        
        ''' softmax '''
        with vs.variable_scope("alpha"):
            Alpha = tf.nn.softmax(Beta, axis=1)
            
            if self.renorm: 
                mask = self.get_mask()
                Alpha = softmax_rescale(Alpha, mask=tf.expand_dims(mask, -1), axis=1)
                
            self.log_tensor(Alpha, name='Alpha')
#             ## old
#             alpha = tf.nn.softmax(beta, axis=1)
#             alpha = softmax_rescale(alpha, mask=mask, axis=1)
#             self.log_tensor_shape(alpha, name='alpha')
        
        ''' apply attn weights '''
        Output = tf.einsum('ijk,ijz->ikz', inputs, Alpha)
        self.log_tensor(Output, name='Output')
#         ## old
#         output = tf.einsum('ijk,ij->ik', inputs, alpha)
#         self.log_tensor_shape(output, name='output')
        
        #d = self.FLAGS.attn_depth * self.FLAGS.rnn_size * (2 if self.FLAGS.bidirectional else 1)
        d = tf.reduce_prod(tf.shape(Output)[1:])
        Output = tf.reshape(Output, [self.batch_size, d])#tf.reshape(Output, [-1, d])
        
        self.log_tensor(Output, name='Output_2')
        
        #################################
        ''' Loss '''
        A_T = Alpha
        A = tf.transpose(A_T, perm=[0, 2, 1])
        tile_eye = tf.tile(tf.eye(R), [self.batch_size, 1])
        tile_eye = tf.reshape(tile_eye, [-1, R, R])
        AA_T = tf.matmul(A, A_T) - tile_eye
        self._penalty = tf.square(tf.norm(AA_T, axis=[-2, -1], ord='fro'))
        
        #################################
        self._alpha = Alpha
        return Output
    

###################################################################################################
''' https://github.com/davidsvaughn/hierarchical-attention-networks/blob/master/model_components.py
'''
    
class task_specific_attention(AttnBase):
    def __init__(self, FLAGS,
                 name="task_specific_attention"):
        super(task_specific_attention, self).__init__(name=name,
                                                      FLAGS=FLAGS,
                                                      seq_len=seq_len)
        
    def _build(self, inputs):
        output_size = self.FLAGS.attn_size
        assert len(inputs.get_shape()) == 3 and inputs.get_shape()[-1].value is not None
    
        with tf.variable_scope(scope or 'attention') as scope:
            attention_context_vector = tf.get_variable(name='attention_context_vector',
                                                       shape=[output_size],
                                                       initializer=initializer,
                                                       dtype=tf.float32)
            input_projection = layers.fully_connected(inputs, output_size,
                                                      activation_fn=activation_fn,
                                                      scope=scope)
            
            keepdims=False
            vector_attn = tf.reduce_sum(tf.multiply(input_projection, attention_context_vector), axis=2, keepdims=keepdims)
            #mask = tf.cast(tf.abs(tf.reduce_sum(input_projection, axis=2, keepdims=keepdims))>0, tf.float32)
            
            ''' softmax '''
            attention_weights = tf.nn.softmax(vector_attn, axis=1)
            #attention_weights = tf.contrib.sparsemax.sparsemax(vector_attn)
            
            if self.renorm:
                attention_weights = softmax_rescale(attention_weights, mask=self.get_mask(), dim=1)
            
            if not keepdims:
                attention_weights = tf.expand_dims( attention_weights, -1)
            outputs = tf.reduce_sum(tf.multiply(inputs, attention_weights), axis=1)
            
            tf.summary.histogram('{}_outputs'.format('task_specific_attention'), outputs)
            
            self._alpha = attention_weights
            return outputs


class HighwayAttention(AttnBase):
    def __init__(self, FLAGS,
                 seq_len=None,
                 apply=True,
                 name="HighwayAttention"):
        super(HighwayAttention, self).__init__(name=name,
                                               FLAGS=FLAGS,
                                               seq_len=seq_len)
        self.apply = apply
    
    def _build(self, inputs):
        A = self.FLAGS.attn_size
        #A = self.FLAGS.attn_depth * self.FLAGS.rnn_size * (2 if self.FLAGS.bidirectional else 1)
        D = inputs.shape[-1].value # D value - hidden size of the RNN layer
        A = D
        
        ''' initializers '''
        w_init, b_init = default_initializers(std=self.FLAGS.attn_std, bias=self.FLAGS.attn_b)
        initializers={ 'w':w_init, 'b':b_init }
        
        ''' Linear Projection Layer (Key, Ws1) '''
#         lin_module = mc.Linear(output_size=A, initializers=initializers) 
#         K = lin_module(inputs)
        
        ''' highway layer ? '''
        highway = mc.Highway(output_size=A, 
                             initializers=initializers,
                             #bias=-1.0,
                             )
        K = highway(inputs)
        
        ''' Query '''
        q = tf.get_variable('q', shape=[A], initializer=w_init)
        beta = tf.einsum('ijk,k->ij', K, q)#beta = tf.tensordot(k, q, axes=1)
        self.log_tensor(K, name='K')
        self.log_tensor(beta, name='beta')
        
        ''' softmax '''
        with vs.variable_scope("alpha"):
            alpha = tf.nn.softmax(beta, axis=1)
            if self.renorm: alpha = softmax_rescale(alpha, mask=self.get_mask(), axis=1)
            self.log_tensor(alpha, name='alpha')
            
        if not self.apply:
            return alpha
        
        ''' apply attn weights '''
        output = tf.einsum('ijk,ij->ik', inputs, alpha)
#         output = tf.einsum('ijk,ij->ik', K, alpha)
        
        ''' return '''
        self.log_tensor(output, name='output')
        self._alpha = alpha
        return output


''' simple sonnet wrapper for rnn pooling'''
class Pool(snt.AbstractModule):
    def __init__(self,
                 FLAGS=None,
                 seq_len=None,
                 final_rnn_state=None,
                 name="pool"):
        super(Pool, self).__init__(name=name)
        self.FLAGS = FLAGS
        self.pad = FLAGS.pad
        self.seq_len = seq_len
        self.final_rnn_state = final_rnn_state
        self.name = name
        self.t = []
        self.ts = []
    
    def _build(self, inputs):
        if self.FLAGS.attn_size>0:
            with tf.variable_scope('attention') as scope:
                attn_module = Attention2D if self.FLAGS.attn_depth>1 else Attention
                self._attn = attn_module(self.FLAGS, self.seq_len)#, final_rnn_state=self.final_rnn_state)
                output = self._attn(inputs)
                self.update_tensors(self._attn)
            
        elif self.FLAGS.attn_size==0:
#             output = tf.reduce_mean(inputs, axis=-2)#output = tf.reduce_mean(inputs, axis=1)      
            if self.seq_len==None:
                output = tf.reduce_mean(inputs, axis=-2)#output = tf.reduce_mean(inputs, axis=1)
            else:
                output = U.dynamic_mean(inputs, self.seq_len, pad=self.pad)
        
        elif self.FLAGS.attn_size==-1:
            output = tf.reduce_max(inputs, axis=-2)
            
        else: ## self.FLAGS.attn_size == -2    otherwise just use final rnn state
            output = self.final_rnn_state
#             output = tf.gather_nd(inputs, tf.stack([tf.range(self.FLAGS.batch_size), self.seq_len-1], axis=1))
        
        tf.summary.histogram('{}_output'.format(self.name), output)

        return output
    
    @property
    def attn(self):
        self._ensure_is_connected()
        try:
            return self._attn
        except AttributeError:
            return None
        
    @property
    def alpha(self):
        self._ensure_is_connected()
        try:
            return self.attn.alpha
        except AttributeError:
            return tf.constant([])
        
    @property
    def penalty(self):
        self._ensure_is_connected()
        try:
            return self.attn.penalty
        except AttributeError:
            return tf.constant( 0, dtype=tf.float32 )
        
    @property
    def tensors(self):
        self._ensure_is_connected()
        return self.t
    
    @property
    def tensor_shapes(self):
        self._ensure_is_connected()
        return self.ts
    
    def update_tensors(self, mod):
        if mod.tensors is not None: 
            self.t.extend(mod.tensors)
        if mod.tensor_shapes is not None: 
            self.ts.extend(mod.tensor_shapes)