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


'''
https://github.com/deepmind/sonnet/blob/master/sonnet/examples/rnn_shakespeare.py
'''
def init_dict(initializer, keys):
    if initializer!=None:
        if U.isnum(initializer):
            initializer = tf.constant_initializer(initializer)
        return {k: initializer for k in keys}
    return None

class Linear(snt.AbstractModule):
    def __init__(self, output_size,
                 act=tf.tanh,
                 bias=True,
                 initializers=None,
                 name="linear"):
        super(Linear, self).__init__(name=name)
        self.output_size = output_size
        self.act = act
        self.bias = bias
        self.name = name
        self.w_init = None
        self.b_init = None
        if initializers is not None:
            if "w" in initializers:
                self.w_init = initializers["w"]
            if "b" in initializers and self.bias:
                self.b_init = initializers["b"]
                
    def _build(self, inputs):
        d = inputs.shape[-1].value     
        self.w = tf.get_variable('w', shape=[d, self.output_size], initializer=self.w_init)
        if self.bias:
            self.b = tf.get_variable('b', shape=[self.output_size], initializer=self.b_init)
        
        ''' build einsum index equation '''
        dim = len(inputs.shape)
        q1 = 'ij,jk'
        q2 = 'ik'
        for i in range(dim-2):
            c = chr(i+97)
            q1 = c+q1
            q2 = c+q2
        q = '{}->{}'.format(q1,q2)
        #print(q)
        
        ''' multiply '''
        output = tf.einsum(q, inputs, self.w)
        if self.bias:
            output = output + self.b
        
        ''' activation '''
        if self.act is not None:
            output = self.act(output)
      
        return output

class WordEmbed(snt.AbstractModule):
    def __init__(self, vocab_size=None, embed_dim=None, initial_matrix=None, trainable=True, name="word_embed"):
        super(WordEmbed, self).__init__(name=name)
        self._vocab_size = vocab_size# word_vocab.size
        self._embed_dim = embed_dim
        self._trainable = trainable
        if initial_matrix:
            self._vocab_size = initial_matrix.shape[0]
            self._embed_dim = initial_matrix.shape[1]
        
        with self._enter_variable_scope():# cuz in init (not build)...
            self._embedding = snt.Embed(vocab_size=self._vocab_size,
                                        embed_dim=self._embed_dim,
                                        trainable=self._trainable,
                                        name="internal_embed")
    
    # inputs shape = [batch_size, ?]
    # inputs = word_idx, output = input_embedded
    def _build(self, inputs):
        return self._embedding(inputs)

''' NOTE: RETURNS RESHAPED TENSOR!!! --> refactor? '''
class CharEmbed(snt.AbstractModule):
    def __init__(self, vocab_size, embed_dim, max_word_length=None, initializer=None, trainable=True, name="char_embed"):
        super(CharEmbed, self).__init__(name=name)
        self._vocab_size = vocab_size# char_vocab.size
        self._embed_dim = embed_dim
        self._max_word_length = max_word_length
        self._initializers = init_dict(initializer, ['embeddings'])
        self._trainable = trainable
        
        with self._enter_variable_scope():# cuz in init (not build)...
            self._char_embedding = snt.Embed(vocab_size=self._vocab_size,
                                             embed_dim=self._embed_dim,
                                             trainable=self._trainable,
                                             #initializers={'embeddings':tf.constant_initializer(0.)},
                                             name="internal_embed")
    
    # inputs shape = [batch_size, num_word_steps, max_word_length] (num_unroll_steps)
    # inputs = char ids, output = input_embedded
    ## or ##
    # inputs shape = [batch_size, num_sentence_steps, num_word_steps, max_word_length]
    # inputs = char ids, output = input_embedded
    
    def _build(self, inputs):
        output = self._char_embedding(inputs)
        
        max_word_length = self._max_word_length
        if max_word_length==None:
            max_word_length = tf.shape(inputs)[-1]
            
        ''' NOTE: RETURNS RESHAPED TENSOR!!! --> refactor? '''
            
        output = tf.reshape(output, [-1, max_word_length, self._embed_dim])
        self._clear_padding_op = tf.scatter_update(self._char_embedding.embeddings,
                                                   [0],
                                                   tf.constant(0.0, shape=[1, self._embed_dim]))
        return output
    
    @property
    def embeddings(self):
        self._ensure_is_connected()
        return self._char_embedding.embeddings
    
    @property
    def clear_padding_op(self):
        self._ensure_is_connected()
        return self._clear_padding_op
    
    def clear_padding(self, sess):
        sess.run(self.clear_padding_op)
      
    def initialize_to(self, sess, v):
        self._ensure_is_connected()
        sess.run(tf.assign(self.embeddings, v))


''' Time Delay Neural Network'''
class TDNN(snt.AbstractModule):
    def __init__(self, kernels, kernel_features, initializer=None, name="tdnn"):
        super(TDNN, self).__init__(name=name)
        self._kernels = kernels
        self._kernel_features = kernel_features
        self._initializers = init_dict(initializer, ['w','b'])
        assert len(self._kernels) == len(self._kernel_features), 'Kernel and Features must have the same size'
        
    def _build(self, inputs):
        #max_word_length = inputs.get_shape()[1]
        max_word_length = tf.shape(inputs)[1]#xxx1
        embed_size = inputs.get_shape()[-1]
        
        inputs = tf.expand_dims(inputs, 1)
        
        layers = []
        self.conv_layers = []
        for kernel_size, kernel_feature_size in zip(self._kernels, self._kernel_features):

            ## [batch_size x max_word_length x embed_size x kernel_feature_size]
            conv_fxn = snt.Conv2D(output_channels=kernel_feature_size,
                                  kernel_shape=[1, kernel_size],# ?? [kernel_size,1] ??
                                  initializers=self._initializers,
                                  padding='VALID')
            conv = conv_fxn(inputs)
            self.conv_layers.append(conv_fxn)
            
#             ## [batch_size x 1 x 1 x kernel_feature_size]
#             reduced_length = max_word_length - kernel_size + 1
#             pool = tf.nn.max_pool(tf.tanh(conv), 
#                                   ksize= [1,1,reduced_length,1], 
#                                   strides= [1,1,1,1],
#                                   padding= 'VALID')

            # https://stackoverflow.com/questions/43574076/tensorflow-maxpool-with-dynamic-ksize#xxx2
            pool = tf.reduce_max(tf.tanh(conv),
                                 axis=2,
                                 keepdims=True
                                 )
            layers.append(tf.squeeze(pool, [1, 2]))
            
        if len(self._kernels) > 1:
            output = tf.concat(layers, 1)
        else:
            output = layers[0]
        return output
    
    def initialize_conv_layer_to(self, sess, i, w, b):
        self._ensure_is_connected()
        sess.run(tf.assign(self.conv_layers[i].w, w))
        sess.run(tf.assign(self.conv_layers[i].b, b))

class Highway(snt.AbstractModule):
    def __init__(self, output_size,
                 num_layers=1,
                 bias=-2.0,
                 f=tf.nn.relu,
                 initializer=None,
                 initializers=None,
                 name="highway"):
        super(Highway, self).__init__(name=name)
        self._output_size = output_size
        self._num_layers = num_layers
        self._bias = bias
        self._f = f
        if initializers is not None:
            self._initializers = initializers
        else:
            self._initializers = init_dict(initializer, ['w','b'])
        
    def _build(self, inputs):
        self.lin_g, self.lin_t = [],[]
        for idx in range(self._num_layers):
            
#             ''' original (for char LM/embeddings) '''
#             lin_in_g = snt.Linear(output_size=self._output_size, initializers=self._initializers, name="lin_in_g")
#             lin_in_t = snt.Linear(output_size=self._output_size, initializers=self._initializers, name="lin_in_t")
            
            ''' test '''
            lin_in_g = Linear(output_size=self._output_size, act=None, initializers=self._initializers, name="lin_in_g")
            lin_in_t = Linear(output_size=self._output_size, act=None, initializers=self._initializers, name="lin_in_t")
            
            self.lin_g.append(lin_in_g)
            self.lin_t.append(lin_in_t)
            
            g = self._f(lin_in_g(inputs))
            t = tf.sigmoid(lin_in_t(inputs) + self._bias)

            output = t * g + (1. - t) * inputs
            inputs = output
            
        return output
    
    def initialize_lin_layers_to(self, sess, Lg, Lt):
        self._ensure_is_connected()
        i=0
        for g, t in zip(Lg, Lt):
            sess.run(tf.assign(self.lin_g[i].w, g[0]))
            sess.run(tf.assign(self.lin_g[i].b, g[1]))
            sess.run(tf.assign(self.lin_t[i].w, t[0]))
            sess.run(tf.assign(self.lin_t[i].b, t[1]))
            i+=1
        
###############################################################################

''' simple sonnet wrapper for dropout'''
class Dropout(snt.AbstractModule):
    def __init__(self, keep_prob=None, name="dropout"):
        super(Dropout, self).__init__(name=name)
        self._keep_prob = keep_prob
        
        if keep_prob is None:
            with self._enter_variable_scope():
                self._keep_prob = tf.placeholder_with_default(1.0, shape=())
    
    def _build(self, inputs):
        return tf.nn.dropout(inputs, keep_prob=self._keep_prob)
    
    @property
    def keep_prob(self):
        self._ensure_is_connected()
        return self._keep_prob


''' simple sonnet wrapper for reshaping'''
class Reshape(snt.AbstractModule):
    def __init__(self,
                 batch_size=128,
                 num_unroll_steps=None,
                 name="reshape"):
        super(Reshape, self).__init__(name=name)
        self.batch_size = batch_size
        self.num_unroll_steps = num_unroll_steps
    
    def _build(self, inputs):
        dim = inputs.get_shape().as_list()[1]
        return tf.reshape(inputs, [self.batch_size, self.num_unroll_steps, dim])


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


def char_cnn_embedding(inputs, 
                       char_vocab_size,
                       char_embed_size,
                       kernel_widths,
                       kernel_features,
                       sparse=True,
                       output_shape=None,
                       stack_output_dims=1,
                       max_word_length=None,
                       scope=None):
    ''' adding scope makes slower!?!?!?!?!? '''
#     with tf.variable_scope(scope or 'char_cnn_embedding') as scope:
        
    input_shape = tf.shape(inputs)
    flat_dim = tf.reduce_prod(input_shape[0:-1])
    if not max_word_length: max_word_length = input_shape[-1]
    
    ## char embed ##
    char_embed_module = CharEmbed(vocab_size=char_vocab_size,
                                  embed_dim=char_embed_size, 
                                  max_word_length=max_word_length,
                                  name='char_embed_b')
    outputs = char_embed_module(inputs)
    
    ## gather ##
    if sparse:
        bool_idx =  tf.not_equal( tf.reduce_max( tf.reshape(inputs, [-1, max_word_length]), axis=-1), tf.constant( 0, dtype=tf.int32 ))
        idx = tf.cast( tf.where(bool_idx), tf.int32 )
        outputs = tf.gather_nd(outputs, idx)
    
    ## tdnn ##
    tdnn_module = TDNN(kernel_widths,
                       kernel_features,
                       initializer=0,
                       name='TDNN')
    outputs = tdnn_module(outputs)
    
    #dim = tf.shape(outputs)[-1]
    #dim = sum(kernel_features)
    dim = outputs.get_shape().as_list()[-1]
    
    ## scatter ##
    if sparse:
        flat_shape = tf.cast( [ flat_dim, dim ] , tf.int32 )
        outputs = tf.scatter_nd(indices=idx,
                                updates=outputs,
                                shape=flat_shape)
    ## reshape ##
    if output_shape is None:
        #output_shape = [tf.reduce_prod(input_shape[:2]), input_shape[2], dim]
        output_shape = [tf.reduce_prod(input_shape[:stack_output_dims])]
        for i in range(stack_output_dims, input_shape.shape[0]-1):
            output_shape += [input_shape[i]]
        output_shape += [dim]
    outputs = tf.reshape(outputs, output_shape)
    return outputs


class Combine(snt.AbstractModule):
    def __init__(self, 
                 aux_inputs,
                 aux_dim,
                 input_dim,
                 FLAGS=None,
                 name="Combine"):
        super(Combine, self).__init__(name=name)
        self.t = []
        self.ts = []
        self.aux_inputs = aux_inputs
        self.log_tensor(self.aux_inputs, name='aux_inputs')
        self.aux_dim = aux_dim
        self.input_dim = input_dim
    
    def log_tensor(self, x, name, prefix='comb_'):
        self.t.append((U.string2tf(prefix + name), x))
        self.log_tensor_shape(x, name)
        
    def log_tensor_shape(self, x, name, prefix='comb_'):
        self.ts.append((U.string2tf(prefix + name), tf.shape(x)))
    
    @property
    def tensors(self):
        return self.t
    
    @property
    def tensor_shapes(self):
        return self.ts
    
    def concat(self, inputs):
        output = tf.concat([inputs, self.aux_inputs], axis=1)
        self.log_tensor(output, name='output')
        ############
        (batch_dim, vector_dim) = tf.unstack(tf.shape(self.aux_inputs))
        output_shape = tf.cast( [batch_dim, self.input_dim + self.aux_dim] , tf.int32 )
        output = tf.reshape( output, shape=output_shape )
        ############
        output = tf.layers.dense(output, 100, activation=tf.nn.relu)
        return output
    
    def gate1(self, inputs):
        lin_module = Linear(output_size=self.aux_dim)
        K = lin_module(inputs)
        beta = tf.einsum('ij,ij->i', K, self.aux_inputs)
        output = tf.concat([inputs, self.aux_inputs * tf.expand_dims(beta, -1)], axis=1)
        
        ############
        (batch_dim, vector_dim) = tf.unstack(tf.shape(self.aux_inputs))
        output_shape = tf.cast( [batch_dim, self.input_dim + self.aux_dim] , tf.int32 )
        output = tf.reshape( output, shape=output_shape )
        
        ############
        output = tf.layers.dense(output, 100, activation=tf.nn.relu)
        return output
    
    def gate2(self, inputs):
        lin_module = Linear(output_size=self.aux_dim)
        K = lin_module(inputs)
        beta = tf.einsum('ij,ij->i', K, self.aux_inputs)
        output = K + self.aux_inputs*tf.expand_dims(beta, -1)
        
        ############
        (batch_dim, vector_dim) = tf.unstack(tf.shape(self.aux_inputs))
        output_shape = tf.cast( [batch_dim, self.aux_dim] , tf.int32 )
        output = tf.reshape( output, shape=output_shape )
        
        ############
        #output = tf.layers.dense(output, 100, activation=tf.nn.relu)
        return output
    
    def gate3(self, inputs):
        d = 100
        
        (batch_dim, vector_dim) = tf.unstack(tf.shape(self.aux_inputs))
        batch_dim = inputs.shape[0].value
        
        #aux_shape = tf.cast( [batch_dim, vector_dim] , tf.int32 )
        #self.aux_inputs = tf.reshape( self.aux_inputs, shape=aux_shape )
        
        self.aux_inputs.set_shape(tf.TensorShape([batch_dim, 512]))
        
        zW = Linear(output_size=d, act=None, bias=False)
        z1 = zW(inputs)
        zU = Linear(output_size=d, act=None, bias=False)
        z2 = zU(self.aux_inputs)
        z = tf.sigmoid(z1 + z2)
        
        xW = Linear(output_size=d, act=tf.nn.tanh, bias=True)
        x1 = xW(inputs)
        xU = Linear(output_size=d, act=tf.nn.tanh, bias=True)
        x2 = xU(self.aux_inputs)

        output = z*x1 + (1-z)*x2
        #output = x1 + z**x2
        
        ############
        (batch_dim, vector_dim) = tf.unstack(tf.shape(self.aux_inputs))
        output_shape = tf.cast( [batch_dim, d] , tf.int32 )
        output = tf.reshape( output, shape=output_shape )
        ############

        return output
    
    def _build(self, inputs):
        #return self.concat(inputs)
        #return self.gate1(inputs)
        #return self.gate2(inputs)
        return self.gate3(inputs)
        
        
        ##################################################################
        
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