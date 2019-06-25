from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os, sys, time, collections
import numpy as np
import sonnet as snt
import tensorflow as tf

from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.ops import variable_scope as vs
import tensorflow.contrib.layers as layers

import utils as U

from model.rwa_cell import RWACell
from nlp.tensorflow_with_latest_papers.rnn_cell_modern import HighwayRNNCell

from rum.RUM import RUMCell

# from nlp.rhn.rhn import RHNCell2 as RHNCell
# from nlp.tensorflow_with_latest_papers import rnn_cell_modern
# from nlp.rnn_cells.MultiplicativeLSTM import MultiplicativeLSTMCell
# from nlp.rnn_cells.sru import SRUCell
# from nlp.rwa.rda_cell import RDACell
# from nlp.ran.ran_cell import RANCell

###############################################################################

def rnn_unit(args):
    kwargs = {}
    if args.unit=='snt.lstm':
        rnn = snt.LSTM
        kwargs = { 'forget_bias':args.FLAGS.forget_bias }
    elif args.unit=='lstm':
        rnn = tf.nn.rnn_cell.LSTMCell
        kwargs = { 'forget_bias':args.FLAGS.forget_bias, 'reuse':False, 'state_is_tuple':True }
    elif args.unit=='snt.gru':
        rnn = snt.GRU
    elif args.unit=='gru':
        rnn = tf.nn.rnn_cell.GRUCell
        kwargs = { 'reuse':False }
    elif args.unit=='rum':
        rnn = RUMCell
    elif args.unit=='rwa':
        rnn = RWACell
#         decay_rate = [0.0]*args.rnn_size
#         #decay_rate = [0.693/10]*150 + [0.0]*150
#         #decay_rate = [0.693]*75 + [0.693/10]*75 + [0.693/100]*75 + [0.0]*75
#         decay_rate = tf.Variable(tf.constant(decay_rate, dtype=tf.float32), trainable=True, name='decay_rate')
#         
#         #std = 0.001
#         #decay_rate = tf.get_variable('decay_rate', shape=[args.rnn_size], initializer=tf.truncated_normal_initializer(mean=2*std,stddev=std))
#         
#         kwargs = { 'decay_rate':decay_rate }
    elif args.unit=='rwa_bn':
        rnn = RWACell
        kwargs = { 'normalize':True }
    elif args.unit=='rhn':
        rnn = HighwayRNNCell
        kwargs = { 'num_highway_layers' : args.FLAGS.rhn_highway_layers,
                   'use_inputs_on_each_layer' : args.FLAGS.rhn_inputs,
                   'use_kronecker_reparameterization' : args.FLAGS.rhn_kronecker }
    elif args.unit=='rhn2':
        rnn = RHNCell
        # num_units, in_size, is_training, depth=3
        kwargs = { 'depth' : args.FLAGS.rhn_highway_layers }
#     elif args.unit=='ran':
#         rnn = RANCell
#     elif args.unit=='ran_ln':
#         rnn = RANCell
#         kwargs = { 'normalize':True }
#     elif args.unit=='rda':
#         rnn = RDACell
#     elif args.unit=='rda_bn':
#         rnn = RDACell
#         kwargs = { 'normalize':True }
#     elif args.unit=='lru':
#         rnn = LRUCell
#     elif args.unit=='sru':
#         rnn = SRUCell
#         #kwargs = { 'state_is_tuple':False }
#     elif args.unit=='hlstm':
#         rnn = HyperLnLSTMCell
#         kwargs = {'is_layer_norm':True,
#                   'state_is_tuple':False,
#                   'hyper_num_units':128,
#                   'hyper_embedding_size':32,
#                   }
#     elif args.unit=='mlstm':
#         rnn = MultiplicativeLSTMCell
#         kwargs = { 'forget_bias':args.forget_bias }
    return rnn, kwargs

def get_initial_state(cell, args, batch_size=None):
    if batch_size==None:
        batch_size = tf.shape(args.inputs)[0]
    
    if args.FLAGS.train_initial_state:
        if args.unit.startswith('snt'):
            return cell.initial_state(batch_size, tf.float32, trainable=True)
        print('TRAINABLE INITIAL STATE NOT YET IMPLEMENTED FOR: {} !!!'.format(args.unit))
#         else:
#             initializer = r2rt.make_variable_state_initializer()
#             return r2rt.get_initial_cell_state(cell, initializer, args.batch_size, tf.float32)
    return cell.zero_state(batch_size, tf.float32)

def create_rnn_cell(args, scope=None, batch_size=None):
    rnn, kwargs = rnn_unit(args)
    
    if scope!=None:
        with tf.variable_scope(scope):
            cell = rnn(args.rnn_size, **kwargs)
            initial_state = get_initial_state(cell, args, batch_size=batch_size)
    else:
        cell = rnn(args.rnn_size, **kwargs)
        initial_state = get_initial_state(cell, args, batch_size=batch_size)
    
    #cell = tf.contrib.rnn.ResidualWrapper(cell)
    #cell = tf.contrib.rnn.HighwayWrapper(cell)
    #cell = tf.contrib.rnn.AttentionCellWrapper(cell, attn_length=10, attn_size=100)
    
    if args.rnn_dropout and abs(args.dropout)>0:
        if args.dropout<0:
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=args._keep_prob)
        else:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=args._keep_prob)
        
        ''' variational_recurrent '''
        #variational_recurrent=True
        
    return cell, initial_state

def get_final_state(final_state, unit=None):
        h,c = 0,1
        f = h
        if unit=='snt.lstm':
            return final_state[f]
        elif unit=='lstm':
            return final_state[f]
            #return final_state.c
        elif unit=='hlstm':
            return final_state[1]
        elif unit=='rda':
            return final_state.h
        elif unit.startswith('rwa'):
            try:
                return final_state.h
            except AttributeError:
                return final_state[2]
        else:
            return final_state
        
def collapse_final_state_layers(final_state_layers, unit=None):
    states = [get_final_state(s, unit) for s in final_state_layers]
    #return tf.concat(states, axis=1)
    return states[-1]

#####################################################

class DeepRNN(snt.AbstractModule):
    def __init__(self, 
                 FLAGS=None,
                 seq_len=None,
                 keep_prob=None,
                 forget_bias=0.,
                 pad=None,
                 name="deep_rnn"):
        super(DeepRNN, self).__init__(name=name)
        self.FLAGS = FLAGS
        self.rnn_size = FLAGS.rnn_size
        self.num_layers = FLAGS.rnn_layers
        self.batch_size = FLAGS.batch_size
        self.dropout = FLAGS.dropout
        self.rnn_dropout = FLAGS.rnn_dropout
        self.train_initial_state = FLAGS.train_initial_state
        self.unit = FLAGS.rnn_cell
        self._seq_len = seq_len
        self._keep_prob = keep_prob
        self.forget_bias = forget_bias
        self.pad = pad if pad else FLAGS.pad
        self.name = name
        
        with self._enter_variable_scope():
            if keep_prob is None:
                self._keep_prob = tf.placeholder_with_default(1.0-abs(self.dropout), shape=())
            if seq_len is None:
                #self._seq_len = tf.placeholder(tf.int32, [None])# [self.batch_size]
                self._seq_len = tf.placeholder(shape=(None,), dtype=tf.int32, name='seq_len')

    def _build(self, inputs):
        self.inputs = inputs#self.batch_size = tf.shape(inputs)[0]
        
        if self.num_layers > 1:
            cells = [create_rnn_cell(self, scope='layer{}'.format(i)) for i in range(self.num_layers)]
            cells, states = zip(*cells)
            cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            self._initial_rnn_state = tuple(states)
        else:
            cell, self._initial_rnn_state = create_rnn_cell(self)

        if self.rnn_dropout and self.dropout<0:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self._keep_prob)
        
        sequence_length = self._seq_len if self.pad=='post' else None
        
        output, self._final_rnn_state = tf.nn.dynamic_rnn(cell,
                                                          inputs,
                                                          dtype=tf.float32,
                                                          sequence_length=sequence_length,
                                                          initial_state=self._initial_rnn_state
                                                          )
        
        tf.summary.histogram('{}_output'.format(self.name), output)
        
        return output#, final_rnn_state
    
    @property
    def seq_len(self):
        return self._seq_len
    
    @property
    def keep_prob(self):
        return self._keep_prob
    
    @property
    def initial_rnn_state(self):
        self._ensure_is_connected()
        return self._initial_rnn_state
    
    @property
    def final_rnn_state(self):
        self._ensure_is_connected()
        if self.num_layers > 1:
            return collapse_final_state_layers(self._final_rnn_state, self.unit)
        else:
            return get_final_state(self._final_rnn_state, self.unit)


###############################################################################

class DeepBiRNN(snt.AbstractModule):
    def __init__(self,
                 FLAGS=None,
                 seq_len=None,
                 keep_prob=None,
                 forget_bias=0.,
                 pad=None,
                 name="deep_bi_rnn"):
        super(DeepBiRNN, self).__init__(name=name)
        self.FLAGS = FLAGS
        self.rnn_size = FLAGS.rnn_size
        self.num_layers = FLAGS.rnn_layers
        self.batch_size = FLAGS.batch_size
        self.dropout = FLAGS.dropout
        self.rnn_dropout = FLAGS.rnn_dropout
        self.train_initial_state = FLAGS.train_initial_state
        self.unit = FLAGS.rnn_cell
        self._seq_len = seq_len
        self._keep_prob = keep_prob
        self.forget_bias = forget_bias
        self.pad = pad if pad else FLAGS.pad
        self.name = name
        
        with self._enter_variable_scope():
            if keep_prob is None:
                self._keep_prob = tf.placeholder_with_default(1.0-abs(self.dropout), shape=())
            if seq_len is None:
                #self._seq_len = tf.placeholder(tf.int32, [None])# [self.batch_size]
                self._seq_len = tf.placeholder(shape=(None,), dtype=tf.int32, name='seq_len')

    def _build(self, inputs):
        self.inputs = inputs
        
        with tf.variable_scope('fwd'):
            self.fwd_rnn = DeepRNN(FLAGS=self.FLAGS, seq_len=self._seq_len, keep_prob=self._keep_prob, pad=self.pad)
            fwd_outputs = self.fwd_rnn(inputs)
            
        with tf.variable_scope('bwd'):
            self.bwd_rnn = DeepRNN(FLAGS=self.FLAGS, seq_len=self._seq_len, keep_prob=self._keep_prob, pad=self.pad)
            bwd_outputs = self.bwd_rnn(padded_reverse(inputs, self._seq_len, pad=self.pad))
            bwd_outputs = padded_reverse(bwd_outputs, self._seq_len, pad=self.pad)
        
        outputs = tf.concat([fwd_outputs, bwd_outputs], axis=2)
        
        tf.summary.histogram('{}_outputs'.format(self.name), outputs)
        
        return outputs
    
    @property
    def seq_len(self):
        return self._seq_len
    
    @property
    def keep_prob(self):
        return self._keep_prob
    
    @property
    def final_rnn_state(self):
        self._ensure_is_connected()
#         fwd = collapse_final_state_layers(self.output_state_fw, self.unit)
#         bwd = collapse_final_state_layers(self.output_state_bw, self.unit)
        fwd = self.fwd_rnn.final_rnn_state
        bwd = self.bwd_rnn.final_rnn_state
        #return tf.concat([fwd, bwd], axis=1)
        return tf.concat([fwd, bwd], axis=2)
    

###############################################################################
''' multi-layer bi-directional dynamic rnn '''

class DeepBiRNN_v1(snt.AbstractModule):
    def __init__(self,
                 FLAGS=None,
                 seq_len=None,
                 keep_prob=None,
                 forget_bias=0.,
                 pad=None,
                 name="deep_bi_rnn_v1"):
        super(DeepBiRNN_v1, self).__init__(name=name)
        self.FLAGS = FLAGS
        self.rnn_size = FLAGS.rnn_size
        self.num_layers = FLAGS.rnn_layers
        self.batch_size = FLAGS.batch_size
        self.dropout = FLAGS.dropout
        self.rnn_dropout = FLAGS.rnn_dropout
        self.train_initial_state = FLAGS.train_initial_state
        self.unit = FLAGS.rnn_cell
        self._seq_len = seq_len
        self._keep_prob = keep_prob
        self.forget_bias = forget_bias
        self.pad = pad if pad else FLAGS.pad
        self.name = name
        
        with self._enter_variable_scope():
            if keep_prob is None:
                self._keep_prob = tf.placeholder_with_default(1.0-abs(self.dropout), shape=())
            if seq_len is None:
                #self._seq_len = tf.placeholder(tf.int32, [self.batch_size])
                self._seq_len = tf.placeholder(shape=(None,), dtype=tf.int32, name='seq_len')

    def _build(self, inputs):
        self.inputs = inputs
        
        with tf.variable_scope('fwd'):
            cells_fw = [create_rnn_cell(self, scope='layer{}'.format(i)) for i in range(self.num_layers)]
        with tf.variable_scope('bwd'):
            cells_bw = [create_rnn_cell(self, scope='layer{}'.format(i)) for i in range(self.num_layers)]
        
        cells_fw, initial_states_fw = zip(*cells_fw)
        cells_bw, initial_states_bw = zip(*cells_bw)
        
        cells_fw = list(cells_fw)
        cells_bw = list(cells_bw)
        initial_states_fw = list(initial_states_fw)
        initial_states_bw = list(initial_states_bw)
        
        sequence_length = self._seq_len if self.pad=='post' else None
            
        outputs, self.output_state_fw, self.output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw, cells_bw, inputs,
                                                                                                             initial_states_fw=initial_states_fw,
                                                                                                             initial_states_bw=initial_states_bw,
                                                                                                             sequence_length=sequence_length,
                                                                                                             dtype=tf.float32)
        return outputs
    
    @property
    def seq_len(self):
        return self._seq_len
    
    @property
    def keep_prob(self):
        return self._keep_prob
    
    @property
    def final_rnn_state(self):
        self._ensure_is_connected()
        fwd = collapse_final_state_layers(self.output_state_fw, self.unit)
        bwd = collapse_final_state_layers(self.output_state_bw, self.unit)
        return tf.concat([fwd, bwd], axis=1)


###############################################################################

''' sonnet wrapper for bidirectional_rnn (below) '''
class BiRNN(snt.AbstractModule):
    def __init__(self,
                 FLAGS=None,
                 seq_len=None,
                 keep_prob=None,
                 pad=None,
                 name="BiRNN"):
        super(BiRNN, self).__init__(name=name)
        self.FLAGS = FLAGS
        self.rnn_size = FLAGS.rnn_size
        self.unit = FLAGS.rnn_cell
        self._seq_len = seq_len
        self._keep_prob = keep_prob
        self.pad = pad if pad else FLAGS.pad
        self.dropout = FLAGS.dropout
        self.rnn_dropout = FLAGS.rnn_dropout
    
    def _build(self, inputs):
        cell, _ = create_rnn_cell(self, #scope=scope, 
                                  batch_size=tf.shape(inputs)[0])
        output, self._final_rnn_state = bidirectional_rnn(
            cell, cell,
            inputs, 
            input_lengths=self.seq_len,
            pad=self.pad,
            #scope=scope
            )
        
        return output
        
    @property
    def seq_len(self):
        return self._seq_len
    
    @property
    def keep_prob(self):
        return self._keep_prob
    
    @property
    def final_rnn_state(self):
        self._ensure_is_connected()
        #return self._final_rnn_state
        return get_final_state(self._final_rnn_state, self.unit)

''' https://github.com/davidsvaughn/hierarchical-attention-networks/blob/master/model_components.py
'''

try:
    from tensorflow.contrib.rnn import LSTMStateTuple
except ImportError:
    LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple
    
def bidirectional_rnn(cell_fw, cell_bw, 
                      inputs_embedded, 
                      input_lengths=None,
                      pad='post',
                      scope=None):
    """Bidirectional RNN with concatenated outputs and states"""
    with tf.variable_scope(scope or "birnn") as scope:
        ((fw_outputs,
          bw_outputs),
         (fw_state,
          bw_state)) = (
            #tf.nn.bidirectional_dynamic_rnn(
            bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=inputs_embedded,
                sequence_length=input_lengths,
                pad=pad,
                dtype=tf.float32,
                swap_memory=True,
                scope=scope))
          
        outputs = tf.concat((fw_outputs, bw_outputs), 2)
        
#         sum = tf.add(fw_outputs, bw_outputs)
#         diff = tf.subtract(fw_outputs, bw_outputs)
#         prod = tf.multiply(fw_outputs, bw_outputs)
#         outputs = tf.concat((outputs, sum, diff, prod), 2)
        
        tf.summary.histogram('{}_outputs'.format('bidirectional_rnn'), outputs)

        def concatenate_state(fw_state, bw_state):
            if isinstance(fw_state, LSTMStateTuple):
                state_c = tf.concat(
                    (fw_state.c, bw_state.c), 1, name='bidirectional_concat_c')
                state_h = tf.concat(
                    (fw_state.h, bw_state.h), 1, name='bidirectional_concat_h')
                state = LSTMStateTuple(c=state_c, h=state_h)
                return state
            elif isinstance(fw_state, tf.Tensor):
                state = tf.concat((fw_state, bw_state), 1,
                                  name='bidirectional_concat')
                return state
            elif (isinstance(fw_state, tuple) and
                    isinstance(bw_state, tuple) and
                    len(fw_state) == len(bw_state)):
                # multilayer
                state = tuple(concatenate_state(fw, bw)
                              for fw, bw in zip(fw_state, bw_state))
                return state

            else:
                raise ValueError(
                    'unknown state type: {}'.format((fw_state, bw_state)))


        state = concatenate_state(fw_state, bw_state)
        return outputs, state

###############################################################################

from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs

# pylint: disable=protected-access
_like_rnncell = rnn_cell_impl._like_rnncell
# pylint: enable=protected-access

def padded_reverse(x, seq_len, batch_dim=0, seq_dim=1, pad='post'):
    if pad=='pre': x = tf.reverse(x, [seq_dim])
    x = tf.reverse_sequence(x, seq_len, batch_dim=batch_dim, seq_dim=seq_dim)
    if pad=='pre': x = tf.reverse(x, [seq_dim])
    return x

def bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=None,
                              pad='post',
                              initial_state_fw=None, initial_state_bw=None,
                              dtype=None, parallel_iterations=None,
                              swap_memory=False, time_major=False, scope=None):
    """Creates a dynamic version of bidirectional recurrent neural network.
    
    Takes input and builds independent forward and backward RNNs. The input_size
    of forward and backward cell must match. The initial state for both directions
    is zero by default (but can be set optionally) and no intermediate states are
    ever returned -- the network is fully unrolled for the given (passed in)
    length(s) of the sequence(s) or completely unrolled if length(s) is not
    given.
    
    Args:
      cell_fw: An instance of RNNCell, to be used for forward direction.
      cell_bw: An instance of RNNCell, to be used for backward direction.
      inputs: The RNN inputs.
        If time_major == False (default), this must be a tensor of shape:
          `[batch_size, max_time, ...]`, or a nested tuple of such elements.
        If time_major == True, this must be a tensor of shape:
          `[max_time, batch_size, ...]`, or a nested tuple of such elements.
      sequence_length: (optional) An int32/int64 vector, size `[batch_size]`,
        containing the actual lengths for each of the sequences in the batch.
        If not provided, all batch entries are assumed to be full sequences; and
        time reversal is applied from time `0` to `max_time` for each sequence.
      initial_state_fw: (optional) An initial state for the forward RNN.
        This must be a tensor of appropriate type and shape
        `[batch_size, cell_fw.state_size]`.
        If `cell_fw.state_size` is a tuple, this should be a tuple of
        tensors having shapes `[batch_size, s] for s in cell_fw.state_size`.
      initial_state_bw: (optional) Same as for `initial_state_fw`, but using
        the corresponding properties of `cell_bw`.
      dtype: (optional) The data type for the initial states and expected output.
        Required if initial_states are not provided or RNN states have a
        heterogeneous dtype.
      parallel_iterations: (Default: 32).  The number of iterations to run in
        parallel.  Those operations which do not have any temporal dependency
        and can be run in parallel, will be.  This parameter trades off
        time for space.  Values >> 1 use more memory but take less time,
        while smaller values use less memory but computations take longer.
      swap_memory: Transparently swap the tensors produced in forward inference
        but needed for back prop from GPU to CPU.  This allows training RNNs
        which would typically not fit on a single GPU, with very minimal (or no)
        performance penalty.
      time_major: The shape format of the `inputs` and `outputs` Tensors.
        If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
        If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
        Using `time_major = True` is a bit more efficient because it avoids
        transposes at the beginning and end of the RNN calculation.  However,
        most TensorFlow data is batch-major, so by default this function
        accepts input and emits output in batch-major form.
      scope: VariableScope for the created subgraph; defaults to
        "bidirectional_rnn"
    
    Returns:
      A tuple (outputs, output_states) where:
        outputs: A tuple (output_fw, output_bw) containing the forward and
          the backward rnn output `Tensor`.
          If time_major == False (default),
            output_fw will be a `Tensor` shaped:
            `[batch_size, max_time, cell_fw.output_size]`
            and output_bw will be a `Tensor` shaped:
            `[batch_size, max_time, cell_bw.output_size]`.
          If time_major == True,
            output_fw will be a `Tensor` shaped:
            `[max_time, batch_size, cell_fw.output_size]`
            and output_bw will be a `Tensor` shaped:
            `[max_time, batch_size, cell_bw.output_size]`.
          It returns a tuple instead of a single concatenated `Tensor`, unlike
          in the `bidirectional_rnn`. If the concatenated one is preferred,
          the forward and backward outputs can be concatenated as
          `tf.concat(outputs, 2)`.
        output_states: A tuple (output_state_fw, output_state_bw) containing
          the forward and the backward final states of bidirectional rnn.
    
    Raises:
      TypeError: If `cell_fw` or `cell_bw` is not an instance of `RNNCell`.
    """

    if not _like_rnncell(cell_fw):
        raise TypeError("cell_fw must be an instance of RNNCell")
    if not _like_rnncell(cell_bw):
        raise TypeError("cell_bw must be an instance of RNNCell")
    
    def _reverse(input_, seq_lengths, seq_dim, batch_dim):
        if seq_lengths is not None:
            #return array_ops.reverse_sequence(input=input_, seq_lengths=seq_lengths, seq_dim=seq_dim, batch_dim=batch_dim)
            return padded_reverse(input_, seq_lengths, batch_dim=batch_dim, seq_dim=seq_dim, pad=pad)
        else:
            return array_ops.reverse(input_, axis=[seq_dim])
            
    with vs.variable_scope(scope or "bidirectional_rnn"):
        
        rnn_seq_length = sequence_length if pad=='post' else None
        
        # Forward direction
        with vs.variable_scope("fw") as fw_scope:
            output_fw, output_state_fw = dynamic_rnn(
                cell=cell_fw, inputs=inputs, sequence_length=rnn_seq_length,
                initial_state=initial_state_fw, dtype=dtype,
                parallel_iterations=parallel_iterations, swap_memory=swap_memory,
                time_major=time_major, 
                scope=fw_scope
                )

        # Backward direction
        if not time_major:
            time_dim = 1
            batch_dim = 0
        else:
            time_dim = 0
            batch_dim = 1

        with vs.variable_scope("bw") as bw_scope:
            inputs_reverse = _reverse(
                inputs, seq_lengths=sequence_length,
                seq_dim=time_dim, batch_dim=batch_dim)
            tmp, output_state_bw = dynamic_rnn(
                cell=cell_bw, inputs=inputs_reverse, sequence_length=rnn_seq_length,
                initial_state=initial_state_bw, dtype=dtype,
                parallel_iterations=parallel_iterations, swap_memory=swap_memory,
                time_major=time_major, 
                scope=bw_scope
                )

            output_bw = _reverse(
                tmp, seq_lengths=sequence_length,
                seq_dim=time_dim, batch_dim=batch_dim)

    outputs = (output_fw, output_bw)
    output_states = (output_state_fw, output_state_bw)
    
    return (outputs, output_states)
