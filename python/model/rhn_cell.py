from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math, numpy as np
from six.moves import xrange 
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell


use_weight_normalization_default = False
def linear(args, output_size, bias, bias_start=0.0, use_l2_loss = False, use_weight_normalization = use_weight_normalization_default, scope=None, timestep = -1, weight_initializer = None, orthogonal_scale_factor = 1.1,
  use_kronecker_reparameterization=False): 
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".
    use_kronecker_reparameterization: reparameterizes weight matrix with 2x2 krocker product

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  # assert args #was causing error in upgraded tensorflow
  if not isinstance(args, (list, tuple)):
    args = [args]


  if use_kronecker_reparameterization:
    use_weight_normalization = False #we don't want to use weight norm with kronecker matrices

  if len(args) > 1 and use_weight_normalization: raise ValueError('you can not use weight_normalization with multiple inputs because the euclidean norm will be incorrect -- besides, you should be using multiple integration instead!!!')

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
      total_arg_size += shape[1]

  if use_l2_loss:
    l_regularizer = tf.contrib.layers.l2_regularizer(1e-5)
  else:
    l_regularizer = None

  # Now the computation.
  with tf.variable_scope(scope or "Linear"):
    if use_kronecker_reparameterization:
      if len(shapes) > 1:
        kro_matrix_list = []
        for i,shape in enumerate(shapes):
          kro_matrix_list.append(
            _kronecker_product(shape[1], output_size, name="Matrix.{}".format(i)))
        matrix = tf.concat(kro_matrix_list, axis=0)

      else:
        matrix = _kronecker_product(total_arg_size, output_size, name="Matrix")
    else:
      matrix = tf.get_variable("Matrix", [total_arg_size, output_size], 
                      initializer = tf.uniform_unit_scaling_initializer(), regularizer = l_regularizer)
    if use_weight_normalization: matrix = weight_normalization(matrix, timestep = timestep)





    if len(args) == 1:
      res = tf.matmul(args[0], matrix)
    else:
      res = tf.matmul(tf.concat(args,1), matrix)

    if not bias:
      return res
    bias_term = tf.get_variable("Bias", [output_size],
                                initializer=tf.constant_initializer(bias_start), regularizer = l_regularizer)

  return res + bias_term

class HighwayRNNCell(RNNCell):
  """Highway RNN Network with multiplicative_integration"""

  def __init__(self, num_units, 
               num_highway_layers=3, 
               use_inputs_on_each_layer=False,
               use_kronecker_reparameterization=False):
    self._num_units = num_units
    self.num_highway_layers = num_highway_layers
    self.use_inputs_on_each_layer = use_inputs_on_each_layer
    self._use_kronecker_reparameterization=use_kronecker_reparameterization

    if self._use_kronecker_reparameterization:
      tf.logging.warn("Using Kronecker Reparmeterization on Highway RNN Cell")


  @property
  def input_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def __call__(self, inputs, state, timestep = 0, scope=None):
    current_state = state
    for highway_layer in xrange(self.num_highway_layers):
      with tf.variable_scope('highway_factor_'+str(highway_layer)):
        if self.use_inputs_on_each_layer or highway_layer == 0:
          highway_factor = tf.tanh(linear([inputs, current_state], self._num_units, True,
            use_kronecker_reparameterization=self._use_kronecker_reparameterization))
        else:
          highway_factor = tf.tanh(linear([current_state], self._num_units, True,
            use_kronecker_reparameterization=self._use_kronecker_reparameterization))
      with tf.variable_scope('gate_for_highway_factor_'+str(highway_layer)):
        if self.use_inputs_on_each_layer or highway_layer == 0:
          gate_for_highway_factor = tf.sigmoid(linear([inputs, current_state], self._num_units,
           True, -3.0, use_kronecker_reparameterization=self._use_kronecker_reparameterization))
        else:
          gate_for_highway_factor = tf.sigmoid(linear([current_state], self._num_units, True,
           -3.0, use_kronecker_reparameterization=self._use_kronecker_reparameterization))

        gate_for_hidden_factor = 1.0 - gate_for_highway_factor

      current_state = highway_factor * gate_for_highway_factor + current_state * gate_for_hidden_factor

    return current_state, current_state