""" Custom Keras Inputlayer that allows 3 tensors to be passed: [input, state, seq_lens] """
# pylint: disable=protected-access
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras.distribute import distributed_training_utils
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.util.tf_export import keras_export


class CustomMultiInputLayer(base_layer.Layer):
  """Layer to be used as an entry point into a Network (a graph of layers).

  It can either wrap an existing tensor (pass an `input_tensor` argument)
  or create its a placeholder tensor (pass arguments `input_shape`, and
  optionally, `dtype`).

  It is generally recommend to use the functional layer API via `Input`,
  (which creates an `InputLayer`) without directly using `InputLayer`.

  Arguments:
      input_shape: Shape tuple (not including the batch axis), or `TensorShape`
        instance (not including the batch axis).
      batch_size: Optional input batch size (integer or None).
      dtype: Datatype of the input.
      input_tensor: Optional tensor to use as layer input
          instead of creating a placeholder.
      sparse: Boolean, whether the placeholder created
          is meant to be sparse.
      name: Name of the layer (string).
  """

  def __init__(self,
               input_shape=None,
               batch_size=None,
               dtype=None,
               input_tensor=None,
               state=None,
               state_shape=None,
               seq_lens=None,
               sparse=False,
               name=None,
               **kwargs):
    strategy = distribution_strategy_context.get_strategy()
    if strategy and batch_size is not None and \
        distributed_training_utils.global_batch_size_supported(strategy):
      if batch_size % strategy.num_replicas_in_sync != 0:
        raise ValueError('The `batch_size` argument value {} cannot be '
                         'divisible by number of replicas {}'.format(
                             batch_size, strategy.num_replicas_in_sync))
      batch_size = batch_size // strategy.num_replicas_in_sync

    if 'batch_input_shape' in kwargs:
      batch_input_shape = kwargs.pop('batch_input_shape')
      if input_shape and batch_input_shape:
        raise ValueError('Only provide the input_shape OR '
                         'batch_input_shape argument to '
                         'InputLayer, not both at the same time.')
      batch_size = batch_input_shape[0]
      input_shape = batch_input_shape[1:]
    if 'state' in kwargs:
        state = kwargs.pop('state')
    if 'state_shape' in kwargs:
        state_shape = kwargs.pop('state_shape')
    if 'seq_lens' in kwargs:
        seq_lens = kwargs.pop('seq_lens')
    if kwargs:
      raise ValueError('Unrecognized keyword arguments:', kwargs.keys())

    if not name:
      prefix = 'input'
      name = prefix + '_' + str(backend.get_uid(prefix))

    if not dtype:
      if input_tensor is None:
        dtype = backend.floatx()
      else:
        dtype = backend.dtype(input_tensor)
    elif input_tensor is not None and input_tensor.dtype != dtype:
      raise ValueError('`input_tensor.dtype` differs from `dtype`: %s vs. %s' %
                       (input_tensor.dtype, dtype))
    super(CustomMultiInputLayer, self).__init__(dtype=dtype, name=name)
    self.built = True
    self.sparse = sparse
    self.batch_size = batch_size
    self.supports_masking = True

    if isinstance(input_shape, tensor_shape.TensorShape):
      input_shape = tuple(input_shape.as_list())
    elif isinstance(input_shape, int):
      input_shape = (input_shape,)

    if isinstance(state_shape, tensor_shape.TensorShape):
      state_shape = tuple(state_shape.as_list())
    elif isinstance(state_shape, int):
      state_shape = (state_shape,)

    # Input tensor
    if input_tensor is None:
      if input_shape is not None:
        batch_input_shape = (batch_size,) + tuple(input_shape)
      else:
        batch_input_shape = None
      graph = backend.get_graph()
      with graph.as_default():
        # In graph mode, create a graph placeholder to call the layer on.
        if sparse:
          input_tensor = backend.placeholder(
              shape=batch_input_shape,
              dtype=dtype,
              name=self.name+'_input_tensor',
              sparse=True)
        else:
          input_tensor = backend.placeholder(
              shape=batch_input_shape,
              dtype=dtype,
              name=self.name+'_input_tensor')

      self.is_placeholder = True
      self._batch_input_shape = batch_input_shape
    else:
      if not tf_utils.is_symbolic_tensor(input_tensor):
        raise ValueError('You should not pass an EagerTensor to `input_tensor` input of InputLayer. Instead of creating '
                         'an InputLayer, you should instantiate your model and directly call it on your input.')
      self.is_placeholder = False
      self._batch_input_shape = tuple(input_tensor.shape.as_list())

    # State
    if state is None:
      if state_shape is not None:
          if isinstance(state_shape,list):
            batch_state_shape = [(batch_size,)+(_state_shape,) for _state_shape in state_shape]
          else:
            batch_state_shape = (batch_size,)+(state_shape,)
      else:
        raise ValueError('You must define a state_shape but state_shape is {state_shape}')
      graph = backend.get_graph()
      with graph.as_default():
        # In graph mode, create a graph placeholder to call the layer on.
        if sparse:
          state = [backend.placeholder(
              shape=_batch_state_shape,
              dtype=tf.float32,
              name=self.name+f'_state{i}',
              sparse=True) for i, _batch_state_shape in enumerate(batch_state_shape)]
        else:
          state = [backend.placeholder(
              shape=_batch_state_shape,
              dtype=tf.float32,
              name=self.name+f'_state{i}') for i, _batch_state_shape in enumerate(batch_state_shape)]

      self.is_state_placeholder = True
      self.is_state_list = True if isinstance(state,list) else False
      self._batch_state_shape = batch_state_shape
    else:
      if not tf_utils.is_symbolic_tensor(state):
        raise ValueError('You should not pass an EagerTensor to `state` input of InputLayer. Instead of creating an '
                         'InputLayer, you should instantiate your model and directly call it on your input.')
      self.is_state_placeholder = False
      self._batch_state_shape = tuple(state.shape.as_list())

    # Seq_lens
    if seq_lens is None:
      batch_seq_lens_shape = (batch_size,) + (1,)
      graph = backend.get_graph()
      with graph.as_default():
        # In graph mode, create a graph placeholder to call the layer on.
        if sparse:
          seq_lens = backend.placeholder(
              shape=batch_seq_lens_shape,
              dtype=tf.uint32,
              name=self.name+'_seq_lens',
              sparse=True)
        else:
          seq_lens = backend.placeholder(
              shape=batch_seq_lens_shape,
              dtype=tf.uint32,
              name=self.name+'_seq_lens')

      self.is_seq_lens_placeholder = True
      self._batch_seq_lens_shape = batch_seq_lens_shape
    else:
      if not tf_utils.is_symbolic_tensor(seq_lens):
        raise ValueError('You should not pass an EagerTensor to `seq_lens` input of InputLayer. Instead of creating an '
                         'InputLayer, you should instantiate your model and directly call it on your input.')
      self.is_seq_lens_placeholder = False
      self._batch_seq_lens_shape = tuple(seq_lens.shape.as_list())

    # Create an input node to add to self.outbound_node
    # and set output_tensors' _keras_history.
    input_tensor._keras_history = base_layer.KerasHistory(self, 0, 0)
    input_tensor._keras_mask = None
    base_layer.Node(
        self,
        inbound_layers=[],
        node_indices=[],
        tensor_indices=[],
        input_tensors=[input_tensor],
        output_tensors=[input_tensor, state, seq_lens])

  def get_config(self):
    config = {
        'batch_input_shape': self._batch_input_shape,
        'dtype': self.dtype,
        'sparse': self.sparse,
        'name': self.name
    }
    return config
