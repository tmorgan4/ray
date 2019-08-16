from tensorflow.python.keras.engine.base_layer import Layer
import numpy as np


class NonMasking(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(NonMasking, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape = input_shape

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        return x

    def get_output_shape_for(self, input_shape):
        return input_shape


# states_per_layer is 1 for GRU, 2 for LSTM
def get_state_shape(lstm_cell_size, lstm_layer_factors=(1.0,), num_states_per_layer=2):
    _layer_sizes = np.dot(lstm_cell_size, lstm_layer_factors).astype(np.int32)
    return [_layer_sizes[i//num_states_per_layer] for i in range(len(_layer_sizes)*num_states_per_layer)]
