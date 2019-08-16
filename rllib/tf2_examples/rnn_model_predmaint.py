from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.misc import normc_initializer
from ray.rllib.models.lstm import add_time_dimension
from ray.rllib.models import ModelCatalog
import tensorflow as tf
tf.disable_v2_behavior()
import numpy as np

# TODO:
# ModelCatalog.register_custom_model("RNNmodel_PredMaint", RnnModel_3)
# state_shape = get_state_shape(lstm_cell_size=???, lstm_layer_factors=(1. / 16., 1. / 16., 1. / 4.), num_states_per_layer=2)


class RnnModel_3(TFModelV2):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kw):
        super(RnnModel_3, self).__init__(obs_space, action_space, num_outputs, model_config, name, **kw)
        self.lstm_layer_sizes = model_config['state_shape']

        self.inputs = tf.keras.layers.Input(shape=(None, obs_space.shape[0]), name='InputLayer')
        x = tf.keras.layers.LSTM(self.lstm_layer_sizes[0], return_sequences=True, kernel_initializer='TruncatedNormal')(
            self.inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.05)(x)

        for _ in range(2):
            x = tf.keras.layers.LSTM(self.lstm_layer_sizes[2], kernel_initializer='TruncatedNormal',
                                     return_sequences=True)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU()(x)
            x = tf.keras.layers.Dropout(0.05)(x)

        x = tf.keras.layers.LSTM(self.lstm_layer_sizes[4], kernel_initializer='TruncatedNormal',
                                 return_sequences=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.05)(x)

        for _ in range(1):
            x = tf.keras.layers.Dense(4, kernel_initializer='TruncatedNormal')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU()(x)
            x = tf.keras.layers.Dropout(0.05)(x)

        for _ in range(8):
            x = tf.keras.layers.Dense(256, kernel_initializer='TruncatedNormal')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU()(x)
            x = tf.keras.layers.Dropout(0.05)(x)

        for _ in range(1):
            x = tf.keras.layers.Dense(64, kernel_initializer='TruncatedNormal')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU()(x)
            x = tf.keras.layers.Dropout(0.05)(x)

        x = tf.keras.layers.Dense(32, kernel_initializer='TruncatedNormal')(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.LeakyReLU()(x)
        # x = tf.keras.layers.Dropout(0.05)(x)

        model_out = tf.keras.layers.Dense(num_outputs, name="model_out", activation=None,
                                          kernel_initializer=normc_initializer(0.01))(x)
        value_out = tf.keras.layers.Dense(1, name="value_out", activation=None,
                                          kernel_initializer=normc_initializer(0.01))(x)

        self.base_model = tf.keras.Model(self.inputs, [model_out, value_out])
        self.register_variables(self.base_model.variables)
        self.base_model.summary()

    # Implement the core forward method
    def forward(self, input_dict, state, seq_lens):
        self.prev_input = input_dict
        x = input_dict['obs']

        if x._rank() < 3:
            x = add_time_dimension(x, seq_lens)

        model_out, self._value_out = self.base_model(x)

        return model_out, state

    def get_initial_state(self):
        # return self.initial_state
        return [np.zeros(self.lstm_layer_sizes[0], np.float32), np.zeros(self.lstm_layer_sizes[0], np.float32),
                np.zeros(self.lstm_layer_sizes[1], np.float32), np.zeros(self.lstm_layer_sizes[1], np.float32),
                np.zeros(self.lstm_layer_sizes[2], np.float32), np.zeros(self.lstm_layer_sizes[2], np.float32)]
        # return []

    def value_function(self):
        return tf.reshape(self._value_out, [-1])
        # return tf.squeeze(self._value_out, axis=-1, name='_value_out_squeeze')
        # return self._value_out
