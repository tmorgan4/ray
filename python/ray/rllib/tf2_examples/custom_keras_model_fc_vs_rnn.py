"""Example of using a custom ModelV2 Keras-style model.

TODO(ekl): add this to docs once ModelV2 is fully implemented.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import ray
from ray import tune
from ray.rllib.models.lstm import add_time_dimension
from ray.rllib.models import ModelCatalog
from ray.rllib.models.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf
from ray.rllib.models.model import restore_original_dimensions
import numpy as np
# tf = try_import_tf()
import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="IMPALA")  # "SimpleQ")  # Try PG, PPO, DQN
parser.add_argument("--stop", type=int, default=200)


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

class FcModel_A(TFModelV2):
    """ Uses standard functional model API"""
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(FcModel_A, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        dense = tf.keras.layers.Dense(128, name="dense", activation=tf.nn.relu, kernel_initializer=normc_initializer(1.0))(self.inputs)
        model_out = tf.keras.layers.Dense(num_outputs, name="model_out", activation=None, kernel_initializer=normc_initializer(0.01))(dense)
        value_out = tf.keras.layers.Dense(1, name="value_out", activation=None, kernel_initializer=normc_initializer(0.01))(dense)
        self.base_model = tf.keras.Model(self.inputs, [model_out, value_out])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        self.prev_input = input_dict
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class FcModel_B(TFModelV2):
    """ Calls each layer invidually in forward """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(FcModel_B, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self.dense = tf.keras.layers.Dense(128, name="dense", input_shape=obs_space.shape, activation=tf.nn.relu, kernel_initializer=normc_initializer(1.0))
        self.model_out = tf.keras.layers.Dense(num_outputs, name="model_out", activation=None, kernel_initializer=normc_initializer(0.01))
        self.value_out = tf.keras.layers.Dense(1, name="value_out", activation=None, kernel_initializer=normc_initializer(0.01))
        self.register_variables(tf.trainable_variables(scope=None))

    def forward(self, input_dict, state, seq_lens):
        self.prev_input = input_dict
        x = input_dict['obs']
        x = self.dense(x)
        model_out = self.model_out(x)
        self._value_out = self.value_out(x)
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class RnnModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kw):
        super(RnnModel, self).__init__(obs_space, action_space, num_outputs, model_config, name, **kw)
        self.input_layer = tf.keras.layers.Input(shape=(None, obs_space.shape[0]), name='inputLayer')
        self.dense_input_layer = tf.keras.layers.Dense(
            model_config['fcnet_hiddens'][0],
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0),
            name='denseInputLayer')(self.input_layer)
        masking_layer = tf.keras.layers.Masking(mask_value=0.0)(self.dense_input_layer)
        rnn_layer = tf.keras.layers.LSTM(
            model_config['lstm_cell_size'],
            return_sequences=True,
            return_state=True,
            name='rnnLayer')(masking_layer)
        # TODO reshape layer does not accept mask
        reshape_layer = tf.reshape(rnn_layer[0], shape=(-1, model_config['lstm_cell_size']), name='rnn_out_reshape')
        dense_layer_1 = tf.keras.layers.Dense(
            model_config['fcnet_hiddens'][0],
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0),
            name='denseLayer1')(reshape_layer)
        state = [dense_layer_1[1], dense_layer_1[2]]
        logits_layer = tf.keras.layers.Dense(
            self.num_outputs,
            activation=tf.keras.activations.linear,
            kernel_initializer=normc_initializer(0.01),
            name='logitsLayer')(dense_layer_1)
        value_layer = tf.keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer=normc_initializer(0.01),
            name='valueLayer')(dense_layer_1)
        # self.register_variables(tf.trainable_variables(scope=None))
        self.base_model = tf.keras.Model(inputs=self.input_layer, outputs=[logits_layer, value_layer, state])
        self.register_variables(self.base_model.variables)
        self.base_model.summary()
        self.dense_before_rnn = True

    # Implement the core forward method
    def forward(self, input_dict, state, seq_lens):
        self.prev_input = input_dict
        x = input_dict['obs']

        if x._rank() < 3:
            x = add_time_dimension(x, seq_lens)

        logits, self._value_out, state = self.base_model(x)

        return logits, state

    def get_initial_state(self):
        # return self.initial_state
        return [np.zeros(self.model_config['lstm_cell_size'], np.float32),
                np.zeros(self.model_config['lstm_cell_size'], np.float32)]
        # return []

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class RnnModel_2(TFModelV2):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kw):
        super(RnnModel_2, self).__init__(obs_space, action_space, num_outputs, model_config, name, **kw)
        self.lstm_layer_sizes = model_config['state_shape']

        self.inputs = tf.keras.layers.Input(shape=(None, obs_space.shape[0]), name='InputLayer')
        x = tf.keras.layers.LSTM(self.lstm_layer_sizes[0], return_sequences=True, kernel_initializer='TruncatedNormal')(self.inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.05)(x)

        for _ in range(2):
            x = tf.keras.layers.LSTM(self.lstm_layer_sizes[2], kernel_initializer='TruncatedNormal', return_sequences=True)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU()(x)
            x = tf.keras.layers.Dropout(0.05)(x)

        x = tf.keras.layers.LSTM(self.lstm_layer_sizes[4], kernel_initializer='TruncatedNormal', return_sequences=False)(x)
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

        model_out = tf.keras.layers.Dense(num_outputs, name="model_out", activation=None, kernel_initializer=normc_initializer(0.01))(x)
        value_out = tf.keras.layers.Dense(1, name="value_out", activation=None, kernel_initializer=normc_initializer(0.01))(x)
        
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


# only used for "RNNmodel_PredMaint" to determine layer sizing
# states_per_layer is 1 for GRU, 2 for LSTM
def get_state_shape(lstm_layer_factors=(1.0,), num_states_per_layer=2):
    _layer_sizes = np.dot(lstm_cell_size, lstm_layer_factors).astype(np.int32)
    return [_layer_sizes[i//num_states_per_layer] for i in range(len(_layer_sizes)*num_states_per_layer)]


if __name__ == "__main__":
    local_mode = True
    lstm_cell_size = 128
    model_idx = 2
    ray.init(num_cpus=0 if local_mode else None,
             local_mode=local_mode,
             object_store_memory=int(2e9),
             redis_max_memory=int(1e9),
             logging_level='DEBUG' if local_mode is True else 'INFO')
    args = parser.parse_args()
    args.run = "PPO" if local_mode is True else args.run  # IMPALA hangs in local mode
    if model_idx == 2:
        state_shape = get_state_shape(lstm_layer_factors=(1.0,), num_states_per_layer=2)
    elif model_idx == 3:
        state_shape = get_state_shape(lstm_layer_factors=(1./16., 1./16., 1./4.), num_states_per_layer=2)
    else:
        state_shape = []
    ModelCatalog.register_custom_model("FCmodel_ModelAPI", FcModel_A)
    ModelCatalog.register_custom_model("FCmodel_LayerAPI", FcModel_B)
    ModelCatalog.register_custom_model("RNNmodel", RnnModel)
    ModelCatalog.register_custom_model("RNNmodel_PredMaint", RnnModel_2)
    model_list = ["FCmodel_ModelAPI", "FCmodel_LayerAPI", "RNNmodel", "RNNmodel_PredMaint"]
    print(f'\n***** RUNNING MODEL  --->  {model_list[model_idx]} *****\n')
    tune.run(
        args.run,
        stop={"episode_reward_mean": args.stop, "timesteps_total": 1e6},
        config={
                "env": "CartPole-v0",
                "num_workers": 0 if local_mode is True else 4,
                "num_gpus": 0,
                "num_sgd_iter": 5 if args.run is "PPO" else 1,
                "vf_loss_coeff": 0.0001 if args.run is "PPO" else 0.01,
                'observation_filter': 'MeanStdFilter',
                "model": {"custom_model": model_list[model_idx],  # tune.grid_search(model_list)
                          "fcnet_hiddens": [128, 128],
                          "lstm_cell_size": lstm_cell_size,
                          "state_shape": state_shape,
                          "vf_share_layers": True,
                          "max_seq_len": 1,  # test rnn statefulness
                          }
        })

