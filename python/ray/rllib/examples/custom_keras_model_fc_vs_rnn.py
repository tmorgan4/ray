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
from ray.rllib.agents.dqn.distributional_q_model import DistributionalQModel
from ray.rllib.utils import try_import_tf
from ray.rllib.models.model import restore_original_dimensions
import numpy as np
tf = try_import_tf()

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="IMPALA")  # "SimpleQ")  # Try PG, PPO, DQN
parser.add_argument("--stop", type=int, default=200)

class MyFcModel(TFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MyFcModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        layer_1 = tf.keras.layers.Dense(128, name="my_layer1", activation=tf.nn.relu, kernel_initializer=normc_initializer(1.0))(self.inputs)
        layer_out = tf.keras.layers.Dense(num_outputs, name="my_out", activation=None, kernel_initializer=normc_initializer(0.01))(layer_1)
        value_out = tf.keras.layers.Dense(1, name="value_out", activation=None, kernel_initializer=normc_initializer(0.01))(layer_1)
        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        self.prev_input = input_dict
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class MyRnnModel(TFModelV2):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kw):
        super(MyRnnModel, self).__init__(obs_space, action_space, num_outputs, model_config, name, **kw)
        # self.input_layer = tf.keras.layers.Input(shape=(None, obs_space.shape[0]), name='inputLayer')
        # lstm_cell = tf.keras.layers.LSTMCell(model_config['lstm_cell_size'])
        self.rnn_layer = tf.keras.layers.LSTM(
            model_config['lstm_cell_size'],
            batch_input_shape=(None, model_config['max_seq_len'], obs_space.shape[0]),
            return_sequences=True,
            return_state=True,
            name='rnnLayer')
        self.dense_layer_1 = tf.keras.layers.Dense(
            model_config['fcnet_hiddens'][0],
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0),
            name='denseLayer1')
        self.logits_layer = tf.keras.layers.Dense(
            self.num_outputs,
            activation=tf.keras.activations.linear,
            kernel_initializer=normc_initializer(0.01),
            name='logitsLayer')
        self.value_layer = tf.keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer=normc_initializer(0.01),
            name='valueLayer')
        self.register_variables(tf.trainable_variables(scope=None))
        # self.base_model = tf.keras.Model(self.inputs, layer_out)
        # self.register_variables(self.base_model.variables)

    # Implement the core forward method
    def forward(self, input_dict, state, seq_lens):
        self.prev_input = input_dict
        x = add_time_dimension(input_dict['obs'], seq_lens)
        mask = tf.sequence_mask(seq_lens, name='mask') if seq_lens is not None else None
        x, state_h, state_c = self.rnn_layer(x, mask=mask, initial_state=state)
        x = tf.reshape(x, [-1, self.rnn_layer.cell.units])
        x = self.dense_layer_1(x)  # only pass first of output of lstm layer
        logits = self.logits_layer(x)
        self._value_out = self.value_layer(x)

        # self.base_model = tf.keras.Model(inputs=add_time_dimension(input_dict['obs'], seq_lens), outputs=last_layer)
        # self.base_model.summary()

        # mask = tf.sequence_mask(seq_lens, self.model_config['max_seq_len'])
        # self.lstm_layer.reset_states(states=state if state else None)
        # model_out, states = self.base_model(inputs=inputs, states=states, seq_lens=seq_lens)

        return logits, [state_h, state_c]

    def get_initial_state(self):
        # return self.initial_state
        return [np.zeros(self.model_config['lstm_cell_size'], np.float32),
                np.zeros(self.model_config['lstm_cell_size'], np.float32)]
        # return []

    def __call__(self, input_dict, state, seq_lens):
        # if not state:
        #     state = None

        restored = input_dict.copy()
        restored["obs"] = restore_original_dimensions(input_dict["obs"], self.obs_space, self.framework)
        restored["obs_flat"] = input_dict["obs"]
        outputs, states = self.forward(input_dict=restored, state=state, seq_lens=seq_lens)

        try:
            shape = outputs.shape
        except AttributeError:
            raise ValueError(f"Output is not a tensor: {outputs}")
        else:
            if len(shape) != 2 or shape[1] != self.num_outputs:
                raise ValueError(f"Expected output shape of [None, {self.num_outputs}], got {shape}")
        if not isinstance(states, list):
            raise ValueError(f"States output is not a list: {states}")

        return outputs, states

    def value_function(self):
        return tf.reshape(self._value_out, [-1])






if __name__ == "__main__":
    local_mode = False
    ray.init(num_cpus=0 if local_mode else None, local_mode=local_mode)
    args = parser.parse_args()
    ModelCatalog.register_custom_model("keras_model", MyKerasModel)
    lstm_cell_size = 128
    tune.run(
        args.run,
        stop={"episode_reward_mean": args.stop},
        config={
            "env": "CartPole-v0",
            "num_workers": 0 if local_mode is True else 4,
            "num_gpus": 0,
            "num_sgd_iter": 3 if args.run is "PPO" else 1,
            'observation_filter': 'MeanStdFilter',
            "model": {"custom_model": "keras_model",
                      "lstm_cell_size": lstm_cell_size,
                      "state_shape": [lstm_cell_size, lstm_cell_size]},
        })
