from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import ray
from ray import tune
from ray.rllib.models.lstm import add_time_dimension
from ray.rllib.models import ModelCatalog
from ray.rllib.models.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
import numpy as np
from stateless_cartpole import CartPoleStatelessEnv
import tensorflow as tf
tf.disable_v2_behavior()


class FcModel_A(TFModelV2):
    """ Uses standard keras functional model API """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(FcModel_A, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        dense = tf.keras.layers.Dense(128, name="dense", activation=tf.nn.relu, kernel_initializer=normc_initializer(1.0))(self.inputs)
        model_out = tf.keras.layers.Dense(num_outputs, name="model_out", activation=None, kernel_initializer=normc_initializer(0.01))(dense)
        value_out = tf.keras.layers.Dense(1, name="value_out", activation=None, kernel_initializer=normc_initializer(0.01))(dense)
        self.base_model = tf.keras.Model(self.inputs, [model_out, value_out])
        self.register_variables(self.base_model.variables)
        self.base_model.summary()

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class FcModel_B(TFModelV2):
    """ Calls each layer individually in forward() """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(FcModel_B, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self.dense = tf.keras.layers.Dense(128, name="dense", input_shape=obs_space.shape, activation=tf.nn.relu, kernel_initializer=normc_initializer(1.0))
        self.model_out = tf.keras.layers.Dense(num_outputs, name="model_out", activation=None, kernel_initializer=normc_initializer(0.01))
        self.value_out = tf.keras.layers.Dense(1, name="value_out", activation=None, kernel_initializer=normc_initializer(0.01))
        self.register_variables(tf.trainable_variables(scope=None))

    def forward(self, input_dict, state, seq_lens):
        x = input_dict['obs']
        x = self.dense(x)
        model_out = self.model_out(x)
        self._value_out = self.value_out(x)
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class RnnModel_1(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kw):
        super(RnnModel_1, self).__init__(obs_space, action_space, num_outputs, model_config, name, **kw)
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
        # TODO reshape layer does not accept mask which is propogated through model if masking layer is used upstream
        reshape_layer = tf.keras.layers.Reshape(
            target_shape=(model_config['lstm_cell_size'],))(rnn_layer[0])
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
        tf.keras.utils.plot_model(self.base_model, show_layer_names=True, show_shapes=True)
        # self.base_model.summary()
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
        return [np.zeros(self.model_config['lstm_cell_size'], np.float32),
                np.zeros(self.model_config['lstm_cell_size'], np.float32)]

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class RnnModel_2(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kw):
        super(RnnModel_2, self).__init__(obs_space, action_space, num_outputs, model_config, name, **kw)
        self.dense_layer_1 = tf.keras.layers.Dense(
            units=model_config['fcnet_hiddens'][0],
            batch_input_shape=(None, None, obs_space.shape[0]),
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0),
            name='denseLayer1')
        self.lstm_layer = tf.keras.layers.LSTM(
            units=model_config['lstm_cell_size'],
            return_sequences=True,
            return_state=True,
            name='lstmLayer')
        self.reshape_layer = tf.keras.layers.Reshape(  # not actually used, see notes in forward()
            target_shape=(self.model_config['lstm_cell_size'],),
            name='reshapeLayer')
        self.dense_layer_2 = tf.keras.layers.Dense(
            units=model_config['fcnet_hiddens'][0],
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0),
            name='denseLayer2')
        self.logits_layer = tf.keras.layers.Dense(
            units=self.num_outputs,
            activation=tf.keras.activations.linear,
            kernel_initializer=normc_initializer(0.01),
            name='logitsLayer')
        self.value_layer = tf.keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer=normc_initializer(0.01),
            name='valueLayer')

        # TODO: tf.trainable_variables() returns empty list, variables not initialized yet (except in eager mode)
        self.register_variables(tf.trainable_variables())

    # Implement the core forward method
    def forward(self, input_dict, state, seq_lens):
        # define inputs
        x = input_dict['obs']

        # add time dimension if input has only 2 dimensions
        if x._rank() < 3:
            x = add_time_dimension(x, seq_lens)

        x = self.dense_layer_1(x)

        if self.model_config['max_seq_len'] > 1:
            mask = tf.sequence_mask(seq_lens)  # 'max_seq_len' is optional, otherwise inferred from longest seq
            if self.model_config['custom_options']['initialize_lstm_with_prev_state']:
                x, state_h, state_c = self.lstm_layer(x, mask=mask, initial_state=state)
            else:
                x, state_h, state_c = self.lstm_layer(x, mask=mask, initial_state=None)
        else:  # no mask needed if max_seq_len is 1
            if self.model_config['custom_options']['initialize_lstm_with_prev_state']:
                x, state_h, state_c = self.lstm_layer(x, mask=None, initial_state=state)
            else:
                x, state_h, state_c = self.lstm_layer(x, mask=None, initial_state=None)

        x, state_h, state_c = self.lstm_layer(
            x,
            mask=tf.sequence_mask(seq_lens) if self.model_config['max_seq_len'] > 1 else None,
            initial_state=state if self.model_config['custom_options']['initialize_lstm_with_prev_state'] is True else None)

        # reshape LSTM output to remove time dim
        x = tf.reshape(x, [-1, self.model_config['lstm_cell_size']])  # this works
        # x = self.reshape_layer(rnn_out[0])  # this does not work

        x = self.dense_layer_2(x)
        logits = self.logits_layer(x)
        self._value_out = self.value_layer(x)
        state = [state_h, state_c]

        return logits, state

    def get_initial_state(self):
        return [np.zeros(self.model_config['lstm_cell_size'], np.float32),
                np.zeros(self.model_config['lstm_cell_size'], np.float32)]

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


if __name__ == "__main__":
    algorithm = "PPO"  # Try PG, PPO, DQN, IMPALA (IMPALA does not work with max_seq_len=1, issue with vtrace)
    run_stateless_cartpole = True  # if True use modified stateless version else use standard 'CartPole-v0'
    initialize_lstm_with_prev_state = True  # if True restore the state from prevous batch, otherwise inialize with zeros
    lstm_cell_size = 32
    model_idx = 2
    local_mode = True if getattr(sys, 'gettrace', None)() is not None else False  # run ray locally in single process if python in debug mode
    _algorithm = algorithm if local_mode is False else "PPO"  # IMPALA hangs in local mode, set to PPO
    ray.init(num_cpus=0 if local_mode else None,
             local_mode=local_mode,
             object_store_memory=int(2e9),
             redis_max_memory=int(1e9),
             logging_level='DEBUG' if local_mode is True else 'INFO')
    ModelCatalog.register_custom_model("FCmodel_ModelAPI", FcModel_A)
    ModelCatalog.register_custom_model("FCmodel_LayerAPI", FcModel_B)
    ModelCatalog.register_custom_model("RNNmodel_1", RnnModel_1)
    ModelCatalog.register_custom_model("RNNmodel_2", RnnModel_2)
    model_list = ["FCmodel_ModelAPI", "FCmodel_LayerAPI", "RNNmodel_1", "RNNmodel_2"]
    print(f'\nRUNNING  --->  MODEL: {model_list[model_idx]}, ALGORITHM: {_algorithm}, LOCAL_MODE: {local_mode}\n')
    tune.register_env("CartPoleStateless", lambda _: CartPoleStatelessEnv())
    tune.run(
        _algorithm,
        stop={"episode_reward_mean": 200, "timesteps_total": 1e6},
        config={
            "env": "CartPoleStateless" if run_stateless_cartpole else 'CartPole-v0',
            "num_workers": 0 if local_mode is True else 4,
            "num_gpus": 0,
            "num_sgd_iter": 5 if _algorithm is "PPO" else 1,
            "vf_loss_coeff": 0.0001 if _algorithm is "PPO" else 0.01,
            'observation_filter': 'MeanStdFilter',
            "horizon": 200,  # CartPoleStatelessEnv does not have built in time limit
            "model": {
                "custom_model": model_list[model_idx],
                "custom_options": {"initialize_lstm_with_prev_state": initialize_lstm_with_prev_state},
                "fcnet_hiddens": [64, 64],
                "lstm_cell_size": lstm_cell_size,
                "use_lstm": False,  # dont use built in lstm
                "state_shape": [lstm_cell_size, lstm_cell_size] if "RNN" in model_list[model_idx] else [],
                "vf_share_layers": True,
                "max_seq_len": 1,
            }
        })

