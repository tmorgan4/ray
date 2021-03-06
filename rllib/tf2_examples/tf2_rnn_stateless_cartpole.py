""" A minimal RNN model showing basic functionality and potential issues with TFModelV2 in RLLIB.
Due to the stateless nature of this CartPole environment the only way this environment can be solved is to
look at multiple steps however we limit max_seq_len to 1 step.  If the environment is solved it proves that rnn states
are being passed between batches properly and rnn is reinitialized with previous state to 'stitch' together sequences"""

import sys
import ray
from ray import tune
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.rnn_sequencing import add_time_dimension
import numpy as np
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
from ray.rllib.utils import try_import_tf
tf = try_import_tf()


### misc suggestions ###
# Standardize use of seq_len and seq_lens (both are used, leading to confusion)

'''copied from tf.python.keras.engine.training.Model docs (slightly modified)
There are two ways to instantiate a `Model`:
#####################################################################
  Method #1 (RLLIB custom model "MaskingLayerRNNmodel" shown below)
#####################################################################
With the "functional API", where you start from `Input`, you chain layer calls to specify 
the model's forward pass, and finally you create your model from inputs and outputs:

EXAMPLE USAGE:
    inputs = tf.keras.Input(shape=(3,))
    x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
    outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

#####################################################################
  Method #2 (RLLIB custom model "SequenceMaskRNNmodel" shown below)  
#####################################################################
By subclassing the `Model` class: in that case, you should define your layers in `__init__` 
and you should implement the model's forward pass in `call`.

EXAMPLE USAGE:
    class MyModel(tf.keras.Model):
    
    def __init__(self):
      super(MyModel, self).__init__()
      self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
      self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
    
    def call(self, inputs):
      x = self.dense1(inputs)
      return self.dense2(x)
    
    model = MyModel()'''


class CartPoleStatelessEnv(gym.Env):
    """ Partially observed variant of the CartPole gym environment.
    We delete the velocity component of the state, so that it can only be solved by a LSTM policy.

    Original CartPole-v0 (unmodified) Specs:
        Observation:
            Type: Box(4)
            Num	Observation	Min	Max
            0	Cart Position	-2.4	2.4
            1	Cart Velocity	-Inf	Inf
            2	Pole Angle	~ -41.8°	~ 41.8°
            3	Pole Velocity At Tip	-Inf	Inf

        Actions:
            Type: Discrete(2)
            Num	Action
            0	Push cart to the left
            1	Push cart to the right

        Stateless CartPole (this env) Specs:
        Observation:
            Type: Box(2)
            Num	Observation	Min	Max
            0	Cart Position	-2.4	2.4
            2	Pole Angle	~ -41.8°	~ 41.8°

        Actions:
            Type: Discrete(2)
            Num	Action
            0	Push cart to the left
            1	Push cart to the right

    Env copied from this example with minor changes:
    https://github.com/ray-project/ray/blob/master/python/ray/rllib/examples/cartpole_lstm.py """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 60}

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        high = np.array([self.x_threshold * 2, self.theta_threshold_radians * 2])
        self.action_space = spaces.Discrete(2)
        # self.action_space = spaces.MultiDiscrete([2,])  # TODO issue with using multidiscrete in rllib
        self.observation_space = spaces.Dict({'cart_position': spaces.Box(-high[0], high[0], dtype=np.float32),
                                              'pole_angle': spaces.Box(-high[1], high[1], dtype=np.float32)})
        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        action = action[0]  # TODO: experimenting with action spaces
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = (temp - self.polemass_length * thetaacc * costheta / self.total_mass)
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = (x, x_dot, theta, theta_dot)
        done = (x < -self.x_threshold or x > self.x_threshold
                or theta < -self.theta_threshold_radians
                or theta > self.theta_threshold_radians)
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:  # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done > 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. "
                            "You should always call 'reset()' once you receive 'done = True' -- any further steps are "
                            "undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        # rv = np.r_[self.state[0], self.state[2]]
        rv = {'cart_position': self.state[0], 'pole_angle': self.state[2]}
        return rv, reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4, ))
        self.steps_beyond_done = None
        # rv = np.r_[self.state[0], self.state[2]]
        rv = {'cart_position': self.state[0], 'pole_angle': self.state[2]}
        return rv


class MaskingLayerRNNmodel(TFModelV2):
    """
    Method #1 - THIS FAILS!!!

    Overview:
     - self.base_model() is defined in __init__
     - model is called in forward() using "outputs = self.base_model(inputs)"

    Known issues:
     - there is no way to pass state or seq_lens when calling self.base_model() in forward()
     - when calling self.base_model(inputs) in forward(), only allowed **kwargs are 'trainable', 'dtype', and 'dynamic'
       --> prevents being able to pass state or seq_lens as kwargs using this API
     - validate_kwargs() is called at tensorflow.python.keras.engine.network line 191 which prevents other kwargs
     - Attempt to create separate tf.keras.layers.Input() layers to pass state and seq_len as inputs did not work

    Possible solutions:
     - eliminate need to pass seq_len if tf.keras.layers.Masking() is used before input to RNN instead of
       tf.sequence_mask() which is only passed to RNN
     - Masking layer would automatically recognize inputs in which all values are mask_value, hence it does not need seq_len
     - BUT it breaks layers that do not support mask such as Reshape() as mask is automatically propagated to all
       downstream layers (try running this model)
     - could eliminate need to pass state if stateful=True in RNN to automatically carry over previous states and
       instead tell it when to reset states
     - however, to call rnn_layer.reset_states() it needs either:
        A) fixed batch dim, defined at layer creation
        B) pass batch_dim when calling RNN_layer.reset_states(), but batch_dim does not seem not easily accessible
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kw):
        super(MaskingLayerRNNmodel, self).__init__(obs_space, action_space, num_outputs, model_config, name, **kw)
        self.initialize_lstm_with_prev_state = model_config['custom_options']['initialize_lstm_with_prev_state']
        self.input_layer = tf.keras.layers.Input(
            shape=(None, obs_space.shape[0]),
            name='inputLayer')
        dense_layer_1 = tf.keras.layers.Dense(
            model_config['fcnet_hiddens'][0],
            activation=tf.nn.tanh,
            name='denseLayer1')(self.input_layer)
        masking_layer = tf.keras.layers.Masking(
            mask_value=0.0)(dense_layer_1)
        lstm_out, state_h, state_c = tf.keras.layers.LSTM(
            model_config['lstm_cell_size'],
            return_sequences=True,
            return_state=True,
            name='lstmLayer')(inputs=masking_layer,
                              mask=None,  # mask is provided by Masking() layer instead of this arg
                              initial_state=None)  # note that initial_states=None (not correct), how could we pass 'state' here?
        # if we had access to batch shape, we could set stateful=True in LSTM and call reset_states() instead of passing state
        # reshape_layer does not accept mask which is propogated through model if Masking() is used upstream, FAILS!
        reshape_layer = tf.keras.layers.Reshape(
            target_shape=(model_config['lstm_cell_size'],))(lstm_out)
        dense_layer_2 = tf.keras.layers.Dense(
            model_config['fcnet_hiddens'][1],
            activation=tf.nn.tanh,
            name='denseLayer2')(reshape_layer)
        logits_layer = tf.keras.layers.Dense(
            self.num_outputs,
            activation=tf.keras.activations.linear,
            name='logitsLayer')(dense_layer_2)
        value_layer = tf.keras.layers.Dense(
            1,
            activation=None,
            name='valueLayer')(dense_layer_2)
        state = [state_h, state_c]

        self.base_model = tf.keras.Model(inputs=self.input_layer, outputs=[logits_layer, value_layer, state])
        self.register_variables(self.base_model.variables)
        self.base_model.summary()

    # Implement the core forward method
    def forward(self, input_dict, state, seq_lens):
        x = input_dict['obs_flat']
        if x._rank() < 3:
            x = add_time_dimension(x, seq_lens)
        logits, self._value_out, state = self.base_model(x)

        return logits, state

    def get_initial_state(self):
        return [np.zeros(self.model_config['lstm_cell_size'], np.float32),
                np.zeros(self.model_config['lstm_cell_size'], np.float32)]

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class SequenceMaskRNNmodel(TFModelV2):
    """
    Method #2 - THIS WORKS!!!

    Overview:
     - all layers are defined in __init__ but called manually in forward() giving us access to input and output tensors
       at every layer

     Known issues:
     - tf.trainable_variables() returns empty list because variables are not initialized yet
       --> no variables are registered with RLLIB (does this matter?)
     - cant print model.summary() to check shapes since no model is defined, very useful for debugging shapes
     - ModelV2 docstring says computation graph is created every call to forward(), is this an issue?
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kw):
        super(SequenceMaskRNNmodel, self).__init__(obs_space, action_space, num_outputs, model_config, name, **kw)
        self.initialize_lstm_with_prev_state = self.model_config['custom_options']['initialize_lstm_with_prev_state']
        self.dense_layer_1 = tf.keras.layers.Dense(
            units=model_config['fcnet_hiddens'][0],
            batch_input_shape=(None, None, obs_space.shape[0]),
            activation=tf.nn.relu,
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
            units=model_config['fcnet_hiddens'][1],
            activation=tf.nn.tanh,
            name='denseLayer2')
        self.logits_layer = tf.keras.layers.Dense(
            units=self.num_outputs,
            activation=tf.keras.activations.linear,
            name='logitsLayer')
        self.value_layer = tf.keras.layers.Dense(
            1,
            activation=None,
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

        # all layers are callable giving us complete control over input and output tensors at each step
        # note that mask is not used if max_seq_len is 1 as it creates empty tensors, the env can still be solved
        # but using a mask would surely increase algorithm performance
        x = self.dense_layer_1(x)
        x, state_h, state_c = self.lstm_layer(
            inputs=x,
            mask=tf.sequence_mask(seq_lens) if self.model_config['max_seq_len'] > 1 else None,
            initial_state=state if self.initialize_lstm_with_prev_state is True else None)

        # reshape LSTM output to remove time dim
        x = tf.reshape(x, [-1, self.model_config['lstm_cell_size']])  # this works
        # x = self.reshape_layer(x)  # this does not work

        # call remaining layers
        x = self.dense_layer_2(x)
        logits = self.logits_layer(x)
        self._value_out = self.value_layer(x)

        # save state to list, must match shape of "state_shape" in config
        state = [state_h, state_c]

        return logits, state

    def get_initial_state(self):
        return [np.zeros(self.model_config['lstm_cell_size'], np.float32),
                np.zeros(self.model_config['lstm_cell_size'], np.float32)]

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class InputLayerRNNmodel(TFModelV2):
    """
    Currently this only works if max_seq_len == 1, otherwise there is state shape mismatch between lstm outputs and
    states (that is off by the multiple of max_seq_len)

    Overview:
     - Create a unique tf.keras.layers.Input() placeholder for each tensor in [obs, state_h, state_c, seq_lens]

    Known issues:
     - state must be split into separate state_h and state_c tensors because placeholders only accept single tensors,
       (not list of states)

    Possible solutions:
     - TODO
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kw):
        super(InputLayerRNNmodel, self).__init__(obs_space, action_space, num_outputs, model_config, name, **kw)
        self.initialize_lstm_with_prev_state = model_config['custom_options']['initialize_lstm_with_prev_state']
        inputs = tf.keras.layers.Input(batch_shape=(None, None, obs_space.shape[0]), name='inputs')
        seq_lens = tf.keras.layers.Input(batch_shape=(None,), name='seq_lens')
        state_h_in = tf.keras.layers.Input(batch_shape=(None, model_config['lstm_cell_size']), name='state_h_in')
        state_c_in = tf.keras.layers.Input(batch_shape=(None, model_config['lstm_cell_size']), name='state_c_in')
        dense_layer_1 = tf.keras.layers.Dense(
            model_config['fcnet_hiddens'][0],
            activation=tf.nn.tanh,
            name='denseLayer1')(inputs)
        mask = tf.sequence_mask(seq_lens, model_config['max_seq_len']) if model_config['max_seq_len'] > 1 else None
        lstm = tf.keras.layers.LSTM(
            model_config['lstm_cell_size'],
            return_sequences=True,
            return_state=True,
            name='lstmLayer')
        lstm_outputs, state_h_out, state_c_out = lstm(inputs=dense_layer_1,
                                                      mask=mask,
                                                      initial_state=[state_h_in, state_c_in] if self.initialize_lstm_with_prev_state is True else None)
        # reshape_layer = tf.reshape(lstm[0], [-1, model_config['lstm_cell_size']])
        reshape_layer = tf.keras.layers.Reshape(
            target_shape=(model_config['lstm_cell_size'],))(lstm_outputs)
        dense_layer_2 = tf.keras.layers.Dense(
            model_config['fcnet_hiddens'][1],
            activation=tf.nn.tanh,
            name='denseLayer2')(reshape_layer)
        logits_layer = tf.keras.layers.Dense(
            self.num_outputs,
            activation=tf.keras.activations.linear,
            name='logitsLayer')(dense_layer_2)
        value_layer = tf.keras.layers.Dense(
            1,
            activation=None,
            name='valueLayer')(dense_layer_2)
        self.base_model = tf.keras.Model(inputs=[inputs, seq_lens, state_h_in, state_c_in],
                                         outputs=[logits_layer, value_layer, state_h_out, state_c_out])
        self.register_variables(self.base_model.variables)
        self.base_model.summary()
        # json = self.base_model.to_json()
        # self.base_model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.RMSprop())
        # self.base_model.save('InputLayerRNNmodel.h5')  # https://github.com/tensorflow/tensorflow/issues/27543

    # Implement the core forward method
    def forward(self, input_dict, state, seq_lens):
        x = input_dict['obs']
        if x._rank() < 3:
            x = add_time_dimension(x, seq_lens)
        logits, self._value_out, state_h, state_c = self.base_model(inputs=[x, seq_lens, state[0], state[1]])

        return logits, state

    def get_initial_state(self):
        return [np.zeros(self.model_config['lstm_cell_size'], np.float32),
                np.zeros(self.model_config['lstm_cell_size'], np.float32)]

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class InputLayerRNNmodel_v2(TFModelV2):
    """
    Alternate version of InputLayerRNNmodel with changes made by Eric L.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kw):
        super(InputLayerRNNmodel_v2, self).__init__(obs_space, action_space, num_outputs, model_config, name, **kw)
        self.initialize_lstm_with_prev_state = model_config['custom_options']['initialize_lstm_with_prev_state']
        self.input_layer = tf.keras.layers.Input(shape=(None, obs_space.shape[0]),
                                                 name='inputLayer')
        self.state_in_c = tf.keras.layers.Input(shape=(model_config['lstm_cell_size']),
                                                name='c_in')
        self.state_in_h = tf.keras.layers.Input(shape=(model_config['lstm_cell_size']),
                                                name='h_in')
        self.seq_in = tf.keras.layers.Input(shape=(),
                                            name='seqLens')
        dense_layer_1 = tf.keras.layers.Dense(model_config['fcnet_hiddens'][0],
                                              activation=tf.nn.tanh,
                                              name='denseLayer1')(self.input_layer)
        lstm_out, state_h, state_c = tf.keras.layers.LSTM(
            model_config['lstm_cell_size'],
            return_sequences=True,
            return_state=True,
            name='lstmLayer')(inputs=dense_layer_1,
                              mask=tf.sequence_mask(self.seq_in) if self.model_config['max_seq_len'] > 1 else None,
                              initial_state=[self.state_in_c, self.state_in_h])
        dense_layer_2 = tf.keras.layers.Dense(
            model_config['fcnet_hiddens'][1],
            activation=tf.nn.tanh,
            name='denseLayer2')(lstm_out)
        logits_layer = tf.keras.layers.Dense(
            self.num_outputs,
            activation=tf.keras.activations.linear,
            name='logitsLayer')(dense_layer_2)
        value_layer = tf.keras.layers.Dense(
            1,
            activation=None,
            name='valueLayer')(dense_layer_2)

        self.base_model = tf.keras.Model(inputs=[self.input_layer, self.seq_in, self.state_in_h, self.state_in_c],
                                         outputs=[logits_layer, value_layer, state_h, state_c])
        self.register_variables(self.base_model.variables)
        self.base_model.summary()
        # json = self.base_model.to_json()
        # self.base_model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.RMSprop())
        # self.base_model.save('InputLayerRNNmodel_v2.h5')  # https://github.com/tensorflow/tensorflow/issues/27543

    # Implement the core forward method
    def forward(self, input_dict, state, seq_lens):
        x = input_dict['obs_flat']
        if x._rank() < 3:
            x = add_time_dimension(x, seq_lens)
        logits, self._value_out, h_state, c_state = self.base_model((x, seq_lens, state[0], state[1]))
        return tf.reshape(logits, [-1, self.num_outputs]), [h_state, c_state]

    def get_initial_state(self):
        initializer = self.model_config['custom_options'].get('rnn_state_initializer', 'zeros')
        if initializer == 'zeros':
            return [np.zeros(self.model_config['lstm_cell_size'], np.float32),
                    np.zeros(self.model_config['lstm_cell_size'], np.float32)]
        elif initializer == 'random_uniform':
            return [np.random.uniform(-1.0, 1.0, self.model_config['lstm_cell_size']).astype(np.float32),
                    np.random.uniform(-1.0, 1.0, self.model_config['lstm_cell_size']).astype(np.float32)]
        elif initializer == 'random_normal':
            return [np.random.normal(0, 0.1, self.model_config['lstm_cell_size']).astype(np.float32),
                    np.random.normal(0, 0.1, self.model_config['lstm_cell_size']).astype(np.float32)]
        else:
            print(f'RNN state initializer argument not recognized!')


    def value_function(self):
        return tf.reshape(self._value_out, [-1])


if __name__ == "__main__":
    initialize_lstm_with_prev_state = True  # if True restore the state from prevous batch, otherwise inialize with zeros
    lstm_cell_size = 32

    # uncomment one of these models, "SequenceMaskRNNmodel is the only model that works properly
    # default_model = "MaskingLayerRNNmodel"
    # default_model = "SequenceMaskRNNmodel"
    # default_model = "InputLayerRNNmodel"
    default_model = "InputLayerRNNmodel_v2"

    local_mode = True if getattr(sys, 'gettrace', None)() is not None else False  # run ray locally in single process if python in debug mode
    ray.init(num_cpus=0 if local_mode else None,
             local_mode=local_mode,
             object_store_memory=int(2e9),
             redis_max_memory=int(1e9),
             logging_level='DEBUG' if local_mode is True else 'INFO')
    ModelCatalog.register_custom_model("MaskingLayerRNNmodel", MaskingLayerRNNmodel)
    ModelCatalog.register_custom_model("SequenceMaskRNNmodel", SequenceMaskRNNmodel)
    ModelCatalog.register_custom_model("InputLayerRNNmodel", InputLayerRNNmodel)
    ModelCatalog.register_custom_model("InputLayerRNNmodel_v2", InputLayerRNNmodel_v2)
    tune.register_env("CartPoleStateless", lambda _: CartPoleStatelessEnv())
    tune.run(
        "PPO",
        stop={"episode_reward_mean": 200,
              "timesteps_total": 1e6},
        config={
            "env": "CartPoleStateless",
            "num_envs_per_worker": 1,
            "num_workers": 0 if local_mode is True else 2,
            "num_gpus": 0,
            "num_sgd_iter": 3,
            "vf_loss_coeff": 1e-4,
            'observation_filter': 'MeanStdFilter',
            "horizon": 200,  # CartPoleStatelessEnv does not have built in time limit
            "model": {
                "no_final_linear": True,
                "custom_model": default_model,
                "custom_options": {"initialize_lstm_with_prev_state": initialize_lstm_with_prev_state,
                                   "rnn_state_initializer": tune.grid_search([
                                       'zeros',
                                       # 'random_uniform',
                                       # 'random_normal',
                                   ])},
                "fcnet_hiddens": [64, 64],
                "lstm_cell_size": lstm_cell_size,
                "use_lstm": False,
                "state_shape": [lstm_cell_size, lstm_cell_size],
                "vf_share_layers": True,
                "max_seq_len": 3,
            }
        }
    )

