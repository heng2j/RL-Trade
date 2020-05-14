# Reference Code 
# https://gist.github.com/arsalanaf/d10e0c9e2422dba94c91e478831acb12
# https://github.com/Stable-Baselines-Team/stable-baselines-tf2
# https://github.com/notadamking/Stock-Trading-Visualization




import gym

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import IPython.display as Display
import PIL.Image as Image

# from stable_baselines.common.policies import MlpPolicy
# Using 
from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines import PPO

from env.StockTradingEnv import StockTradingEnv

import pandas as pd
import numpy as np
import random


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM, Input
from tensorflow.keras.layers import Add, Multiply
from tensorflow.keras.optimizers import RMSprop, Adam

from collections import deque



# determines how to assign values to each state, i.e. takes the state
# and action (two-input model) and determines the corresponding value

class ActorCritic:
    def __init__(self, env, sess):
        self.env = env
        self.sess = sess

        self.learning_rate = 0.001
        self.epsilon = 1
        self.epsilon_decay = .9995
        self.gamma = .95
        self.tau = .125

        # ===================================================================== #
        #                               Actor Model                             #
        # Chain rule: find the gradient of chaging the actor network params in  #
        # getting closest to the final value network predictions, i.e. de/dA    #
        # Calculate de/dA as = de/dC * dC/dA, where e is error, C critic, A act #
        # ===================================================================== #

        self.memory = deque(maxlen=2000)
        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()

        self.actor_critic_grad = tf.placeholder(tf.float32,
                                                [None, self.env.action_space.shape[0]])  # where we will feed de/dC (from critic)
        
        # print("self.env.observation_space.shape: ", self.env.observation_space.shape)
        # print("self.env.action_space.shape: ", self.env.action_space.shape[0])
        actor_model_weights = self.actor_model.trainable_weights

        # print("self.actor_critic_grad shape: ", self.actor_critic_grad.shape)
        # print("actor_model_weights len: ", len(actor_model_weights))
        # print("self.actor_model.output shape: ", self.actor_model.output.shape)


        self.actor_grads = tf.gradients(self.actor_model.output,
                                        actor_model_weights, -self.actor_critic_grad)  # dC/dA (from actor)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).apply_gradients(grads)

        # ===================================================================== #
        #                              Critic Model                             #
        # ===================================================================== #

        self.critic_state_input, self.critic_action_input, \
        self.critic_model = self.create_critic_model()

        print(        self.critic_model.summary())

        _, _, self.target_critic_model = self.create_critic_model()

        print(        self.target_critic_model.summary())

        self.critic_grads = tf.gradients(self.critic_model.output,
                                         self.critic_action_input)  # where we calcaulte de/dC for feeding above

        # Initialize for later gradient calculations
        self.sess.run(tf.initialize_all_variables())

    # ========================================================================= #
    #                              Model Definitions                            #
    # ========================================================================= #

    def create_actor_model(self):
        state_input = Input(shape=self.env.observation_space.shape[1:])


        h1 = Dense(24, activation='relu')(state_input)
        h2 = Dense(48, activation='relu')(h1)
        h3 = Dense(24, activation='relu')(h2)
        output = Dense(self.env.action_space.shape[0], kernel_initializer='lecun_uniform', activation='relu')(h3)

        # print("self.env.action_space.shape[0]: ", self.env.action_space.shape[0])
        # print("Output.shape: ", output.shape)

        model = Model(state_input, output)
        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, model

    def create_critic_model(self):
        state_input = Input(shape=self.env.observation_space.shape[1:])


        state_h1 = Dense(24, activation='relu')(state_input)
        state_h2 = Dense(48)(state_h1)

        action_input = Input(shape=self.env.action_space.shape) # Hack the shape for error ValueError: Error when checking input: expected input_4 to have shape (2,) but got array with shape (1,)
        # action_input = Input(shape=(1,))
        action_h1 = Dense(48)(action_input)

        merged = Add()([state_h2, action_h1])
        merged_h1 = Dense(24, activation='relu')(merged)
        output = Dense(1, activation='relu')(merged_h1)
        model = Model([state_input, action_input], output)

        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model

    # ========================================================================= #
    #                               Model Training                              #
    # ========================================================================= #

    def remember(self, cur_state, action, reward, new_state, done):
        self.memory.append([cur_state, action, reward, new_state, done])

    def _train_actor(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, _ = sample
            predicted_action = self.actor_model.predict(cur_state)
            grads = self.sess.run(self.critic_grads, feed_dict={
                self.critic_state_input:  cur_state,
                self.critic_action_input: predicted_action
            })[0]

            self.sess.run(self.optimize, feed_dict={
                self.actor_state_input: cur_state,
                self.actor_critic_grad: grads
            })

    def _train_critic(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, done = sample
            # print("done: ", done)  
            # print('doen type', int(done[0]))   

            # if not done:
            if int(done[0]) == 0:
                # print("Not Done!!!")
                target_action = self.target_actor_model.predict(new_state)
                # print("_train_critic target_action", target_action)
                # print("_train_critic new_state shape", new_state.shape)
                # print("_train_critic target_action shape", target_action.shape)
                # print("_train_critic target_action shape", target_action.shape)
                future_reward = self.target_critic_model.predict(
                    [new_state, target_action])[0][0]
                reward += self.gamma * future_reward

            # print("done: ", done)            
            # print("_train_critic cur_state shape", cur_state.shape)
            # print("_train_critic action", action)
            # print("_train_critic action shape", action.shape)
            # print("_train_critic reward", reward)

            self.critic_model.fit([cur_state, action], reward, verbose=0)

    def train(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        rewards = []
        samples = random.sample(self.memory, batch_size)
        self._train_critic(samples)
        self._train_actor(samples)

    # ========================================================================= #
    #                         Target Model Updating                             #
    # ========================================================================= #

    def _update_actor_target(self):
        actor_model_weights = self.actor_model.get_weights()
        actor_target_weights = self.target_actor_model.get_weights()

        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]
        self.target_actor_model.set_weights(actor_target_weights)

    def _update_critic_target(self):
        critic_model_weights = self.critic_model.get_weights()
        critic_target_weights = self.target_critic_model.get_weights()

        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]
        self.target_critic_model.set_weights(critic_target_weights)

    def update_target(self):
        self._update_actor_target()
        self._update_critic_target()

    # ========================================================================= #
    #                              Model Predictions                            #
    # ========================================================================= #

    def act(self, cur_state):
        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            samples = []
            for i in range(10):
                samples.append(self.env.action_space.sample())
            return np.array(samples)

        # return np.amax(self.actor_model.predict(cur_state), axis=0)    # np.argmax(self.actor_model.predict(state)[0])
        return self.actor_model.predict(cur_state)

    def save_model(self, fn):
        self.actor_model.save(fn)
              

df = pd.read_csv('./data/MSFT.csv')
df = df.sort_values('Date')

replay_size = 10
trials  = 10
trial_len = 300
Domain_Randomization_Interval = 300
# filename = 'base_line_LSTM_render.txt'
# filename = 'base_line_A2C_V1_render.txt'
filename = 'UDR_base_line_A2C_V1_render.txt'


# export_summary_stat_path = './run_summary/base_line_A2C_V1_run_summary.csv'
export_summary_stat_path = './run_summary/UDR_base_line_A2C_V1_run_summary.csv'

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: StockTradingEnv(df, render_mode='None', filename=filename, export_summary_stat_path=export_summary_stat_path, replay_size=replay_size,trial_len=trial_len, Domain_Randomization_Interval=Domain_Randomization_Interval) ])

sess = tf.compat.v1.Session()
actor_critic = ActorCritic(env, sess)



for trial in range(trials): 

    cur_state = env.reset()
    cur_state = cur_state.reshape(cur_state[0].shape)
    
    for step in range(trial_len):
        action = actor_critic.act(cur_state)


        new_state, reward, done, summary_stat = env.step(action)
        new_state = new_state.reshape(new_state[0].shape)
        # print("Step reward: ", reward)

        reward = reward*10 if not done else -10 # TODO - Need to adjust this for better training / Maybe using other algorithm may help
        
        # TODO - Need to modify the enviroment to generatte continuous rewards
        rewards = []
        for i in range(10):
            rewards.append(random.uniform((int(reward) - 10),(int(reward) + 10)))
        
        env.render(title="MSFT")

        actor_critic.remember(cur_state, action, np.array(rewards), new_state, done)
        # else:
        #     actor_critic.remember(cur_state, np.array([action]), reward, new_state, done)


        actor_critic.train()
        actor_critic.update_target() # iterates target model

        cur_state = new_state
        if done:
            break

    print("Completed trial #{} ".format(trial))
    actor_critic.epsilon = 1.0 - (trial * 0.1)
    print("Reset Agent's epsilon to:  ", actor_critic.epsilon)


columns = ['step', 'date', 'balance', 'shares_held', 'total_shares_sold',
            'cost_basis', 'total_sales_value', 'net_worth', 'max_net_worth',
            'cur_reward', 'cur_action', 'profit'
            ]

df = pd.DataFrame(summary_stat[0],columns=columns)
df.to_csv(export_summary_stat_path)

# model_code = 'baseline_A2C_V1_{0}_iterations_{1}_steps_each'.format(trials,trial_len)
model_code = 'UDR_baseline_A2C_V1_{0}_iterations_{1}_steps_each'.format(trials,trial_len)

actor_critic.actor_model.save_weights("./models/model_{}.h5".format(model_code))

