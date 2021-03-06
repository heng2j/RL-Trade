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




from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM
from tensorflow.keras.optimizers import RMSprop, Adam

from collections import deque

class DQN:
    def __init__(self, env, inputshape=(5,42)):
        self.env     = env
        self.memory  = deque(maxlen=20000)
        
        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = .125

        self.inputshape = inputshape

        self.model        = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model   = Sequential()
       # state_shape  = list(self.env.observation_space.shape.items())[0][1]
        #Reshaping for LSTM 
        #state_shape=np.array(state_shape)
        #state_shape= np.reshape(state_shape, (30,4,1))
        '''
        model.add(Dense(24, input_dim=state_shape[1], activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate))
        '''
        model.add(LSTM(64,
               input_shape=self.inputshape,
               #return_sequences=True,
               stateful=False
               ))
        model.add(Dropout(0.5))
        
        #model.add(LSTM(64,
                       #input_shape=(1,4),
                       #return_sequences=False,
        #               stateful=False
        #               ))
        model.add(Dropout(0.5))
        

        # print("self.env.action_space: ", self.env.action_space)
        # print(self.env.action_space.shape[0])

        model.add(Dense(self.env.action_space.shape[0], kernel_initializer='lecun_uniform'))
        model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

        rms = RMSprop()
        adam = Adam()
        model.compile(loss='mse', optimizer=adam)
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            # print("Sampled action space")
            return self.env.action_space.sample()
        else:
            result = np.argmax(self.model.predict(state)[0])
            print("self.model.predict(state): ", self.model.predict(state))
            if result == 0:
                return [0, 0]
            elif result == 1:
                return [1, 0]
            else:
                return result
        # return np.argmax(self.model.predict(state)[0])

   
    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)
              
    def show_rendered_image(self, rgb_array):
        """
        Convert numpy array to RGB image using PILLOW and
        show it inline using IPykernel.
        """
        Display.display(Image.fromarray(rgb_array))

    def render_all_modes(self, env):
        """
        Retrieve and show environment renderings
        for all supported modes.
        """
        for mode in self.env.metadata['render.modes']:
            print('[{}] mode:'.format(mode))
            self.show_rendered_image(self.env.render(mode))






# TODO 
# Confirm how to train the agent
# Modiify traiing dataset
# Set up test set 
# - Modify training
# Add more osticles  











df = pd.read_csv('./data/MSFT.csv')
df = df.sort_values('Date')

replay_size = 10
trials  = 5
trial_len = 100
Domain_Randomization_Interval = None
filename = 'base_line_render.txt'
# filename = 'UDR_render.txt'


# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: StockTradingEnv(df, render_mode='file', filename=filename, replay_size=replay_size, Domain_Randomization_Interval=Domain_Randomization_Interval) ])

obs = env.reset()

print("obs_shape: ", obs.shape)


gamma   = 0.9
epsilon = .95


# updateTargetNetwork = 1000
dqn_agent = DQN(env=env, inputshape=obs.shape[1:])
steps = []


for trial in range(trials): 

    cur_state = obs = env.reset()
    
    for step in range(trial_len):
        action = dqn_agent.act(cur_state)
        print("Outter action: ", action)
        print(type(action))
        # TODO - Not sure why will return scalar 0 or 1
        if action is 0:
            action = [0, 0]
            print("0 action: ", action)
        elif action is 1:
            action = [1, 0]
            print("1 action: ", action)

        new_state, reward, done, info = env.step([action])
        reward = reward*10 if not done else -10 # TODO - Need to adjust this for better training / Maybe using other algorithm may help
        env.render(title="MSFT")
        # new_state =list(new_state.items())[0][1]
        # new_state= np.reshape(new_state, (30,4,1))
        dqn_agent.target_train() # iterates target model

        cur_state = new_state
        if done:
            break

print("Completed trial #{} ".format(trial))
# dqn_agent.render_all_modes(env)
model_code = 'baseline_{0}_iterations_{1}_steps_each'.format(trials,trial_len)
# model_code = 'UDR_{0}_iterations_{1}_steps_each'.format(trials,Domain_Randomization_Interval)

dqn_agent.save_model("model_{}.model".format(model_code))
        
