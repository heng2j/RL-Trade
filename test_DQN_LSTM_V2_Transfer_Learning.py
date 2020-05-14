
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
    def __init__(self, env, inputshape=(5,82)):
        self.env     = env
        self.memory  = deque(maxlen=20000)
        
        self.gamma = 0.85
        self.epsilon = 0.5
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = .125
        self.batch_size = 1

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

        model.add(LSTM(units=64, return_sequences=True, batch_input_shape=tuple([self.batch_size]+ list(self.inputshape)), unroll=False, stateful=False))
        
        model.add(Dropout(0.2))  
        model.add(LSTM(units=64, return_sequences=True))  
        model.add(Dropout(0.2))
        model.add(LSTM(units=64, return_sequences=True))  
        model.add(Dropout(0.2))
        model.add(LSTM(units=64))  
        model.add(Dropout(0.2))  


        # print("self.env.action_space: ", self.env.action_space)
        # print(self.env.action_space.shape[0])

        model.add(Dense(self.env.action_space.shape[0], kernel_initializer='lecun_uniform'))
        model.add(Activation('relu')) #linear output so we can have range of real-valued outputs

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
            predict = self.model.predict(state)
            result = np.argmax(predict[0])
            # print("self.model.predict(state): ", predict)
            if result == 0:
                return [0, 0]
            elif result == 1:
                return [1, 0]
            else:
                return result
        # return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])
    
    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size: 
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                # print("Q_future: ", Q_future)
                # print("action: ", action)
                # print("target: ", target)
                # print("reward: ", reward)

                # print("reward + Q_future * self.gamma: ", reward + Q_future * self.gamma)
                target[0][0] = reward + Q_future * self.gamma

                # print("target after: ", target)
            self.model.fit(state, target, epochs=1, verbose=0)


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










model_path = './model_baseline_LSTM_5_iterations_100_steps_each.model'
model_path = './models/model_UDR_baseline_LSTM_V2_5_iterations_200_steps_each.model'
# model_path = './models/model_baseline_LSTM_V2_5_iterations_200_steps_each.model'

model_path =  './models/model_UDR_baseline_LSTM_V2_Transfer_Learning_5_iterations_200_steps_each.model'

# df = pd.read_csv('./data/MSFT.csv')
df = pd.read_csv('./data/AAPL_sub_Financial_Crisis.csv')
# df = pd.read_csv('./data/MSFT_sub_COVID_Crisis.csv')

df = df.sort_values('Date')


# export_summary_stat_path = './run_summary/base_line_A2C_V1_run_summary.csv'
# export_summary_stat_path = './run_summary/Test_baseline_LSTM_V2_run_summary.csv'
export_summary_stat_path = './run_summary/Test_UDR_baseline_LSTM_V2_Transfer_Learning_run_summary_t4.csv'
# export_summary_stat_path = './run_summary/Test_UDR_baseline_LSTM_V2_Transfer_Learning_MSFT_COVID_run_summary_t3.csv'




replay_size = 10
trials  = 2
trial_len = 10
Domain_Randomization_Interval = None

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: StockTradingEnv(df, render_mode='None',  export_summary_stat_path=export_summary_stat_path, replay_size=replay_size,trial_len=trial_len, Domain_Randomization_Interval=Domain_Randomization_Interval) ])

dqn_agent = DQN(env=env)
dqn_agent.model= load_model(model_path)




obs = obs = env.reset()

for _ in range(len(df)):
# for _ in range(3):

    action = dqn_agent.act(obs)
    # print("Outer action: ", action)
    # print("type action: ", type(action))
    if action is 0:
        action = [0, 0]
        print("0 action: ", action)
    if action is 1:
        action = [1, 0]
        print("1 action: ", action)
    obs, rewards, done, summary_stat = env.step([action])
    env.render(title="APPL-2008 Financial_Crisis")



columns = ['step', 'date', 'balance', 'shares_held', 'total_shares_sold',
            'cost_basis', 'total_sales_value', 'net_worth', 'max_net_worth',
            'cur_reward', 'cur_action', 'profit'
            ]

# print("summary_stat: ", summary_stat[0])
df = pd.DataFrame(summary_stat[0],columns=columns)
df.to_csv(export_summary_stat_path)
