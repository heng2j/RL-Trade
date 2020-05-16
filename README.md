# RL-Trade

Still under construction.... 

Sample results:

Semi-continuous DQN ( V1 with LSTM) Agent navigating the 2008 financial crisis 

![Early stage in 2007](https://github.com/heng2j/RL-Trade/blob/master/for_readme/RL-Trade-Result_Late.gif?raw=true)


![Later stage in 2009](https://github.com/heng2j/RL-Trade/blob/master/for_readme/RL-Trade-Result_Early.gif?raw=true)

---
## Testing Results
Visualizing the performance of the agent to navigating the 2008 financial crisis. We held out the time-sereis of Sep 2007 to Sep 2009 for testing.

Transfer learning was done. The agent's model was first train with 20 years (Jan 1998 to May 2020) of daily OHLCV Microsoft stock data with Uniform Domain Randomization. Then train again on same 20 years period of time forApple's stock data. And then test on Apple's stock data during 08 financial crisis.

The agent was able to generate some decent purchasing and selling policy to buy low sell high.

![Testing Transfer Learning Results for 08 Financial crisis](https://github.com/heng2j/RL-Trade/blob/master/for_readme/TransferLearning08.png?raw=true)


Test the same agent to run during the COVID crisis in 2020.
![Testing Transfer Learning Results for 2020 COVID-19 crisis](https://github.com/heng2j/RL-Trade/blob/master/for_readme/Transferlearning%20results.png?raw=true)



---
## In Summary

This is jus an experiment to apply RL for simple trading environment. With very simple policy model. The performance of the agent is pretty stochastic and require a lot of training cycles. And much higher resolution of training data, and more realistic environment and reward design to train an agent to do decent job.

In a short period of time. the agent pretty much perform very similar due to it is bounded by the epsilon value 
![Multiple runs Testing Transfer Learning Results for 2020 COVID-19 crisis](https://github.com/heng2j/RL-Trade/blob/master/for_readme/Multiple%20runs%20with%20transfer%20learning.png?raw=true)

In a much longer period of time, it is behave varay. 
![Multiple runs Testing Transfer Learning Results for 08 Financial crisis](https://github.com/heng2j/RL-Trade/blob/master/for_readme/Multiple%20runs%20.png?raw=true)




References:



