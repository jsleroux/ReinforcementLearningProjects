### Introduction
This repository contains experiments that I did to learn more about reinforcement learning.

### [Reinforcement Learning module in UC Berkeley CS188 Intro to AI](cs188/q3)
My first introduction to the subject was from the [CS188 course from Berkeley](http://ai.berkeley.edu/home.html).The learning framework is made around a pacman game, so it's very intuitive, visual and fun! The learning objectives were the following:
1- Markov Decision Process (Value Iteration)
2- Q-Learning and approximate Q-Learning

### [Multi-Armed bandit](Multi_Armed_Bandits.ipynb)
A multi-armed bandit example. The objectives of this project were:
- Reproduced Sutton and Barton's famous 10-armed Testbed
- Implementation of an E-greedy policies (epsilon of 0, 0.1 and 0.01)
- Incremental implementation (less memory hungry)

### OpenAI gyms
I solved a few OpenAI gym with the help of some reinforcement learning algorithms,

#### [Cartpole-v0 and Q-Learning](Q_learning_CartPole_v0.ipynb)
- Decrease the learning rate with adaptive learning
- Discretize the state space into buckets

#### [FrozenLake-v0 and Markov Decision Processes](FrozenLake_Markov_Decision_Process.ipynb)
- Value iteration algorithm 

#### [BlackJack-v0 and MonteCarlo](Monte_Carlo_BlackJack.ipynb)
- Prediction problem. The objective is simply to learn a state-value function for a simple policy, which is to stick when user count >= 20.
- Control problem. The objective is to approximate the optimal policy. This version works without the 'Exploring Starts' because we can't decide a specific state where to begin. We have to do full episodes.
