# Reinforcement learning



## Markov Decision Process
A [Markov Decision Process](https://en.wikipedia.org/wiki/Markov_decision_process) is a model where an agent moves between a number of states. At each time step, the agent has to choose an action *a* among those available in its current state *s*. The process then randomly selects a new state *s'* according the to action transition probabilities, and gives the agent an immediate reward *r*. The goal of the agent is to maximize its cumulative reward.

The agent acts according to a policy &pi;, a function that selects the action to be taken in each state. Seeking the optimal policy which gives the highest expected reward is an optimization problem which may be solved with the [Value iteration and Policy iteration](ROB311_MDP_Value_Policy_iteration.ipynb) algorithms.


## Multi-armed bandits

## Adversarial bandits

## Contextual bandits

## Q-learning and Deep Q learning
