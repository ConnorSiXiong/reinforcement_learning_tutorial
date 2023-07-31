import gymnasium as gym
import random
import numpy as np

policy_q = [[1, 0, 0, 0.],
            [0, 0, 0, 1.],
            [1., 0., 0., 0.],
            [0.,
             0.,
             0.,
             1.],
            [1., 0., 0., 0.],
            [1.,
             0.,
             0.,
             0.],
            [1., 0., 0., 0.],
            [1.,
             0.,
             0.,
             0.],
            [0., 0., 0., 1.],
            [0.,
             1.,
             0.,
             0.],
            [1., 0., 0., 0.],
            [1.,
             0.,
             0.,
             0.],
            [1., 0., 0., 0.],
            [0.,
             0.,
             1.,
             0.],
            [0., 1., 0., 0.],
            [1.,
             0.,
             0.,
             0.]]
np.set_printoptions(suppress=True)
# env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="human")
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)

direction_dic = {0: "Move left",
                 1: "Move down",
                 2: "Move right",
                 3: "Move up"}

nA = env.action_space.n
nS = env.observation_space.n

num_states = nS
num_actions = nA

Q_table = np.zeros((nS, nA))
alpha = 0.9
gamma = 0.9

epsilon = 1
epsilon_decay_rate = 0.0001


def choose_action(state):
    if Q_table[state].sum() == 0 or np.random.uniform() < epsilon:
        a = env.action_space.sample()
    else:
        a = np.argmax(Q_table[state, :])
    return int(a)


last_state = None
game_round = 0

pass_moves = []
pass_times = 0
total_times = 0
num_episodes = 20000
error_list = []
reward_list = []
for i in range(num_episodes):
    error_list.append(np.sum(np.abs(policy_q - Q_table)) / 16)
    state, info = env.reset()
    # action = choose_action_training(state)
    action = choose_action(state)
    done = False
    count_rewards = 0
    while not done:  # max round
        next_state, reward, terminated, truncated, _ = env.step(action)

        reward = float(reward)

        # next_action = choose_action_training(next_state)
        next_action = choose_action(next_state)

        Q_table[state, action] = Q_table[state, action] + alpha * (
                reward + gamma * Q_table[next_state, next_action] - Q_table[state, action])

        state = next_state
        action = next_action

        if terminated or truncated or reward == 1:
            done = True
            if reward:
                count_rewards += 1
    reward_list.append(count_rewards)
    epsilon = max(epsilon - epsilon_decay_rate, 0)
print(Q_table)

import matplotlib.pyplot as plt

x = [i for i in range(num_episodes)]
y = error_list
plt.plot(x, y)
plt.xlabel('Iterations')
plt.ylabel('Loss Value')
plt.show()

sum_rewards = np.zeros(num_episodes)

for t in range(num_episodes):
    sum_rewards[t] = np.sum(reward_list[max(0, t - 100):(t + 1)])
plt.plot(sum_rewards)
plt.show()

success_times = 0
for i in range(1000):
    state, info = env.reset()
    done = False
    while not done:
        action = np.argmax(Q_table[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        state = next_state
        if terminated or truncated or reward == 1:
            done = True
            if reward == 1:
                success_times += 1
print(success_times)
