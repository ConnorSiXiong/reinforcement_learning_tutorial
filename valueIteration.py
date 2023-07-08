import random

import gymnasium as gym
import pprint

import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="human")
env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True, render_mode="rgb_array")
# env = gym.make('FrozenLake-v1', desc=generate_random_map(size=8, seed=100), is_slippery=True, render_mode="rgb_array")
observation, info = env.reset()

nA = env.action_space.n
nS = env.observation_space.n

num_states = nS
num_actions = nA

GAMMA = 0.9

V = np.zeros(nS)
policy = np.ones((nS, nA)) / nA
THETA = 1e-4


def get_action_probabilities(cur_state):
    return policy[cur_state]


def choose_action(cur_state):
    return np.argmax(get_action_probabilities(cur_state))


def value_iteration():
    delta = 0.1
    while delta > THETA:
        delta = 0
        for s in range(num_states):
            value = V[s]

            expected_value_list = np.zeros(num_actions)
            for a in range(num_actions):
                expected_value_of_action_a = 0
                for prob, next_state, r, _ in env.P[s][a]:
                    expected_value_of_action_a += prob * (r + GAMMA * V[next_state])
                expected_value_list[a] = expected_value_of_action_a

            best_action = np.argmax(expected_value_list)
            best_expected_value = max(expected_value_list)
            V[s] = best_expected_value

            delta = max(delta, abs(value - best_expected_value))


def update_policy():
    for s in range(num_states):
        expected_value_list = np.zeros(num_actions)
        for a in range(num_actions):
            expected_value_of_action_a = 0
            for prob, next_state, r, _ in env.P[s][a]:
                expected_value_of_action_a += prob * (r + GAMMA * V[next_state])
            expected_value_list[a] = expected_value_of_action_a
        policy[s] = np.eye(num_actions)[np.argmax(expected_value_list)]


value_iteration()
update_policy()

success = 0
for i in range(1000):
    for j in range(100):
        # action = env.action_space.sample()  # agent policy that uses the observation and info
        action = choose_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated or reward == 1:
            # env = gym.make('FrozenLake-v1', desc=None, is_slippery=True, render_mode="rgb_array")
            observation, info = env.reset()
            if reward == 1:
                success += 1
            break

env.close()
print('success:', success)
