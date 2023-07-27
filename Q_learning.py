import gymnasium as gym
import random
import numpy as np

np.set_printoptions(suppress=True)
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)

observation, info = env.reset()

direction_dic = {0: "Move left",
                 1: "Move down",
                 2: "Move right",
                 3: "Move up"}
nA = env.action_space.n
nS = env.observation_space.n

num_states = nS
num_actions = nA

V = np.zeros(nS)
Q_table = np.zeros((nS, nA))
alpha = 0.1
gamma = 0.9


def choose_action(state):
    if Q_table[state].sum() == 0 or np.random.uniform() > 0.9:
        a = env.action_space.sample()
    else:
        a = np.argmax(Q_table[state, :])
    return int(a)


last_state = None
game_round = 0

pass_moves = []
pass_times = 0
total_times = 0
r1 = 0.9
r2 = 0.99
# training
for i in range(100000):
    state, info = env.reset()
    done = False
    r1 *= r2
    while not done:
        action = choose_action(state)

        next_state, reward, terminated, truncated, _ = env.step(action)

        reward = float(reward)

        Q_table[state, action] = Q_table[state, action] + alpha * (
                reward + gamma * Q_table[next_state, :].max() - Q_table[state, action])

        state = next_state
        if terminated or truncated:
            done = True

print(Q_table)

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
