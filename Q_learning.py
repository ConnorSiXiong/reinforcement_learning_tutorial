import gymnasium as gym
import random
import numpy as np

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
seed = 20
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


def choose_action_training(state):
    if Q_table[state].sum() == 0 or np.random.uniform() > 0.9:
        while True:
            a = env.action_space.sample()
            _, s, r, _ = env.P[state][a][0]
            if s != state:
                break
    else:
        a = np.argmax(Q_table[state])
    return int(a)


def choose_action(state):
    if Q_table[state].sum() == 0:
        while True:
            a = np.random.randint(num_actions)
            _, s, r, _ = env.P[state][a][0]
            if s != state:
                break
    else:
        a = np.argmax(Q_table[state])

    return int(a)


last_state = None
game_round = 0

pass_moves = []
pass_times = 0
total_times = 0

# training
for i in range(100000):
    game_round += 1
    cur_state = observation
    action = choose_action_training(cur_state)
    next_state, reward, terminated, truncated, _ = env.step(action)

    reward = float(reward)

    Q_table[cur_state, action] = Q_table[cur_state, action] + alpha * (
            reward + gamma * Q_table[next_state, :].max() - Q_table[cur_state, action])
    # action = env.action_space.sample()  # agent policy that uses the observation and info

    observation = next_state
    if terminated or truncated:
        observation, info = env.reset()

print(Q_table)

observation, info = env.reset()
moves = []
cur_moves = []
for i in range(100000):
    if observation == 0:
        cur_moves = []

    game_round += 1
    action = choose_action(observation)
    cur_moves.append(direction_dic[action])
    # action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    # print('observation', observation)
    if terminated or truncated or game_round == 50:
        observation, info = env.reset()
        total_times += 1
        moves.append(cur_moves)
        if reward == 1:
            pass_moves.append(game_round)
            pass_times += 1
        game_round = 0

env.close()
from statistics import mean

# print(mean(pass_moves))
print(mean(pass_moves))
print('pass_times', pass_times)
print('total_times', total_times)
print(moves[100])
