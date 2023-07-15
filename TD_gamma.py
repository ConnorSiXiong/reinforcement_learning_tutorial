import gymnasium as gym
import numpy as np

np.set_printoptions(suppress=True)
interval = 1000

alpha = 0.01
gamma = 0.9
lambda_ = 0.6
num_episodes = 100000
shape = 4

# Initialize Frozen Lake environment
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="rgb_array")
state, info = env.reset()

# Initialize value function and eligibility trace
V = np.zeros(env.observation_space.n)
E = np.zeros(env.observation_space.n)
true_value = np.array([0.06821027, 0.06083811, 0.07398942, 0.05534566, 0.09123541, 0.
    , 0.11198184, 0., 0.14493477, 0.2471585, 0.29937932, 0.,
              0., 0.37969132, 0.63889194, 0.])
error_list = []

for i in range(num_episodes):
    for j in range(100):
        # Choose action randomly (exploratory policy)
        action = env.action_space.sample()

        # Take action and get reward and next state
        next_state, reward, terminated, truncated, _ = env.step(action)
        # Compute TD Error
        td_error = reward + gamma * V[next_state] - V[state]
        # print(td_error)

        # Update eligibility trace
        E[state] += 1

        # Update value function and eligibility trace
        for s in range(env.observation_space.n):
            V[s] = V[s] + alpha * td_error * E[s]
            E[s] = gamma * lambda_ * E[s]

        # Update current state
        state = next_state

        if terminated or truncated or reward == 1:
            state, info = env.reset()
            error_list.append(np.average(np.abs(true_value - V)))
            break

# Print estimated value function
# print(V.reshape(shape, shape))
# print(E.reshape(shape, shape))

import matplotlib.pyplot as plt

x = [i for i in range(len(error_list))]
y = error_list
plt.plot(x, y)
plt.show()
