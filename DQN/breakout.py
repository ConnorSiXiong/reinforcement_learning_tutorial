"""
This is the breakout training v1.0.
The model learns limited info from the env.

This is dispatched.
"""
import gymnasium as gym
import torch
import random
import numpy as np

from Agent import DQNAgent

from config import *
from collections import deque
from ImageProcessing import process_image
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Set up the Breakout environment
# env = gym.make('ALE/Breakout-v5', render_mode="human", frameskip=1)
env = gym.make('ALE/Breakout-v5', render_mode="rgb_array")
state, info = env.reset(seed=50)

# Define actions for the Breakout game
# The available actions are 0 (no-op), 1 (fire), 2 (move right), and 3 (move left)
ACTIONS = [0, 1, 2, 3]

agent = DQNAgent(env, memory_capacity)

states_buffer = deque(maxlen=k_frames)
terminated_counter = 0
# Play the game for a few steps
num_episodes = 20000
reward_list = []
reward_counter = 0
for i in range(num_episodes):
    # Choose a random action (you can implement a RL agent later to select actions intelligently)

    action = agent.choose_action()

    # Perform the action and observe the outcome
    next_state, reward, terminated, _, info = env.step(action)
    if reward:
        print('reward_counter', reward_counter)
        reward_counter += 1

    if terminated:

        terminated_counter += 1
        print('reward_counter', reward_counter)
        reward_list.append(reward_counter)
        reward_counter = 0
        env.reset(seed=50)

    # Add experience
    process_state = process_image(state)
    process_state = torch.tensor(process_state, dtype=torch.float32, device=device)

    height, width = process_state.shape[-2], process_state.shape[-1]
    desired_height, desired_width = 84, 84
    pad_height = max(desired_height - height, 0)
    pad_width = max(desired_width - width, 0)
    padding = (0, pad_width, 0, pad_height)
    process_state = F.pad(process_state, padding)
    states_buffer.append(process_state)
    stack_states = np.stack(states_buffer, axis=2)
    stack_states_torch = torch.Tensor(stack_states)

    process_next_state = process_image(next_state)
    process_next_state = torch.tensor(process_next_state, dtype=torch.float32, device=device)
    process_next_state = F.pad(process_next_state, padding)
    states_buffer.append(process_next_state)
    next_stack_states = np.stack(states_buffer, axis=2)
    next_stack_states_torch = torch.Tensor(next_stack_states)

    if i >= k_frames + 1:
        agent.add([stack_states_torch, action, reward, next_stack_states_torch, terminated])

    agent.train()

    # if i % 50000 == 0:
    #     print('hey')
    #     agent.update()

    if terminated_counter == 10:
        print('round:', i)
        agent.update()
        terminated_counter = 0

    state = next_state

# Close the environment when done
env.close()

import matplotlib.pyplot as plt
sum_rewards = np.zeros(num_episodes)

for t in range(num_episodes):
    sum_rewards[t] = np.sum(reward_list[max(0, t - 100):(t + 1)])
plt.plot(sum_rewards)
plt.show()
