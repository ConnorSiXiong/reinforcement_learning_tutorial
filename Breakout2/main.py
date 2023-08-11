import gymnasium as gym
import numpy as np
from collections import deque
import torch
from Imager import process_image

from ReplayMemory import ReplayMemory
"""
Pytorch:
channel, height, width
"""

torch.set_num_threads(4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

k_frames = 4

env = gym.make('ALE/Breakout-v5', render_mode="human", frameskip=1)

# Define actions for the Breakout game
# The available actions are 0 (no-op), 1 (fire), 2 (move right), and 3 (move left)
ACTIONS = [0, 1, 2, 3]

terminated_counter = 0
# Play the game for a few steps
num_episodes = 700
reward_list = []
memory_buffer_capacity = 100000

game_frame_height = 81
game_frame_width = 72
game_num_actions = 4

replay_memory = ReplayMemory(k_frames+1, memory_buffer_capacity, game_frame_height, game_frame_width)

round_counter = 0
states_buffer = deque(maxlen=k_frames + 1)

# Todo: currently, image process -> each frame, which is not efficient. Move the process_image in stack_states

for i in range(num_episodes):
    # Choose a random action (you can implement a RL agent later to select actions intelligently)
    state, info = env.reset()

    state = process_image(state)
    print(state.shape)
    terminated = False
    count_rewards = 0
    pre_reward = 0
    while not terminated and count_rewards <= 200:
        round_counter += 1
        action = env.action_space.sample()
        next_state, reward, terminated, _, info = env.step(action)

        next_state = process_image(next_state)

        states_buffer.append(state)
        if round_counter >= k_frames + 1:
            stack_states = torch.cat(list(states_buffer), dim=0).unsqueeze(0)
            cur_stack_states = torch.cat(list(states_buffer)[1:], dim=0).unsqueeze(0)
            print(stack_states.shape)
            print(cur_stack_states.shape)
            replay_memory.add(stack_states, action, reward, terminated)

            for k in replay_memory.sample(2):
                print(k.shape)
        if terminated:
            print('end')
        state = next_state
# Close the environment when done
env.close()
