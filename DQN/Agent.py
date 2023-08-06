import random
import copy
import numpy as np
from collections import deque

import torch
import torch.optim as optim

from config import *

from DQN_Architecture import DQN


class DQNAgent:
    def __init__(self, game_env, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.env = game_env

        self.Q = DQN(game_num_actions, k_frames, game_frame_height, game_frame_width)
        self.target_Q = None
        self.optimizer = optim.Adam(self.Q.parameters(), lr=0.0001)

    def choose_action(self, state):
        if np.random.uniform() < epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = self.target_Q(state)
            return torch.argmax(q_values).item()

    def train(self):
        """
        D = {e_1, e_2, ..., e_t}
        e_t = (s_t,a_t,r_t,s_{t+1}, terminated_t)

        e_t = (s_t,a_t,r_t,s_{t+1})

        """
        minibatch = self.sample_experiences(batch_size=batch)

        state_batch = torch.cat([info[0] for info in minibatch], dim=0)
        action_batch = torch.tensor([info[1] for info in minibatch])
        reward_batch = torch.tensor([info[2] for info in minibatch])
        next_state = torch.cat([info[3] for info in minibatch], dim=0)
        terminated_batch = torch.tensor([info[4] for info in minibatch])

        """ 先思考一个要怎么学习
        if one_terminated:
            y_pred = one_reward
        else:
            q_values = self.target_Q(one_next_state)
            gamma = 0.1
            y_pred = one_reward + gamma * torch.argmax(one_pred_q_values).item()

        loss = (y_pred - self.Q(one_state)[one_action]) ^ 2
        
        最后的loss要avg
        """

        # 推广到整个batch
        loss = 0
        for i in range(batch):
            one_state = state_batch[i]
            one_action = action_batch[i]
            one_reward = reward_batch[i]
            one_next_state = next_state[i]
            one_terminated = terminated_batch[i]

            if one_terminated:
                y_pred = one_reward
            else:
                gamma = 0.1
                y_pred = one_reward + gamma * torch.max(self.target_Q(one_next_state)).item()
            loss += (y_pred - self.Q(one_state)[one_action]) ** 2

        loss /= batch

        loss = torch.tensor(loss, requires_grad=True)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update(self):
        self.target_Q = copy.deepcopy(self.Q)

    # ----- ReplayMemory method -----
    def add(self, experience):
        self.buffer.append(experience)

    # ----- ReplayMemory method -----
    def sample_experiences(self, batch_size):
        return random.sample(self.buffer, batch_size)

    # ----- ReplayMemory method -----
    def __len__(self):
        return len(self.buffer)
