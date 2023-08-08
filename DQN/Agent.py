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
        self.target_Q = copy.deepcopy(self.Q)
        self.optimizer = optim.AdamW(self.Q.parameters(), lr=LR, amsgrad=True)

        self.max_epsilon = 1.0
        self.min_epsilon = 0.1
        self.epsilon = self.max_epsilon
        self.epsilon_decay = 0.0001

    def choose_action(self):
        if np.random.uniform() < self.epsilon or self.__len__() == 0:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = self.target_Q(self.get_latest_states_stack())
            return torch.argmax(q_values).item()

    def train(self):
        """
        D = {e_1, e_2, ..., e_t}
        e_t = (s_t,a_t,r_t,s_{t+1}, terminated_t)

        e_t = (s_t,a_t,r_t,s_{t+1})

        """
        if len(self.buffer) < batch:
            return

        minibatch = self.sample_experiences(batch_size=batch)
        # print(len(minibatch))
        # print(minibatch[0][0].shape)
        # print(minibatch[1][0].shape)
        # print(minibatch[2][0].shape)
        # print(minibatch[3][0].shape)
        state_batch = torch.cat([info[0].unsqueeze(0) for info in minibatch], dim=0)
        action_batch = torch.tensor([info[1] for info in minibatch])
        reward_batch = torch.tensor([info[2] for info in minibatch])
        next_state = torch.cat([info[3].unsqueeze(0) for info in minibatch], dim=0)
        terminated_batch = torch.tensor([info[4] for info in minibatch])
        # print('state_batch', state_batch.shape)
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
            one_state = one_state.permute(2, 0, 1)
            one_state = one_state.unsqueeze(0)

            one_action = action_batch[i]
            one_reward = reward_batch[i]
            one_next_state = next_state[i]
            one_next_state = one_next_state.permute(2, 0, 1)
            one_next_state = one_next_state.unsqueeze(0)

            one_terminated = terminated_batch[i]
            # print('one_state', one_state.shape)
            # print('one_next_state', one_next_state.shape)
            # print('one_reward', one_reward)
            # print('one_action', one_action)
            # print('one_terminated', one_terminated)
            if one_terminated:
                y_pred = one_reward
            else:
                gamma = 0.1
                y_pred = one_reward + gamma * torch.max(self.target_Q(one_next_state)).item()

            # print(self.Q(one_state))
            # print(self.Q(one_state).shape)
            # print(y_pred)

            loss += (y_pred - self.Q(one_state)[0, one_action]) ** 2

        loss /= batch
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.Q.parameters(), 100)
        self.optimizer.step()
        self.epsilon = max(
            self.min_epsilon, self.epsilon - (
                    self.max_epsilon - self.min_epsilon
            ) * self.epsilon_decay
        )

    def update(self):
        print('model update')
        self.target_Q = copy.deepcopy(self.Q)

    # ----- ReplayMemory method -----
    def add(self, experience):
        self.buffer.append(experience)

    # ----- ReplayMemory method -----
    def sample_experiences(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def get_latest_states_stack(self):
        states_stack = self.buffer[-1][0]
        states_stack = states_stack.permute(2, 0, 1)
        states_stack = states_stack.unsqueeze(0)
        return states_stack

    # ----- ReplayMemory method -----
    def __len__(self):
        return len(self.buffer)
