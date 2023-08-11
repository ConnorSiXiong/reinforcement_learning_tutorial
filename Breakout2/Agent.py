import torch
import torch.nn.functional as F
import torch.optim as optim

from ReplayMemory import ReplayMemory

import numpy as np


class Agent:
    def __init__(self, game_env, game_num_actions, replay_memory: ReplayMemory, batch_size):
        self.env = game_env
        self.Q = DQN(game_num_actions, k_frames, game_frame_height, game_frame_width).to(device)
        self.target_Q = DQN(game_num_actions, k_frames, game_frame_height, game_frame_width).to(device)
        self.optimizer = optim.RMSprop(self.Q.parameters(), lr=0.00025, eps=0.001, alpha=0.95)
        self.replay_memory = replay_memory
        self.batch_size = batch_size

        self.gamma = 0.99

    def choose_action(self):
        if np.random.uniform() < self.epsilon or self.__len__() == 0:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = self.target_Q(self.get_latest_states_stack())
            return torch.argmax(q_values).item()

    def train(self):
        minibatch_states, minibatch_next_states, minibatch_actions, minibatch_rewards, minibatch_terminates = \
            self.replay_memory.sample(self.batch_size)

        self.optimizer.zero_grad()

        q_values = self.Q(minibatch_states).gather(1, minibatch_actions)
        next_q_values = self.target_Q(minibatch_next_states).max(dim=1, keepdim=True)[0].detach()
        mask = 1 - minibatch_terminates
        target = (minibatch_rewards + self.gamma * next_q_values * mask)
        loss = F.smooth_l1_loss(q_values, target)

        loss.backward()

        torch.nn.utils.clip_grad_value_(self.Q.parameters(), 1)
        self.optimizer.step()

    def update(self):
        print('model update')
        self.target_Q.load_state_dict(self.Q.state_dict())

    def save(self):
        pass
