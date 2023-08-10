"""
This is the breakout training v2.0.
The model learns limited info from the env.

This is not well organized and dispatched.

The input size is (81, 72), no padding.
"""
import gymnasium as gym
import numpy as np
import random
import copy
from collections import deque
import torch.nn.functional as F
import torch.optim as optim

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

epsilon = 1
epsilon_decay_rate = 0.0001
memory_capacity = 20000
N = 1000000

k_frames = 4
game_frame_height = 81
game_frame_width = 72
game_num_actions = 4

target_Q_update_steps = 10000  # equal to C
C = 10000

batch = 32
LR = 0.00025

gray_image = np.random.random((256, 256))
# Display grayscale image
plt.imshow(gray_image, cmap='gray')
plt.show()


class DQN(nn.Module):
    def __init__(self, num_actions=game_num_actions, input_channels=k_frames, height=game_frame_height,
                 width=game_frame_width):
        super(DQN, self).__init__()
        self.first_hidden_layer = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=32,
                      kernel_size=8,
                      stride=4),
            nn.ReLU())

        self.second_hidden_layer = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=4,
                      stride=2),
            nn.ReLU())

        self.third_hidden_layer = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1),
            nn.ReLU())
        """
        Solution 1
        conv_out_size = self._get_conv_output_size((input_channels, height, width))

        self.final_hidden_layer = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU()
        )
        """
        # Solution 2
        # self.final_hidden_layer = nn.Sequential(
        #     nn.Linear(3136, 512),
        #     nn.ReLU()
        # )
        self.final_hidden_layer = nn.Sequential(
            nn.Linear(1920, 512),
            nn.ReLU()
        )

        self.output_layer = nn.Linear(512, num_actions)

    def forward(self, x):
        x = self.first_hidden_layer(x)
        x = self.second_hidden_layer(x)
        x = self.third_hidden_layer(x)

        x = torch.flatten(x, start_dim=1)

        x = self.final_hidden_layer(x)
        x = self.output_layer(x)
        return x

    def _get_conv_output_size(self, shape):
        # Function to calculate the output size of the convolutional layers
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = self.first_hidden_layer(x)
            x = self.second_hidden_layer(x)
            x = self.third_hidden_layer(x)
            print(x.view(1, -1).size())
            print(64 * 7 * 7)
            return x.view(1, -1).size(1)


class DQNAgent:
    def __init__(self, game_env, capacity):

        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.env = game_env

        self.Q = DQN(game_num_actions, k_frames, game_frame_height, game_frame_width).to(device)

        self.target_Q = copy.deepcopy(self.Q).to(device)
        self.target_Q.eval()

        # self.optimizer = optim.AdamW(self.Q.parameters(), lr=LR, amsgrad=True)
        self.optimizer = optim.RMSprop(self.Q.parameters(), lr=LR, eps=0.001, alpha=0.95)
        self.max_epsilon = 1.0
        self.min_epsilon = 0.0001
        self.epsilon = self.max_epsilon
        self.epsilon_decay = 0.000001
        self.gamma = 0.99

    def choose_action(self):
        if np.random.uniform() < self.epsilon or self.__len__() == 0:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = self.target_Q(self.get_latest_states_stack())
            return torch.argmax(q_values).item()

    def train(self):
        if len(self.buffer) < batch:
            return
        minibatch = self.sample_experiences(batch_size=batch)
        state_batch = torch.cat([info[0] for info in minibatch]).to(device)
        action_batch = torch.LongTensor([info[1] for info in minibatch]).reshape(-1, 1).to(device)
        reward_batch = torch.FloatTensor([info[2] for info in minibatch]).reshape(-1, 1).to(device)
        next_state_batch = torch.cat([info[3] for info in minibatch]).to(device)
        terminated_batch = torch.LongTensor([info[4] for info in minibatch]).reshape(-1, 1).to(device)

        self.optimizer.zero_grad()

        q_values = self.Q(state_batch).gather(1, action_batch)
        next_q_values = self.target_Q(next_state_batch).max(dim=1, keepdim=True)[0].detach()
        mask = 1 - terminated_batch
        target = (reward_batch + self.gamma * next_q_values * mask).to(device)

        loss = F.smooth_l1_loss(q_values, target)

        loss.backward()
        torch.nn.utils.clip_grad_value_(self.Q.parameters(), 100)
        self.optimizer.step()

        self.epsilon = max(self.min_epsilon, self.epsilon - (self.max_epsilon - self.min_epsilon) * self.epsilon_decay)

    def train2(self):
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
        state_batch = torch.cat([info[0] for info in minibatch], dim=0).to(device)
        action_batch = torch.LongTensor([info[1] for info in minibatch]).to(device)
        reward_batch = torch.FloatTensor([info[2] for info in minibatch]).to(device)
        next_state = torch.cat([info[3] for info in minibatch], dim=0).to(device)
        terminated_batch = torch.LongTensor([info[4] for info in minibatch]).to(device)
        # print('state_batch', state_batch.shape)
        """ 先思考一个要怎么学习
        if one_terminated:
            y_pred = one_reward
        else:
            q_values = self.target_Q(one_next_state)
            gamma = 0.99
            y_pred = one_reward + gamma * torch.argmax(one_pred_q_values).item()

        loss = (y_pred - self.Q(one_state)[one_action]) ^ 2

        最后的loss要avg
        """

        # 推广到整个batch
        loss = 0
        for i in range(batch):
            one_state = state_batch[i].unsqueeze(0).to(device)

            one_action = action_batch[i]
            one_reward = reward_batch[i]
            one_next_state = next_state[i].unsqueeze(0).to(device)

            one_terminated = terminated_batch[i]
            if one_terminated:
                y_pred = one_reward
            else:
                gamma = 0.99
                y_pred = one_reward + gamma * torch.max(self.target_Q(one_next_state)).item()
            y_pred = y_pred.to(device)

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
        self.target_Q.load_state_dict(self.Q.state_dict())

    # ----- ReplayMemory method -----
    def add(self, experience):
        if experience[2] != 0:
            self.buffer.append(experience)
        else:
            if random.random() > 0.6:
                self.buffer.append(experience)
            # self.buffer.append(experience)

    # ----- ReplayMemory method -----
    def sample_experiences(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def get_latest_states_stack(self):
        states_stack = self.buffer[-1][0]
        return states_stack

    # ----- ReplayMemory method -----
    def __len__(self):
        return len(self.buffer)


def clipping(image):
    return image[31:193, 8:152, :]


def grayscale(image):
    return np.mean(image, axis=2, keepdims=False)


def down_sampling(image):
    return image[::2, ::2]


def padding(image):
    desired_shape = (84, 84)
    pad_rows = (desired_shape[0] - image.shape[0]) // 2
    pad_cols = (desired_shape[1] - image.shape[1]) // 2
    return np.pad(image, ((pad_rows + 1, pad_rows), (pad_cols, pad_cols)), mode='constant', constant_values=0)


def normalization(image):
    return np.ascontiguousarray(image, dtype=np.float32) / 255.0


def process_image(image):
    image = clipping(image)
    image = grayscale(image)
    image = down_sampling(image)
    image = normalization(image)
    # image = padding(image)

    return torch.tensor(image, dtype=torch.float32, device=device).unsqueeze(0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 777


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


torch.manual_seed(seed)
np.random.seed(seed)
seed_torch(seed)
random.seed(seed)

# Set up the Breakout environment
# env = gym.make('ALE/Breakout-v5', render_mode="human", frameskip=1)
env = gym.make('ALE/Breakout-v5', render_mode="rgb_array", frameskip=1)
state, info = env.reset(seed=seed)
gray_image = np.random.random((256, 256))


# Define actions for the Breakout game
# The available actions are 0 (no-op), 1 (fire), 2 (move right), and 3 (move left)
ACTIONS = [0, 1, 2, 3]

agent = DQNAgent(env, memory_capacity)

states_buffer = deque(maxlen=k_frames)
terminated_counter = 0
# Play the game for a few steps
num_episodes = 1000000
reward_list = []


def get_mean_reward_N(epi, r_list):
    sum_rewards = np.zeros(epi)

    for t in range(epi):
        sum_rewards[t] = np.sum(r_list[max(0, t - 100):(t + 1)])
    plt.plot(sum_rewards)
    plt.show()


process_state = process_image(state)

generation = 0
pre_life = 5
count_rewards = 0

counter = 0
for i in range(num_episodes):
    # Choose a random action (you can implement a RL agent later to select actions intelligently)

    state, info = env.reset(seed=seed)
    terminated = False
    count_rewards = 0
    pre_reward = 0
    while not terminated and count_rewards <= 200:
        # Perform the action and observe the outcome
        action = agent.choose_action()
        next_state, reward, terminated, _, info = env.step(action)
        counter += 1
        if reward == 1:
            lives = info['lives']
            count_rewards += 1
            reward += pre_reward
            pre_reward = 1
            print(f'Epoch: {counter}, Get an reward at life {lives} in Generation {i}')
            with open('training_records2.txt', 'a') as f:
                f.write(f'Epoch: {counter}, Get an reward at life {lives} in Generation {i}\n')
        if info['lives'] == 5:
            life = 5
            pre_life = 5
        else:
            if info['lives'] != pre_life:
                reward = -0.5
                pre_life = info['lives']
                generation += 1
                pre_reward = 0

        # Add experience
        states_buffer.append(process_state)
        stack_states = torch.stack(list(states_buffer), dim=1)
        stack_states_torch = torch.Tensor(stack_states)

        next_state = process_image(next_state)
        next_stack_states = torch.stack(list(states_buffer), dim=1)
        next_stack_states_torch = torch.Tensor(next_stack_states)

        if i >= k_frames + 1:
            agent.add([stack_states_torch, action, reward, next_stack_states_torch, terminated])
        agent.train()

        state = next_state
        if terminated and count_rewards:
            print(f"Generation {i} get rewards: {count_rewards}")

            with open('training_records.txt', 'a') as f:
                f.write(f"Generation {i} get rewards: {count_rewards}\n")

    reward_list.append(count_rewards)
    if i % 100 == 0 and i != 0:
        get_mean_reward_N(i, reward_list)
    # update model
    if i % 100 == 0 and i != 0:
        print('epsilon', agent.epsilon)
        print(info)
        agent.update()

    if i >= 300 and i % 300 == 0:
        torch.save(agent.target_Q.state_dict(), f'./target_Q_{i}.pth')

# Close the environment when done
env.close()
torch.save(agent.target_Q.state_dict(), './target_Q1.pth')

sum_rewards = np.zeros(num_episodes)

for t in range(num_episodes):
    sum_rewards[t] = np.sum(reward_list[max(0, t - 100):(t + 1)])
plt.plot(sum_rewards)
plt.show()
