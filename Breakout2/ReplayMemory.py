import torch

from config import device


class ReplayMemory:
    """
    This implementation improves the memory usage,

    which is better than deque in collections.
    """

    def __init__(
            self,
            frames,
            capacity,
            width=84,
            height=84,
    ):
        self.capacity = capacity
        self.__pos = 0
        self.__size = 0

        self.states = torch.zeros((capacity, frames, width, height), dtype=torch.long)
        self.actions = torch.zeros((capacity, 1), dtype=torch.uint8)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.uint8)
        self.terminates = torch.zeros((capacity, 1), dtype=torch.bool)

    def add(self,
            stack_states,
            action,
            reward,
            terminated
            ):
        self.states[self.__pos] = stack_states
        self.actions[self.__pos, 0] = action
        self.rewards[self.__pos, 0] = reward
        self.terminates[self.__pos, 0] = terminated

        self.__pos = (self.__pos + 1) % self.capacity
        self.__size = max(self.__size, self.__pos)

    def sample(self, batch_size):
        indies = torch.randint(0, self.__size, (batch_size,))
        minibatch_states = self.states[indies, :4].to(device).float()
        minibatch_next_states = self.states[indies, 1:].to(device).float()
        minibatch_actions = self.actions[indies].to(device)
        minibatch_rewards = self.rewards[indies].to(device)
        minibatch_terminates = self.terminates[indies].to(device)

        return minibatch_states, minibatch_next_states, minibatch_actions, minibatch_rewards, minibatch_terminates

    def __len__(self):
        return self.__size
