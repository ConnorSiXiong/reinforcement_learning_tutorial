import gym
import numpy as np

np.set_printoptions(suppress=True)
"""
Monte carlo
"""
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="rgb_array")


def monte_carlo(env, num_episodes, gamma=0.9):
    V = np.zeros(env.observation_space.n)
    returns = np.zeros(env.observation_space.n)
    returns_count = np.ones(env.observation_space.n)

    for episode in range(num_episodes):
        state = env.reset()

        if isinstance(state, tuple):
            state = state[0]

        episode_states = []
        episode_rewards = []

        done = False
        while not done:
            action = env.action_space.sample()
            next_state, reward, a, b, _ = env.step(action)

            episode_states.append(state)
            episode_rewards.append(reward)

            state = next_state  # Extract state value from tuple
            if a or b:
                done = True
        # print(episode_states)
        # print(episode_rewards)
        G = 0
        for t in range(len(episode_states) - 1, -1, -1):
            G = gamma * G + episode_rewards[t]
            state = episode_states[t]

            returns[state] += G
            returns_count[state] += 1
            V[state] = returns[state] / returns_count[state]
        if episode == 10000 - 1:
            print(returns.reshape(4, 4))
            print(returns_count.reshape(4, 4))
    return V


num_episodes = 10000
V = monte_carlo(env, num_episodes)
print(V.reshape(4, 4))
