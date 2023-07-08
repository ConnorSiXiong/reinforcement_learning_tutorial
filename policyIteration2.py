import gymnasium as gym
import numpy as np

env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True, render_mode="rgb_array")
observation, info = env.reset()

THETA = 1e-4


class PolicyIteration:
    def __init__(self, game_env):
        self.env = game_env
        self.num_actions = self.env.action_space.n
        self.num_states = self.env.observation_space.n
        self.V = np.zeros(self.num_states)
        self.policy = np.ones((self.num_states, self.num_actions)) / self.num_actions

    def get_policy(self):
        return self.policy

    def get_action_probabilities(self, cur_state):
        return self.policy[cur_state]

    def choose_action(self, cur_state):
        return np.argmax(self.get_action_probabilities(cur_state))

    def estimate_policy(self, delta, gamma):
        while delta > THETA:
            delta = 0
            for s in range(self.num_states):
                value = self.V[s]
                expected_value = 0
                a = self.choose_action(s)
                for prob, next_state, r, _ in self.env.P[s][a]:
                    expected_value += prob * (r + gamma * self.V[next_state])

                delta = max(delta, abs(value - expected_value))
                self.V[s] = expected_value

    def improve_policy(self, gamma):
        is_policy_stable = True
        for s in range(self.num_states):
            old_action = self.choose_action(s)
            action_reward = np.zeros(self.num_actions)

            for a in range(self.num_actions):
                action_expect_reward = 0
                for prob, next_state, r, _ in self.env.P[s][a]:
                    action_expect_reward += prob * (r + gamma * self.V[next_state])
                action_reward[a] = action_expect_reward

            best_action = np.argmax(action_reward)

            self.policy[s] = np.eye(self.num_actions)[best_action]

            if old_action != best_action:
                is_policy_stable = False
        return is_policy_stable

    def execute(self, delta=0.1, gamma=0.9):
        while True:
            self.estimate_policy(delta=delta, gamma=gamma)
            is_stable = self.improve_policy(gamma=gamma)
            if is_stable:
                break


def choose_action(policy, s):
    return np.argmax(policy[s])


p = PolicyIteration(env)
p.execute()
policy = p.get_policy()

success = 0
for i in range(1000):
    for j in range(100):
        # action = env.action_space.sample()  # agent policy that uses the observation and info
        # action = policy[observation]
        action = choose_action(policy, observation)

        observation, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated or reward == 1:
            observation, info = env.reset()
            if reward == 1:
                success += 1
            break

env.close()
print('success:', success)
