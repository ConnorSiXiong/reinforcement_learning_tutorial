import gymnasium as gym
import numpy as np

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="rgb_array")
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

    def estimate_policy(self, gamma):

        delta = np.inf
        while delta > THETA:
            v_old = self.V.copy()
            for s in range(self.num_states):
                transitions = np.array([self.env.P[s][a] for a in range(self.num_actions)]).reshape(self.num_actions,
                                                                                                    -1, 4)
                action_prob = self.policy[s].reshape(4, 1)

                # Solution 1
                # probs = transitions[:, :, 0]
                # rewards = transitions[:, :, 2]
                # next_states = transitions[:, :, 1].astype(int)
                #
                # V_s = np.sum(action_prob * probs * (rewards + gamma * v_old[next_states]))
                # self.V[s] = V_s

                # Solution 2
                probabilities, next_states, rewards, _ = transitions.transpose(2, 0, 1)
                self.V[s] = np.sum(action_prob * probabilities * (rewards + gamma * v_old[next_states.astype(int)]))
                delta = np.max(np.abs(self.V - v_old))

    def improve_policy(self, gamma):
        is_policy_stable = True
        for s in range(self.num_states):
            old_action = np.argmax(self.policy[s])

            # Expanding transitions to a consistent size
            transitions = np.array([self.env.P[s][a] for a in range(self.num_actions)]).reshape(self.num_actions, -1, 4)
            probabilities, next_states, rewards, _ = transitions.transpose(2, 0, 1)

            action_expect_rewards = np.sum(probabilities * (rewards + gamma * self.V[next_states.astype(int)]), axis=1)

            best_action = np.argmax(action_expect_rewards)

            self.policy[s] = np.eye(self.num_actions)[best_action]

            if old_action != best_action:
                is_policy_stable = False

        return is_policy_stable

    def execute(self, delta=0.1, gamma=0.9):
        while True:
            self.estimate_policy(gamma=gamma)
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
