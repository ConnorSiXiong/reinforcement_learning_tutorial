import gymnasium as gym

# Set up the Breakout environment
env = gym.make('ALE/Breakout-v5', render_mode="human")
env.reset()

# Define actions for the Breakout game
# The available actions are 0 (no-op), 1 (fire), 2 (move right), and 3 (move left)
ACTIONS = [0, 1, 2, 3]


# Play the game for a few steps
for i in range(100000):
    # Choose a random action (you can implement a RL agent later to select actions intelligently)
    action = env.action_space.sample()

    # Perform the action and observe the outcome
    observation, reward, terminated, truncated, info = env.step(action)
    print('observation', observation.shape)
    print('reward', reward)
    print("terminated", terminated)
    print("truncated", truncated)
    print('info', info)
    if reward:
        print('xxx reward', reward)
    # Render the game (optional, you can comment this out to run the game in the background)

    # Check if the game is over (done=True) and reset if necessary
    if terminated or truncated:
        print('observation', observation.shape)
        print(observation)
        print('reward', reward)
        print("aaa terminated", terminated)
        print("aaa truncated", truncated)
        print('info', info)
        env.reset()
# Close the environment when done
env.close()

from ReplayMemory import ReplayMemory
from DQN_Architecture import DQN

N = 1000000
k_frames = 4
game_frame_height = 84
game_frame_width = 84
game_num_actions = 4
target_Q_update_steps = 10000
batch = 32
# if __name__ == "__main__":
#     D = ReplayMemory(N)
#     Q = DQN(game_num_actions, k_frames, game_frame_height, game_frame_width)
#     target_Q = DQN(game_num_actions, k_frames, game_frame_height, game_frame_width)
