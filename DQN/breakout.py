import gymnasium as gym

# Set up the Breakout environment
env = gym.make('ALE/Breakout-v5', render_mode="human")
env.reset()

# Define actions for the Breakout game
# The available actions are 0 (no-op), 1 (fire), 2 (move right), and 3 (move left)
ACTIONS = [0, 1, 2, 3]


# Play the game for a few steps
for _ in range(100000):
    # Choose a random action (you can implement a RL agent later to select actions intelligently)
    action = env.action_space.sample()

    # Perform the action and observe the outcome
    observation, reward, terminated, truncated, info = env.step(action)
    print('reward', reward)
    print("terminated", terminated)
    print("truncated", truncated)
    print('info', info)
    # Render the game (optional, you can comment this out to run the game in the background)

    # Check if the game is over (done=True) and reset if necessary
    if terminated or truncated:
        env.reset()
# Close the environment when done
env.close()
