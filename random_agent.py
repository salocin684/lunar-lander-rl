"""Random agent baseline for LunarLander-v3.

obs gives 8 values:
    1. x position
    2. y position
    3. x velocity
    4. y velocity
    5. angle/tilt
    6. angular velocity (how fast it is spinning)
    7. left leg touching the ground (0 or 1)
    8. right leg touching the ground (0 or 1)
    
Mean return over 100 episodes: -198.4
Best: -2.3, Worst: -505.9

which show pure randomness and no intelligences at all
"""

import gymnasium as gym
import numpy as np

env = gym.make("LunarLander-v3") # fast, no window (for measuring)
# env = gym.make("LunarLander-v3", render_mode="human") # slow, opens a window (to watch)

returns = []
for episode in range(100):
    obs, info = env.reset(seed=episode) # .reset() makes a new run 
    total_reward = 0
    done = False

    while not done:
        action = env.action_space.sample() # Action has 4 choices: (0) = do nothing, (1) = fire left engine, (2) = fire main engine, (3) = fire right engine
        obs, reward, terminated, truncated, info = env.step(action) # env.step(action) do this action for 1 frame
        total_reward += reward # Positive if you move towards pad, slow down or stay level, firing engine has small negative effect, crash = -100 points, rest on ground = +100 and +10 for each leg touching the ground
        done = terminated or truncated
    
    returns.append(total_reward)

print(f"Mean return over 100 episodes: {np.mean(returns):.1f}")
print(f"Best: {np.max(returns):.1f}, Worst: {np.min(returns):.1f}")
env.close()