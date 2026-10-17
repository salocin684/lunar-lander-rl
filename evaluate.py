"""Evaluate the trained DQN agent on LunarLander-v3.

Loads dqn_lunar.pt, plays with epsilon = 0 (pure exploitation, no learning).
1. Measures the real average return over 100 episodes (fast, no window)
2. Shows 5 episodes in a window so you can watch it land

Mean return over 100 episodes: 249.4
Best: 317.1, Worst: -19.4
Landings above 200: 85/100
watch episode 0: return 310.3
watch episode 1: return 255.6
watch episode 2: return 237.9
watch episode 3: return 239.5
watch episode 4: return 257.4

used a lot of side thrusts on the ground to get more in the middle
"""

import gymnasium as gym
import numpy as np
import torch

from dqn import Agent

agent = Agent()
agent.q_net.load_state_dict(torch.load("dqn_lunar.pt")) # load the trained brain
agent.epsilon = 0.0 # no exploration, only the network decides

def play(env, seed):
    obs, info = env.reset(seed=seed)
    total_reward = 0
    done = False
    while not done:
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    return total_reward

# 1. Measure: 100 episodes, same seeds as the random baseline
env = gym.make("LunarLander-v3")
returns = [play(env, seed) for seed in range(100)]
env.close()

print(f"Mean return over 100 episodes: {np.mean(returns):.1f}")
print(f"Best: {np.max(returns):.1f}, Worst: {np.min(returns):.1f}")
print(f"Landings above 200: {sum(r >= 200 for r in returns)}/100")

# 2. Watch: 5 episodes with a window to actually see it but keep some speed
env = gym.make("LunarLander-v3", render_mode="human")
for seed in range(5):
    print(f"watch episode {seed}: return {play(env, seed):.1f}")
env.close()