"""Train a DQN agent on LunarLander-v3

Each step: pick an action --> step env --> store eperience --> learn from a batch
Each episode: decay epsilon, log the return
Solved when the average return over the last 100 episodes is >= 200
"""

import gymnasium as gym
import numpy as np
import torch

from collections import deque
from dqn import Agent, ReplayBuffer

env = gym.make("LunarLander-v3")
agent = Agent()
buffer = ReplayBuffer()

n_episodes = 1500
warmup_steps = 1000 # collect experience before learning starts
target_update_every = 1000 # steps between target network refreshes

recent_returns = deque(maxlen=100)
total_steps = 0

for episode in range(n_episodes):
    obs, info = env.reset()
    episode_return = 0
    done = False
    
    while not done:
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        buffer.push(obs, action, reward, next_obs, terminated)
        
        obs = next_obs
        episode_return += reward
        total_steps += 1
        done = terminated or truncated
        
        if len(buffer) >= warmup_steps:
            agent.learn(buffer) # one learning step per env step
        
        if total_steps % target_update_every == 0:
            agent.update_target() # refresh the frozen copy
        
    agent.decay_epsilon() # explore a bit less next episode
    recent_returns.append(episode_return)
    avg = np.mean(recent_returns)
    
    if episode % 10 == 0:
        print(f"episode {episode:4d} | return {episode_return:7.1f} | avg100 {avg:7.1f} | epsilon {agent.epsilon:.3f}")
    
    if len(recent_returns) == 100 and avg >= 200:
        print(f"\nSolved at episode{episode}! avg100 = {avg:.1f}")
        break

torch.save(agent.q_net.state_dict(), "dqn_lunar.pt") # save the trained brain
env.close()