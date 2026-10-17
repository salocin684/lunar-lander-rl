"""DQN building blocks for LunarLander-v3.

QNetwork: 8 observation values in -> 4 Q-values out (one per action)
ReplayBuffer: stores past experiences, returns random batches
Agent: picks actions with epsilon-greedy (explore vs exploit)

obs (8): x, y, vx, vy, angle, angular velocity, left leg, right leg
actions (4): 0 nothing, 1 left engine, 2 main engine, 3 right engine
"""

import torch
import torch.nn as nn
import random
import numpy as np
import gymnasium as gym
from collections import deque

# The brain: estimates how good each action is in a give state
class QNetwork(nn.Module):
    def __init__(self, obs_size=8, n_actions=4, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden), # 8 -> 128
            nn.ReLU(), # non-linearity
            nn.Linear(hidden, hidden), # 128 -> 128
            nn.ReLU(), # non-linearity
            nn.Linear(hidden, n_actions), # 128 -> 4, no ReLU: Q-values can be negative
        )
    
    def forward(self, x):
        return self.net(x)

# The memory: stores experiences and hands out random batches
class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity) # oldest experience drops off when full
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size) # random = less correlated data
        states, actions, rewards, next_states, dones = zip(*batch) # list of tuples --> 5 groups
        return(
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64), # int used as index
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )
    
    def __len__(self):
        return len(self.buffer)

# The decision maker: explores randomly or picks the best action
class Agent:
    def __init__(self, obs_size=8, n_actions=4):
        self.n_actions = n_actions
        self.q_net = QNetwork(obs_size, n_actions) # the brain it uses
        
        self.epsilon = 1.0 # start fully random
        self.epsilon_min = 0.01 # always keep 1% exploration
        self.epsilon_decay = 0.995 # shrink epsilon after each episode
    
    def select_action(self, state):
        if random.random() < self.epsilon: # get a ranom number in [0,1)
            return random.randrange(self.n_actions) # explore: random action 0-3
        
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():   # no training here, just asking: skip gradients
            q_values = self.q_net(state) # 4 Q-values
        return int(q_values.argmax()) # exploit: index of highest Q-value
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay) # shrink but not below the min

# Test: action choiches at different stages of training (still untrained)
if __name__ == "__main__":
    agent = Agent()
    fake_state = np.zeros(8, dtype=np.float32) # same fake state every time
    
    for episode in [0, 100, 300, 500, 1000]:
        agent.epsilon = max(agent.epsilon_min, 0.995 ** episode) # epsilon at that episode
        actions = [agent.select_action(fake_state) for _ in range(1000)]
        print(f"episode {episode:4d} | epsilon {agent.epsilon:.3f} | counts {np.bincount(actions, minlength=4)}")


# Test: fill the buffer with 500 random steps and check the batch shapes
# if __name__ == "__main__":
#     env = gym.make("LunarLander-v3")
#     buffer = ReplayBuffer()
    
#     obs, info = env.reset(seed=0)
#     for _ in range(500):
#         action = env.action_space.sample()
#         next_obs, reward, terminated, truncated, info = env.step(action)
#         buffer.push(obs, action, reward, next_obs, terminated) # terminated, not truncated
#         obs = next_obs
#         if terminated or truncated:
#             obs, info = env.reset()
    
#     states, actions, rewards, next_states, dones = buffer.sample(64)
#     print("buffer size: ", len(buffer))
#     print("states: ", states.shape, "actions: ", actions.shape, "rewards: ", rewards.shape)


# Test: Base  

#if __name__ == "__main__":
#    q_net = QNetwork()
#    fake_obs = torch.zeros(8)
#    print(q_net(fake_obs))