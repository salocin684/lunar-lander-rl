# lunar-lander-rl

## Project info

My first RL project. I'm training an agent from scratch to land the lunar lander from Gymnasium ([LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/)) using a DQN (Deep Q-Network) written in PyTorch.

The agent gets 8 numbers (position, velocity, angle, leg contact) and has to choose between 4 actions: do nothing, fire left engine, fire main engine or fire right engine. The env counts as solved when the average return over 100 episodes is 200 or higher.

Files:
- `random_agent.py`: random agent baseline, to know what score "no intelligence" gets
- `dqn.py`: the building blocks (network, replay buffer, agent)
- `train.py`: the training loop, saves the trained model to `dqn_lunar.pt`

## How to run it

Python 3.12 recommended (3.10.0 crashes with gymnasium)

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install swig
    pip install "gymnasium[box2d]" torch numpy

Then:

    python random_agent.py
    python train.py

## Results

| Agent | Mean return (100 episodes) |
|---|---|
| Random | -198.4 |
| DQN | (training...) |

## What i learned

### the RL loop

Every RL project is built around the same loop: observe the state, pick an action, get a reward and the next state, repeat until the episode ends.

- `env.reset()` starts a new episode
- `env.step(action)` does the action for 1 frame and gives back the new state, the reward, and if the episode is over
- The return is all the rewards of 1 episode added up, thats the number we want to push above 200

The env doesnt care who picks the actions, me, random noise or a neural network. RL is basically replacing the random action with something that learns.

### exploration vs exploitation

- Exploit: do what you currently think is best to collect reward now
- Explore: try something else because your current belief might be wrong

RL agents are quickly happy with what they have and what works so they wont explore new paths cause they have no reason, thats why we need to implement some exploration so it can find a more optimal path if that exists

The standard fix is ε-greedy:
- With probability ε pick a random action (explore)
- Otherwise pick the action with the highest Q-value (exploit)

We then decay ε over time, we start high cause the network knows nothing yet and slowly drop it because after a while the networks guesses are trustworthy enough.
So in short early training is all exploration, and late training is mostly exploitation

Important: decaying ε doesnt make the agent smarter, it only makes it trust its network more. If the network learned nothing a low ε just means it confidently does something dumb.

### Q-values

A Q-value is the estimate of how good an action is in a state, so the total return you'll get if you take that action now and play well after. If you know the Q-values playing is easy, just pick the highest one.

Nobody gives you the Q-values so a neural network estimates them: 8 numbers in, 4 Q-values out. No ReLU on the last layer cause Q-values can be negative (a crash is -100).

### replay buffer

Training on each step right after it happens works bad because consecutive frames are almost the same, so the network overfits on whatever is happening right now. And every experience gets used once and thrown away.

So we store experiences (state, action, reward, next_state, done) in a big memory and train on random batches from it. Random mixing breaks the correlation and every experience can be reused a lot of times.

### the learning step

For a batch of experiences:
- guess: what Q-value does the network give the action that was taken
- target: reward + γ × best Q-value of the next state
- loss: the gap between guess and target, then adjust the weights to make it smaller

If the episode ended (crash or landing) there is no future so the target is just the reward. Thats why we store `terminated` and not `truncated`, running out of time doesnt mean the future is worth 0.

The target network is a frozen copy of the network that we use to calculate the targets, and we only refresh it every 1000 steps. Without it the network chases a target that moves every time it updates itself and training gets unstable.

### loss is not progress

In normal supervised learning the loss goes down nicely. In RL the targets come from the networks own guesses so they keep shifting, and the loss jumps around. The real measure of progress is the episode return, not the loss.