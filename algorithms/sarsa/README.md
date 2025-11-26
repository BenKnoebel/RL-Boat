# SARSA Implementation

## Algorithm Overview

SARSA (State-Action-Reward-State-Action) is an on-policy TD control algorithm.

**Key Features:**
- On-policy: learns from actions actually taken
- TD(0) updates: one-step temporal difference
- ε-greedy exploration
- Function approximation (linear or neural network)

## Files to Implement

### 1. `sarsa_agent.py` - Main SARSA Agent
Core agent logic implementing SARSA algorithm.

**Key methods:**
- `select_action(state, epsilon)`: ε-greedy action selection
- `update(s, a, r, s', a')`: SARSA update rule
- `save()` / `load()`: Model persistence

### 2. `train.py` - Training Script
Main training loop.

**Should include:**
- Environment creation
- Agent initialization
- Training loop (episodes)
- Logging and checkpointing
- Evaluation during training

### 3. `config.py` - Hyperparameters
Centralized configuration.

**Key parameters:**
- Learning rate (α)
- Discount factor (γ)
- Epsilon (ε) for exploration
- Feature representation (if using function approximation)

## Pseudocode

```
Initialize Q(s,a) arbitrarily
For episode = 1 to M:
    Initialize state s
    Choose action a from s using ε-greedy policy
    For t = 1 to T:
        Take action a, observe r, s'
        Choose action a' from s' using ε-greedy policy
        Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        s ← s'
        a ← a'
```

## Implementation Options

### Option 1: Tabular SARSA
- Discretize continuous state space
- Use Q-table (dictionary)
- Simple but may need coarse discretization

### Option 2: Linear Function Approximation
- Hand-craft features from state
- Linear weights for each (feature, action) pair
- More scalable than tabular

### Option 3: Neural Network Approximation
- Use neural network for Q(s,a)
- Similar to DQN but on-policy
- Most flexible representation

## State Discretization (for Tabular)

If using tabular SARSA, discretize state:

```python
def discretize_state(state):
    # state = [x, y, angle, vx, vy, omega]
    x_bin = int((state[0] + 50) / 10)  # 10 bins for x
    y_bin = int((state[1] + 50) / 10)  # 10 bins for y
    angle_bin = int((state[2] + np.pi) / (np.pi/4))  # 8 bins for angle
    # Could add velocity bins if needed
    return (x_bin, y_bin, angle_bin)
```

## Tips

1. **Start with coarse discretization**: Too fine = too many states
2. **Learning rate**: α = 0.1 is a good starting point
3. **Epsilon decay**: ε = 1.0 → 0.1 over training
4. **Discount factor**: γ = 0.95-0.99
5. **Compare on-policy vs off-policy**: SARSA should be more stable but potentially slower

## Expected Performance

After ~1000-2000 episodes:
- Success rate: >70% (may be lower than DQN)
- Average reward: >-50
- More stable learning than off-policy methods

## Advantages vs DQN

- **Simpler**: No replay buffer or target network
- **More stable**: On-policy is inherently more stable
- **Faster per step**: Less computation per update

## Disadvantages vs DQN

- **Sample efficiency**: Doesn't reuse past experiences
- **Performance ceiling**: May not reach same performance as DQN
- **State representation**: Discretization can limit performance

## Debugging

Common issues:
- **Not learning**: Discretization may be too coarse/fine
- **Unstable**: Reduce learning rate
- **Slow convergence**: Try function approximation instead of tabular

## References

- Sutton & Barto: Reinforcement Learning, Chapter 6.4
- Rummery & Niranjan (1994): "On-line Q-learning using connectionist systems"
