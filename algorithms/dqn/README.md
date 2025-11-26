# Deep Q-Network (DQN) Implementation

## Algorithm Overview

DQN is an off-policy, value-based method that uses deep neural networks to approximate the Q-function.

**Key Features:**
- Experience replay buffer
- Target network (for stability)
- ε-greedy exploration
- Mini-batch training

## Files to Implement

### 1. `network.py` - Q-Network Architecture
Define your neural network that approximates Q(s, a).

**Input**: State vector (6 dimensions)
**Output**: Q-values for all actions (9 dimensions)

### 2. `replay_buffer.py` - Experience Replay
Store and sample transitions (s, a, r, s', done).

**Key methods:**
- `add(transition)`: Store experience
- `sample(batch_size)`: Sample random batch
- `__len__()`: Return buffer size

### 3. `dqn_agent.py` - Main DQN Agent
Core agent logic implementing DQN algorithm.

**Key methods:**
- `select_action(state, epsilon)`: ε-greedy action selection
- `update(batch)`: Update Q-network using batch
- `update_target_network()`: Copy weights to target network
- `save()` / `load()`: Model persistence

### 4. `train.py` - Training Script
Main training loop.

**Should include:**
- Environment creation
- Agent initialization
- Training loop (episodes)
- Logging and checkpointing
- Evaluation during training

### 5. `config.py` - Hyperparameters
Centralized configuration.

**Key parameters:**
- Learning rate
- Discount factor (gamma)
- Epsilon (start, end, decay)
- Batch size
- Buffer size
- Target network update frequency

## Pseudocode

```
Initialize replay buffer D
Initialize Q-network with random weights θ
Initialize target network with weights θ⁻ = θ

For episode = 1 to M:
    Initialize state s
    For t = 1 to T:
        Select action a using ε-greedy policy
        Execute action a, observe r, s'
        Store transition (s, a, r, s', done) in D
        Sample random minibatch from D
        Compute target y = r + γ * max_a' Q(s', a'; θ⁻)
        Update θ by minimizing loss: (y - Q(s, a; θ))²
        Every C steps: θ⁻ ← θ
        s ← s'
```

## Tips

1. **Network Architecture**: Start simple (2-3 hidden layers with 64-128 units)
2. **Replay Buffer**: Size 10,000-50,000 should be sufficient
3. **Epsilon Decay**: Start with ε=1.0, decay to 0.01 over ~500 episodes
4. **Target Update**: Update every 100-1000 steps
5. **Learning Rate**: Try 0.001 or 0.0001
6. **Discount Factor**: γ = 0.99 is typical

## Expected Performance

After ~1000 episodes:
- Success rate: >80%
- Average reward: >0
- Can navigate to goal in <100 steps

## Debugging

Common issues:
- **Diverging Q-values**: Reduce learning rate, increase target update frequency
- **Not learning**: Check epsilon decay, verify reward signal
- **Overfitting**: Increase buffer size, add dropout

## References

- Mnih et al. (2013): "Playing Atari with Deep Reinforcement Learning"
- Mnih et al. (2015): "Human-level control through deep reinforcement learning" (Nature)
