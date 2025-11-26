# Policy Gradient Methods (Stretch Goal)

## Algorithm Overview

Policy gradient methods directly optimize the policy π(a|s) without learning a value function.

**Popular variants:**
- **REINFORCE**: Basic policy gradient with Monte Carlo returns
- **Actor-Critic**: Policy gradient with learned value function baseline
- **PPO**: Proximal Policy Optimization (advanced, more stable)

## Files to Implement

### 1. `pg_agent.py` - Policy Gradient Agent
Core agent implementing policy gradient algorithm.

**Key methods:**
- `select_action(state)`: Sample action from policy
- `compute_returns(rewards)`: Compute discounted returns
- `update(trajectories)`: Update policy using gradient
- `save()` / `load()`: Model persistence

### 2. `train.py` - Training Script
Main training loop.

**Should include:**
- Environment creation
- Agent initialization
- Trajectory collection
- Policy updates
- Logging and evaluation

### 3. `config.py` - Hyperparameters
Centralized configuration.

**Key parameters:**
- Learning rate
- Discount factor (γ)
- Number of episodes per update
- Baseline (if using actor-critic)

## REINFORCE Pseudocode

```
Initialize policy network π(a|s;θ) with random weights θ

For episode = 1 to M:
    Generate episode τ = (s₀, a₀, r₁, s₁, ..., sₜ) following π
    For each step t in episode:
        Compute return Gₜ = Σᵢ₌ₜᵀ γⁱ⁻ᵗ rᵢ
        Update θ: θ ← θ + α·∇ₐlog π(aₜ|sₜ;θ)·Gₜ
```

## Actor-Critic Pseudocode

```
Initialize policy network π(a|s;θ_π) and value network V(s;θ_v)

For episode = 1 to M:
    Initialize state s
    For t = 1 to T:
        Sample action a ~ π(·|s;θ_π)
        Take action a, observe r, s'
        Compute TD error: δ = r + γV(s';θ_v) - V(s;θ_v)
        Update value network: θ_v ← θ_v + α_v·δ·∇V(s;θ_v)
        Update policy: θ_π ← θ_π + α_π·∇log π(a|s;θ_π)·δ
        s ← s'
```

## Implementation Options

### Option 1: REINFORCE (Simpler)
- **Pros**: Simple, unbiased gradient estimates
- **Cons**: High variance, slow convergence
- **Good for**: First implementation, understanding basics

### Option 2: REINFORCE with Baseline
- **Pros**: Reduced variance, faster convergence
- **Cons**: Need to learn baseline (value function)
- **Good for**: Better performance than vanilla REINFORCE

### Option 3: Actor-Critic
- **Pros**: Lower variance, faster convergence, online updates
- **Cons**: More complex, two networks to train
- **Good for**: Best performance

### Option 4: PPO (Advanced)
- **Pros**: State-of-the-art, very stable
- **Cons**: Most complex implementation
- **Good for**: If you have extra time

## Policy Network Architecture

```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=6, action_dim=9):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        return torch.softmax(logits, dim=-1)  # Action probabilities
```

## Tips

1. **Start with REINFORCE**: Simpler to debug
2. **Learning rate**: α = 0.001-0.01 (higher than value-based methods)
3. **Discount factor**: γ = 0.99
4. **Batch updates**: Collect multiple episodes before updating
5. **Entropy bonus**: Add entropy term to encourage exploration
6. **Normalize returns**: Helps with gradient stability

## Expected Performance

After ~2000-5000 episodes:
- Success rate: >80% (with good implementation)
- Average reward: >0
- Potentially smoother policies than value-based methods

## Advantages

- **Direct optimization**: Optimizes what you care about (policy)
- **Continuous actions**: Can easily extend to continuous action spaces
- **Stochastic policies**: Can learn stochastic optimal policies

## Disadvantages

- **Sample efficiency**: Often needs more samples than DQN
- **Variance**: High gradient variance without careful design
- **Sensitive**: More sensitive to hyperparameters

## Debugging

Common issues:
- **Policy collapse**: All actions get same probability → add entropy bonus
- **High variance**: Use baseline, reduce learning rate
- **Slow learning**: Increase batch size, tune learning rate
- **NaN gradients**: Add gradient clipping

## Advanced Techniques

If you have time:
- **GAE (Generalized Advantage Estimation)**: Better advantage estimation
- **Trust region methods**: PPO or TRPO for stability
- **Natural policy gradient**: Better optimization geometry

## References

- Sutton & Barto: Reinforcement Learning, Chapter 13
- Williams (1992): "Simple statistical gradient-following algorithms"
- Schulman et al. (2017): "Proximal Policy Optimization Algorithms"
- Mnih et al. (2016): "Asynchronous Methods for Deep Reinforcement Learning" (A3C)

## Note

This is a **stretch goal**. Focus on DQN and SARSA first, then implement this if time permits!
