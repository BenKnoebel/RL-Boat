# RL Algorithms

This folder contains your RL algorithm implementations for the boat navigation task.

## Proposed Algorithms

Based on your project proposal, you will implement:

1. **DQN (Deep Q-Network)** - Deep value iteration
2. **SARSA** - On-policy TD control
3. **Policy Gradient Methods** (stretch goal)

## Getting Started

Each algorithm should have its own folder with:
- Agent implementation
- Training script
- Configuration file
- (Optional) Additional components (replay buffer, networks, etc.)

## Recommended Structure

```
algorithms/
├── dqn/
│   ├── dqn_agent.py      # Main agent class
│   ├── train.py          # Training script
│   ├── config.py         # Hyperparameters
│   └── README.md         # Algorithm-specific notes
├── sarsa/
│   ├── sarsa_agent.py
│   ├── train.py
│   ├── config.py
│   └── README.md
└── policy_gradient/
    ├── pg_agent.py
    ├── train.py
    ├── config.py
    └── README.md
```

## Environment Interface

All algorithms should work with the standard Gym environment interface:

```python
from envs.boat_env import BoatEnv

# Create environment
env = BoatEnv(goal_position=[20.0, 20.0])

# Reset
state, info = env.reset()

# Step
next_state, reward, terminated, truncated, info = env.step(action)
```

**State**: `np.array([x, y, angle, vx, vy, omega])` - shape (6,)
**Action**: `int` in range [0, 8] - 9 discrete actions
**Reward**: `float` - see README for reward structure

## Evaluation Metrics

Your algorithms should track:
- **Success rate**: % of episodes reaching the goal
- **Average reward**: Mean episode reward
- **Learning curve**: Reward vs training steps
- **Convergence**: Steps to reach target performance
- **Optimality**: Path length to goal

## Tips

1. **Start without obstacles**: Set `mask_path=None` for initial training
2. **Tune hyperparameters**: Learning rate, epsilon, discount factor, etc.
3. **Visualize learning**: Plot training curves
4. **Compare algorithms**: Same random seeds for fair comparison
5. **Add obstacles gradually**: Train without, then with obstacles

## Baseline Performance

Random policy baseline (no obstacles, goal at [20, 20]):
- Success rate: ~5-10%
- Average reward: ~-200 to -500
- Average steps: ~500 (max steps)

Your trained policies should significantly outperform this baseline.

## Implementation Order

Recommended order:
1. **SARSA** - Simpler to implement, good baseline
2. **DQN** - More complex but potentially better performance
3. **Policy Gradient** - Stretch goal if time permits

## Resources

- Sutton & Barto: Reinforcement Learning (Chapter 6 for SARSA, Chapter 10 for on-policy approximation)
- DQN Paper: "Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013)
- Stable-Baselines3: Reference implementations (don't copy, but learn from)

Good luck! 🎯
