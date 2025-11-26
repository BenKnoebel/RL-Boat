# Project Structure

## Directory Layout

```
RL-Boat/
├── envs/
│   ├── __init__.py
│   ├── boat_env.py          # Main Gym environment implementation
│   └── renderer.py          # Matplotlib visualization renderer
│
├── algorithms/              # RL algorithm implementations (YOU IMPLEMENT THESE)
│   ├── dqn/                 # Deep Q-Network implementation
│   ├── sarsa/               # SARSA implementation
│   └── policy_gradient/     # Policy gradient methods
│
├── examples/
│   ├── test_environment.py  # Random agent testing
│   ├── test_square_pattern.py  # Controlled square pattern demo
│   └── test_friction.py     # Friction testing
│
├── utils/                   # Utility functions (helper code)
│
├── react-website/           # HTML visualization interface
│   ├── index.html           # Visualization UI
│   └── image.png            # Background map
│
├── evaluate.py              # Model evaluation script
├── mask_utils.py            # Mask/obstacle utilities
├── image_mask.png           # Obstacle mask for environment
├── requirements.txt         # Project dependencies
└── README.md
```

## Environment Features

### Core Environment (`envs/boat_env.py`)

**State Space (6D):**
- `x, y`: Position in 2D plane
- `angle`: Boat orientation (radians)
- `vx, vy`: Linear velocities
- `omega`: Angular velocity

**Action Space (9 discrete actions):**
- 0-8: Different combinations of left/right rudder actions

**Reward Structure:**
- `-1` per step (time penalty)
- `+100` for reaching goal (terminates)
- `-10` for going out of bounds (terminates)
- `-25` for being in obstacle/black zone (does NOT terminate)

**New Features:**
- **Obstacle Support**: Load black/white mask images
- **Custom Goal**: Set fixed goal position
- **Coordinate System**: Bottom-left origin, supports non-square masks
- **Validation**: Start/goal positions validated to not be in obstacles

### Using Obstacles

```python
from envs.boat_env import BoatEnv

# Without obstacles (basic navigation)
env = BoatEnv(
    goal_position=[20.0, 20.0]
)

# With obstacles (use mask)
env = BoatEnv(
    goal_position=[30.0, 30.0],
    mask_path="image_mask.png"  # Black = obstacle, White = navigable
)
```

## Algorithm Implementation Structure

You should implement your RL algorithms in the `algorithms/` folder. Here's the recommended structure:

### DQN (Deep Q-Network)

```
algorithms/dqn/
├── __init__.py
├── dqn_agent.py        # Main DQN agent class
├── replay_buffer.py    # Experience replay buffer
├── network.py          # Q-network architecture
├── train.py            # Training script
└── config.py           # Hyperparameters
```

### SARSA

```
algorithms/sarsa/
├── __init__.py
├── sarsa_agent.py      # SARSA agent class
├── train.py            # Training script
└── config.py           # Hyperparameters
```

### Policy Gradient

```
algorithms/policy_gradient/
├── __init__.py
├── pg_agent.py         # Policy gradient agent
├── train.py            # Training script
└── config.py           # Hyperparameters
```

## Workflow

1. **Test Environment**: Run `examples/test_environment.py` to verify setup
2. **Implement Algorithm**: Create your agent in `algorithms/<method>/`
3. **Train**: Run your training script
4. **Evaluate**: Use `evaluate.py` to test trained models
5. **Visualize**: Use `react-website/index.html` for live visualization

## Utility Functions (`utils/`)

Create helper functions here:
- Plotting functions
- Data logging
- Model saving/loading
- Hyperparameter tuning utilities

## Example Training Script Structure

```python
# algorithms/dqn/train.py
from envs.boat_env import BoatEnv
from algorithms.dqn.dqn_agent import DQNAgent

# Create environment
env = BoatEnv(
    goal_position=[20.0, 20.0],
    mask_path=None,  # or "image_mask.png" for obstacles
    max_steps=500
)

# Create agent
agent = DQNAgent(
    state_dim=6,
    action_dim=9,
    learning_rate=0.001
)

# Training loop
for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.store_transition(state, action, reward, next_state, done)
        agent.train()

        state = next_state
        episode_reward += reward

    print(f"Episode {episode}: Reward = {episode_reward}")

# Save trained model
agent.save("models/dqn_model.pt")
```

## Evaluation

The `evaluate.py` script provides a template for evaluating your trained models:

```python
# Your algorithm should be compatible with this interface
python evaluate.py models/your_model.pt --episodes 10
```

## Visualization

Use the HTML visualization in `react-website/` to see your agent in action:

1. Run your trained agent
2. Stream boat state to API endpoint
3. Open `react-website/index.html` in browser

See visualization server examples for implementation details.

## Tips

1. **Start Simple**: Test without obstacles first
2. **Use Examples**: Reference `examples/` for basic usage
3. **Modular Code**: Keep agent, training, and evaluation separate
4. **Save Progress**: Checkpoint models during training
5. **Compare Methods**: Implement multiple algorithms to compare performance

## Next Steps

1. Review `envs/boat_env.py` to understand the environment
2. Run `examples/test_environment.py` to see it in action
3. Choose an algorithm and create its folder structure
4. Implement your agent
5. Train and evaluate
6. Compare results with your teammates

Good luck! 🚀
