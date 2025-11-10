# RL-Boat

This is a repository for the final project of Harvard's Reinforcement Learning and Optimal Control Class ES158 taught by Heng Yang

Team members: Florian Schechner, Benjamin Knoebel del Olmo

## Project Overview

This project implements reinforcement learning algorithms to control a rowing boat with two independent rudders navigating in a 2D environment.

## Problem Definition

**Agent:** A rowing boat with two independent rudders (left and right)

**Action Space:**
- Discrete, 9 possible actions (3² combinations)
- Each rudder can: row forward, row backward, or remain idle
- Results in forward/backward motion or rotation depending on rudder coordination

**State Space:**
- Continuous 2D coordinates (x, y) on a plane
- Infinite dimensionality (continuous state space)

**Environment:**
- 2D plane representing a river or lake
- Boat dynamics defined through force and momentum balances
- Each rudder applies force with a specific lever arm
- Boat motion determined by mass and inertia approximations

**Problem Formulation:**
- Infinite horizon Markov Decision Process (MDP)
- Continuous state space, discrete action space
- Agent has no prior knowledge of dynamics (simulator provides feedback)

**Objective:**
- Primary goal: Navigate optimally to a target location
- Reward function: Constant negative reward until goal is reached (episode termination)
- Potential extensions: obstacles, water currents, and other environmental complexities

## Proposed Methods

We aim to implement and compare two families of reinforcement learning algorithms:

1. **Generalized Policy Iteration (GPI) with Function Approximation**
   - Semi-Gradient SARSA

2. **Generalized Value Iteration (GVI) with Function Approximation**
   - Deep Q-Network (DQN)
  
3. As a strech goal, we will try to look into Policy Gradient Methods

Additional methods may be explored to validate theoretical expectations regarding stability and performance.

## Goals and Evaluation Metrics

**Primary Goals:**
- Implement and benchmark multiple RL methods on the boat navigation problem
- Compare algorithm performance and solution optimality

**Evaluation Metrics:**
- Success rate
- Required learning iterations to achieve target success rate
- Optimality of learned policies
- Stability and convergence behavior

**Stretch Goals:**
- Test algorithms in varied environments (obstacles, currents)
- Extended comparative analysis across multiple methods and metrics

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/BenKnoebel/RL-Boat.git
cd RL-Boat
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Testing the Environment

Run the test scripts to verify the environment is working correctly:

```bash
# Test with random agent and verify dynamics
python examples/test_environment.py

# Test controlled movement in a square pattern (great for visualization!)
python examples/test_square_pattern.py
```

## Project Structure

```
RL-Boat/
├── envs/
│   ├── __init__.py
│   ├── boat_env.py          # Main Gym environment implementation
│   └── renderer.py          # Matplotlib visualization renderer
├── examples/
│   ├── test_environment.py  # Random agent testing
│   └── test_square_pattern.py  # Controlled square pattern demo
├── requirements.txt         # Project dependencies
└── README.md
```

## Environment Details 

The `BoatEnv` implements a Gymnasium-compatible environment with the following specifications:

**State Vector (6 dimensions):**
- `x, y`: Position coordinates in 2D plane (meters)
- `angle`: Boat orientation in radians (θ)
- `velocity_x, velocity_y`: Linear velocities in global frame (m/s)
- `angular_velocity`: Angular velocity (ω, radians/second)

**Actions (9 discrete options):**
- 0: Both rudders idle
- 1: Left forward, Right idle
- 2: Left backward, Right idle
- 3: Left idle, Right forward
- 4: Left idle, Right backward
- 5: Both forward
- 6: Both backward
- 7: Left forward, Right backward (rotate right)
- 8: Left backward, Right forward (rotate left)

**Rewards:**
- `-1` per timestep (encourage efficiency)
- `+100` when goal is reached
- `-10` if boat goes out of bounds

**Episode Termination:**
- Goal reached (within goal radius)
- Out of bounds
- Maximum steps exceeded (default: 500)

**Dynamics:**
The boat dynamics are simulated using physics-based equations:
- **Angular dynamics**: Torque from differential rudder forces creates angular acceleration (τ = I·α)
- **Linear dynamics**: Net thrust from both rudders in boat's forward direction
- **Friction**: Applied to both linear and angular velocities
- **Integration**: Euler integration with configurable time step (default: 0.1s)

The state evolves according to:
- Position: `x(t+dt) = x(t) + vx·dt`, `y(t+dt) = y(t) + vy·dt`
- Angle: `θ(t+dt) = θ(t) + ω·dt`
- Linear velocity: `v(t+dt) = v(t) + a·dt` (with friction)
- Angular velocity: `ω(t+dt) = ω(t) + α·dt` (with friction)

## Visualization

The environment includes a real-time 2D visualization using matplotlib:

**Visual Elements:**
- **Red square**: The rowing boat
- **Dark red arrow**: Boat orientation/heading
- **Green circle**: Goal area
- **Green star**: Goal center point
- **Blue line**: Boat's trajectory path
- **Info displays**: Distance to goal and step count

The visualization updates automatically during episode execution and displays in a separate window.

## Usage Examples

### Basic Usage (No Visualization)

```python
from envs.boat_env import BoatEnv
import numpy as np

# Create environment without visualization
env = BoatEnv(goal_position=np.array([20.0, 20.0]))

# Reset environment
state, info = env.reset()

# Run episode
done = False
while not done:
    action = env.action_space.sample()  # Random action
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
```

### With Real-Time Visualization

```python
from envs.boat_env import BoatEnv
import numpy as np

# Create environment WITH visualization
env = BoatEnv(
    goal_position=np.array([20.0, 20.0]),
    render_mode='human'  # Enable visualization
)

# Reset and run - visualization appears automatically
state, info = env.reset()

done = False
while not done:
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
```
