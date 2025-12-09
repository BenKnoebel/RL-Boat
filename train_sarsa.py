"""
Semi-Gradient SARSA Training Script for Boat Environment

This script trains a Semi-Gradient SARSA agent to navigate from (0,0) to (5,5) using
the boat environment with two independent rudders.

SARSA (State-Action-Reward-State-Action) is an on-policy TD control algorithm that:
- Updates Q(S,A) based on the actual next action A' taken (not max)
- Uses function approximation (neural network) for continuous state spaces
- Updates weights after every step (true online learning)
- No experience replay buffer (unlike DQN)
- No target network (unlike DQN)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from datetime import datetime
import os
from envs.boat_env import BoatEnv


class SARSANetwork(nn.Module):
    """
    Neural network for Q-value approximation in SARSA.

    Architecture:
        - Input: 6-dimensional state space (x, y, angle, vx, vy, omega)
        - Hidden layers: 3 fully connected layers with ReLU activation
        - Output: Q-values for 9 discrete actions
    """

    def __init__(self, state_dim=6, action_dim=9, hidden_dim=128):
        super(SARSANetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class SARSAAgent:
    """
    Semi-Gradient SARSA Agent for on-policy TD control.

    Key differences from DQN:
    - No replay buffer (online learning)
    - No target network (single network)
    - Updates based on actual next action, not max
    - True on-policy learning
    """

    def __init__(self, state_dim=6, action_dim=9, learning_rate=0.0005,
                 gamma=0.99, epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay=0.995):
        """
        Initialize SARSA Agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            learning_rate: Learning rate for optimizer (typically lower than DQN)
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Exponential decay rate for epsilon
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Q-Network (single network, no target network)
        self.q_network = SARSANetwork(state_dim, action_dim, hidden_dim=128).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Training statistics
        self.training_step = 0

        print(f"SARSA Agent initialized: lr={learning_rate}, gamma={gamma}")

    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            training: If True, use epsilon-greedy; otherwise greedy

        Returns:
            Selected action
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        # Greedy action selection
        with torch.no_grad():
            state_tensor = torch.from_numpy(np.array(state, dtype=np.float32)).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()

    def update(self, state, action, reward, next_state, next_action, done):
        """
        SARSA update rule: Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]

        Key difference from Q-learning/DQN: Uses actual next_action A', not max over actions.

        Args:
            state: Current state S
            action: Current action A
            reward: Reward R
            next_state: Next state S'
            next_action: Next action A' (actually taken, not max)
            done: Whether episode is done

        Returns:
            TD error (for monitoring)
        """
        # Convert to tensors
        state_tensor = torch.from_numpy(np.array(state, dtype=np.float32)).unsqueeze(0).to(self.device)
        next_state_tensor = torch.from_numpy(np.array(next_state, dtype=np.float32)).unsqueeze(0).to(self.device)

        # Current Q(S,A)
        q_values = self.q_network(state_tensor)
        current_q = q_values[0, action]

        # Next Q(S',A') - using the actual next action (SARSA)
        with torch.no_grad():
            next_q_values = self.q_network(next_state_tensor)
            next_q = next_q_values[0, next_action] if not done else 0.0

        # TD target: R + γQ(S',A')
        target_q = reward + self.gamma * next_q

        # TD error
        td_error = target_q - current_q

        # Loss: MSE between current Q and target Q
        loss = F.mse_loss(current_q, target_q)

        # Gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        self.training_step += 1

        return td_error.item()

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """Save model checkpoint."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }, filepath)

    def load(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']


def train_sarsa(env, agent, num_episodes=500, max_steps=500, print_freq=10, save_freq=50):
    """
    Train the SARSA agent on the boat environment.

    Args:
        env: Boat environment
        agent: SARSA agent
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        print_freq: Frequency to print training progress
        save_freq: Frequency to save model checkpoints

    Returns:
        Training statistics dictionary
    """
    print("\n" + "="*70)
    print(" "*20 + "SARSA TRAINING STARTED")
    print("="*70)
    print(f"Algorithm: Semi-Gradient SARSA (On-Policy TD Control)")
    print(f"Goal: Navigate from (0,0) to ({env.goal_position[0]:.0f},{env.goal_position[1]:.0f})")
    print(f"Episodes: {num_episodes} | Max Steps: {max_steps}")
    print(f"Device: {agent.device}")
    print("="*70 + "\n")

    # Statistics
    episode_rewards = []
    episode_lengths = []
    episode_td_errors = []
    success_count = 0
    best_reward = float('-inf')

    # Create checkpoint directory
    os.makedirs('checkpoints_sarsa', exist_ok=True)

    for episode in range(1, num_episodes + 1):
        state, info = env.reset()

        # Select initial action A using epsilon-greedy
        action = agent.select_action(state, training=True)

        episode_reward = 0
        episode_td_error = []
        done = False
        step = 0

        while not done and step < max_steps:
            # Take action A, observe R, S'
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Select next action A' using epsilon-greedy (on-policy)
            next_action = agent.select_action(next_state, training=True)

            # SARSA update: Q(S,A) using (S, A, R, S', A')
            td_error = agent.update(state, action, reward, next_state, next_action, done)
            episode_td_error.append(abs(td_error))

            episode_reward += reward

            # Move to next state and action
            state = next_state
            action = next_action  # Key: use the selected action (on-policy)
            step += 1

        # Decay epsilon after each episode
        agent.decay_epsilon()

        # Track statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(step)
        avg_td_error = np.mean(episode_td_error) if episode_td_error else 0
        episode_td_errors.append(avg_td_error)

        # Check if goal was reached
        if info['distance_to_goal'] <= env.goal_radius:
            success_count += 1

        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save('checkpoints_sarsa/best_model.pth')

        # Print progress
        if episode % print_freq == 0:
            avg_reward = np.mean(episode_rewards[-print_freq:])
            avg_length = np.mean(episode_lengths[-print_freq:])
            avg_td = np.mean(episode_td_errors[-print_freq:])
            success_rate = (success_count / episode) * 100

            print(f"Episode {episode:4d}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:7.2f} | "
                  f"Avg Length: {avg_length:5.1f} | "
                  f"TD Error: {avg_td:6.4f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Success Rate: {success_rate:5.2f}%")

        # Save checkpoint periodically
        if episode % save_freq == 0:
            agent.save(f'checkpoints_sarsa/checkpoint_ep{episode}.pth')

    # Save final model
    agent.save('checkpoints_sarsa/final_model.pth')

    print("\n" + "="*70)
    print(" "*20 + "TRAINING COMPLETED")
    print("="*70)
    print(f"Total Episodes: {num_episodes}")
    print(f"Final Success Rate: {(success_count/num_episodes)*100:.2f}%")
    print(f"Best Episode Reward: {best_reward:.2f}")
    print(f"Average Episode Reward: {np.mean(episode_rewards):.2f}")
    print("="*70 + "\n")

    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_td_errors': episode_td_errors,
        'success_rate': success_count / num_episodes
    }


def visualize_policy(env, agent, num_points=20, save_path='sarsa_policy_visualization.png'):
    """
    Visualize the learned policy by showing action selections across the state space.

    Args:
        env: Boat environment
        agent: Trained SARSA agent
        num_points: Number of grid points in each dimension
        save_path: Path to save visualization
    """
    print("Generating SARSA policy visualization...")

    # Create grid of positions
    x_range = np.linspace(-env.bounds * 0.8, env.bounds * 0.8, num_points)
    y_range = np.linspace(-env.bounds * 0.8, env.bounds * 0.8, num_points)

    # Action names for legend
    action_names = [
        "Both Idle", "L-Fwd, R-Idle", "L-Back, R-Idle",
        "L-Idle, R-Fwd", "L-Idle, R-Back", "Both Forward",
        "Both Backward", "Rotate Right", "Rotate Left"
    ]

    # Action colors
    colors = plt.cm.tab10(np.linspace(0, 1, 9))

    fig, ax = plt.subplots(figsize=(12, 10))

    # For each grid point, determine the best action
    for x in x_range:
        for y in y_range:
            # Calculate angle towards goal
            dx = env.goal_position[0] - x
            dy = env.goal_position[1] - y
            angle_to_goal = np.arctan2(dy, dx)

            # Create state (position, angle pointing to goal, zero velocities)
            state = np.array([x, y, angle_to_goal, 0.0, 0.0, 0.0])

            # Get best action from agent (greedy)
            action = agent.select_action(state, training=False)

            # Plot arrow indicating action
            arrow_length = 2.0
            if action == 0:  # Both idle
                ax.scatter(x, y, c=[colors[action]], s=30, alpha=0.6)
            elif action == 5:  # Both forward
                ax.arrow(x, y, arrow_length * np.cos(angle_to_goal),
                        arrow_length * np.sin(angle_to_goal),
                        head_width=1, head_length=0.5, fc=colors[action],
                        ec=colors[action], alpha=0.6)
            elif action == 6:  # Both backward
                ax.arrow(x, y, -arrow_length * np.cos(angle_to_goal),
                        -arrow_length * np.sin(angle_to_goal),
                        head_width=1, head_length=0.5, fc=colors[action],
                        ec=colors[action], alpha=0.6)
            elif action == 7:  # Rotate right
                from matplotlib.patches import Circle
                circle = Circle((x, y), radius=1.0, color=colors[action],
                              fill=False, linewidth=2, alpha=0.6)
                ax.add_patch(circle)
            elif action == 8:  # Rotate left
                circle = Circle((x, y), radius=1.0, color=colors[action],
                              fill=True, alpha=0.3)
                ax.add_patch(circle)
            else:  # Single rudder actions
                ax.scatter(x, y, c=[colors[action]], s=50, marker='s', alpha=0.6)

    # Plot start and goal
    ax.scatter(0, 0, c='green', s=200, marker='*', label='Start (0,0)',
              edgecolors='black', linewidths=2, zorder=10)
    ax.scatter(env.goal_position[0], env.goal_position[1], c='red', s=200,
              marker='*', label=f'Goal ({env.goal_position[0]:.0f},{env.goal_position[1]:.0f})',
              edgecolors='black', linewidths=2, zorder=10)

    # Goal radius circle
    from matplotlib.patches import Circle
    goal_circle = Circle(env.goal_position, env.goal_radius, color='red',
                         fill=False, linestyle='--', linewidth=2, label='Goal Radius')
    ax.add_patch(goal_circle)

    # Create custom legend for actions
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], label=action_names[i])
                      for i in range(9)]
    legend_elements.extend([
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='green',
                   markersize=15, label='Start (0,0)'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
                   markersize=15, label=f'Goal ({env.goal_position[0]:.0f},{env.goal_position[1]:.0f})')
    ])

    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1),
             fontsize=9, framealpha=0.9)

    ax.set_xlim(-env.bounds, env.bounds)
    ax.set_ylim(-env.bounds, env.bounds)
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title('Learned SARSA Policy Visualization', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Policy visualization saved to: {save_path}")
    plt.close()


def plot_training_curves(stats, save_path='sarsa_training_curves.png'):
    """
    Plot training curves showing rewards, episode lengths, and TD errors.

    Args:
        stats: Training statistics dictionary
        save_path: Path to save plot
    """
    print("Generating training curves...")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Episode rewards
    axes[0].plot(stats['episode_rewards'], alpha=0.3, color='blue', label='Episode Reward')
    # Moving average
    window = 20
    if len(stats['episode_rewards']) >= window:
        moving_avg = np.convolve(stats['episode_rewards'],
                                np.ones(window)/window, mode='valid')
        axes[0].plot(range(window-1, len(stats['episode_rewards'])),
                    moving_avg, color='red', linewidth=2, label=f'{window}-Episode Moving Avg')
    axes[0].set_xlabel('Episode', fontsize=11)
    axes[0].set_ylabel('Total Reward', fontsize=11)
    axes[0].set_title('Episode Rewards Over Training (SARSA)', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Episode lengths
    axes[1].plot(stats['episode_lengths'], alpha=0.3, color='green', label='Episode Length')
    if len(stats['episode_lengths']) >= window:
        moving_avg = np.convolve(stats['episode_lengths'],
                                np.ones(window)/window, mode='valid')
        axes[1].plot(range(window-1, len(stats['episode_lengths'])),
                    moving_avg, color='red', linewidth=2, label=f'{window}-Episode Moving Avg')
    axes[1].set_xlabel('Episode', fontsize=11)
    axes[1].set_ylabel('Steps', fontsize=11)
    axes[1].set_title('Episode Lengths Over Training', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # TD errors
    axes[2].plot(stats['episode_td_errors'], alpha=0.3, color='purple', label='Episode Avg TD Error')
    if len(stats['episode_td_errors']) >= window:
        moving_avg = np.convolve(stats['episode_td_errors'],
                                np.ones(window)/window, mode='valid')
        axes[2].plot(range(window-1, len(stats['episode_td_errors'])),
                    moving_avg, color='red', linewidth=2, label=f'{window}-Episode Moving Avg')
    axes[2].set_xlabel('Episode', fontsize=11)
    axes[2].set_ylabel('TD Error', fontsize=11)
    axes[2].set_title('Temporal Difference Error Over Time', fontsize=12, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {save_path}")
    plt.close()


def evaluate_agent(env, agent, num_episodes=10, render=False):
    """
    Evaluate trained agent performance.

    Args:
        env: Boat environment
        agent: Trained SARSA agent
        num_episodes: Number of evaluation episodes
        render: Whether to render evaluation

    Returns:
        Evaluation statistics
    """
    print("\n" + "="*70)
    print(" "*25 + "EVALUATION")
    print("="*70)

    success_count = 0
    episode_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False
        step = 0

        while not done:
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step += 1

            if render and env.render_mode == 'human':
                env.render()

        episode_rewards.append(episode_reward)
        episode_lengths.append(step)

        if info['distance_to_goal'] <= env.goal_radius:
            success_count += 1
            status = "SUCCESS"
        else:
            status = "FAILED"

        print(f"Episode {episode+1:2d}: Reward={episode_reward:7.2f} | "
              f"Length={step:3d} | Distance={info['distance_to_goal']:5.2f} | "
              f"Status={status}")

    print("="*70)
    print(f"Success Rate: {(success_count/num_episodes)*100:.2f}%")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print("="*70 + "\n")

    return {
        'success_rate': success_count / num_episodes,
        'avg_reward': np.mean(episode_rewards),
        'avg_length': np.mean(episode_lengths)
    }


def main():
    """Main training function."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    # Create environment with goal at (5, 5)
    env = BoatEnv(
        goal_position=[5.0, 5.0],
        bounds=50.0,
        max_steps=500,
        render_mode=None  # Set to 'human' for visualization during training
    )

    print("Environment created successfully!")
    print(f"State space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Goal position: {env.goal_position}")

    # Create SARSA agent
    agent = SARSAAgent(
        state_dim=6,
        action_dim=9,
        learning_rate=0.0005,  # Lower than DQN due to online learning
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995
    )

    # Train agent
    stats = train_sarsa(
        env=env,
        agent=agent,
        num_episodes=500,
        max_steps=500,
        print_freq=10,
        save_freq=50
    )

    # Plot training curves
    plot_training_curves(stats, save_path='sarsa_training_curves.png')

    # Visualize learned policy
    visualize_policy(env, agent, num_points=20, save_path='sarsa_policy_visualization.png')

    # Evaluate trained agent
    eval_stats = evaluate_agent(env, agent, num_episodes=10, render=False)

    env.close()
    print("\nTraining complete! Check 'checkpoints_sarsa/' directory for saved models.")
    print("Visualizations saved: 'sarsa_training_curves.png' and 'sarsa_policy_visualization.png'")


if __name__ == "__main__":
    main()
