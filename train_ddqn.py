"""
Double Deep Q-Network (DDQN) Training Script for Boat Environment

This script trains a Double DQN agent to navigate from (0,0) to (5,5) using
the boat environment with two independent rudders.

DOUBLE DQN KEY DIFFERENCES FROM DQN:
- Action selection: Uses online network to SELECT the best action
- Action evaluation: Uses target network to EVALUATE that action
- This decoupling reduces overestimation bias in Q-value estimates
- Formula: Q_target = r + γ * Q_target(s', argmax_a Q_online(s', a))

Standard DQN uses: Q_target = r + γ * max_a Q_target(s', a)
This leads to overestimation because the same network both selects and evaluates.

PERFORMANCE OPTIMIZATIONS:
- Larger batch size (256 vs 64) for better GPU utilization
- Training every N steps (train_freq=4) instead of every step
- Non-blocking tensor transfers to GPU
- Multiple gradient steps per training call
- Reduced CPU-GPU synchronization points

Reference: "Deep Reinforcement Learning with Double Q-learning" (van Hasselt et al., 2015)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt
from datetime import datetime
import os
from envs.boat_env import BoatEnv


class DDQNNetwork(nn.Module):
    """
    Neural network for Q-value approximation in Double DQN.

    Architecture:
    - Input: 6D state (x, y, angle, vx, vy, angular_velocity)
    - Hidden: 3 fully connected layers with ReLU activation
    - Output: 9D Q-values (one for each action)
    """

    def __init__(self, state_dim=6, action_dim=9, hidden_dim=128):
        super(DDQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions."""

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a random batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


class DoubleDQNAgent:
    """
    Double DQN Agent with experience replay and target network.
    Uses Double Q-learning to reduce overestimation bias.
    Optimized for GPU training with reduced transfer overhead.
    """

    def __init__(self, state_dim=6, action_dim=9, learning_rate=0.001,
                 gamma=0.99, epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay=0.995, buffer_size=100000, batch_size=256,
                 target_update_freq=10, train_freq=4, num_gradient_steps=1):
        """
        Initialize Double DQN Agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Exponential decay rate for epsilon
            buffer_size: Size of replay buffer
            batch_size: Batch size for training (increased for better GPU utilization)
            target_update_freq: Frequency (episodes) to update target network
            train_freq: Train every N steps (reduces GPU transfer overhead)
            num_gradient_steps: Number of gradient steps per training call
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_freq = train_freq
        self.num_gradient_steps = num_gradient_steps

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        print(f"Algorithm: Double DQN (reduces overestimation bias)")
        print(f"Optimizations: batch_size={batch_size}, train_freq={train_freq}, gradient_steps={num_gradient_steps}")

        # Q-Network (online) and Target Network
        self.q_network = DDQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network = DDQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer and replay buffer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Training statistics
        self.training_step = 0
        self.step_count = 0

    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy.
        Optimized to reduce GPU transfer overhead.

        Args:
            state: Current state
            training: If True, use epsilon-greedy; otherwise greedy

        Returns:
            Selected action
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        # Optimized: create tensor directly on GPU when possible
        with torch.no_grad():
            state_tensor = torch.from_numpy(np.array(state, dtype=np.float32)).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()

    def train_step(self):
        """
        Perform multiple training steps using DOUBLE DQN update rule.

        KEY DIFFERENCE FROM STANDARD DQN:
        - Standard DQN: target = r + γ * max_a Q_target(s', a)
        - Double DQN: target = r + γ * Q_target(s', argmax_a Q_online(s', a))

        This decoupling reduces overestimation bias by separating action
        selection (online network) from action evaluation (target network).

        Returns:
            Average loss value for monitoring (as tensor, no CPU-GPU sync)
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Perform multiple gradient steps per training call for better efficiency
        losses = []
        for _ in range(self.num_gradient_steps):
            # Sample batch from replay buffer
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

            # Optimized: Convert to tensors with non_blocking transfer
            states = torch.from_numpy(states).float().to(self.device, non_blocking=True)
            actions = torch.from_numpy(actions).long().to(self.device, non_blocking=True)
            rewards = torch.from_numpy(rewards).float().to(self.device, non_blocking=True)
            next_states = torch.from_numpy(next_states).float().to(self.device, non_blocking=True)
            dones = torch.from_numpy(dones).float().to(self.device, non_blocking=True)

            # Compute current Q values
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

            # ==================== DOUBLE DQN UPDATE ====================
            # Standard DQN would do: max_next_q = target_network(next_states).max(1)[0]
            # Double DQN does the following:

            with torch.no_grad():
                # Step 1: Use ONLINE network to SELECT the best action for next state
                next_actions = self.q_network(next_states).argmax(1)

                # Step 2: Use TARGET network to EVALUATE that selected action
                next_q_values = self.target_network(next_states)
                max_next_q_values = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze()

                # Compute target Q values
                target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
            # ============================================================

            # Compute loss
            loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
            self.optimizer.step()

            self.training_step += 1
            losses.append(loss.detach())

        # Return average loss as tensor (avoid .item() for less CPU-GPU sync)
        return torch.stack(losses).mean() if losses else None

    def update_target_network(self):
        """Copy weights from Q-network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """Save model checkpoint."""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }
        torch.save(checkpoint, filepath)

    def load(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']


def train_ddqn(env, agent, num_episodes=1000, max_steps=500, print_freq=10, save_freq=50):
    """
    Train the Double DQN agent on the boat environment.

    Args:
        env: Boat environment
        agent: Double DQN agent
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        print_freq: Frequency to print training progress
        save_freq: Frequency to save model checkpoints

    Returns:
        Training statistics dictionary
    """
    print("\n" + "="*70)
    print(" "*18 + "DOUBLE DQN TRAINING STARTED")
    print("="*70)
    print(f"Algorithm: Double DQN (van Hasselt et al., 2015)")
    print(f"Key Feature: Decoupled action selection and evaluation")
    print(f"Goal: Navigate from (0,0) to ({env.goal_position[0]:.0f},{env.goal_position[1]:.0f})")
    print(f"Episodes: {num_episodes} | Max Steps: {max_steps}")
    print(f"Device: {agent.device}")
    print("="*70 + "\n")

    # Statistics
    episode_rewards = []
    episode_lengths = []
    episode_losses = []
    success_count = 0
    best_reward = float('-inf')

    # Rolling success rate tracker (last 50 episodes)
    recent_successes = deque(maxlen=50)

    # Create checkpoint directory
    os.makedirs('checkpoints_ddqn', exist_ok=True)

    for episode in range(1, num_episodes + 1):
        state, info = env.reset()
        episode_reward = 0
        episode_loss = []
        done = False
        step = 0

        while not done and step < max_steps:
            # Select and perform action
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition in replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, float(done))

            # Optimized: Train agent only every train_freq steps (reduces GPU overhead)
            agent.step_count += 1
            if agent.step_count % agent.train_freq == 0:
                loss = agent.train_step()
                if loss is not None:
                    # Convert tensor to float only when needed for tracking
                    episode_loss.append(loss.item() if torch.is_tensor(loss) else loss)

            episode_reward += reward
            state = next_state
            step += 1

        # Update target network periodically
        if episode % agent.target_update_freq == 0:
            agent.update_target_network()

        # Decay epsilon
        agent.decay_epsilon()

        # Track statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(step)
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        episode_losses.append(avg_loss)

        # Check if goal was reached
        goal_reached = info['distance_to_goal'] <= env.goal_radius
        if goal_reached:
            success_count += 1

        # Track success for rolling window
        recent_successes.append(1 if goal_reached else 0)

        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save('checkpoints_ddqn/best_model.pth')

        # Print progress
        if episode % print_freq == 0:
            avg_reward = np.mean(episode_rewards[-print_freq:])
            avg_length = np.mean(episode_lengths[-print_freq:])
            avg_loss = np.mean(episode_losses[-print_freq:])
            # Rolling success rate over last 50 episodes
            success_rate = (np.mean(recent_successes) * 100) if len(recent_successes) > 0 else 0.0

            print(f"Episode {episode:4d}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:7.2f} | "
                  f"Avg Length: {avg_length:5.1f} | "
                  f"Loss: {avg_loss:6.4f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Success Rate (Last 50): {success_rate:5.2f}%")

        # Save checkpoint periodically
        if episode % save_freq == 0:
            agent.save(f'checkpoints_ddqn/checkpoint_ep{episode}.pth')

    # Save final model
    agent.save('checkpoints_ddqn/final_model.pth')

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)

    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_losses': episode_losses,
        'success_rate': (np.mean(recent_successes) * 100) if len(recent_successes) > 0 else 0.0
    }


def plot_training_curves(stats, save_path='training_curves_ddqn.png'):
    """
    Plot training curves for episode rewards, lengths, and losses.

    Args:
        stats: Dictionary containing training statistics
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    episode_rewards = stats['episode_rewards']
    episode_lengths = stats['episode_lengths']
    episode_losses = stats['episode_losses']

    episodes = np.arange(1, len(episode_rewards) + 1)

    # Plot 1: Episode Rewards
    ax = axes[0, 0]
    ax.plot(episodes, episode_rewards, alpha=0.3, color='blue', label='Raw')
    # Moving average
    window = 50
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax.plot(episodes[window-1:], moving_avg, color='red', linewidth=2, label=f'Moving Avg ({window})')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Reward', fontsize=12)
    ax.set_title('Episode Rewards (Double DQN)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Episode Lengths
    ax = axes[0, 1]
    ax.plot(episodes, episode_lengths, alpha=0.3, color='green', label='Raw')
    # Moving average
    if len(episode_lengths) >= window:
        moving_avg = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
        ax.plot(episodes[window-1:], moving_avg, color='red', linewidth=2, label=f'Moving Avg ({window})')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Episode Length (steps)', fontsize=12)
    ax.set_title('Episode Lengths (Double DQN)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Training Loss
    ax = axes[1, 0]
    ax.plot(episodes, episode_losses, alpha=0.3, color='purple', label='Raw')
    # Moving average
    if len(episode_losses) >= window:
        moving_avg = np.convolve(episode_losses, np.ones(window)/window, mode='valid')
        ax.plot(episodes[window-1:], moving_avg, color='red', linewidth=2, label=f'Moving Avg ({window})')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss (Double DQN)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Success Rate (Rolling)
    ax = axes[1, 1]
    # Calculate rolling success rate (last 50 episodes)
    rolling_window = 50
    success_rates = []
    for i in range(len(episode_rewards)):
        start_idx = max(0, i - rolling_window + 1)
        recent_episodes = episode_rewards[start_idx:i+1]
        # Assume success if reward > 0 (simplified)
        successes = sum(1 for r in recent_episodes if r > 0)
        success_rates.append(100 * successes / len(recent_episodes))

    ax.plot(episodes, success_rates, color='orange', linewidth=2)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title(f'Rolling Success Rate (Last {rolling_window} Episodes)', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3)
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='100%')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining curves saved to: {save_path}")
    plt.close()


def visualize_policy(env, agent, save_path='policy_visualization_ddqn.png', grid_size=50):
    """
    Visualize the learned policy by showing the preferred action at different positions.

    Args:
        env: Boat environment
        agent: Trained Double DQN agent
        save_path: Path to save visualization
        grid_size: Number of grid points per dimension
    """
    print("\nGenerating policy visualization...")

    # Create grid of positions
    x = np.linspace(-env.bounds, env.bounds, grid_size)
    y = np.linspace(-env.bounds, env.bounds, grid_size)
    X, Y = np.meshgrid(x, y)

    # For each position, determine the best action (pointing towards goal)
    # We'll use a fixed angle pointing towards the goal
    actions = np.zeros_like(X, dtype=int)
    q_values_max = np.zeros_like(X)

    goal_pos = env.goal_position

    for i in range(grid_size):
        for j in range(grid_size):
            pos_x, pos_y = X[i, j], Y[i, j]

            # Calculate angle towards goal
            dx = goal_pos[0] - pos_x
            dy = goal_pos[1] - pos_y
            angle_to_goal = np.arctan2(dy, dx)

            # Create state: [x, y, angle, vx, vy, omega]
            state = np.array([pos_x, pos_y, angle_to_goal, 0.0, 0.0, 0.0], dtype=np.float32)

            # Get Q-values from agent
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).unsqueeze(0).to(agent.device)
                q_vals = agent.q_network(state_tensor).cpu().numpy()[0]

            actions[i, j] = np.argmax(q_vals)
            q_values_max[i, j] = np.max(q_vals)

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: Action map
    action_names = [
        "Idle", "L-Fwd", "L-Back", "R-Fwd", "R-Back",
        "Both-F", "Both-B", "Rot-R", "Rot-L"
    ]

    im1 = ax1.imshow(actions, extent=[-env.bounds, env.bounds, -env.bounds, env.bounds],
                     origin='lower', cmap='tab10', alpha=0.7, vmin=0, vmax=8)
    ax1.scatter(goal_pos[0], goal_pos[1], c='red', s=500, marker='*',
               edgecolors='black', linewidths=3, label='Goal', zorder=10)
    ax1.scatter(0, 0, c='green', s=300, marker='o',
               edgecolors='black', linewidths=2, label='Start (approx)', zorder=10)

    # Add colorbar with action labels
    cbar1 = plt.colorbar(im1, ax=ax1, ticks=np.arange(9))
    cbar1.set_label('Action', fontsize=12)
    cbar1.ax.set_yticklabels(action_names, fontsize=9)

    ax1.set_xlabel('X Position (m)', fontsize=12)
    ax1.set_ylabel('Y Position (m)', fontsize=12)
    ax1.set_title('Learned Policy: Best Action at Each Position\n(Double DQN)',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Plot 2: Q-value heatmap
    im2 = ax2.imshow(q_values_max, extent=[-env.bounds, env.bounds, -env.bounds, env.bounds],
                     origin='lower', cmap='viridis', alpha=0.8)
    ax2.scatter(goal_pos[0], goal_pos[1], c='red', s=500, marker='*',
               edgecolors='black', linewidths=3, label='Goal', zorder=10)
    ax2.scatter(0, 0, c='green', s=300, marker='o',
               edgecolors='black', linewidths=2, label='Start (approx)', zorder=10)

    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Max Q-Value', fontsize=12)

    ax2.set_xlabel('X Position (m)', fontsize=12)
    ax2.set_ylabel('Y Position (m)', fontsize=12)
    ax2.set_title('Value Function: Max Q-Value at Each Position\n(Double DQN)',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Policy visualization saved to: {save_path}")
    plt.close()


def evaluate_agent(env, agent, num_episodes=10, render=False):
    """
    Evaluate the trained agent.

    Args:
        env: Boat environment
        agent: Trained Double DQN agent
        num_episodes: Number of episodes to evaluate
        render: Whether to render episodes

    Returns:
        Evaluation statistics
    """
    print("\n" + "="*70)
    print(" "*22 + "EVALUATION STARTED")
    print("="*70)

    episode_rewards = []
    episode_lengths = []
    successes = 0

    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False
        steps = 0

        while not done:
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1

            if render:
                env.render()

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)

        # Check if goal reached
        if info['distance_to_goal'] <= env.goal_radius:
            successes += 1

        status = "SUCCESS" if info['distance_to_goal'] <= env.goal_radius else "FAILED"
        print(f"Episode {episode+1:2d}: Reward = {episode_reward:7.2f}, "
              f"Steps = {steps:3d}, Status = {status}")

    print("-"*70)
    print(f"Average Reward: {np.mean(episode_rewards):7.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):7.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Success Rate: {successes}/{num_episodes} ({100*successes/num_episodes:.1f}%)")
    print("="*70 + "\n")

    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'success_rate': successes / num_episodes
    }


def main():
    """Main training function."""
    print("\n" + "="*70)
    print(" "*15 + "DOUBLE DQN BOAT NAVIGATION TRAINING")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # Create environment
    env = BoatEnv(
        goal_position=[5.0, 5.0],
        bounds=50.0,
        max_steps=500,
        goal_radius=1.0,
        render_mode=None  # Set to 'human' for visualization during training
    )

    print("Environment created successfully!")
    print(f"State space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Goal position: {env.goal_position}")

    # Create Double DQN agent with optimized hyperparameters
    agent = DoubleDQNAgent(
        state_dim=6,
        action_dim=9,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.9996,
        buffer_size=100000,
        batch_size=200,           # Increased from 64 for better GPU utilization
        target_update_freq=10,
        train_freq=4,             # Train every 4 steps instead of every step
        num_gradient_steps=1      # Number of gradient updates per training call
    )

    # Train agent
    stats = train_ddqn(
        env=env,
        agent=agent,
        num_episodes=3000,
        max_steps=500,
        print_freq=10,
        save_freq=50
    )

    # Plot training curves
    plot_training_curves(stats, save_path='training_curves_ddqn.png')

    # Visualize learned policy
    visualize_policy(env, agent, save_path='policy_visualization_ddqn.png')

    # Evaluate trained agent
    eval_stats = evaluate_agent(env, agent, num_episodes=10, render=False)

    env.close()
    print("\nTraining complete! Check 'checkpoints_ddqn/' directory for saved models.")
    print("Visualizations saved: 'training_curves_ddqn.png' and 'policy_visualization_ddqn.png'")


if __name__ == "__main__":
    main()
