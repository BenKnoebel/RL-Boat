"""
Evaluation Script for Trained Double DQN Models

This script loads a trained Double DQN model and evaluates it on the boat environment,
with comprehensive trajectory visualization.

Usage:
    python evaluate_ddqn.py <model_path> [options]

Example:
    python evaluate_ddqn.py checkpoints_ddqn/best_model.pth --episodes 5 --save-trajectories
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow
import os
from envs.boat_env import BoatEnv


class DDQNNetwork(nn.Module):
    """
    Neural network for Q-value approximation in Double DQN.
    Must match the architecture used in training.
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


class DoubleDQNAgent:
    """
    Double DQN Agent for evaluation (inference only).
    """

    def __init__(self, state_dim=6, action_dim=9):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load Q-network only (no need for target network during evaluation)
        self.q_network = DDQNNetwork(state_dim, action_dim).to(self.device)
        self.q_network.eval()

    def select_action(self, state, deterministic=True):
        """
        Select action using the learned policy.

        Args:
            state: Current state
            deterministic: If True, always select best action (no exploration)

        Returns:
            Selected action
        """
        with torch.no_grad():
            state_tensor = torch.from_numpy(np.array(state, dtype=np.float32)).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()

    def load(self, filepath):
        """Load model checkpoint."""
        print(f"Loading Double DQN model from: {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        print(f"Model loaded successfully on device: {self.device}")

        # Print model info if available
        if 'epsilon' in checkpoint:
            print(f"Model training epsilon: {checkpoint['epsilon']:.4f}")
        if 'training_step' in checkpoint:
            print(f"Model training steps: {checkpoint['training_step']}")


def run_episode(agent, env, render=False):
    """
    Run a single episode and record trajectory.

    Args:
        agent: Trained Double DQN agent
        env: Boat environment
        render: Whether to render the episode

    Returns:
        Dictionary containing trajectory data
    """
    state, info = env.reset()
    done = False
    step = 0

    # Track trajectory
    trajectory = {
        'states': [state.copy()],
        'actions': [],
        'rewards': [],
        'positions': [state[:2].copy()],
        'angles': [state[2]],
        'total_reward': 0,
        'total_steps': 0,
        'success': False
    }

    while not done:
        action = agent.select_action(state, deterministic=True)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Record trajectory
        trajectory['actions'].append(action)
        trajectory['rewards'].append(reward)
        trajectory['states'].append(next_state.copy())
        trajectory['positions'].append(next_state[:2].copy())
        trajectory['angles'].append(next_state[2])
        trajectory['total_reward'] += reward

        if render:
            env.render()

        state = next_state
        step += 1

    trajectory['total_steps'] = step
    trajectory['success'] = info['distance_to_goal'] <= env.goal_radius
    trajectory['final_distance'] = info['distance_to_goal']
    trajectory['goal_position'] = info['goal_position']

    return trajectory


def visualize_trajectory(trajectory, env, save_path='trajectory_visualization_ddqn.png'):
    """
    Visualize a single episode trajectory with zoomed view.

    Args:
        trajectory: Dictionary containing trajectory data
        env: Environment (for goal position and bounds)
        save_path: Path to save visualization
    """
    print(f"Generating trajectory visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    positions = np.array(trajectory['positions'])
    angles = np.array(trajectory['angles'])
    actions = np.array(trajectory['actions'])
    rewards = np.array(trajectory['rewards'])
    goal_pos = trajectory['goal_position']

    # Action names for legend
    action_names = [
        "Both Idle", "L-Fwd, R-Idle", "L-Back, R-Idle",
        "L-Idle, R-Fwd", "L-Idle, R-Back", "Both Forward",
        "Both Backward", "Rotate Right", "Rotate Left"
    ]

    # Color map for actions
    colors = plt.cm.tab10(np.linspace(0, 1, 9))

    # Calculate zoom bounds based on trajectory
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()

    # Add margin around trajectory (20% of range or at least 2 units)
    x_range = max(x_max - x_min, 4.0)
    y_range = max(y_max - y_min, 4.0)
    margin_x = max(x_range * 0.2, 2.0)
    margin_y = max(y_range * 0.2, 2.0)

    zoom_x_min = x_min - margin_x
    zoom_x_max = x_max + margin_x
    zoom_y_min = y_min - margin_y
    zoom_y_max = y_max + margin_y

    # --- Plot 1: Trajectory with actions (ZOOMED) ---
    ax1 = axes[0, 0]

    # Plot trajectory line
    ax1.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.3, linewidth=2, label='Trajectory')

    # Plot positions colored by action
    for i in range(len(actions)):
        ax1.scatter(positions[i, 0], positions[i, 1], c=[colors[actions[i]]],
                   s=30, alpha=0.6, edgecolors='none')

    # Plot start and goal
    ax1.scatter(positions[0, 0], positions[0, 1], c='green', s=300, marker='*',
               edgecolors='black', linewidths=2, label='Start', zorder=10)
    ax1.scatter(goal_pos[0], goal_pos[1], c='red', s=300, marker='*',
               edgecolors='black', linewidths=2, label='Goal', zorder=10)

    # Plot goal radius
    goal_circle = Circle(goal_pos, env.goal_radius, color='red', fill=False,
                        linestyle='--', linewidth=2, label='Goal Radius')
    ax1.add_patch(goal_circle)

    # Plot final position
    ax1.scatter(positions[-1, 0], positions[-1, 1], c='orange', s=200, marker='X',
               label='Final Position', edgecolors='black', linewidths=2, zorder=9)

    # Apply zoom to focus on trajectory
    ax1.set_xlim(zoom_x_min, zoom_x_max)
    ax1.set_ylim(zoom_y_min, zoom_y_max)
    ax1.set_xlabel('X Position (m)', fontsize=11)
    ax1.set_ylabel('Y Position (m)', fontsize=11)
    ax1.set_title('Trajectory with Actions (Zoomed) - Double DQN', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')

    # --- Plot 2: Boat orientation along trajectory (ZOOMED) ---
    ax2 = axes[0, 1]

    # Plot trajectory
    ax2.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.3, linewidth=2)

    # Plot boat orientation at intervals (scale arrow size based on zoom)
    step_interval = max(1, len(positions) // 20)  # Show ~20 orientations
    arrow_scale = min(x_range, y_range) * 0.05  # Scale arrows to zoom level
    for i in range(0, len(positions), step_interval):
        x, y = positions[i]
        angle = angles[i]
        # Draw arrow showing boat direction
        dx = arrow_scale * np.cos(angle)
        dy = arrow_scale * np.sin(angle)
        arrow = FancyArrow(x, y, dx, dy, width=arrow_scale*0.25, head_width=arrow_scale*0.75,
                          head_length=arrow_scale*0.5, fc='blue', ec='darkblue', alpha=0.6)
        ax2.add_patch(arrow)

    # Plot start and goal
    ax2.scatter(positions[0, 0], positions[0, 1], c='green', s=300, marker='*',
               edgecolors='black', linewidths=2, zorder=10)
    ax2.scatter(goal_pos[0], goal_pos[1], c='red', s=300, marker='*',
               edgecolors='black', linewidths=2, zorder=10)

    goal_circle = Circle(goal_pos, env.goal_radius, color='red', fill=False,
                        linestyle='--', linewidth=2)
    ax2.add_patch(goal_circle)

    # Apply zoom to focus on trajectory
    ax2.set_xlim(zoom_x_min, zoom_x_max)
    ax2.set_ylim(zoom_y_min, zoom_y_max)
    ax2.set_xlabel('X Position (m)', fontsize=11)
    ax2.set_ylabel('Y Position (m)', fontsize=11)
    ax2.set_title('Boat Orientation Along Trajectory (Zoomed) - Double DQN', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')

    # --- Plot 3: Rewards over time ---
    ax3 = axes[1, 0]

    steps = np.arange(len(rewards))
    cumulative_reward = np.cumsum(rewards)

    ax3_twin = ax3.twinx()
    ax3.plot(steps, rewards, 'b-', alpha=0.6, linewidth=1.5, label='Step Reward')
    ax3_twin.plot(steps, cumulative_reward, 'r-', linewidth=2, label='Cumulative Reward')

    ax3.set_xlabel('Step', fontsize=11)
    ax3.set_ylabel('Step Reward', fontsize=11, color='b')
    ax3_twin.set_ylabel('Cumulative Reward', fontsize=11, color='r')
    ax3.set_title('Rewards Over Time - Double DQN', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='y', labelcolor='b')
    ax3_twin.tick_params(axis='y', labelcolor='r')
    ax3.grid(True, alpha=0.3)

    # Combine legends
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9)

    # --- Plot 4: Action distribution ---
    ax4 = axes[1, 1]

    # Count action occurrences
    action_counts = np.bincount(actions, minlength=9)
    action_percentages = (action_counts / len(actions)) * 100

    # Create bar plot
    bars = ax4.bar(range(9), action_percentages, color=[colors[i] for i in range(9)],
                   edgecolor='black', linewidth=1)

    # Add percentage labels on top of bars
    for i, (bar, pct) in enumerate(zip(bars, action_percentages)):
        if pct > 0:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{pct:.1f}%\n({action_counts[i]})',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax4.set_xlabel('Action', fontsize=11)
    ax4.set_ylabel('Usage Percentage (%)', fontsize=11)
    ax4.set_title('Action Distribution - Double DQN', fontsize=12, fontweight='bold')
    ax4.set_xticks(range(9))

    # Create short labels for x-axis
    short_action_labels = [
        "Idle", "L-Fwd", "L-Back",
        "R-Fwd", "R-Back", "Both-F",
        "Both-B", "Rot-R", "Rot-L"
    ]
    ax4.set_xticklabels(short_action_labels, fontsize=8, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add full action names as legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], edgecolor='black',
                            label=f'{i}: {action_names[i]}') for i in range(9)]
    ax4.legend(handles=legend_elements, loc='upper left', fontsize=7,
              ncol=1, framealpha=0.9, bbox_to_anchor=(1.02, 1))

    # Add statistics as text box
    stats_text = (
        f"Episode Statistics:\n"
        f"  Algorithm: Double DQN\n"
        f"  Total Steps: {trajectory['total_steps']}\n"
        f"  Total Reward: {trajectory['total_reward']:.2f}\n"
        f"  Success: {'Yes ✓' if trajectory['success'] else 'No ✗'}\n"
        f"  Final Distance: {trajectory['final_distance']:.2f}m\n"
        f"  Goal Position: ({goal_pos[0]:.1f}, {goal_pos[1]:.1f})"
    )
    fig.text(0.01, 0.01, stats_text, fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, pad=0.5),
             verticalalignment='bottom')

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space for stats box
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Trajectory visualization saved to: {save_path}")
    plt.close()


def evaluate_model(agent, env, n_episodes=10, render=False, save_trajectories=False, output_dir='evaluation_results_ddqn'):
    """
    Evaluate trained Double DQN model over multiple episodes.

    Args:
        agent: Trained Double DQN agent
        env: Boat environment
        n_episodes: Number of episodes to evaluate
        render: Whether to render episodes
        save_trajectories: Whether to save trajectory visualizations
        output_dir: Directory to save trajectory visualizations

    Returns:
        Evaluation statistics
    """
    print("\n" + "="*70)
    print(" "*18 + "DOUBLE DQN MODEL EVALUATION")
    print("="*70)

    if save_trajectories:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Trajectory visualizations will be saved to: {output_dir}/")

    episode_rewards = []
    episode_lengths = []
    episode_distances = []
    successes = 0

    for episode in range(n_episodes):
        trajectory = run_episode(agent, env, render=render)

        episode_rewards.append(trajectory['total_reward'])
        episode_lengths.append(trajectory['total_steps'])
        episode_distances.append(trajectory['final_distance'])

        if trajectory['success']:
            successes += 1

        # Save trajectory visualization if requested
        if save_trajectories:
            save_path = os.path.join(output_dir, f"trajectory_ep{episode+1}.png")
            visualize_trajectory(trajectory, env, save_path=save_path)

        status = "SUCCESS ✓" if trajectory['success'] else "FAILED ✗"
        print(f"Episode {episode+1:2d}/{n_episodes}: "
              f"Reward = {trajectory['total_reward']:7.2f}, "
              f"Steps = {trajectory['total_steps']:3d}, "
              f"Final Dist = {trajectory['final_distance']:5.2f}m, "
              f"Status = {status}")

    print("-"*70)
    print(f"Average Reward:      {np.mean(episode_rewards):7.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Length:      {np.mean(episode_lengths):7.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Average Final Dist:  {np.mean(episode_distances):7.2f} ± {np.std(episode_distances):.2f}m")
    print(f"Success Rate:        {successes}/{n_episodes} ({100*successes/n_episodes:.1f}%)")
    print("="*70 + "\n")

    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'mean_distance': np.mean(episode_distances),
        'std_distance': np.std(episode_distances),
        'success_rate': successes / n_episodes,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_distances': episode_distances
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Double DQN model on the boat environment")
    parser.add_argument("model_path", type=str, help="Path to the trained Double DQN model (.pth file)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes (default: 10)")
    parser.add_argument("--render", action="store_true",
                        help="Render the episodes (not recommended for many episodes)")
    parser.add_argument("--save-trajectories", action="store_true",
                        help="Save trajectory visualizations for each episode")
    parser.add_argument("--output-dir", type=str, default="evaluation_results_ddqn",
                        help="Directory to save trajectory visualizations (default: evaluation_results_ddqn)")
    parser.add_argument("--goal", type=float, nargs=2, default=[5.0, 5.0],
                        help="Goal position [x y] (default: 5.0 5.0)")
    parser.add_argument("--bounds", type=float, default=50.0,
                        help="Environment bounds (default: 50.0)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    print("\n" + "="*70)
    print(" "*15 + "DOUBLE DQN MODEL EVALUATION SCRIPT")
    print("="*70)
    print(f"Model Path: {args.model_path}")
    print(f"Episodes: {args.episodes}")
    print(f"Goal Position: {args.goal}")
    print(f"Environment Bounds: ±{args.bounds}m")
    if args.seed is not None:
        print(f"Random Seed: {args.seed}")
    print("="*70)

    # Create environment
    env = BoatEnv(
        goal_position=args.goal,
        bounds=args.bounds,
        max_steps=500,
        goal_radius=1.0,
        render_mode='human' if args.render else None
    )

    if args.seed is not None:
        env.reset(seed=args.seed)

    # Create and load agent
    agent = DoubleDQNAgent(state_dim=6, action_dim=9)
    agent.load(args.model_path)

    # Evaluate model
    stats = evaluate_model(
        agent,
        env,
        n_episodes=args.episodes,
        render=args.render,
        save_trajectories=args.save_trajectories,
        output_dir=args.output_dir
    )

    # Cleanup
    env.close()

    print("Evaluation complete!")
    if args.save_trajectories:
        print(f"Trajectory visualizations saved to: {args.output_dir}/")

    return stats


if __name__ == "__main__":
    main()
