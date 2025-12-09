"""
Evaluation script for custom DQN trained models.

This script loads a trained custom DQN model and evaluates it on the boat environment,
with optional visualization using the existing rendering pipeline.
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow
import os
from envs.boat_env import BoatEnv
from train_dqn import DQNAgent, DQNNetwork


def load_dqn_checkpoint(checkpoint_path, state_dim=6, action_dim=9):
    """
    Load a trained DQN model from checkpoint.

    Args:
        checkpoint_path: Path to the saved checkpoint (.pth file)
        state_dim: Dimension of state space
        action_dim: Number of discrete actions

    Returns:
        agent: Loaded DQN agent
    """
    print(f"Loading DQN model from: {checkpoint_path}")

    # Check if file exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Create agent (epsilon will be loaded from checkpoint)
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=0.0,  # Will be overwritten by checkpoint
        epsilon_end=0.0,
        epsilon_decay=1.0,
        batch_size=256,
        target_update_freq=10,
        train_freq=4,
        num_gradient_steps=1
    )

    # Load checkpoint
    agent.load(checkpoint_path)

    # Set to evaluation mode (no exploration)
    agent.epsilon = 0.0
    agent.q_network.eval()

    print(f"Model loaded successfully!")
    print(f"Training steps: {agent.training_step}")

    return agent


def evaluate_episode(env, agent, render=False, save_trajectory=False):
    """
    Evaluate a single episode.

    Args:
        env: Boat environment
        agent: Trained DQN agent
        render: Whether to render during evaluation
        save_trajectory: Whether to save trajectory data

    Returns:
        dict: Episode statistics and trajectory data
    """
    state, info = env.reset()
    episode_reward = 0
    done = False
    step = 0

    # Store trajectory for visualization
    trajectory = {
        'states': [state.copy()],
        'actions': [],
        'rewards': [],
        'positions': [(state[0], state[1])],
        'angles': [state[2]]
    }

    while not done:
        # Select action (deterministic policy)
        action = agent.select_action(state, training=False)

        # Take step in environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Track episode data
        episode_reward += reward
        step += 1

        # Store trajectory data
        if save_trajectory:
            trajectory['states'].append(next_state.copy())
            trajectory['actions'].append(action)
            trajectory['rewards'].append(reward)
            trajectory['positions'].append((next_state[0], next_state[1]))
            trajectory['angles'].append(next_state[2])

        # Render if requested
        if render:
            env.render()

        state = next_state

    # Add final info
    trajectory['total_reward'] = episode_reward
    trajectory['total_steps'] = step
    trajectory['success'] = terminated and info['distance_to_goal'] <= env.goal_radius
    trajectory['final_distance'] = info['distance_to_goal']
    trajectory['goal_position'] = info['goal_position']

    return trajectory


def visualize_trajectory(trajectory, env, save_path='trajectory_visualization.png'):
    """
    Visualize a single episode trajectory with zoomed view.

    Args:
        trajectory: Dictionary containing trajectory data
        env: Environment (for goal position and bounds)
        save_path: Path to save visualization
    """
    print(f"Generating trajectory visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Extract data
    positions = np.array(trajectory['positions'])
    angles = np.array(trajectory['angles'])
    rewards = np.array(trajectory['rewards'])
    actions = np.array(trajectory['actions'])
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
                   s=50, alpha=0.6, edgecolors='black', linewidth=0.5)

    # Plot start and goal
    ax1.scatter(positions[0, 0], positions[0, 1], c='green', s=300, marker='*',
               label='Start', edgecolors='black', linewidths=2, zorder=10)
    ax1.scatter(goal_pos[0], goal_pos[1], c='red', s=300, marker='*',
               label='Goal', edgecolors='black', linewidths=2, zorder=10)

    # Goal radius circle
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
    ax1.set_title('Trajectory with Actions (Zoomed)', fontsize=12, fontweight='bold')
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
    ax2.set_title('Boat Orientation Along Trajectory (Zoomed)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')

    # --- Plot 3: Rewards over time ---
    ax3 = axes[1, 0]

    steps = np.arange(len(rewards))
    cumulative_reward = np.cumsum(rewards)

    ax3_twin = ax3.twinx()

    # Plot step rewards
    ax3.bar(steps, rewards, alpha=0.6, color='steelblue', label='Step Reward')
    # Plot cumulative reward
    ax3_twin.plot(steps, cumulative_reward, color='red', linewidth=2,
                 marker='o', markersize=3, label='Cumulative Reward')

    ax3.set_xlabel('Time Step', fontsize=11)
    ax3.set_ylabel('Step Reward', fontsize=11, color='steelblue')
    ax3_twin.set_ylabel('Cumulative Reward', fontsize=11, color='red')
    ax3.tick_params(axis='y', labelcolor='steelblue')
    ax3_twin.tick_params(axis='y', labelcolor='red')
    ax3.set_title('Rewards Over Time', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

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
    ax4.set_title('Action Distribution', fontsize=12, fontweight='bold')
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


def evaluate_model(agent, env, n_episodes=10, render=False, save_trajectories=False, output_dir='evaluation_results'):
    """
    Evaluate trained DQN model over multiple episodes.

    Args:
        agent: Trained DQN agent
        env: Boat environment
        n_episodes: Number of evaluation episodes
        render: Whether to render episodes
        save_trajectories: Whether to save trajectory visualizations
        output_dir: Directory to save results

    Returns:
        dict: Evaluation statistics
    """
    print("\n" + "="*70)
    print(" "*25 + "EVALUATION")
    print("="*70)
    print(f"Episodes: {n_episodes}")
    print(f"Goal Position: ({env.goal_position[0]:.1f}, {env.goal_position[1]:.1f})")
    print(f"Goal Radius: {env.goal_radius:.1f}m")
    print("="*70 + "\n")

    # Create output directory if saving trajectories
    if save_trajectories:
        os.makedirs(output_dir, exist_ok=True)

    episode_rewards = []
    episode_lengths = []
    success_count = 0
    final_distances = []
    all_trajectories = []

    for episode in range(n_episodes):
        print(f"Evaluating episode {episode + 1}/{n_episodes}...", end=' ')

        # Run episode
        trajectory = evaluate_episode(env, agent, render=render, save_trajectory=save_trajectories)

        # Store statistics
        episode_rewards.append(trajectory['total_reward'])
        episode_lengths.append(trajectory['total_steps'])
        final_distances.append(trajectory['final_distance'])
        all_trajectories.append(trajectory)

        if trajectory['success']:
            success_count += 1
            status = "SUCCESS ✓"
        else:
            status = "FAILED ✗"

        print(f"Reward: {trajectory['total_reward']:7.2f} | "
              f"Steps: {trajectory['total_steps']:3d} | "
              f"Distance: {trajectory['final_distance']:5.2f}m | "
              f"Status: {status}")

        # Save trajectory visualization
        if save_trajectories:
            traj_path = os.path.join(output_dir, f'trajectory_ep{episode+1}.png')
            visualize_trajectory(trajectory, env, save_path=traj_path)

    # Print summary statistics
    print("\n" + "="*70)
    print(" "*25 + "SUMMARY")
    print("="*70)
    print(f"Success Rate:      {success_count}/{n_episodes} ({100*success_count/n_episodes:.1f}%)")
    print(f"Average Reward:    {np.mean(episode_rewards):7.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Length:    {np.mean(episode_lengths):7.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Average Distance:  {np.mean(final_distances):7.2f} ± {np.std(final_distances):.2f}m")
    print(f"Best Reward:       {np.max(episode_rewards):7.2f}")
    print(f"Worst Reward:      {np.min(episode_rewards):7.2f}")
    print("="*70 + "\n")

    return {
        'success_rate': success_count / n_episodes,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'mean_distance': np.mean(final_distances),
        'std_distance': np.std(final_distances),
        'trajectories': all_trajectories if save_trajectories else None
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained custom DQN model on the boat environment")

    # Model arguments
    parser.add_argument("checkpoint_path", type=str, help="Path to the trained model checkpoint (.pth file)")

    # Environment arguments
    parser.add_argument("--goal", type=float, nargs=2, default=[5.0, 5.0],
                        help="Goal position [x y] (default: 5.0 5.0)")
    parser.add_argument("--bounds", type=float, default=50.0,
                        help="Environment bounds (default: 50.0)")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Maximum steps per episode (default: 500)")
    parser.add_argument("--goal-radius", type=float, default=1.0,
                        help="Goal radius (default: 1.0)")

    # Evaluation arguments
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes (default: 10)")
    parser.add_argument("--render", action="store_true",
                        help="Render episodes in real-time")
    parser.add_argument("--save-trajectories", action="store_true",
                        help="Save trajectory visualizations for each episode")
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                        help="Directory to save evaluation results (default: evaluation_results)")

    args = parser.parse_args()

    # Load model
    agent = load_dqn_checkpoint(args.checkpoint_path)

    # Create environment
    print(f"\nCreating environment:")
    print(f"  Goal: ({args.goal[0]}, {args.goal[1]})")
    print(f"  Bounds: ±{args.bounds}")
    print(f"  Max Steps: {args.max_steps}")
    print(f"  Goal Radius: {args.goal_radius}")

    env = BoatEnv(
        goal_position=args.goal,
        bounds=args.bounds,
        max_steps=args.max_steps,
        goal_radius=args.goal_radius,
        render_mode='human' if args.render else None
    )

    # Evaluate
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

    return stats


if __name__ == "__main__":
    main()
