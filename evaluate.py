"""
Evaluation script for trained RL models.

This script loads a trained model and evaluates it on the boat environment,
optionally rendering the episodes for visualization.
"""

import argparse
import numpy as np
from stable_baselines3 import DQN, PPO, A2C
from envs.boat_env import BoatEnv


def load_model(model_path, algorithm="DQN"):
    """
    Load a trained model.

    Args:
        model_path: Path to the saved model
        algorithm: Algorithm name (DQN, PPO, A2C)

    Returns:
        model: Loaded model
    """
    print(f"Loading {algorithm} model from: {model_path}")

    if algorithm.upper() == "DQN":
        model = DQN.load(model_path)
    elif algorithm.upper() == "PPO":
        model = PPO.load(model_path)
    elif algorithm.upper() == "A2C":
        model = A2C.load(model_path)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    return model


def evaluate_model(model, env, n_episodes=10, render=False, deterministic=True):
    """
    Evaluate a trained model.

    Args:
        model: Trained RL model
        env: Environment to evaluate on
        n_episodes: Number of episodes to run
        render: Whether to render the episodes
        deterministic: Use deterministic policy

    Returns:
        dict: Evaluation statistics
    """
    episode_rewards = []
    episode_lengths = []
    successes = 0
    black_zone_hits = 0

    print("\n" + "=" * 60)
    print(f"EVALUATING MODEL - {n_episodes} Episodes")
    print("=" * 60)

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        hit_black_zone = False

        while not done:
            # Predict action using the trained model
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            steps += 1
            done = terminated or truncated

            # Track if black zone was hit
            if reward == -25.0:
                hit_black_zone = True

            if render:
                env.render()

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)

        # Success if terminated with positive reward (reached goal)
        success = terminated and reward == 100.0
        if success:
            successes += 1

        if hit_black_zone:
            black_zone_hits += 1

        status = "SUCCESS" if success else ("TRUNCATED" if truncated else "FAILED")
        print(f"Episode {episode + 1:2d}: Reward = {episode_reward:7.2f}, Steps = {steps:3d}, "
              f"Status = {status:9s}, Hit Black Zone = {hit_black_zone}")

    print("-" * 60)
    print(f"Average Reward:    {np.mean(episode_rewards):7.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Length:    {np.mean(episode_lengths):7.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Success Rate:      {successes}/{n_episodes} ({100*successes/n_episodes:.1f}%)")
    print(f"Black Zone Hits:   {black_zone_hits}/{n_episodes} ({100*black_zone_hits/n_episodes:.1f}%)")
    print("=" * 60 + "\n")

    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "success_rate": successes / n_episodes,
        "black_zone_hit_rate": black_zone_hits / n_episodes
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained RL model on the boat environment")
    parser.add_argument("model_path", type=str, help="Path to the trained model")
    parser.add_argument("--algorithm", type=str, default="DQN", choices=["DQN", "PPO", "A2C"],
                        help="RL algorithm used (default: DQN)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes (default: 10)")
    parser.add_argument("--render", action="store_true",
                        help="Render the episodes")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic policy instead of deterministic")
    parser.add_argument("--goal", type=float, nargs=2, default=[30.0, 30.0],
                        help="Goal position [x y] (default: 30.0 30.0)")
    parser.add_argument("--mask", type=str, default="image_mask.png",
                        help="Path to obstacle mask (default: image_mask.png)")
    parser.add_argument("--no-obstacles", action="store_true",
                        help="Disable obstacles (ignore mask)")

    args = parser.parse_args()

    # Load model
    model = load_model(args.model_path, args.algorithm)

    # Create environment
    print(f"\nCreating environment with goal at {args.goal}")
    print(f"Obstacles: {'DISABLED' if args.no_obstacles else 'ENABLED'}")
    if not args.no_obstacles:
        print(f"Mask Path: {args.mask}")

    env = BoatEnv(
        goal_position=args.goal,
        mask_path=None if args.no_obstacles else args.mask,
        bounds=50.0,
        max_steps=500,
        goal_radius=1.0,
        render_mode='human' if args.render else None
    )

    # Evaluate
    stats = evaluate_model(
        model,
        env,
        n_episodes=args.episodes,
        render=args.render,
        deterministic=not args.stochastic
    )

    # Cleanup
    env.close()

    return stats


if __name__ == "__main__":
    main()
