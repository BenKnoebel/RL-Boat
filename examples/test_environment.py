"""
Example script to test the BoatEnv environment.

This script demonstrates:
- Creating the environment
- Running random actions
- Checking the state transitions
- Testing episode completion
"""

import sys
import os
import numpy as np

# Add parent directory to path to import envs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.boat_env import BoatEnv


def test_random_agent(num_episodes=5, render=True):
    """
    Test the environment with a random agent.

    Args:
        num_episodes: Number of episodes to run
        render: Whether to print state information
    """
    # Create environment
    env = BoatEnv(
        goal_position=np.array([20.0, 20.0]),
        goal_radius=2.0,
        max_steps=500
    )

    print("=" * 60)
    print("RL-Boat Environment Test")
    print("=" * 60)
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Goal Position: {env.goal_position}")
    print(f"Goal Radius: {env.goal_radius}")
    print("=" * 60)

    action_names = [
        "Both idle",
        "Left forward, Right idle",
        "Left backward, Right idle",
        "Left idle, Right forward",
        "Left idle, Right backward",
        "Both forward",
        "Both backward",
        "Left forward, Right backward (rotate right)",
        "Left backward, Right forward (rotate left)"
    ]

    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False
        step = 0

        print(f"\n--- Episode {episode + 1} ---")
        print(f"Start Position: ({state[0]:.2f}, {state[1]:.2f})")
        print(f"Goal Position: ({env.goal_position[0]:.2f}, {env.goal_position[1]:.2f})")
        print(f"Initial Distance: {info['distance_to_goal']:.2f}")

        while not done:
            # Sample random action
            action = env.action_space.sample()

            # Take step
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            step += 1

            # Print information every 50 steps or at the end
            if render and (step % 50 == 0 or done):
                print(f"\nStep {step}: Action {action} ({action_names[action]})")
                print(f"  Position: ({state[0]:.2f}, {state[1]:.2f})")
                print(f"  Angle: {np.degrees(state[2]):.1f}°")
                print(f"  Velocity: ({state[3]:.2f}, {state[4]:.2f})")
                print(f"  Distance to goal: {info['distance_to_goal']:.2f}")
                print(f"  Reward: {reward:.2f}")

        print(f"\nEpisode {episode + 1} finished:")
        print(f"  Total Steps: {step}")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Final Distance: {info['distance_to_goal']:.2f}")
        print(f"  Success: {terminated and info['distance_to_goal'] <= env.goal_radius}")

    env.close()
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


def test_specific_actions():
    """
    Test specific action sequences to verify dynamics.
    """
    env = BoatEnv(goal_position=np.array([10.0, 10.0]))

    print("\n" + "=" * 60)
    print("Testing Specific Action Sequences")
    print("=" * 60)

    # Test 1: Both rudders forward
    print("\nTest 1: Both rudders forward (action 5) for 10 steps")
    state, _ = env.reset()
    print(f"Initial state: {state}")

    for _ in range(10):
        state, _, _, _, _ = env.step(5)  # Both forward

    print(f"Final state: {state}")
    print(f"Expected: Boat should move forward in the direction it's facing")

    # Test 2: Rotation with differential rudders
    print("\nTest 2: Left forward, Right backward (action 7) for 10 steps")
    state, _ = env.reset()
    print(f"Initial angle: {np.degrees(state[2]):.1f}°")

    for _ in range(10):
        state, _, _, _, _ = env.step(7)  # Rotate right

    print(f"Final angle: {np.degrees(state[2]):.1f}°")
    print(f"Expected: Boat should rotate (angle should change)")

    # Test 3: Idle action
    print("\nTest 3: Both idle (action 0) for 10 steps")
    state, _ = env.reset()
    # First give it some velocity
    for _ in range(5):
        state, _, _, _, _ = env.step(5)

    initial_velocity = np.linalg.norm(state[3:5])
    print(f"Velocity after acceleration: {initial_velocity:.2f}")

    for _ in range(10):
        state, _, _, _, _ = env.step(0)  # Idle

    final_velocity = np.linalg.norm(state[3:5])
    print(f"Velocity after idling: {final_velocity:.2f}")
    print(f"Expected: Velocity should decrease due to friction")

    print("=" * 60)


if __name__ == "__main__":
    # Test with random agent
    test_random_agent(num_episodes=3, render=True)

    # Test specific actions
    test_specific_actions()
