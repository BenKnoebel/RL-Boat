"""
Test script to verify boat dynamics and visualization with a controlled square pattern.

This script makes the boat:
1. Move forward in a straight line for 5 seconds
2. Rotate 90 degrees to the left
3. Repeat this process 3 more times to complete a square

This demonstrates:
- Forward motion with both rudders
- Controlled rotation
- Visualization tracking over extended movement
"""

import sys
import os
import numpy as np
import time

# Add parent directory to path to import envs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.boat_env import BoatEnv


def test_square_pattern():
    """
    Test the boat by making it move in a square pattern.

    The boat will:
    - Move forward for 5 seconds (both rudders forward)
    - Rotate 90 degrees left (left backward, right forward)
    - Repeat 4 times total
    """
    print("=" * 70)
    print("RL-Boat Square Pattern Test")
    print("=" * 70)
    print("This test demonstrates controlled boat movement:")
    print("- The boat will move forward for 5 seconds")
    print("- Then rotate 90 degrees to the left")
    print("- This repeats 4 times to form a square pattern")
    print()
    print("Watch the visualization to see:")
    print("  - Red square (boat) moving and rotating")
    print("  - Dark red arrow showing orientation")
    print("  - Blue trajectory line forming a square")
    print("=" * 70)
    print()

    # Create environment with visualization and a distant goal so we don't reach it
    env = BoatEnv(
        goal_position=np.array([100.0, 100.0]),  # Far away goal
        goal_radius=2.0,
        max_steps=10000,  # High limit so we don't timeout
        bounds=100.0,  # Larger bounds for the square pattern
        render_mode='human'
    )

    # Reset environment
    state, info = env.reset()
    print(f"Starting Position: ({state[0]:.2f}, {state[1]:.2f})")
    print(f"Starting Angle: {np.degrees(state[2]):.1f}°")
    print()

    # Simulation parameters
    dt = env.dt  # Time step from environment (0.1 seconds)
    forward_duration = 5.0  # seconds
    forward_steps = int(forward_duration / dt)  # Number of steps for 5 seconds

    # Action definitions
    ACTION_BOTH_FORWARD = 5  # Both rudders forward
    ACTION_ROTATE_LEFT = 8   # Left backward, Right forward

    # Complete 4 sides of the square
    for side in range(4):
        print(f"--- Side {side + 1}/4 ---")

        # Record starting position and angle
        start_pos = state[:2].copy()
        start_angle = state[2]

        # Move forward for 5 seconds
        print(f"Moving forward for {forward_duration} seconds...")
        for step in range(forward_steps):
            state, reward, terminated, truncated, info = env.step(ACTION_BOTH_FORWARD)

            if terminated or truncated:
                print("Episode ended early!")
                break

        end_pos = state[:2].copy()
        distance_traveled = np.linalg.norm(end_pos - start_pos)
        print(f"  Traveled {distance_traveled:.2f} meters")
        print(f"  Current position: ({state[0]:.2f}, {state[1]:.2f})")
        print(f"  Current angle: {np.degrees(state[2]):.1f}°")

        # Rotate 90 degrees to the left (if not on the last side)
        if side < 3:
            print(f"Rotating 90° to the left...")
            target_angle = start_angle + np.pi/2  # 90 degrees in radians

            # Normalize target angle to [-pi, pi]
            target_angle = ((target_angle + np.pi) % (2 * np.pi)) - np.pi

            # Rotate until we've turned approximately 90 degrees
            rotation_steps = 0
            max_rotation_steps = 200  # Safety limit

            while rotation_steps < max_rotation_steps:
                state, reward, terminated, truncated, info = env.step(ACTION_ROTATE_LEFT)
                rotation_steps += 1

                # Calculate angle difference
                current_angle = state[2]
                angle_diff = abs(((current_angle - target_angle + np.pi) % (2 * np.pi)) - np.pi)

                # Stop when we're close enough (within 5 degrees)
                if angle_diff < np.radians(5):
                    break

                if terminated or truncated:
                    print("Episode ended early!")
                    break

            print(f"  Rotation complete after {rotation_steps} steps")
            print(f"  New angle: {np.degrees(state[2]):.1f}°")

        print()

    print("=" * 70)
    print("Square Pattern Complete!")
    print("=" * 70)
    print(f"Final Position: ({state[0]:.2f}, {state[1]:.2f})")
    print(f"Final Angle: {np.degrees(state[2]):.1f}°")
    print()
    print("The boat should have traced approximately a square pattern.")
    print("Check the visualization window to see the blue trajectory line.")
    print("=" * 70)

    # Keep window open for a moment
    print()
    print("Keeping visualization window open for 5 seconds...")
    time.sleep(5)

    env.close()


if __name__ == "__main__":
    test_square_pattern()
