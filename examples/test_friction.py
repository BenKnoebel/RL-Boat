"""
Test script to verify friction and drag forces are working correctly.

This script tests:
1. Angular deceleration when idle (should be constant)
2. Linear deceleration when idle (should be constant)
3. Physics consistency
"""

import sys
import os
import numpy as np

# Add parent directory to path to import envs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.boat_env import BoatEnv


def test_angular_friction():
    """
    Test that angular velocity decreases at a constant rate when idle.
    """
    print("=" * 70)
    print("Testing Angular Friction (Linear Damping)")
    print("=" * 70)
    print()

    # Create environment
    env = BoatEnv(
        goal_position=np.array([100.0, 100.0]),
        render_mode=None
    )

    # Reset and build up angular velocity
    state, _ = env.reset()
    print("Building up angular velocity...")
    for _ in range(30):
        state, _, _, _, _ = env.step(8)  # Rotate left

    initial_omega = state[5]
    print(f"Initial angular velocity: {np.degrees(initial_omega):.2f}°/s")
    print()

    # Now idle and measure deceleration
    print("Testing angular deceleration when idle (no rudder input):")
    print("-" * 70)
    print(f"{'Step':>6} | {'ω (°/s)':>12} | {'Δω (°/s)':>12} | {'Angular Accel (°/s²)':>20}")
    print("-" * 70)

    prev_omega = initial_omega
    angular_accels = []

    for step in range(20):
        state, _, _, _, _ = env.step(0)  # Idle
        omega = state[5]
        delta_omega = omega - prev_omega
        angular_accel = delta_omega / env.dt  # α = Δω / Δt

        angular_accels.append(angular_accel)

        print(f"{step:6d} | {np.degrees(omega):12.4f} | {np.degrees(delta_omega):12.4f} | {np.degrees(angular_accel):20.4f}")

        prev_omega = omega

        # Stop when angular velocity is near zero
        if abs(omega) < 0.001:
            print(f"\nAngular velocity reached near-zero at step {step}")
            break

    print("-" * 70)
    print()

    # Check if angular acceleration is approximately constant
    angular_accels = np.array(angular_accels[:-1])  # Exclude last one (might be at zero)
    mean_accel = np.mean(angular_accels)
    std_accel = np.std(angular_accels)

    print(f"Mean angular acceleration: {np.degrees(mean_accel):.4f}°/s²")
    print(f"Std dev of angular acceleration: {np.degrees(std_accel):.4f}°/s²")
    print(f"Coefficient of variation: {(std_accel/abs(mean_accel)*100):.2f}%")
    print()

    if std_accel / abs(mean_accel) < 0.05:  # Less than 5% variation
        print("✓ PASS: Angular deceleration is approximately constant (linear damping)")
    else:
        print("✗ FAIL: Angular deceleration is not constant")

    print()
    env.close()


def test_linear_friction():
    """
    Test that linear velocity decreases at a constant rate when idle.
    """
    print("=" * 70)
    print("Testing Linear Friction (Linear Drag)")
    print("=" * 70)
    print()

    # Create environment
    env = BoatEnv(
        goal_position=np.array([100.0, 100.0]),
        render_mode=None
    )

    # Reset and build up linear velocity
    state, _ = env.reset()
    print("Building up linear velocity...")
    for _ in range(30):
        state, _, _, _, _ = env.step(5)  # Both forward

    initial_speed = np.sqrt(state[3]**2 + state[4]**2)
    print(f"Initial linear speed: {initial_speed:.4f} m/s")
    print()

    # Now idle and measure deceleration
    print("Testing linear deceleration when idle (no rudder input):")
    print("-" * 70)
    print(f"{'Step':>6} | {'Speed (m/s)':>12} | {'ΔSpeed (m/s)':>14} | {'Accel (m/s²)':>16}")
    print("-" * 70)

    prev_speed = initial_speed
    linear_accels = []

    for step in range(20):
        state, _, _, _, _ = env.step(0)  # Idle
        speed = np.sqrt(state[3]**2 + state[4]**2)
        delta_speed = speed - prev_speed
        linear_accel = delta_speed / env.dt  # a = Δv / Δt

        linear_accels.append(linear_accel)

        print(f"{step:6d} | {speed:12.4f} | {delta_speed:14.4f} | {linear_accel:16.4f}")

        prev_speed = speed

        # Stop when speed is near zero
        if speed < 0.001:
            print(f"\nLinear speed reached near-zero at step {step}")
            break

    print("-" * 70)
    print()

    # Check if linear acceleration is approximately constant
    linear_accels = np.array(linear_accels[:-1])  # Exclude last one (might be at zero)
    mean_accel = np.mean(linear_accels)
    std_accel = np.std(linear_accels)

    print(f"Mean linear acceleration: {mean_accel:.4f} m/s²")
    print(f"Std dev of linear acceleration: {std_accel:.4f} m/s²")
    print(f"Coefficient of variation: {(std_accel/abs(mean_accel)*100):.2f}%")
    print()

    if std_accel / abs(mean_accel) < 0.05:  # Less than 5% variation
        print("✓ PASS: Linear deceleration is approximately constant (linear drag)")
    else:
        print("✗ FAIL: Linear deceleration is not constant")

    print()
    env.close()


def test_physics_parameters():
    """
    Display physics parameters for verification.
    """
    print("=" * 70)
    print("Physics Parameters")
    print("=" * 70)
    print()

    env = BoatEnv()

    print(f"Boat dimensions:")
    print(f"  Length: {env.length} m")
    print(f"  Width: {env.width} m")
    print(f"  Mass: {env.boat_mass} kg")
    print()

    print(f"Moment of inertia:")
    print(f"  I = {env.moment_of_inertia:.4f} kg·m²")
    print()

    print(f"Forces and torques:")
    print(f"  Rudder force: {env.rudder_force} N")
    print(f"  Lever arm: {env.lever_arm} m")
    print(f"  Max torque (one rudder): {2 * env.rudder_force * env.lever_arm:.4f} N·m")
    print()

    print(f"Damping coefficients:")
    print(f"  Linear drag: {env.friction_coeff} N·s/m (or kg/s)")
    print(f"  Angular drag: {env.angular_drag_coeff} N·m·s (or kg·m²/s)")
    print()

    print(f"Characteristic values:")
    max_angular_accel = (2 * env.rudder_force * env.lever_arm) / env.moment_of_inertia
    print(f"  Max angular acceleration: {np.degrees(max_angular_accel):.2f}°/s² (both rudders opposing)")
    max_linear_accel = (2 * env.rudder_force) / env.boat_mass
    print(f"  Max linear acceleration: {max_linear_accel:.2f} m/s² (both rudders forward)")
    print()

    angular_damping_time = env.moment_of_inertia / env.angular_drag_coeff
    linear_damping_time = env.boat_mass / env.friction_coeff
    print(f"Damping time constants:")
    print(f"  Angular: {angular_damping_time:.2f} s")
    print(f"  Linear: {linear_damping_time:.2f} s")
    print()

    env.close()


if __name__ == "__main__":
    # Display physics parameters
    test_physics_parameters()

    # Test angular friction
    test_angular_friction()

    # Test linear friction
    test_linear_friction()

    print("=" * 70)
    print("All tests completed!")
    print("=" * 70)
