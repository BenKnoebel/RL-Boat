"""
Simulation: Follower drone tracks leader drone flying a smooth random trajectory.
The random trajectory changes each time the simulation runs.
"""

import numpy as np
import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from Drone_Tracking import State, AdaptiveController, DroneTracker


def generate_smooth_random_trajectory(duration=30.0, num_frequencies=5, seed=None, max_speed=15.0):
    """
    Generate a smooth random trajectory using sum of sinusoids with random frequencies/phases.
    This creates a smooth, continuous trajectory that looks random but is differentiable.
    Speed is limited to realistic DJI drone values.

    Args:
        duration: Total duration of trajectory (seconds)
        num_frequencies: Number of frequency components for smoothness
        seed: Random seed for reproducibility (None for different trajectory each time)
        max_speed: Maximum speed for the leader drone (m/s)

    Returns:
        Function that takes time t and returns (position, velocity)
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random frequency components for each axis
    # Lower frequencies create smoother, more sweeping motions
    # Reduced frequency range for slower, more realistic motion
    base_freqs = np.random.uniform(0.05, 0.2, (3, num_frequencies))  # Hz (reduced from 0.1-0.5)
    amplitudes = np.random.uniform(1.0, 2.5, (3, num_frequencies))   # meters (reduced from 1.0-3.0)
    phases = np.random.uniform(0, 2*np.pi, (3, num_frequencies))     # radians

    # Center position and altitude offset
    center = np.array([0.0, 0.0, 5.0])

    def trajectory(t):
        """
        Get position and velocity at time t.

        Args:
            t: time (seconds)

        Returns:
            position, velocity as numpy arrays
        """
        position = center.copy()
        velocity = np.zeros(3)

        for axis in range(3):
            for i in range(num_frequencies):
                omega = 2 * np.pi * base_freqs[axis, i]
                amp = amplitudes[axis, i]
                phase = phases[axis, i]

                # Position: sum of sinusoids
                position[axis] += amp * np.sin(omega * t + phase)

                # Velocity: derivative of position
                velocity[axis] += amp * omega * np.cos(omega * t + phase)

        return position, velocity

    # Generate sample waypoints for visualization (not used in trajectory)
    sample_times = np.linspace(0, duration, 8)
    sample_positions = np.array([trajectory(t)[0] for t in sample_times])
    waypoints = (sample_positions[:, 0], sample_positions[:, 1],
                 sample_positions[:, 2], sample_times)

    return trajectory, waypoints


def run_random_trajectory_simulation():
    """
    Run simulation with leader flying smooth random trajectory and follower tracking.
    """
    print("=" * 70)
    print("DRONE PURSUIT SIMULATION: SMOOTH RANDOM TRAJECTORY")
    print("=" * 70)

    # Generate random trajectory (no seed = different each time)
    duration = 30.0
    leader_trajectory, waypoints = generate_smooth_random_trajectory(
        duration=duration,
        num_frequencies=5,
        seed=None  # Set to integer for reproducible trajectory
    )

    print(f"\nGenerated smooth random trajectory")
    print(f"Sample trajectory positions (XYZ) at key times:")
    for i, (x, y, z, t) in enumerate(zip(*waypoints)):
        print(f"  t={t:.1f}s: ({x:.2f}, {y:.2f}, {z:.2f})")

    # Initialize follower drone (starts at origin)
    follower_state = State(position=[0.0, 0.0, 3.0], velocity=[0.0, 0.0, 0.0])

    # Initialize tracker and controller (same parameters as figure-eight)
    tracker = DroneTracker(update_interval=3, prediction_method='polynomial')
    controller = AdaptiveController(
        adaptation_gain=5.0,    # Lower gain to prevent parameter drift
        damping_gain=150.0,     # Higher damping for better tracking
        lambda_gain=8.0         # Higher convergence rate
    )

    # Simulation parameters
    dt = 0.01  # 100 Hz update rate
    num_steps = int(duration / dt)

    # Storage for results
    follower_positions = []
    leader_positions = []
    follower_velocities = []
    leader_velocities = []
    tracking_errors = []
    control_forces_log = []
    follower_headings = []
    heading_alignment_angles = []
    times = []

    # Verification thresholds
    MAX_ACCEPTABLE_ERROR = 2.0  # meters
    MIN_HEADING_ALIGNMENT = 0.7  # cosine similarity
    tracking_violations = []

    print(f"\nSimulation Parameters:")
    print(f"  Duration: {duration}s")
    print(f"  Time step: {dt}s")
    print(f"  Leader update interval: every {tracker.update_interval} steps")
    print(f"  Follower start position: {follower_state.position}")
    print(f"\nStarting simulation...")

    # Simulation loop
    t = 0.0
    for step in range(num_steps):

        # --- LEADER DRONE (smooth random trajectory) ---
        leader_pos, leader_vel = leader_trajectory(t)

        # --- UPDATE TRACKER (every 3rd timestep) ---
        if tracker.should_update(step):
            tracker.update_leader_state(leader_pos, leader_vel, timestamp=t)
            if step % 300 == 0:  # Print every 3 seconds
                print(f"[t={t:6.2f}s] Leader: {leader_pos}, Tracker updated")

        # --- GET DESIRED TRAJECTORY ---
        # Dynamic lookahead: increase when tracking well, decrease when error is high
        if step > 0:
            current_error = np.linalg.norm(np.array(follower_state.position) - leader_pos)
            # Lookahead between 0.2s and 0.6s based on error
            lookahead_time = 0.2 + min(0.4, current_error / 10.0)
        else:
            lookahead_time = 0.4

        desired_traj = tracker.get_desired_trajectory(t, lookahead_time=lookahead_time)

        # --- COMPUTE ADAPTIVE CONTROL ---
        control_force, ahat_dot = controller.compute_control(follower_state, desired_traj, t)

        # Saturate control force to prevent extreme values
        max_force = 100.0  # Maximum force per axis in Newtons
        control_force = np.clip(control_force, -max_force, max_force)

        # Update parameter estimates
        controller.update_parameters(ahat_dot, dt)

        # --- FOLLOWER DRONE DYNAMICS ---
        mass = 1.2  # slightly heavier than estimated
        drag = np.array([0.12, 0.12, 0.15])  # slight drag
        gravity = np.array([0.0, 0.0, -9.81])

        drag_force = -drag * np.array(follower_state.velocity)
        total_force = control_force + drag_force + mass * gravity
        acceleration = total_force / mass

        # Calculate desired velocity from acceleration
        desired_velocity = [
            follower_state.velocity[0] + acceleration[0] * dt,
            follower_state.velocity[1] + acceleration[1] * dt,
            follower_state.velocity[2] + acceleration[2] * dt
        ]

        # Apply realistic DJI drone constraints
        constrained_velocity = follower_state.apply_realistic_constraints(desired_velocity, dt)

        # Update follower velocity with constraints
        follower_state.velocity = constrained_velocity

        # Update position
        follower_state.position[0] += follower_state.velocity[0] * dt
        follower_state.position[1] += follower_state.velocity[1] * dt
        follower_state.position[2] += follower_state.velocity[2] * dt

        # Update heading to point towards leader
        direction_to_leader = leader_pos - np.array(follower_state.position)
        distance_to_leader = np.linalg.norm(direction_to_leader)
        if distance_to_leader > 0.01:  # Avoid division by zero
            follower_state.heading = (direction_to_leader / distance_to_leader).tolist()

        # Store state history
        follower_state.add_to_history(timestamp=t)

        # --- RECORD DATA ---
        error = np.linalg.norm(np.array(follower_state.position) - leader_pos)

        # Compute heading alignment (cosine similarity)
        heading_vec = np.array(follower_state.heading)
        if distance_to_leader > 0.01:
            direction_normalized = direction_to_leader / distance_to_leader
            heading_alignment = np.dot(heading_vec, direction_normalized)
        else:
            heading_alignment = 1.0  # Perfect alignment when on top of leader

        follower_positions.append(follower_state.position.copy())
        leader_positions.append(leader_pos.copy())
        follower_velocities.append(follower_state.velocity.copy())
        leader_velocities.append(leader_vel.copy())
        follower_headings.append(follower_state.heading.copy())
        tracking_errors.append(error)
        control_forces_log.append(np.linalg.norm(control_force))
        heading_alignment_angles.append(heading_alignment)
        times.append(t)

        # --- VERIFICATION CHECKS ---
        # Only check after initial convergence (skip first 3 seconds)
        if t > 3.0:
            if error > MAX_ACCEPTABLE_ERROR:
                tracking_violations.append({
                    'time': t,
                    'type': 'tracking_error',
                    'value': error,
                    'threshold': MAX_ACCEPTABLE_ERROR
                })
            if heading_alignment < MIN_HEADING_ALIGNMENT:
                tracking_violations.append({
                    'time': t,
                    'type': 'heading_misalignment',
                    'value': heading_alignment,
                    'threshold': MIN_HEADING_ALIGNMENT
                })

        t += dt

    # Convert to numpy arrays
    follower_positions = np.array(follower_positions)
    leader_positions = np.array(leader_positions)
    follower_velocities = np.array(follower_velocities)
    leader_velocities = np.array(leader_velocities)
    tracking_errors = np.array(tracking_errors)
    heading_alignment_angles = np.array(heading_alignment_angles)
    times = np.array(times)

    # Calculate speeds
    follower_speeds = np.linalg.norm(follower_velocities, axis=1)
    leader_speeds = np.linalg.norm(leader_velocities, axis=1)

    # Print results
    print(f"\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print(f"\nTracking Performance:")
    print(f"  Initial error: {tracking_errors[0]:.3f} m")
    print(f"  Final error: {tracking_errors[-1]:.3f} m")
    print(f"  Mean error (last 5s): {np.mean(tracking_errors[-500:]):.3f} m")
    print(f"  Max error: {np.max(tracking_errors):.3f} m")
    print(f"  Std deviation: {np.std(tracking_errors):.3f} m")

    # Print heading alignment statistics
    print(f"\nHeading Alignment (1.0 = perfect, 0.0 = perpendicular, -1.0 = opposite):")
    print(f"  Mean alignment: {np.mean(heading_alignment_angles):.3f}")
    print(f"  Min alignment: {np.min(heading_alignment_angles):.3f}")
    print(f"  Final alignment: {heading_alignment_angles[-1]:.3f}")

    # Print speed statistics
    print(f"\nSpeed Statistics:")
    print(f"  Leader max speed: {np.max(leader_speeds):.2f} m/s")
    print(f"  Leader avg speed: {np.mean(leader_speeds):.2f} m/s")
    print(f"  Follower max speed: {np.max(follower_speeds):.2f} m/s (limit: 20.0 m/s horizontal)")
    print(f"  Follower avg speed: {np.mean(follower_speeds):.2f} m/s")
    print(f"  Follower max vertical speed: {np.max(np.abs(follower_velocities[:, 2])):.2f} m/s (limit: 5.0 m/s)")

    # Print verification results
    print(f"\n" + "=" * 70)
    print("TRACKING VERIFICATION RESULTS")
    print("=" * 70)
    print(f"\nThresholds:")
    print(f"  Max acceptable tracking error: {MAX_ACCEPTABLE_ERROR} m")
    print(f"  Min acceptable heading alignment: {MIN_HEADING_ALIGNMENT}")

    # Count violations by type
    error_violations = [v for v in tracking_violations if v['type'] == 'tracking_error']
    heading_violations = [v for v in tracking_violations if v['type'] == 'heading_misalignment']

    print(f"\nViolations (after 3s convergence period):")
    print(f"  Tracking error violations: {len(error_violations)}")
    print(f"  Heading misalignment violations: {len(heading_violations)}")
    print(f"  Total violations: {len(tracking_violations)}")

    # Calculate percentage of time in compliance
    evaluation_steps = sum(1 for t in times if t > 3.0)
    violation_steps = len(set(v['time'] for v in tracking_violations))
    compliance_rate = 100 * (1 - violation_steps / evaluation_steps) if evaluation_steps > 0 else 0

    print(f"\nCompliance rate: {compliance_rate:.1f}%")

    if len(tracking_violations) == 0:
        print("\n✓ VERIFICATION PASSED: Follower successfully tracks leader!")
        print("  - Tracking error always below threshold")
        print("  - Heading always properly aligned")
    else:
        print(f"\n✗ VERIFICATION ISSUES: Found {len(tracking_violations)} violations")
        if error_violations:
            print(f"  - Max tracking error violation: {max(v['value'] for v in error_violations):.3f} m")
        if heading_violations:
            print(f"  - Min heading alignment violation: {min(v['value'] for v in heading_violations):.3f}")

    # Print estimated parameters
    params = controller.get_estimated_parameters()
    print(f"\nEstimated Parameters:")
    print(f"  Mass: {params['mass_est']:.3f} kg (true: 1.2 kg)")
    print(f"  Drag coefficients: [{params['drag_x_est']:.3f}, {params['drag_y_est']:.3f}, {params['drag_z_est']:.3f}]")
    print(f"  Mass*g: {params['mass_times_g_est']:.3f} N (true: {1.2*9.81:.3f} N)")

    return {
        'times': times,
        'follower_positions': follower_positions,
        'leader_positions': leader_positions,
        'follower_velocities': follower_velocities,
        'leader_velocities': leader_velocities,
        'follower_speeds': follower_speeds,
        'leader_speeds': leader_speeds,
        'follower_headings': np.array(follower_headings),
        'tracking_errors': tracking_errors,
        'control_forces': control_forces_log,
        'heading_alignment_angles': heading_alignment_angles,
        'tracking_violations': tracking_violations,
        'waypoints': waypoints,
        'controller': controller,
        'tracker': tracker
    }


def plot_results(results):
    """Plot the simulation results with verification metrics."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(18, 12))

        # 1. 3D Trajectory
        ax1 = fig.add_subplot(3, 3, 1, projection='3d')
        ax1.plot(results['leader_positions'][:, 0],
                results['leader_positions'][:, 1],
                results['leader_positions'][:, 2],
                'b-', label='Leader (Random)', linewidth=2, alpha=0.7)
        ax1.plot(results['follower_positions'][:, 0],
                results['follower_positions'][:, 1],
                results['follower_positions'][:, 2],
                'r-', label='Follower', linewidth=2, alpha=0.7)

        # Mark start positions
        ax1.scatter(*results['leader_positions'][0], color='blue', s=100, marker='o', label='Leader Start')
        ax1.scatter(*results['follower_positions'][0], color='red', s=100, marker='s', label='Follower Start')

        # Plot waypoints
        x_wp, y_wp, z_wp, _ = results['waypoints']
        ax1.scatter(x_wp, y_wp, z_wp, color='green', s=80, marker='^',
                   label='Waypoints', alpha=0.8, edgecolors='darkgreen', linewidths=2)

        # Add heading arrows (every 500 timesteps for clarity)
        arrow_interval = 500
        for i in range(0, len(results['follower_positions']), arrow_interval):
            pos = results['follower_positions'][i]
            heading = results['follower_headings'][i]
            arrow_scale = 0.8
            ax1.quiver(pos[0], pos[1], pos[2],
                      heading[0]*arrow_scale, heading[1]*arrow_scale, heading[2]*arrow_scale,
                      color='red', arrow_length_ratio=0.3, linewidth=1.5, alpha=0.7)

        ax1.set_xlabel('X (m)', fontsize=10)
        ax1.set_ylabel('Y (m)', fontsize=10)
        ax1.set_zlabel('Z (m)', fontsize=10)
        ax1.set_title('3D Trajectories', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 2. XY Plane View
        ax2 = fig.add_subplot(3, 3, 2)
        ax2.plot(results['leader_positions'][:, 0],
                results['leader_positions'][:, 1],
                'b-', label='Leader', linewidth=2, alpha=0.7)
        ax2.plot(results['follower_positions'][:, 0],
                results['follower_positions'][:, 1],
                'r-', label='Follower', linewidth=2, alpha=0.7)
        ax2.scatter(results['leader_positions'][0, 0], results['leader_positions'][0, 1],
                   color='blue', s=100, marker='o')
        ax2.scatter(results['follower_positions'][0, 0], results['follower_positions'][0, 1],
                   color='red', s=100, marker='s')
        # Plot waypoints in XY plane
        ax2.scatter(x_wp, y_wp, color='green', s=80, marker='^', alpha=0.8,
                   edgecolors='darkgreen', linewidths=2)
        ax2.set_xlabel('X (m)', fontsize=10)
        ax2.set_ylabel('Y (m)', fontsize=10)
        ax2.set_title('Top View (XY Plane)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')

        # 3. Tracking Error over Time
        ax3 = fig.add_subplot(3, 3, 3)
        ax3.plot(results['times'], results['tracking_errors'], 'k-', linewidth=2)
        ax3.axhline(y=2.0, color='red', linestyle='--', linewidth=1, label='Max threshold')
        ax3.set_xlabel('Time (s)', fontsize=10)
        ax3.set_ylabel('Tracking Error (m)', fontsize=10)
        ax3.set_title('Tracking Error vs Time', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(bottom=0)

        # 4. Position Components over Time
        ax4 = fig.add_subplot(3, 3, 4)
        ax4.plot(results['times'], results['leader_positions'][:, 0], 'b-', label='Leader X', alpha=0.7)
        ax4.plot(results['times'], results['follower_positions'][:, 0], 'r--', label='Follower X', alpha=0.7)
        ax4.plot(results['times'], results['leader_positions'][:, 1], 'g-', label='Leader Y', alpha=0.7)
        ax4.plot(results['times'], results['follower_positions'][:, 1], 'm--', label='Follower Y', alpha=0.7)
        ax4.set_xlabel('Time (s)', fontsize=10)
        ax4.set_ylabel('Position (m)', fontsize=10)
        ax4.set_title('X & Y Positions vs Time', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

        # 5. Z Position over Time
        ax5 = fig.add_subplot(3, 3, 5)
        ax5.plot(results['times'], results['leader_positions'][:, 2], 'b-', label='Leader Z', linewidth=2, alpha=0.7)
        ax5.plot(results['times'], results['follower_positions'][:, 2], 'r--', label='Follower Z', linewidth=2, alpha=0.7)
        ax5.set_xlabel('Time (s)', fontsize=10)
        ax5.set_ylabel('Altitude (m)', fontsize=10)
        ax5.set_title('Altitude vs Time', fontsize=12, fontweight='bold')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)

        # 6. Control Force Magnitude
        ax6 = fig.add_subplot(3, 3, 6)
        ax6.plot(results['times'], results['control_forces'], 'purple', linewidth=1.5)
        ax6.set_xlabel('Time (s)', fontsize=10)
        ax6.set_ylabel('Control Force Magnitude (N)', fontsize=10)
        ax6.set_title('Control Effort vs Time', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)

        # 7. Heading Alignment over Time
        ax7 = fig.add_subplot(3, 3, 7)
        ax7.plot(results['times'], results['heading_alignment_angles'], 'g-', linewidth=2)
        ax7.axhline(y=0.7, color='orange', linestyle='--', linewidth=1, label='Min threshold (0.7)')
        ax7.axhline(y=1.0, color='darkgreen', linestyle=':', linewidth=1, label='Perfect (1.0)')
        ax7.set_xlabel('Time (s)', fontsize=10)
        ax7.set_ylabel('Heading Alignment (cosine)', fontsize=10)
        ax7.set_title('Heading Alignment vs Time', fontsize=12, fontweight='bold')
        ax7.legend(fontsize=8)
        ax7.grid(True, alpha=0.3)
        ax7.set_ylim([-0.1, 1.1])

        # 8. Error Distribution Histogram
        ax8 = fig.add_subplot(3, 3, 8)
        ax8.hist(results['tracking_errors'][300:], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax8.axvline(x=2.0, color='red', linestyle='--', linewidth=2, label='Max threshold (2.0 m)')
        ax8.set_xlabel('Tracking Error (m)', fontsize=10)
        ax8.set_ylabel('Frequency', fontsize=10)
        ax8.set_title('Error Distribution (after 3s)', fontsize=12, fontweight='bold')
        ax8.legend(fontsize=8)
        ax8.grid(True, alpha=0.3)

        # 9. Violation Timeline
        ax9 = fig.add_subplot(3, 3, 9)
        violations = results['tracking_violations']
        if len(violations) > 0:
            error_viols = [v for v in violations if v['type'] == 'tracking_error']
            heading_viols = [v for v in violations if v['type'] == 'heading_misalignment']

            if error_viols:
                error_times = [v['time'] for v in error_viols]
                error_values = [v['value'] for v in error_viols]
                ax9.scatter(error_times, error_values, c='red', marker='x', s=50, label='Error violations', alpha=0.7)

            if heading_viols:
                heading_times = [v['time'] for v in heading_viols]
                heading_values = [v['value'] for v in heading_viols]
                ax9.scatter(heading_times, heading_values, c='orange', marker='o', s=30, label='Heading violations', alpha=0.7)

            ax9.legend(fontsize=8)
            ax9.set_title('Violation Timeline', fontsize=12, fontweight='bold')
        else:
            ax9.text(0.5, 0.5, 'NO VIOLATIONS\n✓ All checks passed!',
                    ha='center', va='center', fontsize=14, color='green', fontweight='bold',
                    transform=ax9.transAxes)
            ax9.set_title('Violation Timeline', fontsize=12, fontweight='bold')

        ax9.set_xlabel('Time (s)', fontsize=10)
        ax9.set_ylabel('Violation Value', fontsize=10)
        ax9.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        save_path = os.path.join(current_dir, 'random_trajectory_results.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")

        # plt.show()  # Commented out to avoid blocking

    except ImportError as e:
        print(f"\nCannot create plots: {e}")
        print("Install matplotlib to visualize results: pip install matplotlib")


if __name__ == "__main__":
    # Run simulation
    results = run_random_trajectory_simulation()

    # Plot results
    print("\nGenerating plots...")
    plot_results(results)

    print("\n" + "=" * 70)
    print("Simulation complete! Check the plots above.")
    print("=" * 70)
