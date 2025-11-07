"""
Example usage of the drone tracking system.
Shows how a follower drone uses adaptive control to track a leader drone.
"""

import numpy as np
import sys
sys.path.append('.')
from Drone_Tracking import State, AdaptiveController, DroneTracker


def example_drone_pursuit():
    """
    Example: Follower drone tracks leader drone with periodic updates.
    """

    # Initialize system components
    follower_state = State(position=[0.0, 0.0, 0.0], velocity=[0.0, 0.0, 0.0])
    tracker = DroneTracker(update_interval=3, prediction_method='polynomial')
    controller = AdaptiveController(adaptation_gain=200.0, damping_gain=200.0, lambda_gain=10.0)

    # Leader drone starts at a different position with some velocity
    leader_pos = np.array([5.0, 5.0, 5.0])
    leader_vel = np.array([1.0, 0.5, 0.2])

    # Simulation parameters
    dt = 0.01  # time step
    duration = 10.0  # seconds
    num_steps = int(duration / dt)

    # Storage for plotting
    follower_positions = []
    leader_positions = []
    times = []

    # Simulation loop
    t = 0.0
    for step in range(num_steps):

        # --- LEADER DRONE (simulate its motion) ---
        # Leader follows a circular path (you can change this to any pattern)
        omega = 0.3
        radius = 5.0
        leader_pos = np.array([
            radius * np.cos(omega * t) + 10,
            radius * np.sin(omega * t) + 10,
            5.0 + 0.5 * np.sin(0.5 * t)
        ])
        leader_vel = np.array([
            -radius * omega * np.sin(omega * t),
            radius * omega * np.cos(omega * t),
            0.25 * np.cos(0.5 * t)
        ])

        # --- UPDATE TRACKER (every 3rd timestep) ---
        if tracker.should_update(step):
            tracker.update_leader_state(leader_pos, leader_vel, timestamp=t)
            print(f"[t={t:.2f}s] Updated leader state: pos={leader_pos}, vel={leader_vel}")

        # --- GET DESIRED TRAJECTORY ---
        desired_traj = tracker.get_desired_trajectory(t, lookahead_time=0.5)

        # --- COMPUTE ADAPTIVE CONTROL ---
        control_force, ahat_dot = controller.compute_control(follower_state, desired_traj, t)

        # Update parameter estimates
        controller.update_parameters(ahat_dot, dt)

        # --- FOLLOWER DRONE DYNAMICS ---
        # Simple dynamics: m * a = F - drag - gravity
        mass = 1.0
        drag = np.array([0.1, 0.1, 0.1])
        gravity = np.array([0.0, 0.0, -9.81])

        drag_force = -drag * np.array(follower_state.velocity)
        total_force = control_force + drag_force + mass * gravity
        acceleration = total_force / mass

        # Update follower state (Euler integration)
        follower_state.velocity[0] += acceleration[0] * dt
        follower_state.velocity[1] += acceleration[1] * dt
        follower_state.velocity[2] += acceleration[2] * dt

        follower_state.position[0] += follower_state.velocity[0] * dt
        follower_state.position[1] += follower_state.velocity[1] * dt
        follower_state.position[2] += follower_state.velocity[2] * dt

        # --- RECORD DATA ---
        follower_positions.append(follower_state.position.copy())
        leader_positions.append(leader_pos.copy())
        times.append(t)

        t += dt

    # Convert to numpy arrays
    follower_positions = np.array(follower_positions)
    leader_positions = np.array(leader_positions)
    times = np.array(times)

    # Print final tracking error
    final_error = np.linalg.norm(follower_positions[-1] - leader_positions[-1])
    print(f"\nFinal tracking error: {final_error:.3f} m")

    # Print estimated parameters
    params = controller.get_estimated_parameters()
    print(f"\nEstimated parameters:")
    print(f"  Mass: {params['mass_est']:.3f} kg (true: 1.0 kg)")
    print(f"  Gravity: {params['gravity_est']:.3f} m/s² (true: 9.81 m/s²)")

    return {
        'times': times,
        'follower_positions': follower_positions,
        'leader_positions': leader_positions,
        'controller': controller,
        'tracker': tracker
    }


if __name__ == "__main__":
    print("Running drone pursuit example...")
    print("=" * 60)
    results = example_drone_pursuit()

    # Optional: Plot results if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(12, 5))

        # 3D trajectory plot
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(results['leader_positions'][:, 0],
                results['leader_positions'][:, 1],
                results['leader_positions'][:, 2],
                'b-', label='Leader', linewidth=2)
        ax1.plot(results['follower_positions'][:, 0],
                results['follower_positions'][:, 1],
                results['follower_positions'][:, 2],
                'r--', label='Follower', linewidth=2)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Trajectories')
        ax1.legend()

        # Tracking error over time
        ax2 = fig.add_subplot(122)
        errors = np.linalg.norm(results['follower_positions'] - results['leader_positions'], axis=1)
        ax2.plot(results['times'], errors, 'k-', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Tracking Error (m)')
        ax2.set_title('Tracking Error vs Time')
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("\nMatplotlib not available. Skipping plots.")
