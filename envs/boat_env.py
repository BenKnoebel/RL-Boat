"""
Rowing Boat Environment for Reinforcement Learning

This environment simulates a rowing boat with two independent rudders
navigating in a 2D plane towards a goal location.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from envs.renderer import BoatRenderer


class BoatEnv(gym.Env):
    """
    A rowing boat environment with two independent rudders.

    Action Space:
        Discrete(9): Each action corresponds to a combination of left and right rudder actions
        - 0: Both idle
        - 1: Left forward, Right idle
        - 2: Left backward, Right idle
        - 3: Left idle, Right forward
        - 4: Left idle, Right backward
        - 5: Both forward
        - 6: Both backward
        - 7: Left forward, Right backward (rotate right)
        - 8: Left backward, Right forward (rotate left)

    State Space:
        Box(6): [x, y, angle, velocity_x, velocity_y, angular_velocity]
        - x, y: Position in 2D plane
        - angle: Boat orientation (radians)
        - velocity_x, velocity_y: Linear velocities in global frame
        - angular_velocity: Angular velocity (radians/second)

    Rewards:
        - Constant negative reward (-1) at each timestep
        - Large positive reward when goal is reached
        - Episode terminates when goal is reached or max steps exceeded
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self,
                 goal_position=None,
                 goal_radius=1.0,
                 max_steps=500,
                 bounds=50.0,
                 boat_mass=10.0,
                 rudder_force=5.0,
                 lever_arm=1.0,
                 friction_coeff=0.1,
                 dt=0.1,
                 render_mode=None):
        """
        Initialize the Boat Environment.

        Args:
            goal_position: Target position [x, y]. If None, random goal is set.
            goal_radius: Distance threshold to consider goal reached
            max_steps: Maximum steps per episode
            bounds: Size of the environment (square from -bounds to +bounds)
            boat_mass: Mass of the boat (kg)
            rudder_force: Force applied by each rudder when rowing (N)
            lever_arm: Distance from boat center to rudder (m)
            friction_coeff: Linear friction coefficient
            dt: Time step for simulation (seconds)
            render_mode: Mode for rendering ('human' for visualization, None for no rendering)
        """
        super().__init__()

        self.render_mode = render_mode

        # Environment parameters
        self.goal_position = np.array(goal_position) if goal_position is not None else None
        self.goal_radius = goal_radius
        self.max_steps = max_steps
        self.bounds = bounds

        # Boat physics parameters
        self.boat_mass = boat_mass
        self.rudder_force = rudder_force
        self.lever_arm = lever_arm
        self.friction_coeff = friction_coeff
        self.dt = dt

        # Moment of inertia (approximated as a rectangular boat)
        self.moment_of_inertia = boat_mass * (lever_arm ** 2)

        # Action space: 9 discrete actions (3^2 combinations)
        self.action_space = spaces.Discrete(9)

        # State space: [x, y, angle, velocity_x, velocity_y, angular_velocity]
        # Using large but bounded continuous space
        self.observation_space = spaces.Box(
            low=np.array([-bounds, -bounds, -np.pi, -10.0, -10.0, -2*np.pi]),
            high=np.array([bounds, bounds, np.pi, 10.0, 10.0, 2*np.pi]),
            dtype=np.float32
        )

        # Action mapping: (left_rudder_action, right_rudder_action)
        # -1: backward, 0: idle, 1: forward
        self.action_map = {
            0: (0, 0),    # Both idle
            1: (1, 0),    # Left forward, Right idle
            2: (-1, 0),   # Left backward, Right idle
            3: (0, 1),    # Left idle, Right forward
            4: (0, -1),   # Left idle, Right backward
            5: (1, 1),    # Both forward
            6: (-1, -1),  # Both backward
            7: (1, -1),   # Left forward, Right backward (rotate right)
            8: (-1, 1),   # Left backward, Right forward (rotate left)
        }

        # State variables
        self.state = None
        self.steps = 0

        # Renderer for visualization
        self.renderer = None
        if self.render_mode == 'human':
            self.renderer = BoatRenderer(bounds=self.bounds)

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.

        Returns:
            observation: Initial state
            info: Additional information
        """
        super().reset(seed=seed)

        # Random starting position near origin
        start_x = self.np_random.uniform(-5, 5)
        start_y = self.np_random.uniform(-5, 5)
        start_angle = self.np_random.uniform(-np.pi, np.pi)

        # Initialize with zero velocities (linear and angular)
        self.state = np.array([start_x, start_y, start_angle, 0.0, 0.0, 0.0], dtype=np.float32)

        # Set random goal if not specified
        if self.goal_position is None:
            goal_x = self.np_random.uniform(-self.bounds * 0.8, self.bounds * 0.8)
            goal_y = self.np_random.uniform(-self.bounds * 0.8, self.bounds * 0.8)
            self.goal_position = np.array([goal_x, goal_y])

        self.steps = 0

        # Reset renderer if visualization is enabled
        if self.renderer is not None:
            self.renderer.reset()

        info = {
            'goal_position': self.goal_position.copy(),
            'distance_to_goal': self._distance_to_goal()
        }

        return self.state.copy(), info

    def step(self, action):
        """
        Execute one time step within the environment.

        Args:
            action: Integer from 0 to 8 representing the chosen action

        Returns:
            observation: New state
            reward: Reward for this step
            terminated: Whether episode has ended
            truncated: Whether episode was truncated (max steps)
            info: Additional information
        """
        assert self.action_space.contains(action), f"Invalid action {action}"

        # Get rudder actions
        left_action, right_action = self.action_map[action]

        # Apply physics simulation
        self._apply_dynamics(left_action, right_action)

        # Update step counter
        self.steps += 1

        # Calculate reward
        distance = self._distance_to_goal()
        reward = -1.0  # Constant negative reward per step

        # Check if goal is reached
        terminated = distance <= self.goal_radius
        if terminated:
            reward = 100.0  # Large positive reward for reaching goal

        # Check if out of bounds
        if abs(self.state[0]) > self.bounds or abs(self.state[1]) > self.bounds:
            terminated = True
            reward = -10.0  # Penalty for going out of bounds

        # Check if max steps exceeded
        truncated = self.steps >= self.max_steps

        info = {
            'distance_to_goal': distance,
            'steps': self.steps,
            'goal_position': self.goal_position.copy()
        }

        # Render if visualization is enabled
        if self.render_mode == 'human':
            self.render()

        return self.state.copy(), reward, terminated, truncated, info

    def _apply_dynamics(self, left_action, right_action):
        """
        Apply boat dynamics based on rudder actions.

        Uses proper physics integration:
        - Angular dynamics: torque from differential rudder forces
        - Linear dynamics: thrust from rudder forces in boat's forward direction
        - Friction: opposes both linear and angular motion

        Args:
            left_action: -1 (backward), 0 (idle), or 1 (forward)
            right_action: -1 (backward), 0 (idle), or 1 (forward)
        """
        x, y, angle, vx, vy, omega = self.state

        # Calculate forces from rudders
        left_force = left_action * self.rudder_force
        right_force = right_action * self.rudder_force

        # Net force in boat's forward direction
        net_force = left_force + right_force

        # Torque from differential forces (positive = counterclockwise)
        # Each rudder is at distance lever_arm from the center
        torque = (left_force - right_force) * self.lever_arm

        # Angular dynamics: τ = I * α
        angular_acceleration = torque / self.moment_of_inertia

        # Apply angular friction (proportional to angular velocity)
        angular_friction = -self.friction_coeff * omega
        angular_acceleration += angular_friction / self.moment_of_inertia

        # Update angular velocity: ω(t+dt) = ω(t) + α * dt
        omega += angular_acceleration * self.dt

        # Update angle: θ(t+dt) = θ(t) + ω * dt
        angle += omega * self.dt

        # Normalize angle to [-pi, pi]
        angle = ((angle + np.pi) % (2 * np.pi)) - np.pi

        # Linear acceleration in boat's local frame (forward direction)
        forward_acceleration = net_force / self.boat_mass

        # Convert to global frame
        ax_global = forward_acceleration * np.cos(angle)
        ay_global = forward_acceleration * np.sin(angle)

        # Apply linear friction (opposes motion)
        ax_global -= self.friction_coeff * vx
        ay_global -= self.friction_coeff * vy

        # Update linear velocities
        vx += ax_global * self.dt
        vy += ay_global * self.dt

        # Update positions
        x += vx * self.dt
        y += vy * self.dt

        # Update state with all 6 components
        self.state = np.array([x, y, angle, vx, vy, omega], dtype=np.float32)

    def _distance_to_goal(self):
        """Calculate Euclidean distance to goal."""
        return np.linalg.norm(self.state[:2] - self.goal_position)

    def render(self):
        """
        Render the environment using matplotlib visualization.
        """
        if self.state is None:
            return None

        if self.render_mode == 'human' and self.renderer is not None:
            distance = self._distance_to_goal()
            self.renderer.render(
                self.state,
                self.goal_position,
                self.goal_radius,
                distance,
                self.steps
            )
        elif self.render_mode is None:
            # Basic console rendering for backward compatibility
            distance = self._distance_to_goal()
            print(f"Position: ({self.state[0]:.2f}, {self.state[1]:.2f}), "
                  f"Angle: {np.degrees(self.state[2]):.1f}°, "
                  f"Angular velocity: {np.degrees(self.state[5]):.2f}°/s, "
                  f"Distance to goal: {distance:.2f}")

        return None

    def close(self):
        """Clean up resources."""
        if self.renderer is not None:
            self.renderer.close()
