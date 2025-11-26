"""
Rowing Boat Environment for Reinforcement Learning

This environment simulates a rowing boat with two independent rudders
navigating in a 2D plane towards a goal location.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from envs.renderer import BoatRenderer
from PIL import Image


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
        - Large positive reward (+100) when goal is reached
        - Penalty (-25) when boat is in a black zone (obstacle)
        - Episode terminates when goal is reached or max steps exceeded
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self,
                 goal_position=None,
                 mask_path=None,
                 width = 1,
                 length = 2.5,
                 goal_radius=1.0,
                 max_steps=500,
                 bounds=50.0,
                 boat_mass=120.0,
                 rudder_force=80.0,
                 lever_arm=0.015,
                 friction_coeff=0.2,
                 angular_drag_coeff=0.8,
                 dt=0.1,
                 render_mode=None):
        """
        Initialize the Boat Environment.

        Args:
            goal_position: Target position [x, y]. If None, random goal is set.
            mask_path: Path to obstacle mask image (black=obstacles, white=navigable). If None, no obstacles.
            width: Width of the boat (m)
            length: Length of the boat (m)
            goal_radius: Distance threshold to consider goal reached (m)
            max_steps: Maximum steps per episode
            bounds: Size of the environment (square from -bounds to +bounds) (m)
            boat_mass: Mass of the boat (kg)
            rudder_force: Force applied by each rudder when rowing (N)
            lever_arm: Distance from boat center to rudder (m)
            friction_coeff: Linear drag coefficient (N·s/m or kg/s)
            angular_drag_coeff: Angular drag coefficient (N·m·s or kg·m²/s)
            dt: Time step for simulation (seconds)
            render_mode: Mode for rendering ('human' for visualization, None for no rendering)
        """
        super().__init__()

        self.render_mode = render_mode

        # Environment parameters
        self.goal_radius = goal_radius
        self.max_steps = max_steps
        self.bounds = bounds
        self.goal_position = None
        self.fixed_goal = False

        # Validate and set goal position if provided
        if goal_position is not None:
            goal_pos = np.array(goal_position)
            # Check if within bounds
            if abs(goal_pos[0]) > self.bounds or abs(goal_pos[1]) > self.bounds:
                raise ValueError(f"Goal position {goal_position} is outside bounds [-{self.bounds}, {self.bounds}]")
            self.goal_position = goal_pos
            self.fixed_goal = True

        # Load obstacle mask if provided
        self.mask = None
        self.mask_scale_x = None
        self.mask_scale_y = None
        if mask_path is not None:
            mask_img = Image.open(mask_path).convert('L')  # Convert to grayscale
            self.mask = np.array(mask_img)
            # Calculate scale: map world coordinates [-bounds, bounds] to image pixels
            # mask.shape[0] is height (y), mask.shape[1] is width (x)
            self.mask_scale_x = self.mask.shape[1] / (2 * self.bounds)
            self.mask_scale_y = self.mask.shape[0] / (2 * self.bounds)

            # Validate that fixed goal is not in a black zone
            if self.fixed_goal and self._is_position_in_black_zone(self.goal_position[0], self.goal_position[1]):
                raise ValueError(f"Goal position {self.goal_position} is inside a black zone (obstacle)")

        # Boat physics parameters
        self.boat_mass = boat_mass
        self.rudder_force = rudder_force
        self.lever_arm = lever_arm
        self.friction_coeff = friction_coeff
        self.angular_drag_coeff = angular_drag_coeff
        self.dt = dt
        self.length = length
        self.width = width

        # Moment of inertia (approximated as a rectangular plate)
        # For a rectangular plate rotating about its center: I = (1/12) * m * (L² + W²)
        self.moment_of_inertia = (1.0/12.0) * boat_mass * (self.length**2 + self.width**2)

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

        # Random starting position near origin, ensuring it's not in a black zone
        max_attempts = 100
        for _ in range(max_attempts):
            start_x = self.np_random.uniform(-5, 5)
            start_y = self.np_random.uniform(-5, 5)
            # Check if position is valid (not in black zone)
            if not self._is_position_in_black_zone(start_x, start_y):
                break
        else:
            # If we couldn't find a valid position after max_attempts, use (0, 0)
            start_x, start_y = 0.0, 0.0

        start_angle = self.np_random.uniform(-np.pi, np.pi)

        # Initialize with zero velocities (linear and angular)
        self.state = np.array([start_x, start_y, start_angle, 0.0, 0.0, 0.0], dtype=np.float32)

        # Set random goal only if not manually specified
        if not self.fixed_goal:
            # Try to find a valid goal position (not in black zone)
            max_attempts = 100
            for _ in range(max_attempts):
                goal_x = self.np_random.uniform(-self.bounds * 0.8, self.bounds * 0.8)
                goal_y = self.np_random.uniform(-self.bounds * 0.8, self.bounds * 0.8)
                # Check if position is valid (not in black zone)
                if not self._is_position_in_black_zone(goal_x, goal_y):
                    self.goal_position = np.array([goal_x, goal_y])
                    break
            else:
                # If we couldn't find a valid position, use bounds edge
                self.goal_position = np.array([self.bounds * 0.7, self.bounds * 0.7])

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
        terminated = False

        # Check if goal is reached
        goal_reached = distance <= self.goal_radius
        if goal_reached:
            reward = 100.0  # Large positive reward for reaching goal
            terminated = True
        # Check if in black zone (obstacle)
        elif self._is_in_black_zone():
            reward = -25.0  # Penalty for hitting obstacle
            # No termination, no reset - just apply penalty
        # Check if out of bounds
        elif abs(self.state[0]) > self.bounds or abs(self.state[1]) > self.bounds:
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
        - Friction/Drag: Linear damping opposes both linear and angular motion

        Physics:
        - Angular: τ = I·α, with damping torque τ_drag = -c_angular·ω
        - Linear: F = m·a, with drag force F_drag = -c_linear·v
        - Integration: Euler method with time step dt

        Args:
            left_action: -1 (backward), 0 (idle), or 1 (forward)
            right_action: -1 (backward), 0 (idle), or 1 (forward)
        """
        x, y, angle, vx, vy, omega = self.state

        # ==================== ANGULAR DYNAMICS ====================
        # Calculate forces from rudders
        left_force = left_action * self.rudder_force
        right_force = right_action * self.rudder_force

        # Torque from differential forces (positive = counterclockwise)
        # Each rudder is at distance lever_arm from the center
        torque = (left_force - right_force) * self.lever_arm

        # Angular drag torque (opposes rotation, linear in omega)
        # This provides constant angular deceleration when idle
        drag_torque = -self.angular_drag_coeff * omega

        # Total torque and angular acceleration: τ_total = I * α
        total_torque = torque + drag_torque
        angular_acceleration = total_torque / self.moment_of_inertia

        # Update angular velocity: ω(t+dt) = ω(t) + α * dt
        omega += angular_acceleration * self.dt

        # Update angle: θ(t+dt) = θ(t) + ω * dt
        angle += omega * self.dt

        # Normalize angle to [-pi, pi]
        angle = ((angle + np.pi) % (2 * np.pi)) - np.pi

        # ==================== LINEAR DYNAMICS ====================
        # Net force in boat's forward direction (local frame)
        net_force = left_force + right_force

        # Linear acceleration in boat's local frame (forward direction)
        # Convert to global frame
        thrust_ax = (net_force / self.boat_mass) * np.cos(angle)
        thrust_ay = (net_force / self.boat_mass) * np.sin(angle)

        # Linear drag force (opposes motion, linear in velocity)
        # F_drag = -c * v  =>  a_drag = -c/m * v
        drag_ax = -self.friction_coeff * vx
        drag_ay = -self.friction_coeff * vy

        # Total acceleration
        ax_global = thrust_ax + drag_ax
        ay_global = thrust_ay + drag_ay

        # Update linear velocities: v(t+dt) = v(t) + a * dt
        vx += ax_global * self.dt
        vy += ay_global * self.dt

        # Update positions: x(t+dt) = x(t) + v * dt
        x += vx * self.dt
        y += vy * self.dt

        # Update state with all 6 components
        self.state = np.array([x, y, angle, vx, vy, omega], dtype=np.float32)

    def _distance_to_goal(self):
        """Calculate Euclidean distance to goal."""
        return np.linalg.norm(self.state[:2] - self.goal_position)

    def _is_position_in_black_zone(self, x, y):
        """
        Check if a given position is in a black zone (obstacle).

        Args:
            x, y: World coordinates to check

        Returns:
            bool: True if in black zone, False otherwise (or if no mask is loaded)
        """
        if self.mask is None:
            return False

        # Convert world coordinates to image pixel coordinates
        # World: bottom-left origin, [-bounds, bounds] for both x and y
        # Image: top-left origin, [0, width] x [0, height]
        pixel_x = int((x + self.bounds) * self.mask_scale_x)
        # Flip y-axis: world y increases upward, image y increases downward
        pixel_y = self.mask.shape[0] - 1 - int((y + self.bounds) * self.mask_scale_y)

        # Check if within mask bounds
        if (pixel_x < 0 or pixel_x >= self.mask.shape[1] or
            pixel_y < 0 or pixel_y >= self.mask.shape[0]):
            return False  # Outside mask is considered safe

        # Black pixels have low values (0), white pixels have high values (255)
        # We consider it a black zone if pixel value is below threshold
        return self.mask[pixel_y, pixel_x] < 128

    def _is_in_black_zone(self):
        """
        Check if the current boat position is in a black zone (obstacle).

        Returns:
            bool: True if in black zone, False otherwise (or if no mask is loaded)
        """
        return self._is_position_in_black_zone(self.state[0], self.state[1])

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
