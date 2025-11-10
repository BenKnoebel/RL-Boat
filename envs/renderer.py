"""
Visualization renderer for the BoatEnv environment.

This module provides a clean separation of rendering logic from the
environment dynamics, following best practices for code organization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrow
from matplotlib.animation import FuncAnimation


class BoatRenderer:
    """
    Handles visualization of the boat environment using matplotlib.

    This class manages a matplotlib figure and updates it to show the
    boat's position, orientation, goal location, and trajectory.
    """

    def __init__(self, bounds=50.0, figsize=(8, 8)):
        """
        Initialize the renderer.

        Args:
            bounds: Size of the environment (square from -bounds to +bounds)
            figsize: Size of the matplotlib figure
        """
        self.bounds = bounds
        self.figsize = figsize

        # Initialize matplotlib components
        self.fig = None
        self.ax = None
        self.boat_patch = None
        self.boat_arrow = None
        self.goal_patch = None
        self.trajectory_line = None
        self.distance_text = None
        self.step_text = None
        self.velocity_text = None

        # Trajectory tracking
        self.trajectory_x = []
        self.trajectory_y = []

        # Flag to check if renderer is initialized
        self.initialized = False

    def initialize(self, goal_position, goal_radius):
        """
        Initialize the matplotlib figure and axes.

        Args:
            goal_position: [x, y] position of the goal
            goal_radius: Radius of the goal circle
        """
        if self.initialized:
            return

        # Create figure and axis
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.ax.set_xlim(-self.bounds, self.bounds)
        self.ax.set_ylim(-self.bounds, self.bounds)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X Position (m)', fontsize=10)
        self.ax.set_ylabel('Y Position (m)', fontsize=10)
        self.ax.set_title('RL-Boat Environment', fontsize=12, fontweight='bold')

        # Create goal marker (green circle)
        self.goal_patch = Circle(
            goal_position,
            goal_radius,
            color='green',
            alpha=0.5,
            label='Goal'
        )
        self.ax.add_patch(self.goal_patch)

        # Add goal center point
        self.ax.plot(goal_position[0], goal_position[1], 'g*',
                    markersize=15, label='Goal Center')

        # Create boat marker (red square)
        # We'll update its position in render()
        boat_size = 1.5
        self.boat_patch = Rectangle(
            (0, 0),  # Will be updated
            boat_size,
            boat_size,
            color='red',
            alpha=0.8,
            label='Boat'
        )
        self.ax.add_patch(self.boat_patch)

        # Create direction arrow to show boat orientation
        self.boat_arrow = None  # Will be created in render()

        # Create trajectory line (starts empty)
        self.trajectory_line, = self.ax.plot(
            [], [],
            'b-',
            alpha=0.3,
            linewidth=1,
            label='Trajectory'
        )

        # Add text displays for distance and step count
        self.distance_text = self.ax.text(
            0.02, 0.98,
            '',
            transform=self.ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9
        )

        self.step_text = self.ax.text(
            0.02, 0.90,
            '',
            transform=self.ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
            fontsize=9
        )

        self.velocity_text = self.ax.text(
            0.02, 0.82,
            '',
            transform=self.ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
            fontsize=9
        )

        # Add legend
        self.ax.legend(loc='upper right', fontsize=8)

        self.initialized = True

        # Initial draw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def render(self, state, goal_position, goal_radius, distance, step):
        """
        Update the visualization with the current state.

        Args:
            state: Current state [x, y, angle, vx, vy, omega]
            goal_position: Position of the goal
            goal_radius: Radius of the goal
            distance: Current distance to goal
            step: Current step number
        """
        if not self.initialized:
            self.initialize(goal_position, goal_radius)

        x, y, angle, vx, vy, omega = state

        # Update boat position (center the square on the boat position)
        boat_size = 1.5
        self.boat_patch.set_xy((x - boat_size/2, y - boat_size/2))

        # Remove old arrow if it exists
        if self.boat_arrow is not None:
            self.boat_arrow.remove()

        # Create new direction arrow
        arrow_length = 2.5
        dx = arrow_length * np.cos(angle)
        dy = arrow_length * np.sin(angle)

        self.boat_arrow = FancyArrow(
            x, y, dx, dy,
            width=0.5,
            head_width=1.2,
            head_length=0.8,
            color='darkred',
            alpha=0.9,
            zorder=10
        )
        self.ax.add_patch(self.boat_arrow)

        # Update trajectory
        self.trajectory_x.append(x)
        self.trajectory_y.append(y)
        self.trajectory_line.set_data(self.trajectory_x, self.trajectory_y)

        # Update text displays
        self.distance_text.set_text(f'Distance to Goal: {distance:.2f} m')
        self.step_text.set_text(f'Step: {step}')

        # Calculate speed for display
        linear_speed = np.sqrt(vx**2 + vy**2)
        self.velocity_text.set_text(
            f'Speed: {linear_speed:.2f} m/s\n'
            f'ω: {np.degrees(omega):.1f}°/s'
        )

        # Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)  # Small pause to allow rendering

    def reset(self):
        """Reset the trajectory tracking."""
        self.trajectory_x = []
        self.trajectory_y = []
        if self.trajectory_line is not None:
            self.trajectory_line.set_data([], [])

    def close(self):
        """Close the matplotlib figure."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.initialized = False

    def update_goal(self, goal_position, goal_radius):
        """
        Update the goal position and radius.

        Args:
            goal_position: New goal position [x, y]
            goal_radius: New goal radius
        """
        if self.goal_patch is not None:
            self.goal_patch.center = goal_position
            self.goal_patch.radius = goal_radius
