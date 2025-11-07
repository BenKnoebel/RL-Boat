# RL-Boat

This is a repository for the final project of Harvard's Reinforcement Learning and Optimal Control Class ES158 taught by Heng Yang

Team members: Florian Schechner, Benjamin Knoebel del Olmo

## Project Overview

This project implements reinforcement learning algorithms to control a rowing boat with two independent rudders navigating in a 2D environment.

## Problem Definition

**Agent:** A rowing boat with two independent rudders (left and right)

**Action Space:**
- Discrete, 9 possible actions (3² combinations)
- Each rudder can: row forward, row backward, or remain idle
- Results in forward/backward motion or rotation depending on rudder coordination

**State Space:**
- Continuous 2D coordinates (x, y) on a plane
- Infinite dimensionality (continuous state space)

**Environment:**
- 2D plane representing a river or lake
- Boat dynamics defined through force and momentum balances
- Each rudder applies force with a specific lever arm
- Boat motion determined by mass and inertia approximations

**Problem Formulation:**
- Infinite horizon Markov Decision Process (MDP)
- Continuous state space, discrete action space
- Agent has no prior knowledge of dynamics (simulator provides feedback)

**Objective:**
- Primary goal: Navigate optimally to a target location
- Reward function: Constant negative reward until goal is reached (episode termination)
- Potential extensions: obstacles, water currents, and other environmental complexities

## Proposed Methods

We aim to implement and compare two families of reinforcement learning algorithms:

1. **Generalized Policy Iteration (GPI) with Function Approximation**
   - Semi-Gradient SARSA

2. **Generalized Value Iteration (GVI) with Function Approximation**
   - Deep Q-Network (DQN)

Additional methods may be explored to validate theoretical expectations regarding stability and performance.

## Goals and Evaluation Metrics

**Primary Goals:**
- Implement and benchmark multiple RL methods on the boat navigation problem
- Compare algorithm performance and solution optimality

**Evaluation Metrics:**
- Success rate
- Required learning iterations to achieve target success rate
- Optimality of learned policies
- Stability and convergence behavior

**Stretch Goals:**
- Implement OpenAI Gym environment for visualization
- Test algorithms in varied environments (obstacles, currents)
- Extended comparative analysis across multiple methods and metrics
