# Robotics Paper Replications

This repository contains my implementations of seminal papers in robotics.

## Project Status

### Path Planning

| Paper / Concept | What it does | Status |
| :--- | :--- | :--- |
| **[Karaman & Frazzoli 2011]**<br>_RRT*: Optimal Path Planning_ | Finds the shortest path through complex obstacles using random sampling. Unlike standard RRT, this version rewires itself to optimize the route over time. | ✅ **Done** |
| **[Dolgov et al. 2008]**<br>_Hybrid A* Path Planning_ | Plans smooth, drivable paths for car-like robots by combining A* grid search with continuous vehicle kinematics. Uses Reeds-Shepp curves for analytic path completion near the goal. | ✅ **Done** |

### Control

| Paper / Concept | What it does | Status |
| :--- | :--- | :--- |
| **[Kanayama et al. 1990]**<br>_Stable Tracking Control_ | A PID-style controller that uses Lyapunov stability to ensure a 2-wheeled robot (Unicycle) smoothly follows a target without oscillating. | ✅ **Done** |
| **[Kong et al. 2015]**<br>*Kinematic vs. Dynamic MPC* | Compares vehicle models for autonomous driving. Demonstrates that Kinematic MPC is robust at low speeds but fails (drifts) at high speeds due to unmodeled tire dynamics, requiring Dynamic models. | ✅ **Done** |

### Reinforcement Learning

| Paper / Concept | What it does | Status |
| :--- | :--- | :--- |
| **[Mnih et al. 2015]**<br>*DQN & Double DQN (Atari)* | The "Hello World" of Deep RL. Uses a CNN and Q-learning to teach an agent to play Atari Breakout directly from raw pixels. Implements Double DQN (Hasselt 2015) for training stability. | ✅ **Done** |
| **[Schulman et al. 2017]**<br>*Proximal Policy Optimization (PPO)* | A policy gradient method that uses clipped surrogate objectives to enable stable, efficient training. The go-to algorithm for continuous control tasks. | **In Progress** |

## Quick Start

This project uses `uv` for dependency management.
Note: Most of these Replications create and output files to a results/ folder in the current working directory. I recommend `cd`'ing into the specific project folder for better organization.

```bash
git clone https://github.com/yourname/Paper_Replications.git
cd Paper_Replications

# Sync dependencies 
uv sync

# Run a demo
cd 01_RRT_Star_Karaman/ 
uv run main.py
```