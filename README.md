# Robotics Paper Replications

This repository contains my implementations of seminal papers in robotics.

## Project Status

| Paper / Concept | What it does | Status |
| :--- | :--- | :--- |
| **[Karaman & Frazzoli 2011]**<br>_RRT*: Optimal Path Planning_ | Finds the shortest path through complex obstacles using random sampling. Unlike standard RRT, this version rewires itself to optimize the route over time. | ✅ **Done** |
| **[Kanayama et al. 1990]**<br>_Stable Tracking Control_ | A PID-style controller that uses Lyapunov stability to ensure a 2-wheeled robot (Unicycle) smoothly follows a target without oscillating. | **In Progress** |

## Quick Start

This project uses `uv` for ultra-fast dependency management.

```bash
# 1. Clone the repo
git clone [https://github.com/yourname/Paper_Replications.git](https://github.com/yourname/Paper_Replications.git)
cd Paper_Replications

# 2. Sync dependencies (installs the local 'common' library automatically)
uv sync

# 3. Run a demo
uv run 01_RRT_Star_Karaman/main.py