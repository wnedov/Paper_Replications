# Robotics Paper Replications

Implementations and experiment reproductions of classic robotics, planning, control, and reinforcement-learning papers.

## Path Planning

### RRT* — Karaman & Frazzoli (2011)

<table>
  <tr>
    <td width="46%" align="center"><img src="01_RRT_Star_Karaman/results/rrta2500.gif" alt="RRT* rewiring a planning tree around obstacles" width="420"></td>
    <td><strong>RRT explores fast. RRT* rewires the tree until the route gets shorter.</strong><br><br>Built: sampling, collision checks, best-parent selection, rewiring, and cost trials.<br><br>Shows: branches straightening as better paths are found.<br><br><code>cd 01_RRT_Star_Karaman && uv run main.py</code></td>
  </tr>
</table>

### Hybrid A* — Dolgov et al. (2008)

<table>
  <tr>
    <td width="46%" align="center"><img src="05_HybridA*_Dolgov/results/example_path.png" alt="Hybrid A* path through an obstacle field" width="420"></td>
    <td><strong>A* finds a route. Hybrid A* finds one a car can actually drive.</strong><br><br>Built: forward/reverse bicycle motion, two heuristics, collision checks, and a Reeds–Shepp finish.<br><br>Shows: a feasible path to the target pose.<br><br><code>cd '05_HybridA*_Dolgov' && uv run main.py</code></td>
  </tr>
</table>

## Control

### Stable tracking control — Kanayama et al. (1990)

<table>
  <tr>
    <td width="46%" align="center"><img src="02_PID_Kanayama/results/kanayama_demo.gif" alt="Kanayama controller following a moving reference" width="420"></td>
    <td><strong>Pose error becomes steering commands that pull the robot back onto its path.</strong><br><br>Built: body-frame error feedback, unicycle dynamics, and error logging.<br><br>Shows: an offset robot converging onto the moving reference.<br><br><code>cd 02_PID_Kanayama && uv run main.py</code></td>
  </tr>
</table>

### Model predictive control — Kong et al. (2015)

<table>
  <tr>
    <td width="46%" align="center"><img src="03_MPC_Kong/results/drift.gif" alt="Kinematic MPC drifting against dynamic vehicle behaviour" width="420"></td>
    <td><strong>MPC predicts ahead. This experiment shows what happens when its vehicle model is too simple.</strong><br><br>Built: constrained horizon optimization and matched/mismatched vehicle simulations.<br><br>Shows: good low-speed tracking, then high-speed drift from tire dynamics.<br><br><code>cd 03_MPC_Kong && uv run main.py</code></td>
  </tr>
</table>

## Reinforcement Learning

### DQN / Double DQN — Mnih et al. (2015), van Hasselt et al. (2015)

<table>
  <tr>
    <td width="46%" align="center"><img src="04_DQN_Minh/results/progress_50M.gif" alt="Double DQN agent playing Atari Breakout" height="300"></td>
    <td><strong>Learn which joystick action is worth most—directly from pixels.</strong><br><br>Built: frame stacking, replay memory, CNN value estimates, target networks, and Double DQN updates.<br><br>Shows: learned Breakout play, including degradation and recovery across checkpoints.<br><br><code>cd 04_DQN_Minh && uv run main.py</code></td>
  </tr>
</table>

### PPO — Schulman et al. (2017)

<table>
  <tr>
    <td width="46%" align="center"><img src="06_PPO_Schulman/results/breakout_gameplay.gif" alt="PPO agent playing Atari Breakout" height="300"></td>
    <td><strong>Improve a policy without letting any single update move it too far.</strong><br><br>Built: rollouts, GAE, clipped updates, CNN Atari policies, and Gaussian continuous-control policies.<br><br>Shows: ~373 Breakout and ~2700 HalfCheetah average return—comparable to a strong reference implementation, not paper-beating.<br><br><code>cd 06_PPO_Schulman && uv run main.py</code></td>
  </tr>
</table>

![PPO Breakout learning curve over 10 million steps](06_PPO_Schulman/results/breakout_final.png)

## Quick Start / Reproducibility

The repository uses Python 3.13 and [`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/wnedov/Paper_Replications.git
cd Paper_Replications
uv sync

cd 01_RRT_Star_Karaman
uv run main.py
```

Run commands from the individual project directories so generated checkpoints, plots, and animations remain with the corresponding experiment. RL training is hardware- and seed-sensitive; the committed plots and evaluation summaries are retained as evidence of the reported runs.

## Repository Structure

```text
01_RRT_Star_Karaman/     RRT and RRT* planning
02_PID_Kanayama/         stable trajectory tracking
03_MPC_Kong/             constrained vehicle MPC
04_DQN_Minh/             DQN and Double DQN on Breakout
05_HybridA*_Dolgov/      Hybrid A* vehicle planning
06_PPO_Schulman/         PPO on Breakout and HalfCheetah
07_UKF_Uhlmann/          UKF reference papers
common/                  shared environments, models, networks, and trajectories
other/                   additional paper references
```

## Attribution / Provenance

The planning, control, and learning experiments are repository implementations of the cited methods, but not every component is written from scratch. In Hybrid A*, [`reeds_shepp_path_planning.py`](05_HybridA*_Dolgov/reeds_shepp_path_planning.py) is adapted from the [PythonRobotics Reeds–Shepp planner](https://github.com/AtsushiSakai/PythonRobotics/tree/master/PathPlanning/ReedsSheppPath); the upstream author attribution remains in the source file.

## References

- Karaman, S. and Frazzoli, E. (2011), [Sampling-based Algorithms for Optimal Motion Planning](01_RRT_Star_Karaman/RRT%2A%20-%20Karaman%20and%20Frazzoli.pdf).
- Kanayama, Y. et al. (1990), [A Stable Tracking Control Method for an Autonomous Mobile Robot](02_PID_Kanayama/PID_Kanayama_1990.pdf).
- Kong, J. et al. (2015), [Kinematic and Dynamic Vehicle Models for Autonomous Driving Control Design](03_MPC_Kong/Kong-2015.pdf).
- Mnih, V. et al. (2015), [Human-level Control through Deep Reinforcement Learning](04_DQN_Minh/Minh2015.pdf), and van Hasselt, H. et al. (2015), [Deep Reinforcement Learning with Double Q-learning](04_DQN_Minh/DoubleQ_Hasselt2015.pdf).
- Dolgov, D. et al. (2008), [Practical Search Techniques in Path Planning for Autonomous Driving](05_HybridA*_Dolgov/HybridA_Dolgov2008.pdf).
- Schulman, J. et al. (2017), [Proximal Policy Optimization Algorithms](06_PPO_Schulman/PPO_Schulman2017.pdf).
