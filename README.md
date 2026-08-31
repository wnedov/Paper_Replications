# Robotics Paper Replications

Implementations and experiment reproductions of classic robotics, planning, control, and reinforcement-learning papers.

## Visual Highlights

<table>
  <tr>
    <td width="33%" align="center"><strong>RRT* planning</strong></td>
    <td width="33%" align="center"><strong>Kanayama tracking control</strong></td>
    <td width="33%" align="center"><strong>PPO on Breakout</strong></td>
  </tr>
  <tr>
    <td align="center"><img src="01_RRT_Star_Karaman/results/rrta2500.gif" alt="RRT* tree expansion and path planning" height="240"></td>
    <td align="center"><img src="02_PID_Kanayama/results/kanayama_demo.gif" alt="Kanayama controller tracking a reference trajectory" height="240"></td>
    <td align="center"><img src="06_PPO_Schulman/results/breakout_gameplay.gif" alt="PPO agent playing Atari Breakout" height="240"></td>
  </tr>
  <tr>
    <td>Sampling, rewiring, and best-path refinement around obstacles.</td>
    <td>Pose-error feedback brings a unicycle robot onto a moving reference.</td>
    <td>A trained clipped-objective policy playing from stacked image observations.</td>
  </tr>
</table>

## Measured / Reproduced Results

| Area | Evidence in this repository |
| --- | --- |
| RRT* | Repeated-trial cost histories reproduce the expected improvement from rewiring relative to RRT. |
| Kanayama control | Saved trajectories and error traces show lateral and heading error converging during reference tracking. |
| MPC | The same controller tracks with a matched kinematic model and exhibits high-speed drift against nonlinear tire dynamics, exposing model mismatch. |
| DQN / Double DQN | Breakout checkpoints, evaluation summaries, learning curves, and failed-run traces document both learned play and training instability. |
| PPO | Saved reproduction runs reach approximately **373 average return on Breakout** and **2700 average return on HalfCheetah**. These results are comparable to a strong reference implementation; they are not presented as outperforming the paper. |

### PPO Breakout learning curve

The saved 10-million-step run shows the policy progressing from near-zero return to a sustained return in the high 300s. The upper panel records optimization loss; the lower panel records episodic return.

![PPO Breakout loss and episodic return over 10 million steps](06_PPO_Schulman/results/breakout_final.png)

## Path Planning

### RRT* — Karaman & Frazzoli (2011)

- **Paper / concept:** Asymptotically optimal sampling-based motion planning through incremental tree rewiring.
- **Implemented:** Collision-checked RRT and RRT*, neighborhood selection, parent optimization, rewiring, path extraction, and repeated-trial cost tracking.
- **Demo / result:** The animation shows tree growth around obstacles; the saved convergence experiment shows RRT* continuing to reduce path cost after a feasible route is found.
- **Run:** `cd 01_RRT_Star_Karaman && uv run main.py`

### Hybrid A* — Dolgov et al. (2008)

- **Paper / concept:** Search over discretized vehicle pose while propagating continuous bicycle-model motion.
- **Implemented:** Forward/reverse motion primitives, collision checking, holonomic and non-holonomic heuristics, direction-change penalties, and analytic Reeds–Shepp connection to the goal.
- **Demo / result:** The saved path reaches the target pose through the obstacle field while respecting car-like motion constraints.
- **Run:** `cd '05_HybridA*_Dolgov' && uv run main.py`

## Control

### Stable tracking control — Kanayama et al. (1990)

- **Paper / concept:** Pose-error feedback for stable trajectory tracking by a nonholonomic mobile robot.
- **Implemented:** Body-frame error transformation, linear/angular velocity control law, unicycle simulation, and lateral/heading error logging.
- **Demo / result:** The robot converges from an offset initial pose and follows the moving reference trajectory; the accompanying plot records error convergence.
- **Run:** `cd 02_PID_Kanayama && uv run main.py`

### Model predictive control — Kong et al. (2015)

- **Paper / concept:** Linearized, constrained MPC for vehicle path tracking and the limits of kinematic models at higher speed.
- **Implemented:** Receding-horizon optimization with state/input costs, actuator constraints, bicycle-model linearization, and matched-versus-mismatched vehicle simulations.
- **Demo / result:** Low-speed tracking succeeds, while the dynamic tire-model experiment drifts at high speed and makes the kinematic-model mismatch visible.
- **Run:** `cd 03_MPC_Kong && uv run main.py`

## Reinforcement Learning

### DQN / Double DQN — Mnih et al. (2015), van Hasselt et al. (2015)

- **Paper / concept:** Value-based control from pixels with replay memory and a target network; Double DQN separates action selection from target evaluation.
- **Implemented:** Atari preprocessing, stacked frames, convolutional Q-network, replay buffer, epsilon-greedy exploration, target checkpoints, and Double DQN targets.
- **Demo / result:** Breakout checkpoints show learning, mid-training degradation, and recovery; retained failed runs make instability visible rather than hiding it.
- **Run:** `cd 04_DQN_Minh && uv run main.py`

### PPO — Schulman et al. (2017)

- **Paper / concept:** On-policy actor–critic learning with a clipped probability-ratio objective.
- **Implemented:** Vectorized rollouts, generalized advantage estimation, clipped policy updates, entropy/value losses, gradient clipping, learning-rate annealing, CNN policies for Atari, and Gaussian policies for continuous control.
- **Demo / result:** Breakout gameplay and the learning curve accompany reproduction results of approximately 373 average return on Breakout and 2700 on HalfCheetah.
- **Run:** `cd 06_PPO_Schulman && uv run main.py --env-type atari --env-id ALE/Breakout-v5 --run-name breakout --total-timesteps 10000000`

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
