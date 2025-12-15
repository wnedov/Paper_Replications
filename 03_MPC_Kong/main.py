import numpy as np
import matplotlib.pyplot as plt
from common.models import BicycleModel
from common.trajectories import ReferenceGhost
from controller import KongMPC

def run_simulation():
    # --- 1. CONFIGURATION ---
    DT = 0.1           # Time step (s)
    N = 10             # Horizon length
    SIM_TIME = 40.0    # Duration (s)
    
    # Weights (Tuned for Kong-style tracking)
    # Penalize position error (x,y) heavily
    Q = np.diag([10.0, 10.0, 10.0, 1.0]) 
    R = np.diag([1.0, 1.0])              # Input effort
    Rbar = np.diag([10.0, 10.0])         # Smoothness (Slew rate)

    # --- 2. INITIALIZATION ---
    # Scale=20 makes the figure 8 approx 60m wide (drivable)
    # T=20 means one loop takes 20 seconds
    ghost = ReferenceGhost(scale=20, T=40) 
    
    # We start the robot exactly where the ghost starts to avoid initial transient jumps
    start_ref = ghost.refState(0) 
    real_car = BicycleModel(x=start_ref[0], y=start_ref[1], psi=start_ref[2], v=start_ref[3])
    
    # The helper model for the MPC to perform linearization
    mpc_model = BicycleModel()
    
    mpc = KongMPC(Q, R, Rbar, N, DT)

    # --- 3. LOGGING ---
    history = {
        't': [], 'v': [], 'accel': [], 'steer': [], 
        'x': [], 'y': [], 'xref': [], 'yref': [], 'pos_error': []
    }

    print("Starting Simulation...")
    
    # --- 4. CONTROL LOOP ---
    for t in np.arange(0, SIM_TIME, DT):
        
        # A. Get Ground Truth & Reference
        current_state = real_car.state
        
        # Note: calling get_horizion (using the spelling from your file)
        xref, uref = ghost.get_horizion(t, DT, N)
        
        # B. Compute Control
        # We pass mpc_model so the controller can linearize internally
        control = mpc.compute_control(current_state, xref, uref, mpc_model, DT)
        
        # C. Apply to Plant (Real World)
        real_car.step(control, DT)
        
        # D. Log Data
        history['t'].append(t)
        history['x'].append(current_state[0])
        history['y'].append(current_state[1])
        history['v'].append(current_state[3])
        history['accel'].append(control[0])
        history['steer'].append(control[1])
        history['xref'].append(xref[0, 0])
        history['yref'].append(xref[1, 0])
        
        # Calculate Euclidean position error
        dist_err = np.sqrt((current_state[0] - xref[0,0])**2 + (current_state[1] - xref[1,0])**2)
        history['pos_error'].append(dist_err)

    return history

def plot_results(h):
    # --- FIGURE 5 REPLICATION (The Dashboard) ---
    fig1, axs = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    fig1.suptitle("Kong et al. Fig 5 Replication (Dashboard)", fontsize=16)

    # 1. Velocity
    axs[0].plot(h['t'], h['v'], 'b-', linewidth=2)
    axs[0].set_ylabel('Speed (m/s)')
    axs[0].grid(True)

    # 2. Acceleration (Check Constraints!)
    axs[1].plot(h['t'], h['accel'], 'r-', linewidth=2)
    axs[1].axhline(1.0, color='k', linestyle='--', label='Max Limit')
    axs[1].axhline(-1.5, color='k', linestyle='--', label='Min Limit')
    axs[1].set_ylabel('Accel (m/s^2)')
    axs[1].legend(loc='right')
    axs[1].grid(True)

    # 3. Steering
    axs[2].plot(h['t'], np.rad2deg(h['steer']), 'g-', linewidth=2)
    axs[2].axhline(37, color='k', linestyle='--')
    axs[2].axhline(-37, color='k', linestyle='--')
    axs[2].set_ylabel('Steer (deg)')
    axs[2].grid(True)

    # 4. Position Error
    axs[3].plot(h['t'], h['pos_error'], 'k-', linewidth=2)
    axs[3].set_ylabel('Error (m)')
    axs[3].set_xlabel('Time (s)')
    axs[3].grid(True)

    # --- FIGURE 6 REPLICATION (The Map) ---
    plt.figure(figsize=(8, 8))
    plt.title("Kong et al. Fig 6 Replication (Tracking Performance)")
    plt.plot(h['xref'], h['yref'], 'k--', label='Ghost Path (Ref)', linewidth=1)
    plt.plot(h['x'], h['y'], 'b-', label='MPC Path (Actual)', linewidth=2, alpha=0.7)
    plt.axis('equal')
    plt.legend()
    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    data = run_simulation()
    plot_results(data)