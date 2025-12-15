import numpy as np
import matplotlib.pyplot as plt
from common.models import BicycleModel
from common.trajectories import ReferenceGhost
from controller import KongMPC

def run_simulation():
    DT = 0.1          
    N = 10           
    SIM_TIME = 40.0   
    
    Q = np.diag([10.0, 10.0, 10.0, 1.0]) 
    R = np.diag([1.0, 1.0])             
    Rbar = np.diag([10.0, 10.0])       

    ghost = ReferenceGhost(scale=20, T=40) 
    
    start_ref = ghost.refState(0) 
    real_car = BicycleModel(x=start_ref[0], y=start_ref[1], psi=start_ref[2], v=start_ref[3])
    
    mpc_model = BicycleModel()
    
    mpc = KongMPC(Q, R, Rbar, N, DT)

    history = {
        't': [], 'v': [], 'accel': [], 'steer': [], 
        'x': [], 'y': [], 'xref': [], 'yref': [], 'pos_error': []
    }

    print("Starting Simulation...")
    
    for t in np.arange(0, SIM_TIME, DT):
        
        current_state = real_car.state
        xref, uref = ghost.get_horizion(t, DT, N)
        control = mpc.compute_control(current_state, xref, uref, mpc_model, DT)
        real_car.step(control, DT)
        

        history['t'].append(t)
        history['x'].append(current_state[0])
        history['y'].append(current_state[1])
        history['v'].append(current_state[3])
        history['accel'].append(control[0])
        history['steer'].append(control[1])
        history['xref'].append(xref[0, 0])
        history['yref'].append(xref[1, 0])
        
        dist_err = np.sqrt((current_state[0] - xref[0,0])**2 + (current_state[1] - xref[1,0])**2)
        history['pos_error'].append(dist_err)

    return history

def plot_results(h):
    fig1, axs = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    fig1.suptitle("Kong et al. Fig 5 Replication (Dashboard)", fontsize=16)

    axs[0].plot(h['t'], h['v'], 'b-', linewidth=2)
    axs[0].set_ylabel('Speed (m/s)')
    axs[0].grid(True)

    axs[1].plot(h['t'], h['accel'], 'r-', linewidth=2)
    axs[1].axhline(1.0, color='k', linestyle='--', label='Max Limit')
    axs[1].axhline(-1.5, color='k', linestyle='--', label='Min Limit')
    axs[1].set_ylabel('Accel (m/s^2)')
    axs[1].legend(loc='right')
    axs[1].grid(True)

    axs[2].plot(h['t'], np.rad2deg(h['steer']), 'g-', linewidth=2)
    axs[2].axhline(37, color='k', linestyle='--')
    axs[2].axhline(-37, color='k', linestyle='--')
    axs[2].set_ylabel('Steer (deg)')
    axs[2].grid(True)

    axs[3].plot(h['t'], h['pos_error'], 'k-', linewidth=2)
    axs[3].set_ylabel('Error (m)')
    axs[3].set_xlabel('Time (s)')
    axs[3].grid(True)

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