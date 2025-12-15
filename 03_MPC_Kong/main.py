import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os

from common.models import BicycleModel, DynamicBicycleModel
from common.trajectories import ReferenceGhost
from controller import KongMPC


def run_kinematic_sim(T_period, scale=20.0, dt=0.1, sim_time=None):
    if sim_time is None:
        sim_time = T_period
    
    N = 10
    Q = np.diag([10.0, 10.0, 10.0, 1.0])
    R = np.diag([1.0, 1.0])
    Rbar = np.diag([10.0, 10.0])
    
    ghost = ReferenceGhost(scale=scale, T=T_period)
    start_ref = ghost.refState(0)
    
    real_car = BicycleModel(x=start_ref[0], y=start_ref[1], psi=start_ref[2], v=start_ref[3])
    mpc_model = BicycleModel()
    mpc = KongMPC(Q, R, Rbar, N, dt)
    
    h = {'t': [], 'x': [], 'y': [], 'psi': [], 'v': [], 
         'xref': [], 'yref': [], 'vref': [], 'accel': [], 'steer': [], 'pos_error': []}
    
    for t in np.arange(0, sim_time, dt):
        state = real_car.state
        xref, uref = ghost.get_horizion(t, dt, N)
        
        try:
            control = mpc.compute_control(state, xref, uref, mpc_model, dt)
        except:
            control = np.array([0.0, 0.0])
        
        real_car.step(control, dt)
        
        err = np.sqrt((state[0] - xref[0,0])**2 + (state[1] - xref[1,0])**2)
        
        h['t'].append(t)
        h['x'].append(state[0])
        h['y'].append(state[1])
        h['psi'].append(state[2])
        h['v'].append(state[3])
        h['xref'].append(xref[0, 0])
        h['yref'].append(xref[1, 0])
        h['vref'].append(xref[3, 0])
        h['accel'].append(control[0])
        h['steer'].append(control[1])
        h['pos_error'].append(err)
    
    for k in h:
        h[k] = np.array(h[k])
    return h


def run_mismatched_sim(mpc_dt, sim_dt=0.1, T_period=35, scale=20.0, sim_time=35):
    N = 10
    Q = np.diag([10.0, 10.0, 10.0, 1.0])
    R = np.diag([1.0, 1.0])
    Rbar = np.diag([10.0, 10.0])
    
    ghost = ReferenceGhost(scale=scale, T=T_period)
    start_ref = ghost.refState(0)
    
    real_car = DynamicBicycleModel(
        x=start_ref[0], y=start_ref[1], psi=start_ref[2],
        vx=start_ref[3], vy=0, r=0, tyre_model='pacejka'
    )
    
    mpc_model = BicycleModel()
    mpc = KongMPC(Q, R, Rbar, N, mpc_dt)
    
    h = {'t': [], 'pos_error': []}
    
    for t in np.arange(0, sim_time, sim_dt):
        state = real_car.state
        mpc_state = np.array([state[0], state[1], state[2], state[3]])
        xref, uref = ghost.get_horizion(t, mpc_dt, N)
        
        try:
            control = mpc.compute_control(mpc_state, xref, uref, mpc_model, mpc_dt)
        except:
            control = np.zeros(2)
        
        real_car.step(control, sim_dt)
        dist_err = np.sqrt((state[0] - xref[0,0])**2 + (state[1] - xref[1,0])**2)
        h['t'].append(t)
        h['pos_error'].append(dist_err)
        
    return h


def run_discretization_study(results_dir="results"):
    
    h_100 = run_mismatched_sim(mpc_dt=0.1, sim_dt=0.1, T_period=25);
    h_200 = run_mismatched_sim(mpc_dt=0.2, sim_dt=0.1, T_period=25)
    mean_100 = np.mean(h_100['pos_error'])
    mean_200 = np.mean(h_200['pos_error'])
    

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Kong Discretization Effect (Dynamic Reality)", fontsize=14, fontweight='bold')
    
    ax.plot(h_100['t'], h_100['pos_error'], '-', lw=1.5, color='#2ecc71', 
            label=f'td = 100ms (mean: {mean_100:.3f}m)')
    ax.plot(h_200['t'], h_200['pos_error'], '-', lw=1.5, color='#e74c3c', 
            label=f'td = 200ms (mean: {mean_200:.3f}m)')
    
    ax.axhline(mean_100, color='#27ae60', ls='--', alpha=0.5)
    ax.axhline(mean_200, color='#c0392b', ls='--', alpha=0.5)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Tracking Error (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    save_path = os.path.join(results_dir, "fig_discretization.png")
    plt.savefig(save_path, dpi=150)


def run_dynamic_sim(T_period, tire_model='pacejka', scale=20.0, dt=0.1, sim_time=None):
    if sim_time is None:
        sim_time = T_period
    
    N = 10
    Q = np.diag([10.0, 10.0, 10.0, 1.0])
    R = np.diag([1.0, 1.0])
    Rbar = np.diag([10.0, 10.0])
    
    ghost = ReferenceGhost(scale=scale, T=T_period)
    start_ref = ghost.refState(0)
    
    real_car = DynamicBicycleModel(
        x=start_ref[0], y=start_ref[1], psi=start_ref[2],
        vx=start_ref[3], vy=0, r=0, tyre_model=tire_model
    )
    
    mpc_model = BicycleModel()
    mpc = KongMPC(Q, R, Rbar, N, dt)
    
    h = {'t': [], 'x': [], 'y': [], 'psi': [], 'v': [], 'vy': [],
         'xref': [], 'yref': [], 'vref': [], 'accel': [], 'steer': [], 'pos_error': []}
    
    for t in np.arange(0, sim_time, dt):
        state = real_car.state
        mpc_state = np.array([state[0], state[1], state[2], state[3]])
        xref, uref = ghost.get_horizion(t, dt, N)
        
        try:
            control = mpc.compute_control(mpc_state, xref, uref, mpc_model, dt)
        except:
            control = np.array([0.0, 0.0])
        
        real_car.step(control, dt)
        
        err = np.sqrt((state[0] - xref[0,0])**2 + (state[1] - xref[1,0])**2)
        
        h['t'].append(t)
        h['x'].append(state[0])
        h['y'].append(state[1])
        h['psi'].append(state[2])
        h['v'].append(state[3])
        h['vy'].append(state[4])
        h['xref'].append(xref[0, 0])
        h['yref'].append(xref[1, 0])
        h['vref'].append(xref[3, 0])
        h['accel'].append(control[0])
        h['steer'].append(control[1])
        h['pos_error'].append(err)
    
    for k in h:
        h[k] = np.array(h[k])
    return h


def plot_dashboard(h, title="Low-Speed Tracking", save_path=None):
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    t = h['t']
    
    axs[0].plot(t, h['vref'], 'k--', lw=1.5, label='Reference')
    axs[0].plot(t, h['v'], 'k-', lw=1.5, label='Measured')
    axs[0].set_ylabel(r'$v$ (m/s)')
    axs[0].legend(loc='upper right')
    axs[0].grid(True, alpha=0.3)
    
    axs[1].plot(t, h['accel'], 'k-', lw=1)
    axs[1].axhline(1.0, color='k', ls='--', lw=0.8)
    axs[1].axhline(-1.5, color='k', ls='--', lw=0.8)
    axs[1].set_ylabel(r'$a$ (m/s²)')
    axs[1].set_ylim([-2, 1.5])
    axs[1].grid(True, alpha=0.3)
    
    axs[2].plot(t, np.rad2deg(h['steer']), 'k-', lw=1)
    axs[2].axhline(37, color='k', ls='--', lw=0.8)
    axs[2].axhline(-37, color='k', ls='--', lw=0.8)
    axs[2].set_ylabel(r'$\delta_f$ (deg)')
    axs[2].grid(True, alpha=0.3)
    
    avg_err = np.mean(h['pos_error'])
    axs[3].plot(t, h['pos_error'], 'k-', lw=1, label=f'Error (avg: {avg_err:.2f}m)')
    axs[3].axhline(avg_err, color='k', ls='--', lw=1)
    axs[3].set_ylabel('Pos. Err. (m)')
    axs[3].set_xlabel('Time (s)')
    axs[3].legend(loc='upper right')
    axs[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_tracking(h, title="Path Tracking", save_path=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.plot(h['xref'], h['yref'], 'k--', lw=2, alpha=0.7, label='Reference')
    ax.plot(h['x'], h['y'], 'b-', lw=2, label='Tracked')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_drift_comparison(h_good, h_bad, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Kinematic MPC Failure at High Speed\n(Tire Saturation / Understeer)", fontsize=14, fontweight='bold')
    
    ax.plot(h_good['xref'], h_good['yref'], 'k--', lw=2, label='Reference', alpha=0.7)
    ax.plot(h_good['x'], h_good['y'], 'b-', lw=2, label='Low Speed (Success)', alpha=0.8)
    ax.plot(h_bad['x'], h_bad['y'], 'r-', lw=2.5, label='High Speed (Drift)', alpha=0.8)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_error_comparison(histories, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Tracking Error Comparison", fontsize=14, fontweight='bold')
    
    max_err = 0
    for label, h in histories:
        err = h['pos_error']
        max_err = max(max_err, np.max(err[np.isfinite(err)]))
        ax.plot(h['t'], err, lw=1.5, label=f"{label} (avg: {np.mean(err):.2f}m)")
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position Error (m)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, min(max_err * 1.1, 5)]) 
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def create_gif(h, filename="drift.gif"):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    margin = 10
    ax.set_xlim([min(h['x'].min(), h['xref'].min()) - margin, 
                 max(h['x'].max(), h['xref'].max()) + margin])
    ax.set_ylim([min(h['y'].min(), h['yref'].min()) - margin,
                 max(h['y'].max(), h['yref'].max()) + margin])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.plot(h['xref'], h['yref'], 'k--', alpha=0.5, lw=1)
    
    trail, = ax.plot([], [], 'r-', lw=1.5, alpha=0.7)
    car, = ax.plot([], [], 'r-', lw=3)
    title = ax.set_title("")
    
    skip = max(1, len(h['t']) // 150)
    
    def update(frame):
        i = frame * skip
        if i >= len(h['x']):
            return car, trail
        x, y, psi = h['x'][i], h['y'][i], h['psi'][i]
        L, W = 3.0, 1.5
        corners = np.array([[L/2, W/2], [L/2, -W/2], [-L/2, -W/2], [-L/2, W/2], [L/2, W/2]])
        R = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])
        rc = (R @ corners.T).T + [x, y]
        car.set_data(rc[:, 0], rc[:, 1])
        trail.set_data(h['x'][:i], h['y'][:i])
        title.set_text(f"t={h['t'][i]:.1f}s | Error={h['pos_error'][i]:.1f}m")
        return car, trail, title
    
    ani = FuncAnimation(fig, update, frames=len(h['t'])//skip, blit=True, interval=50)
    ani.save(filename, writer=PillowWriter(fps=20))
    plt.close(fig)



if __name__ == "__main__":
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    SCALE = 20.0
    DT = 0.1
    
    print("\n Kinematic Path...")
    h_kin = run_kinematic_sim(T_period=40, scale=SCALE, dt=DT, sim_time=40)
    plot_tracking(h_kin, "Kinematic MPC Path Tracking", os.path.join(RESULTS_DIR, "fig_kinematic.png"))

    print("\n Drift Failure...")
    h_slow = run_dynamic_sim(T_period=40, tire_model='pacejka', scale=SCALE, dt=DT, sim_time=40)
    h_fast = run_dynamic_sim(T_period=10, tire_model='pacejka', scale=SCALE, dt=DT, sim_time=20)
    plot_drift_comparison(h_slow, h_fast, os.path.join(RESULTS_DIR, "fig_drift.png"))

    print("\n Dashboard...")
    plot_dashboard(h_slow, "Low-Speed Tracking", os.path.join(RESULTS_DIR, "fig_dashboard.png"))
    
    print("\n Error Comparison...")
    plot_error_comparison([
        ('Kinematic (ideal)', h_kin),
        ('Dynamic + Pacejka (slow)', h_slow),
    ], os.path.join(RESULTS_DIR, "fig_error.png"))
    
    print("\n Animations...")
    create_gif(h_fast, os.path.join(RESULTS_DIR, "drift.gif"))
    create_gif(h_slow, os.path.join(RESULTS_DIR, "success.gif"))

    print("\n Discretization...")
    run_discretization_study(RESULTS_DIR)
    
    print("Done! Generated (in results/)")