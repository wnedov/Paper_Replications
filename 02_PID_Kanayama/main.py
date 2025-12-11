from common.trajectories import ReferenceGhost
from controller import KanayamaControl
from common.models import UnicycleModel
from common.environment import MapEnv
import numpy as np
import matplotlib.pyplot as plt

def save_error_plot(time, ye, thetae):
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.xlim(0, 1.25)
    plt.plot(time, ye, color='#E24A33', linewidth=2)
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.ylabel("Lateral Error (m)")
    plt.title("Kanayama Controller Performance")
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.xlim(0, 1.25)
    plt.plot(time, thetae, color='#348ABD', linewidth=2)
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.ylabel("Heading Error (rad)")
    plt.xlabel("Time (s)")
    plt.grid(True)


    
    plt.show()


def draw_robot(ax, state, color='blue', label=None):
    x, y, theta = state
    body, = ax.plot(x, y, marker='o', color=color, markersize=8, label=label)
    r = 0.3 
    nose_x = x + r * np.cos(theta)
    nose_y = y + r * np.sin(theta)
    nose, = ax.plot([x, nose_x], [y, nose_y], color=color, linewidth=2)
    return body, nose


def main():
    period = 4;
    ghost = ReferenceGhost(startx=0, starty=0, starttheta=0, scale=5, T=period)
    controller = KanayamaControl(Kx=10, Ky=64, Ktheta=16)
    robot = UnicycleModel(x=-2, y=1, theta=np.deg2rad(30))
    env = MapEnv()
    env.ax.set_title("Kanayama Tracking")
    
    hist_thetae = []
    gif_frames = []
    hist_ye, hist_time = [], []

    robot_trail_x, robot_trail_y = [], []
    ghost_trail_x, ghost_trail_y = [], []
    
    trail_robot, = env.ax.plot([], [], color='red', linewidth=1, alpha=0.5, label='Robot Path')
    trail_ghost, = env.ax.plot([], [], color='blue', linestyle='--', linewidth=1, alpha=0.5, label='Ghost Path')
    
    r_body, r_nose = draw_robot(env.ax, robot.state, color='red', label='Robot')
    g_body, g_nose = draw_robot(env.ax, [0,0,0], color='blue', label='Ghost')

    dt = 0.01
    sim_time = period * 2.5
    time_steps = int(sim_time / dt)

    plt.ion()  
    

    
    env.ax.legend(loc='upper right')

    for step in range(time_steps):
        t = step * dt
        ref_state = ghost.refStep(t)
        control_inputs = controller.compute_control(robot.state, ref_state)
        robot.step(control_inputs, dt)

        ex = ref_state[0] - robot.state[0]
        ey = ref_state[1] - robot.state[1]
        theta_rob = robot.state[2]
        
        ye = -ex * np.sin(theta_rob) + ey * np.cos(theta_rob)
        
        hist_ye.append(ye)
        hist_time.append(t)

        theta_rob = robot.state[2]
        theta_e = ref_state[2] - theta_rob
        theta_e = (theta_e + np.pi) % (2 * np.pi) - np.pi
        
        hist_thetae.append(theta_e)

        robot_trail_x.append(robot.state[0])
        robot_trail_y.append(robot.state[1])
        ghost_trail_x.append(ref_state[0])
        ghost_trail_y.append(ref_state[1])

        if step % 2 == 0:
            trail_robot.set_data(robot_trail_x, robot_trail_y)
            trail_ghost.set_data(ghost_trail_x, ghost_trail_y)
            
            r_body.set_data([robot.state[0]], [robot.state[1]])
            r_nose.set_data([robot.state[0], robot.state[0] + 0.5*np.cos(robot.state[2])], 
                            [robot.state[1], robot.state[1] + 0.5*np.sin(robot.state[2])])

            g_body.set_data([ref_state[0]], [ref_state[1]])
            g_nose.set_data([ref_state[0], ref_state[0] + 0.5*np.cos(ref_state[2])], 
                            [ref_state[1], ref_state[1] + 0.5*np.sin(ref_state[2])])
            # if step % 4 == 0:
            #     buf = io.BytesIO()
            #     env.fig.savefig(buf, format='png', dpi=80)
            #     buf.seek(0)
            #     gif_frames.append(Image.open(buf))

            plt.pause(0.001)

    plt.ioff() 
    # print("Saving GIF...")
    # if gif_frames:
    #     gif_frames[0].save(
    #         "kanayama_demo.gif",
    #         save_all=True,
    #         append_images=gif_frames[1:],
    #         optimize=True,
    #         duration=40,
    #         loop=0
    #     )

    save_error_plot(hist_time, hist_ye, hist_thetae)
    plt.show()

    
if __name__ == "__main__":
    main()
    