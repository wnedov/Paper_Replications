from ghost import ReferenceGhost
from controller import KanayamaControl
from common.models import UnicycleModel
from common.environment import MapEnv
import numpy as np
import matplotlib.pyplot as plt


def draw_robot(ax, state, color='blue', label=None):
    x, y, theta = state
    body, = ax.plot(x, y, marker='o', color=color, markersize=8, label=label)
    r = 0.3 
    nose_x = x + r * np.cos(theta)
    nose_y = y + r * np.sin(theta)
    nose, = ax.plot([x, nose_x], [y, nose_y], color=color, linewidth=2)
    return body, nose


def main():
    period = 20;
    ghost = ReferenceGhost(startx=0, starty=0, starttheta=0, scale=5, T=period)
    controller = KanayamaControl(Kx=10, Ky=64, Ktheta=16)
    robot = UnicycleModel(x=-2, y=1, theta=np.deg2rad(30))
    env = MapEnv()

    robot_trail_x, robot_trail_y = [], []
    ghost_trail_x, ghost_trail_y = [], []
    
    trail_robot, = env.ax.plot([], [], color='red', linewidth=1, alpha=0.5, label='Robot Path')
    trail_ghost, = env.ax.plot([], [], color='blue', linestyle='--', linewidth=1, alpha=0.5, label='Ghost Path')
    
    r_body, r_nose = draw_robot(env.ax, robot.state, color='red', label='Robot')
    g_body, g_nose = draw_robot(env.ax, [0,0,0], color='blue', label='Ghost')

    dt = 0.1
    sim_time = period * 5
    time_steps = int(sim_time / dt)

    plt.ion()  # Turn on interactive mode
    

    
    env.ax.legend(loc='upper right')

    for step in range(time_steps):
        t = step * dt
        ref_state = ghost.refStep(t)
        control_inputs = controller.compute_control(robot.state, ref_state)
        robot.step(control_inputs, dt)

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

            plt.pause(0.001)

    plt.ioff() 
    plt.show()

    


if __name__ == "__main__":
    main()
    