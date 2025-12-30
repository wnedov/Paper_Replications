import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry
from common.environment import MapEnv
from rrt import RRT, RRT_A
from scipy.interpolate import splprep, splev

def run_planner(planner_class, env, n_iter, step_size):
    print(f"Running {planner_class.__name__}...")
    planner = planner_class(env, n_iter, step_size)
    
    if hasattr(planner, 'rrt_a'):
        return planner.rrt_a()
    else:
        return planner.rrt()

def draw_results(env, node_list, planner_class, smooth):
    E = [[node.parent.get_loc(), node.get_loc()] for node in node_list if node.parent]
    env.draw_tree(E)

    goal_candidates = [n for n in node_list if env.end.contains(shapely.geometry.Point(n.get_loc()))]
    
    if goal_candidates:
        best_node = min(goal_candidates, key=lambda n: n.cost)
        
        temp_planner = planner_class(env, 1, 1) 
        path = temp_planner.get_path_coords(best_node)
        
        if smooth:
            env.draw_path(path, color='gray', linestyle='--')
            smooth_path_coords = get_smooth_path(path, s=0.5) 
            env.draw_path(smooth_path_coords, color='red')
        else:
            env.draw_path(path, color='red')
    else:
        print(f"[{planner_class.__name__}] No path found!")

def get_smooth_path(path, s=0.5):
    path = np.array(path).T
    if path.shape[1] < 4:
        return path.T
        
    tck, u = splprep(path, s=s)
    u_new = np.linspace(u.min(), u.max(), 100)
    x_new, y_new = splev(u_new, tck)
    
    return np.column_stack((x_new, y_new))

def main():
    # Options: "RRT", "RRT_STAR", "BOTH" - the both option also shows a cost comparison graph
    MODE = "BOTH" 
    SMOOTHING = False #Smooth the final path using Bezier
    ITERATIONS = 5000
    STEP_SIZE = 0.5
    maze_obstacles = [ 
    (6.0, -4.0, 7.0, 10.0),   
    (-2.0, 2.0, 1.0, 3.0),    
    ]


    if MODE == "BOTH":
        env_rrt = MapEnv(obstacles=maze_obstacles)
        env_rrt.draw_elements()
        plt.title("RRT Path Finding")
        nodes_rrt, hist_rrt = run_planner(RRT, env_rrt, ITERATIONS, STEP_SIZE)
        draw_results(env_rrt, nodes_rrt, RRT, smooth=SMOOTHING)

        env_star = MapEnv(obstacles=maze_obstacles)
        env_star.draw_elements()
        plt.title("RRT* Path Finding")
        nodes_star, hist_star = run_planner(RRT_A, env_star, ITERATIONS, STEP_SIZE)
        draw_results(env_star, nodes_star, RRT_A, smooth=SMOOTHING)

        plt.figure(figsize=(10, 5))
        plt.plot(hist_rrt, color='red', label='RRT', linewidth=1.5, alpha=0.7)
        plt.plot(hist_star, color='blue', label='RRT*', linewidth=1.5)
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.legend()
        plt.title("Cost Convergence: RRT vs RRT*")

    elif MODE == "RRT":
        env = MapEnv(obstacles=maze_obstacles)
        env.draw_elements()
        plt.title("RRT Path Finding")
        nodes, history = run_planner(RRT, env, ITERATIONS, STEP_SIZE)
        draw_results(env, nodes, RRT, smooth=SMOOTHING)
        
        plt.figure(figsize=(10, 5))
        plt.plot(history, color='red', label='RRT')
        plt.title("RRT Cost History")

    elif MODE == "RRT_STAR":
        env = MapEnv(obstacles=maze_obstacles)
        env.draw_elements()
        plt.title("RRT* Path Finding")

        nodes, history = run_planner(RRT_A, env, ITERATIONS, STEP_SIZE)
        draw_results(env, nodes, RRT_A, smooth=SMOOTHING)
        
        plt.figure(figsize=(10, 5))
        plt.plot(history, color='blue', label='RRT*')
        plt.title("RRT* Cost History")

    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()