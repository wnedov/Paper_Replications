import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry
from common.environment import MapEnv
from rrt import RRT, RRT_A

def run_planner(planner_class, env, n_iter, step_size):
    print(f"Running {planner_class.__name__}...")
    planner = planner_class(env, n_iter, step_size)
    
    if hasattr(planner, 'rrt_a'):
        return planner.rrt_a()
    else:
        return planner.rrt()

def draw_results(env, node_list, planner_class):
    E = [[node.parent.get_loc(), node.get_loc()] for node in node_list if node.parent]
    env.draw_tree(E)

    goal_candidates = [n for n in node_list if env.end.contains(shapely.geometry.Point(n.get_loc()))]
    
    if goal_candidates:
        best_node = min(goal_candidates, key=lambda n: n.cost)
        
        temp_planner = planner_class(env, 1, 1) 
        path = temp_planner.get_path_coords(best_node)
        
        env.draw_path(path)
    else:
        print(f"[{planner_class.__name__}] No path found!")

def main():
    # Options: "RRT", "RRT_STAR", "BOTH" - the both option also shows a cost comparison graph
    MODE = "RRT_STAR" 
    
    ITERATIONS = 5000
    STEP_SIZE = 0.5

    env = MapEnv() 
    env.draw_elements()

    if MODE == "BOTH":
        temp_env = MapEnv() 
        plt.close(temp_env.fig) 
        _, history_rrt = run_planner(RRT, temp_env, ITERATIONS, STEP_SIZE)

        nodes_star, history_star = run_planner(RRT_A, env, ITERATIONS, STEP_SIZE)
        draw_results(env, nodes_star, RRT_A)

        plt.figure(figsize=(10, 5))
        plt.plot(history_rrt, color='red', label='RRT', linewidth=1.5, alpha=0.7)
        plt.plot(history_star, color='blue', label='RRT*', linewidth=1.5)
        plt.title("Cost Convergence: RRT vs RRT*")

    elif MODE == "RRT":
        nodes, history = run_planner(RRT, env, ITERATIONS, STEP_SIZE)
        draw_results(env, nodes, RRT)
        
        plt.figure(figsize=(10, 5))
        plt.plot(history, color='red', label='RRT')
        plt.title("RRT Cost History")

    elif MODE == "RRT_STAR":
        nodes, history = run_planner(RRT_A, env, ITERATIONS, STEP_SIZE)
        draw_results(env, nodes, RRT_A)
        
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