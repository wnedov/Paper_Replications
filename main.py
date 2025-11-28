from enviroment import *
from rrt import *

def run_planner(planner_class, env, n_iter, step_size):
    planner = planner_class(env, n_iter, step_size)
    
    if hasattr(planner, 'rrt_a'):
        node_list = planner.rrt_a()
    else:
        node_list = planner.rrt()
    
    goal_candidates = []
    for node in node_list:
        p = shapely.geometry.Point(node.get_loc())
        if env.end.contains(p):
            goal_candidates.append(node)
    
    if goal_candidates:
        best_node = min(goal_candidates, key=lambda n: n.cost)
        
        path = planner.get_path_coords(best_node)
        env.draw_path(path)
        env.save_frame(n_iter + 1)
    else:
        print("No path found.")

def main():
    env = MapEnv("tmp")
    
    ITERATIONS = 20000
    STEP_SIZE = 0.5
    SELECTED_ALGORITHM = RRT_A

    run_planner(SELECTED_ALGORITHM, env, ITERATIONS, STEP_SIZE)
    
    plt.show()

if __name__ == "__main__":
    main()