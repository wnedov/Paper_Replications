import numpy as np
import matplotlib.pyplot as plt
from common.environment import MapEnv
from common.models import BicycleModel
from hybridAstar import hybridAstar, DiscreteGrid

def main():

    maze_obstacles = [ 
        (5.0, -4.0, 6.0, 10.0),   
        (-2.0, 2.0, 1.0, 3.0),    
    ]
    env = MapEnv(obstacles=maze_obstacles, start=(-5, -2))
    
    start_state = [-5, -2, 0.0, 0.0]  
    goal_state = [8.0, 8.0, np.pi/2]  
    

    model = BicycleModel()
    grid = DiscreteGrid(resolution_xy=0.5, resolution_deg=15, map_env=env)
    

    planner = hybridAstar(env, start_state, goal_state, model, grid)
    path = planner.hybridastar()
    env.draw_elements()
    
    if path:
        print(f"Path found with {len(path)} steps")
        path_array = np.array(path)
        env.draw_path(path_array, color='blue')
    else:
        print("No path found")
        
    plt.show()

if __name__ == "__main__":
    main()