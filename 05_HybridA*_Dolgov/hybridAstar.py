import numpy as np
import shapely


class Node: 
    def __init__(self, g_cost, f_cost, parent=None, state=None, discrete=None, direction=1): 
        self.state = state       
        self.discrete = discrete  
        self.g_cost = g_cost
        self.f_cost = f_cost 
        self.parent = parent 
        self.direction = direction 

    def __lt__(self, other):
        return self.f_cost < other.f_cost


class DiscreteGrid:

    def __init__(self, resolution_xy, resolution_deg, map_env):
        self.res_xy = resolution_xy
        self.res_theta = np.deg2rad(resolution_deg)
        

        min_x, min_y, max_x, max_y = map_env.border.bounds
        self.x_min = min_x
        self.y_min = min_y
        
        self.width_indices = int((max_x - min_x) / resolution_xy)
        self.height_indices = int((max_y - min_y) / resolution_xy)
        self.theta_indices = int((2 * np.pi) / self.res_theta)

    def get_index(self, state):
        x, y, theta = state

        idx_x = int((x - self.x_min) / self.res_xy)
        idx_y = int((y - self.y_min) / self.res_xy)

        theta_norm = (theta + np.pi) % (2 * np.pi) - np.pi
        theta_shifted = theta_norm + np.pi
        
        idx_theta = int(theta_shifted / self.res_theta) % self.theta_indices

        return (idx_x, idx_y, idx_theta)


class hybridAstar(): 

    def __init__(self, env, start_state, goal_state, model, grid):
        self.env = env
        self.start = start_state
        self.goal = goal_state
        self.model = model
        self.grid = grid
        
        # Configuration constants
        self.dt = 0.2  
        self.steering_inputs = [-0.5, 0, 0.5] 
        self.reverse_penalty = 2.0
        self.change_dir_penalty = 5.0


    def h(self): 
        holonomic = 
        non_holonomic = 
        return max(holonomic, non_holonomic)


    def g(self, parent_node, direction, step_distance): 
        cost = parent_node.g_cost
        
        if direction == 1:
            cost += step_distance
        else:
            cost += step_distance * self.reverse_penalty
            
        if parent_node.direction != direction:
            cost += self.change_dir_penalty
            
        return cost
    
    def check_collision(self, state):
        idx = self.grid.get_index(state)
        if not (0 <= idx[0] < self.grid.width_indices and 0 <= idx[1] < self.grid.height_indices):
            return False
            
        point = shapely.Point(state[0], state[1])
        if self.env.obstacle.contains(point): 
            return False
            
        return True

    def reconstruct_path(self, node):
        path = []
        while node is not None:
            path.append(node.state)
            node = node.parent
        return path[::-1]

    
    def hybridastar(self): 
        



