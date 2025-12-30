import numpy as np
import shapely
import heapq
import reeds_shepp_path_planning as rs



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
        x, y, theta = state[:3]

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
        self.v = 3.0
        self.steering_inputs = [-0.5, 0, 0.5] 
        self.reverse_penalty = 2.0
        self.change_dir_penalty = 5.0

        self.wheelbase = self.model.lf + self.model.lr
        self.min_radius = self.wheelbase / np.tan(max(self.steering_inputs))
        self.maxc = 1.0 / self.min_radius


        self.start_node = Node(0, self.h(self.start), state=self.start, discrete=self.grid.get_index(self.start))
        


    def h(self, state): 
        sx, sy, syaw = state[:3] 
        gx, gy, gyaw = self.goal
        
        px, py, pyaw, mode, course_lengths = rs.reeds_shepp_path_planning(
            sx, sy, syaw, gx, gy, gyaw, self.maxc, step_size=2
        )
        
        if course_lengths is None:
            non_holonomic = float('inf')
        else:
            non_holonomic = sum(abs(l) for l in course_lengths)

        euclid = np.linalg.norm(np.array(self.goal[:2]) - np.array(state[:2]))
        return max(euclid, non_holonomic)


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
            return True
            
        point = shapely.Point(state[0], state[1])
        if self.env.obstacle.contains(point): 
            return True
            
        return False

    def reconstruct_path(self, node):
        path = []
        while node is not None:
            path.append(node.state)
            node = node.parent
        return path[::-1]

    
    def hybridastar(self): 
        open = []
        heapq.heappush(open, (self.start_node.f_cost, self.start_node))
        closed = {}
        closed[self.start_node.discrete] = self.start_node
        current_node = None

        while open:
            current_node = heapq.heappop(open)[1]

            if np.linalg.norm(np.array(current_node.state[:2]) - np.array(self.goal[:2])) < 20.0:
                px, py, pyaw, mode, clen = rs.reeds_shepp_path_planning(
                current_node.state[0], current_node.state[1], current_node.state[2],
                self.goal[0], self.goal[1], self.goal[2], 
                self.maxc, step_size=0.2
                )

                if px is not None:
                    collision = False
                    for i in range(len(px)):
                        if self.check_collision((px[i], py[i], pyaw[i])):
                            collision = True
                            break
                            
                    if not collision:
                        path_so_far = self.reconstruct_path(current_node)
                        analytic_path = []
                        for i in range(len(px)):
                            analytic_path.append([px[i], py[i], pyaw[i], 0.0])
                            
                        return path_so_far + analytic_path
            
            for steering in self.steering_inputs:
                for direction in [1, -1]:
                    total_distance = 0.0

                    state = np.array(current_node.state)
                    state[3] = self.v * direction  
                    control = np.array([0, steering])

                    new_state = self.model.discrete_dynamics(state, control, self.dt)
                    total_distance += np.linalg.norm(new_state[:2] - state[:2]) 
                    state = new_state
                        
                    if self.check_collision(new_state):
                        continue

                    else: 
                        g_cost = self.g(current_node, direction, total_distance) 
                        h_cost = self.h(new_state)
                        new_cost = g_cost + h_cost
                        
                        discrete_idx = self.grid.get_index(state)

                        if discrete_idx in closed and closed[discrete_idx].g_cost <= g_cost: 
                            continue

                       
                        successor_node = Node(g_cost, new_cost, current_node, state, discrete_idx, direction)
                        closed[discrete_idx] = successor_node
                        heapq.heappush(open, (new_cost, successor_node)) 

        if current_node is None:
            return None 