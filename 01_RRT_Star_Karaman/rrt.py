import shapely
import numpy as np
from scipy.spatial import cKDTree

class Node: 

    def __init__(self,x,y,parent=None,cost=0):
        self.loc = np.array([x,y]);
        self.cost = cost;
        self.parent = parent;
    
    def get_parent(self):
        return self.parent; 

    def get_loc(self): 
        return self.loc;

    def __repr__(self):
        return str(self.loc);

    def __eq__(self, value):
        return np.array_equal(self.loc, value.loc)
        
        
class RRT: 

    def __init__(self, env, n, step):
        self.env = env; 
        self.start = Node(0,0);
        self.iterations = n;
        self.step = step;

        self.cost_history = [] 
        self.goal_nodes = []
        self.node_coords = [self.start.get_loc()]
    
    def sample(self):
        lbound = [self.env.border.bounds[0], self.env.border.bounds[1]];
        hbound = [self.env.border.bounds[2], self.env.border.bounds[3]];
        s = np.random.uniform(lbound,hbound);
        p = shapely.geometry.Point(s[0], s[1]);
        return s if not(p.intersects(self.env.obstacle)) else None;

    def NearestNeighbour(self, V, rand): 
        tree = cKDTree(self.node_coords)
        _, i = tree.query(rand, k=1)
        return V[i]

    def Steer(self, nearest, rand):
        if np.linalg.norm(nearest.get_loc() - rand) <= self.step:
            cost = nearest.cost + np.linalg.norm(rand - nearest.loc)
            return Node(rand[0], rand[1], nearest, cost)
        else: 
            vec = (rand - nearest.get_loc())/np.linalg.norm(rand-nearest.get_loc());
            newpos = nearest.get_loc() + vec * self.step

            cost = nearest.cost + np.linalg.norm(newpos - nearest.loc)

            return Node(newpos[0], newpos[1], nearest, cost);
        
    def CollisionTest(self, nearest, new):
        line = shapely.LineString([nearest.get_loc(), new.get_loc()]); 
        
        return not(line.intersects(self.env.obstacle));

    def get_path_coords(self, goal_node):
        path = []
        curr = goal_node
        while curr is not None:
            path.append(curr.get_loc())
            curr = curr.get_parent()
        return np.array(path[::-1])

    def rrt(self): 
        V = [self.start];
        self.node_coords = [self.start.get_loc()]
        for i in range(0, self.iterations):
            rand = self.sample();
            while rand is None: 
                rand = self.sample();
             
            nearest = self.NearestNeighbour(V, rand);
            new = self.Steer(nearest, rand); 
            if self.CollisionTest(nearest, new):
                V.append(new);
                self.node_coords.append(new.get_loc())

                p = shapely.geometry.Point(new.get_loc())
                if self.env.end.contains(p):
                    self.goal_nodes.append(new)

            
            if self.goal_nodes:
                min_cost = min([n.cost for n in self.goal_nodes])
                self.cost_history.append(min_cost)
            else:
                self.cost_history.append(None) 

        E = [[node.parent.get_loc(), node.get_loc()] for node in V if node.parent is not None]    
        self.env.draw_tree(E);
        return V, self.cost_history;

class RRT_A(RRT):

    def NearVertices(self, V, new, r):
        tree = cKDTree(self.node_coords)
        indices = tree.query_ball_point(new.get_loc(), r)
        return [V[i] for i in indices]


    def rrt_a(self):
        V = [self.start];
        E = []
        self.node_coords = [self.start.get_loc()]
        for i in range(0, self.iterations): 
            rand = self.sample();
            while rand is None:
                rand = self.sample()

            nearest = self.NearestNeighbour(V, rand);
            new = self.Steer(nearest, rand); 
            if self.CollisionTest(nearest, new):
                y = 20 # Change according to random formula (in paper)
                r = min(y*(np.log(len(V))/len(V))**(1/2), self.step);
                nearNodes = self.NearVertices(V, new, r)
                V.append(new); 
                self.node_coords.append(new.get_loc())

                cmin = nearest.cost + np.linalg.norm(nearest.get_loc() - new.get_loc())
                xmin = nearest

                for near in nearNodes:
                    if self.CollisionTest(near, new) and (near.cost + np.linalg.norm(near.get_loc() - new.get_loc())) < cmin: 
                        cmin = near.cost + np.linalg.norm(near.get_loc() - new.get_loc())
                        xmin = near

                new.parent = xmin
                new.cost = cmin

                p = shapely.geometry.Point(new.get_loc())
                if self.env.end.contains(p):
                    self.goal_nodes.append(new)

                for near in nearNodes: 
                    cost = new.cost + np.linalg.norm(new.get_loc() - near.get_loc())
                    if self.CollisionTest(new, near) and cost < near.cost:
                        near.parent = new
                        near.cost = cost
            

            if self.goal_nodes:
                min_cost = min([n.cost for n in self.goal_nodes])
                self.cost_history.append(min_cost)
            else:
                self.cost_history.append(None)
                    
        E = [[node.parent.get_loc(), node.get_loc()] for node in V if node.parent is not None]
        self.env.draw_tree(E);
        return V, self.cost_history;
                          
