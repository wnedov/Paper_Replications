import shapely
import numpy as np

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
    
    def sample(self):
        lbound = [self.env.border.bounds[0], self.env.border.bounds[1]];
        hbound = [self.env.border.bounds[2], self.env.border.bounds[3]];
        s = np.random.uniform(lbound,hbound);
        p = shapely.geometry.Point(s[0], s[1]);
        return s if not(p.intersects(self.env.obstacle)) else None;

    def NearestNeighbour(self, V, rand): 
        min = V[0];
        mindist = np.linalg.norm(min.get_loc() - rand);
        for v in V: 
            dist = np.linalg.norm(v.get_loc() - rand);
            if dist < mindist:
                min = v;
                mindist = dist;
        return min

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
        E = []
        for i in range(0, self.iterations):
            rand = self.sample();
            while rand is None: 
                rand = self.sample();
             
            nearest = self.NearestNeighbour(V, rand);
            new = self.Steer(nearest, rand); 
            if self.CollisionTest(nearest, new):
                V.append(new);
                E.append([nearest.get_loc(), new.get_loc()])

                if i % 50 == 0:
                    self.env.update(E)
                    self.env.save_frame(i)

        self.env.update(E);
        self.env.save_frame(self.iterations);
        return V;

class RRT_A(RRT):

    def NearVertices(self, V, new, r):
        return [v for v in V if np.linalg.norm(v.get_loc() - new.get_loc()) < r];


    def rrt_a(self):
        V = [self.start];
        E = []
        for i in range(0, self.iterations): 
            rand = self.sample();
            while rand is None:
                rand = self.sample()

            nearest = self.NearestNeighbour(V, rand);
            new = self.Steer(nearest, rand); 
            if self.CollisionTest(nearest, new):
                y = 20 #Change according to random formula
                r = min(y*(np.log(len(V))/len(V))**(1/2), self.step);
                nearNodes = self.NearVertices(V, new, r)
                V.append(new); 

                cmin = nearest.cost + np.linalg.norm(nearest.get_loc() - new.get_loc())
                xmin = nearest

                for near in nearNodes:
                    if self.CollisionTest(near, new) and (near.cost + np.linalg.norm(near.get_loc() - new.get_loc())) < cmin: 
                        cmin = near.cost + np.linalg.norm(near.get_loc() - new.get_loc())
                        xmin = near

                new.parent = xmin
                new.cost = cmin
                E.append([xmin.get_loc(), new.get_loc()])

                for near in nearNodes: 
                    cost = new.cost + np.linalg.norm(new.get_loc() - near.get_loc())
                    if self.CollisionTest(new, near) and cost < near.cost:
                        parent = near.parent;
                        near.parent = new
                        near.cost = cost
                        E = [[node.parent.get_loc(), node.get_loc()] for node in V if node.parent is not parent]
                    
            if i % 50 == 0:
                self.env.update(E)
                self.env.save_frame(i)
        return V;
                          
