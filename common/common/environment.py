import shapely
import shapely.plotting
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.collections as c
import shutil 
import os

class MapEnv:

    def __init__(self, frame_dir): 
        
        self.frame_dir = "tmp"
        if os.path.exists(self.frame_dir):
            shutil.rmtree(self.frame_dir)
        os.makedirs(self.frame_dir)

        self.border = shapely.box(-10,-10,10,10);
        self.start = (0,0); 
        self.end = shapely.box(7,7,9,9);
        self.obstacle = shapely.box(2.5,4,5,6);

        self.fig, self.ax = plt.subplots();
        self.ax.set_xlim(-10,10);
        self.ax.set_ylim(-10,10);
        self.ax.set_xticks(range(-10,11));
        self.ax.set_yticks(range(-10,11));
        self.ax.set_aspect('equal');
        self.ax.grid(True, which='both', linestyle='--', alpha=0.3)
        self.lines = c.LineCollection([], color="green", alpha=0.5)
        self.ax.add_collection(self.lines);
    
        shapely.plotting.plot_polygon(self.end, self.ax, color="purple", add_points=False);
        shapely.plotting.plot_polygon(self.obstacle, self.ax, color="red", add_points=False);
        self.ax.scatter([self.start[0]], [self.start[1]], color="blue", zorder=5);
    
    def update(self, seg):
        self.lines.set_segments(seg);

    def save_frame(self, frame_id):
        self.fig.savefig(f"tmp/frame_{frame_id:03d}.png", bbox_inches='tight')

    def draw_path(self, path_coords):
        x = path_coords[:, 0] 
        y = path_coords[:, 1] 
        
        self.ax.plot(x, y, color="red", linewidth=3, zorder=10)