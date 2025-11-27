from enviroment import *
from rrt import *


def main():
   env = MapEnv()
   r = RRT(env, 1000, 0.5) 
   result = r.rrt()

   goal_node = None
   for node in result:
       p = shapely.geometry.Point(node.get_loc())
       if env.end.contains(p):
           goal_node = node
           break 
       
   if goal_node:
       print("Goal Reached!")
       path = r.get_path_coords(goal_node)
       env.draw_path(path)
       env.save_frame(1001) 
       
   plt.show()


if __name__ == "__main__":
    main()
