from enviroment import *
from rrt import *


def main():
   env = MapEnv();
   r = RRT(env, 1000, 0.5); 
   result = r.rrt();
   print(result[-1]);
   return 


if __name__ == "__main__":
    main()
