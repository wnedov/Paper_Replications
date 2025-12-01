import matplotlib.pyplot as plt
import numpy as np
from common.environment import MapEnv
from rrt import RRT, RRT_A

def run_experiment():
    env = MapEnv("tmp") 
    ITERATIONS = 10000  
    STEP_SIZE = 0.5

    print("Running RRT...")
    planner_rrt = RRT(env, ITERATIONS, STEP_SIZE)
    _, history_rrt = planner_rrt.rrt()

    print("Running RRT*...")
    planner_star = RRT_A(env, ITERATIONS, STEP_SIZE)
    _, history_star = planner_star.rrt_a()


    plt.figure(figsize=(10, 6))
    x = range(ITERATIONS)
    plt.plot(x, history_rrt, color='red', label='RRT', linewidth=1.5)
    plt.plot(x, history_star, color='blue', label='RRT*', linewidth=1.5)


    plt.xlabel('Number of iterations')
    plt.ylabel('Cost')
    plt.title('RRT vs RRT* Cost Convergence')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Optional: Set Y-axis limits if you have huge spikes at the start
    # plt.ylim(10, 25) 

    plt.show()

if __name__ == "__main__":
    run_experiment()