import numpy as np 
import cvxpy as cp


class KongMPC:

    def __init__(self, Q, R, Rbar):
        self.Q = Q
        self.R = R
        self.Rbar = Rbar
    
    def compute_control(self, state, ref_traj):
        


        
    