import numpy as np 
import cvxpy as cp


class KongMPC:

    def __init__(self, Q, R, Rbar):
        self.Q = Q
        self.R = R
        self.Rbar = Rbar
        self.problem = None
    
    def linearize(self, state, control):
        for s in state: 
            # numerical linearization
            pass
    
    def create_problem(self, state, ref_traj, N):
        xreff, ureff = ref_traj
        x = cp.Variable((4, N + 1))
        u = cp.Variable((2, N))
        
        xref = cp.Parameter((4, N + 1))
        xref.value = xreff
        uref = cp.Parameter((2, N))
        uref.value = ureff

        cost = 0
        constraints = [x[:, 0] == state]
        # also, need to add constraints on steering angle, acceleration?
        constraints += [u[0,:] <= 1.0]
        constraints += [u[0,:] >= -1.5]
        constraints += [cp.abs(self.var_u[1, :]) <= np.deg2rad(37)]

        for k in range(N): 
            A, B = self.linearize(state, uref[:, k]);

            cost += cp.quad_form(x[:, k] - xref[:, k], self.Q)
            cost += cp.quad_form(u[:, k] - uref[:, k], self.R)
            if k > 1:
                cost += cp.quad_form(u[:, k] - u[:, k -1], self.Rbar)

            constraints += [(x[:, k + 1] - xref[:, k+1]) == A @ (x[:, k] - xref[:,k]) + B @ (u[:, k] - uref[:, k])]
        
        self.problem = cp.Problem(cp.Minimize(cost), constraints);

    def compute_control(self, state, ref_traj):
        #update values? 
        self.problem.solve(solver=cp.OSQP)
        #what to return?

            

        