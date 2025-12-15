import numpy as np 
import cvxpy as cp

class KongMPC:

    def __init__(self, Q, R, Rbar, N, dt):
        self.Q = Q
        self.R = R
        self.Rbar = Rbar
        self.N = N
        self.dt = dt

        self.create_problem()
            
    
    def create_problem(self):

        self.x = cp.Variable((4, self.N + 1))
        self.u = cp.Variable((2, self.N))

        self.xref = cp.Parameter((4, self.N + 1))
        self.uref = cp.Parameter((2, self.N))
        self.x0 = cp.Parameter(4)
        self.A = [cp.Parameter((4,4)) for _ in range(self.N)]
        self.B = [cp.Parameter((4,2)) for _ in range(self.N)]
        
        cost = 0
        constraints = [self.x[:, 0] == self.x0]

        # constraints on steering angle, acceleration - change if neccessary
        constraints += [self.u[0,:] <= 1.0]
        constraints += [self.u[0,:] >= -1.5]
        constraints += [cp.abs(self.u[1, :]) <= np.deg2rad(37)]

        for k in range(self.N): 
            cost += cp.quad_form(self.x[:, k] - self.xref[:, k], self.Q)
            cost += cp.quad_form(self.u[:, k] - self.uref[:, k], self.R)
            if k > 0:
                cost += cp.quad_form(self.u[:, k] - self.u[:, k -1], self.Rbar)

            constraints += [(self.x[:, k + 1] - self.xref[:, k+1]) == 
                            self.A[k] @ (self.x[:, k] - self.xref[:,k])
                            + self.B[k] @ (self.u[:, k] - self.uref[:, k])]
        
        self.problem = cp.Problem(cp.Minimize(cost), constraints);

    def compute_control(self, state, xref, uref, model, dt):
        self.create_problem()
        
        self.x0.value = state

        xref_unwrapped = xref.copy()
    
        psi_diff = xref_unwrapped[2, 0] - state[2]
        psi_diff_wrapped = (psi_diff + np.pi) % (2 * np.pi) - np.pi
        xref_unwrapped[2, 0] = state[2] + psi_diff_wrapped
    
        for k in range(1, self.N + 1):
            psi_diff = xref_unwrapped[2, k] - xref_unwrapped[2, k-1]
            psi_diff_wrapped = (psi_diff + np.pi) % (2 * np.pi) - np.pi
            xref_unwrapped[2, k] = xref_unwrapped[2, k-1] + psi_diff_wrapped
    
        self.xref.value = xref_unwrapped
        self.uref.value = uref

        for k in range(self.N):
            self.A[k].value, self.B[k].value = model.linearize(xref[:,k], uref[:,k], dt)

        self.problem.solve(solver=cp.OSQP, warm_start=False, ignore_dpp=True)
        
        if self.problem.status in ['optimal', 'optimal_inaccurate']:
            return self.u[:, 0].value
        else:
            print(f"MPC failed: {self.problem.status}")
            return uref[:, 0]  
