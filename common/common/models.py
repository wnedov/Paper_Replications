import numpy as np

class UnicycleModel: 

    def __init__(self, x=0, y=0, theta=0):
        self.state = np.array([x, y, theta], dtype=float)

    def step(self, control, dt):
        x, y, theta = self.state
        v, omega = control

        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += omega * dt

        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        self.state = np.array([x, y, theta], dtype=float)

        return self.state
    
class BicycleModel: 

    def __init__(self, x=0, y=0, psi=0, v=0, lf=1.5, lr=1.5):
        self.state = np.array([x, y, psi, v], dtype=float)
        self.lf = lf
        self.lr = lr
    
    def step(self, control, dt):
        x, y, psi, v = self.state
        a, delta = control

        beta = np.arctan((self.lr / (self.lf + self.lr)) * np.tan(delta))
        theta = psi + beta

        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        psi += (v / self.lr) * np.sin(beta) * dt
        v += a * dt

        psi = (psi + np.pi) % (2 * np.pi) - np.pi
        self.state = np.array([x, y, psi, v], dtype=float)

        return self.state
    
    def dynamics(self, state, control):
        x, y, psi, v = state
        a, delta = control

        beta = np.arctan((self.lr / (self.lf + self.lr)) * np.tan(delta))
        theta = psi + beta

        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dpsi = (v / self.lr) * np.sin(beta)
        dv = a

        return np.array([dx, dy, dpsi, dv], dtype=float)
    
    def discrete_dynamics(self, state, control, dt):
        return state + self.dynamics(state, control) * dt

    def linearize(self, state, control, dt):
    # TODO: I should probably use torch here, but its numerical for now.
        EPSILON = 1e-5
        n_states = state.shape[0]
        n_controls = control.shape[0]

        A = np.zeros((n_states, n_states))
        B = np.zeros((n_states, n_controls))

        for i in range(n_states):
            dx = np.zeros(n_states)
            dx[i] = EPSILON
            f1 = self.discrete_dynamics(state + dx, control, dt)
            f2 = self.discrete_dynamics(state - dx, control, dt)
            A[:, i] = (f1 - f2) / (2 * EPSILON)

        for i in range(n_controls):
            du = np.zeros(n_controls)
            du[i] = EPSILON
            f1 = self.discrete_dynamics(state, control + du, dt)
            f2 = self.discrete_dynamics(state, control - du, dt)
            B[:, i] = (f1 - f2) / (2 * EPSILON)

        return A, B
    