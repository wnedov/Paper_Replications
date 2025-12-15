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

    def __init__(self, x=0, y=0, psi=0, v=0, lf=1.105, lr=1.738):
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
    
class DynamicBicycleModel(BicycleModel):
    def __init__(self, x=0, y=0, psi=0, vx=0, vy=0, r=0, lf=1.105, lr=1.738, tyre_model='linear'):
        self.state = np.array([x, y, psi, vx, vy, r], dtype=float)
        self.lr = lr 
        self.lf = lf 
        self.tyre_model = tyre_model

        self.m = 1650.0      
        self.Iz = 3000.0     
        self.g = 9.81

        # These are PER TIRE. The step function sums them or multiplies by 2.
        self.Cf = 60000.0   
        self.Cr = 90000.0   

    def step(self, control, dt):
        x, y, psi, vx, vy, r = self.state
        ax, delta = control

        if abs(vx) > 0.1:
            alpha_f = np.arctan2((vy + self.lf * r), vx) - delta
            alpha_r = np.arctan2((vy - self.lr * r), vx)
        else:
            alpha_f = 0.0
            alpha_r = 0.0
        
        Fcf, Fcr = self.tire_forces(alpha_f, alpha_r)

        vx_dot = r * vy + ax 
        vy_dot = -r*vx + 2/self.m * (Fcf * np.cos(delta) + Fcr)
        r_dot = 2/self.Iz * (self.lf * Fcf - self.lr * Fcr)
        x_dot = vx * np.cos(psi) - vy * np.sin(psi)
        y_dot = vx * np.sin(psi) + vy * np.cos(psi)

        x += x_dot * dt
        y += y_dot * dt
        psi += r * dt
        vx += vx_dot * dt
        vy += vy_dot * dt
        r += r_dot * dt

        psi = (psi + np.pi) % (2 * np.pi) - np.pi
        self.state = np.array([x, y, psi, vx, vy, r], dtype=float)
        return self.state
    
    def dynamics(self, state, control):
        x, y, psi, vx, vy, r = state
        ax, delta = control

        if abs(vx) > 0.1:
            alpha_f = np.arctan2((vy + self.lf * r), vx) - delta
            alpha_r = np.arctan2((vy - self.lr * r), vx)
        else:
            alpha_f = 0.0
            alpha_r = 0.0
        
        Fcf, Fcr = self.tire_forces(alpha_f, alpha_r)

        d_vx = r * vy + ax 
        d_vy = -r*vx + 2/self.m * (Fcf * np.cos(delta) + Fcr)
        d_rdot = 2/self.Iz * (self.lf * Fcf - self.lr * Fcr)
        dx = vx * np.cos(psi) - vy * np.sin(psi)
        dy = vx * np.sin(psi) + vy * np.cos(psi)

        return np.array([dx, dy, r, d_vx, d_vy, d_rdot], dtype=float)

    def tire_forces(self, alpha_f, alpha_r):
        # Normalize tire model name to lowercase for robustness
        tire_model = self.tyre_model.lower()
        
        if tire_model == 'linear':
            Fcf = -self.Cf * alpha_f
            Fcr = -self.Cr * alpha_r
        elif tire_model == 'pacejka':
            Fz_f = (self.m * 9.81 * self.lr) / (self.lf + self.lr) / 2
            Fz_r = (self.m * 9.81 * self.lf) / (self.lf + self.lr) / 2
            mu = 1.0
            B = 10.0
            C = 1.3
            Df = mu * Fz_f
            Dr = mu * Fz_r
            E = 0.97
            Fcf = -Df * np.sin(C * np.arctan(B * alpha_f - E * (B * alpha_f - np.arctan(B * alpha_f))))
            Fcr = -Dr * np.sin(C * np.arctan(B * alpha_r - E * (B * alpha_r - np.arctan(B * alpha_r))))
        else:
            raise ValueError(f"Unknown tire model: '{self.tyre_model}'. Use 'linear' or 'pacejka'.")
        
        return Fcf, Fcr