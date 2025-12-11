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
    