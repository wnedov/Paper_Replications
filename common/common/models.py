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
    