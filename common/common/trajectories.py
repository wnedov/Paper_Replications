import numpy as np


class ReferenceGhost:

    def __init__(self, startx= 0, starty = 0, starttheta = 0, scale=1, T=1):
        self.start_x = startx
        self.start_y = starty
        self.start_theta = starttheta
        self.omega = 2 * np.pi / T
        self.scale = scale
        self.L = 3

    def refState(self, t):
        xc = self.scale * 1.5 * np.cos(self.omega * t)
        yc = self.scale * 0.7 * np.sin(self.omega * 2*t)
        
        xdot = -1.5 * self.scale * self.omega * np.sin(self.omega * t)
        ydot = 1.4 * self.scale * self.omega * np.cos(self.omega * 2*t)
        xdott = -1.5 * self.scale * self.omega**2 * np.cos(self.omega * t);
        ydott = -2.8 * self.scale * self.omega**2 * np.sin(self.omega * 2*t);

        c_th = np.cos(self.start_theta)
        s_th = np.sin(self.start_theta)
        rot_matrix = np.array([[c_th, -s_th], 
                               [s_th,  c_th]])
        
        pos_rotated = rot_matrix @ np.array([xc, yc])
        vel_rotated = rot_matrix @ np.array([xdot, ydot])

        xr, yr = pos_rotated + np.array([self.start_x, self.start_y])
        vr = np.sqrt(vel_rotated[0]**2 + vel_rotated[1]**2)
        thetar = np.atan2(vel_rotated[1], vel_rotated[0])
        omegar = (xdot * ydott - ydot * xdott) / (xdot**2 + ydot**2)


        return np.array([xr, yr, thetar, vr, omegar], dtype=float)
    
    def get_horizion(self, t_start, dt, N):

        xref = np.zeros((4, N + 1))
        uref = np.zeros((2, N))

        for k in range(N + 1):
            t = t_start + k * dt

            state = self.refState(t)
            xref[:, k] = state[0:4]

            if k < N:

                v = state[3]
                omega = state[4]

                steerref = np.arctan2(omega * self.L, v)
                state_next = self.refState(t + dt)
                v_next = state_next[3]
                    
                a_ref = (v_next - v) / dt

            
                uref[:, k] = np.array([a_ref, steerref])
        
        return xref, uref

    
