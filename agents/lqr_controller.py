import control
import numpy as np


class LqrController():
    def __init__(self, env):
        self.env = env

    def step(self, observation):
        A = np.array(
            [[0, 1, 0, 0], [0, 0, -self.masspole * self.gravity /
                            self.masscart, 0], [0, 0, 0, 1],
             [0, 0, self.total_mass * self.gravity / (self.length *
                                                      self.masscart), 0]])
        B = np.array([[0], [1 / self.masscart], [0], [-1 / (self.length *
                                                            self.masscart)]])
        Q = 10 * np.eye(4)
        R = 0.01
        K, _, _ = control.lqr(A, B, Q, R)
        action = -K * observation
        return action
