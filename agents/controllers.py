import abc

import control
import numpy as np

from agents.virtual_pid import PID


class Controller(abc.ABC):
    @abc.abstractmethod
    def get_action(self, t, state):
        raise NotImplementedError("get_action")


class LqrController(Controller):
    def __init__(self, env):
        self.env = env

    def get_action(self, t, state):
        A = np.array(
            [[0, 1, 0, 0],
             [0, 0, -self.env.masspole * self.env.gravity / self.env.masscart,
              0],
             [0, 0, 0, 1],
             [0, 0, self.env.total_mass * self.env.gravity /
              (self.env.length * self.env.masscart), 0]
             ])
        B = np.array(
            [[0],
             [1 / self.env.masscart],
             [0],
             [-1 / (self.env.length * self.env.masscart)]])
        Q = 10 * np.eye(4)
        R = 0.01
        K, _, _ = control.lqr(A, B, Q, R)
        action = - np.dot(K[0].T, state)
        return action


class PidController(Controller):
    def __init__(self):
        self.pid = PID(1, 0.1, 0.05, setpoint=0)

    def get_action(self, t, state):
        return self.pid(t, state[0])
