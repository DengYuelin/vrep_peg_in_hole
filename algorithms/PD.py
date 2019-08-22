import numpy as np


class PD:
    def __init__(self):
        self.kp = 0.1
        self.kd = 0.1
        self.uk = np.array([0, 0, 0, 0, 0])
        self.uk_1 = np.array([0, 0, 0, 0, 0])
        self.yk = np.array([0, 0, 0, 0, 0])
        self.ek = np.array([0, 0, 0, 0, 0])
        self.ek_1 = np.array([0, 0, 0, 0, 0])
        self.ek_2 = np.array([0, 0, 0, 0, 0])

    def cal(self, s, rk):
        yk = np.array([s[3], s[4], s[5], s[6], s[7]])
        self.ek = rk - yk
        # discrete PD algorithm
        self.uk = self.uk_1 + self.kp * (self.ek - self.ek_1) + self.kd * (self.ek - 2 * self.ek_1 + self.ek_2)
        # renew variables
        self.ek_2 = self.ek_1
        self.ek_1 = self.ek
        self.uk_1 = self.uk
        action = np.append(self.uk, 0)
        action[0] = - action[0]
        action[3] = action[3] * 10
        action[4] = action[4] * 10
        return action
