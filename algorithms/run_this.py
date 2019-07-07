from envs.vrepenv import ArmEnv
from algorithms.PD import PD
import numpy as np
import time

MAX_EPISODES = 900
MAX_EP_STEPS = 1000
ON_TRAIN = True

# set env
env = ArmEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

# set RL method
rl = PD()

steps = []


def train():
    # start training
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_r = 0.
        for j in range(MAX_EP_STEPS):

            a = rl.cal(s, np.array([0, 0, -40, 0, 0]))
            s, r, done, safe = env.step(a)
            # time.sleep(1)
            print("force", s[3:6], "action", a[0:3])
            if done or j == MAX_EP_STEPS - 1 or safe is False:
                print('Ep: %i | %s | %s | ep_r: %.1f | step: %i' % (i, '---' if not done else 'done', 'unsafe' if not safe else 'safe', ep_r, j))
                break


# not used now
# def eval():


if ON_TRAIN:
    train()
else:
    eval()
