from envs.vrepenv import ArmEnv
from algorithms.PD import PD
import numpy as np

MAX_EPISODES = 900
MAX_EP_STEPS = 200
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

            s, r, done = env.step(a)

            if done or j == MAX_EP_STEPS - 1:
                print('Ep: %i | %s | ep_r: %.1f | step: %i' % (i, '---' if not done else 'done', ep_r, j))
                break


# not used now
#def eval():



if ON_TRAIN:
    train()
else:
    eval()
