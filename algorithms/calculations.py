# Setting according to the defintion of problem
import numpy as np


# boundary for action out put
action_bound = [-20, 20]


def reword(s):
    r = -(abs(s[0]) + abs(s[1]) + abs(s[2]))*10 - (abs(s[3]) + abs(s[4]) + abs(s[5] + 1.5))
    if abs(s[0]) < 0.001 and abs(s[1]) < 0.001 and abs(s[2]) < 0.001:
        done = True
    else:
        done = False
    return r, done


# this function adjust the output of the network in to usable actions
def actions(a, mode):  # here a ∈ action_bound
    if mode:
        a_a = a[0] * 0.001
    else:
        a_a = a[0] * 0.01
    return a_a


# this function checks if the force and torque extends safety value
def safetycheck(s):
    if s[3] >= 100 or s[4]>= 100 or s[5] >= 100:
        return False
    elif s[6] >= 100 or s[7]>= 100 or s[8] >= 100:
        return False
    else:
        return True

