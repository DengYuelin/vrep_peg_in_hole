import numpy as np


# boundary for action
action_bound = [-20, 20]


def reword(s):
    r = 0
    done = 0
    return r, done


# this function adjust the output of the network in to usable actions
def actions(a, mode):  # here a ∈ action_bound
    if mode:
        a_a = a * 0.001
    else:
        a_a = a * 0.01
    return a_a


# this function checks if the force and torque extends safety value
def safetycheck(s):
    if s[3] >= 100 or s[4]>= 100 or s[5] >= 100:
        return False
    elif s[6] >= 100 or s[7]>= 100 or s[8] >= 100:
        return False
    else:
        return True

