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
        a_a = a * 0.0001
    else:
        a_a = a * 0.01
    return a_a

