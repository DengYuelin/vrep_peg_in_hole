import numpy as np
import algorithms.calculations as cal
from support_files import vrep
import time
import copy as cp
from gym import spaces


class ArmEnv(object):

    def __init__(self, step_max=100, fuzzy=False, add_noise=False):

        self.observation_dim = 9
        self.action_dim = 6

        """ state """
        self.state = np.zeros(self.observation_dim)
        self.init_state = np.zeros(self.observation_dim)

        """ action """
        self.action_high_bound = 1
        self.action = np.zeros(self.action_dim)
        self.fuzzy_control = fuzzy

        """ reward """
        self.step_max = step_max
        self.step_max_pos = 15
        self.reward = 1.

        """setting"""
        self.add_noise = add_noise  # or True
        self.pull_terminal = False

        '''Wrong'''
        self.state_high = np.array([50, 50, 0, 5, 5, 6, 1453, 70, 995, 5, 5, 6])
        self.state_low = np.array([-50, -50, -50, -5, -5, -6, 1456, 76, 985, -5, -5, -6])
        self.terminated_state = np.array([30, 30, 30, 2, 2, 2])
        self.observation_space = spaces.Box(low=self.state_low, high=self.state_high,
                                            shape=(self.observation_dim,), dtype=np.float32)
        
        """information for action and state"""
        self.terminated_state = np.array([30, 30, 30, 2, 2, 2])
        self.action_space = spaces.Box(low=-1, high=1,
                                       shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10, high=10,
                                            shape=(self.observation_dim,), dtype=np.float32)

        """fuzzy parameters"""
        # self.fc = fuzzy_control(low_output=np.array([0., 0., 0., 0., 0., 0.]),
        #                         high_output=np.array([0.03, 0.03, 0.004, 0.03, 0.03, 0.03]))

        
        '''vrep init session''' 
        print('Program started')
        vrep.simxFinish(-1)  # just in case, close all opened connections
        # Connect to V-REP, get clientID
        self.clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)  # enter server here
        # the server ID should be consistent with the ID listed in remoteApiConnections.txt, which can be found under
        # the V-REP installation folder
        vrep.c_Synchronous(self.clientID, True)

        if self.clientID != -1:  # confirm connection
            print('Connected to remote API server')

        else:
            exit('Failed connecting to remote API server')

        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_oneshot)

        vrep.simxSetIntegerSignal(self.clientID, "Apimode", 1, vrep.simx_opmode_oneshot)  # activate apimode

        # vrep sensor setup
        # Setup the force sensor
        self.errorCode, self.force_sensor_handle = vrep.simxGetObjectHandle(self.clientID, 'IRB140_connection',
                                                                            vrep.simx_opmode_blocking)
        self.errorCode, self.target_handle = vrep.simxGetObjectHandle(self.clientID, 'Target',
                                                                            vrep.simx_opmode_blocking)
        print("IRB140_connection", self.errorCode, self.force_sensor_handle)
        print("Target", self.errorCode, self.target_handle)
        self.errorCode, self.forceState, self.forceVector, self.torqueVector = \
            vrep.simxReadForceSensor(self.clientID, self.force_sensor_handle, vrep.simx_opmode_streaming)
        self.errorCode, self.position = \
            vrep.simxGetObjectPosition(self.clientID, self.force_sensor_handle, self.target_handle, vrep.simx_opmode_streaming)
        while self.errorCode:
            self.errorCode, self.position = vrep.simxGetObjectPosition(self.clientID, self.force_sensor_handle, self.target_handle,
                                                                            vrep.simx_opmode_buffer)
        print("init force sensor IRB140_connection", self.position, self.forceState)

        # Get hole position info
        self.errorCode, self.hole_handle = vrep.simxGetObjectHandle(self.clientID, 'Hole', vrep.simx_opmode_blocking)
        self.errorCode, self.init_position = vrep.simxGetObjectPosition(self.clientID, self.hole_handle, -1,
                                                                        vrep.simx_opmode_streaming)
        while self.errorCode:
            self.errorCode, self.init_position = vrep.simxGetObjectPosition(self.clientID, self.hole_handle, -1,
                                                                            vrep.simx_opmode_buffer)

        self.errorCode, self.init_orientation = vrep.simxGetObjectOrientation(self.clientID, self.hole_handle, -1,
                                                                              vrep.simx_opmode_streaming)
        while self.errorCode:
            self.errorCode, self.init_orientation = vrep.simxGetObjectOrientation(self.clientID, self.hole_handle, -1,
                                                                                  vrep.simx_opmode_buffer)
        print("init position of hole", self.init_position, self.init_orientation)

        vrep.simxFinish(-1)
        self.clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_oneshot)
        # Get Joint data
        self.Joints = np.zeros((6, 2))
        self.Joint_boundary = np.zeros((6, 2))
        for i in range(6):
            for j in range(2):
                self.errorCode, self.Joints[i][j] = vrep.simxGetFloatSignal(self.clientID,
                                                                            'Interval_{}_{}'.format(i + 1, j + 1),
                                                                            vrep.simx_opmode_streaming)
                while self.errorCode:
                    self.errorCode, self.Joints[i][j] = vrep.simxGetFloatSignal(self.clientID,
                                                                                'Interval_{}_{}'.format(i + 1, j + 1),
                                                                                vrep.simx_opmode_buffer)
                # print(self.errorCode,'Interval_{}_{}'.format(i+1,j+1),self.Joints[i][j])
            self.Joint_boundary[i] = [(self.Joints[i][0] / np.pi * 180),
                                      ((self.Joints[i][0] / np.pi * 180) + (self.Joints[i][1] / np.pi * 180))]
            print("Joint boundary ", i, self.Joint_boundary[i])

        # Setup controllable variables
        self.movementMode = 1  # work under FK(0) or IK(1)

        self.FK = np.zeros(1, dtype=[('Joint1', np.float32), ('Joint2', np.float32), ('Joint3', np.float32),
                                     ('Joint4', np.float32), ('Joint5', np.float32), ('Joint6', np.float32)])
        # initial angle in FK mode
        self.FK['Joint1'] = 0
        self.FK['Joint2'] = 0
        self.FK['Joint3'] = 0
        self.FK['Joint4'] = 0
        self.FK['Joint5'] = -90
        self.FK['Joint6'] = 0

        self.IK = np.zeros(1, dtype=[('Pos_x', np.float32), ('Pos_y', np.float32), ('Pos_z', np.float32),
                                     ('Alpha', np.float32), ('Beta', np.float32), ('Gamma', np.float32)])
        # initial position in IK mode
        self.IK['Pos_x'] = 0
        self.IK['Pos_y'] = 0
        self.IK['Pos_z'] = 0
        self.IK['Alpha'] = 0  # x
        self.IK['Beta'] = 0  # y
        self.IK['Gamma'] = 0  # z

        self.reset()
        # Auxiliary variables

    def step(self, action):
        # set FK or IK
        vrep.simxSetIntegerSignal(self.clientID, "movementMode", self.movementMode, vrep.simx_opmode_oneshot)
        # read force sensor
        self.errorCode, self.forceState, self.forceVector, self.torqueVector = \
            vrep.simxReadForceSensor(self.clientID, self.force_sensor_handle, vrep.simx_opmode_buffer)
        self.errorCode, self.position = \
            vrep.simxGetObjectPosition(self.clientID, self.force_sensor_handle, self.target_handle, vrep.simx_opmode_buffer)

        # calculations
        # adjust action to usable motion
        action = cal.actions(action, self.movementMode)

        # take actions
        if self.movementMode:  # in IK mode
            # do action
            self.IK['Pos_x'] += action[0]
            self.IK['Pos_y'] += action[1]
            self.IK['Pos_z'] += action[2]
            self.IK['Alpha'] += action[4]
            self.IK['Beta'] += action[3]
            self.IK['Gamma'] += action[5]

            # send signal
            vrep.simxSetFloatSignal(self.clientID, "pos_X", self.IK['Pos_x'], vrep.simx_opmode_oneshot)
            vrep.simxSetFloatSignal(self.clientID, "pos_Y", self.IK['Pos_y'], vrep.simx_opmode_oneshot)
            vrep.simxSetFloatSignal(self.clientID, "pos_Z", self.IK['Pos_z'], vrep.simx_opmode_oneshot)
            vrep.simxSetFloatSignal(self.clientID, "Alpha", self.IK['Alpha'], vrep.simx_opmode_oneshot)
            vrep.simxSetFloatSignal(self.clientID, "Beta", self.IK['Beta'], vrep.simx_opmode_oneshot)
            vrep.simxSetFloatSignal(self.clientID, "Gamma", self.IK['Gamma'], vrep.simx_opmode_oneshot)
            time.sleep(0.1)  # wait for action to finish
        else:  # in FK mode
            # do action
            self.FK['Joint1'] += action[0]
            self.FK['Joint2'] += action[1]
            self.FK['Joint3'] += action[2]
            self.FK['Joint4'] += action[3]
            self.FK['Joint5'] += action[4]
            self.FK['Joint6'] += action[5]

            # boundary
            self.FK['Joint1'] = np.clip(self.FK['Joint1'], *self.Joint_boundary[0])
            self.FK['Joint2'] = np.clip(self.FK['Joint2'], *self.Joint_boundary[1])
            self.FK['Joint3'] = np.clip(self.FK['Joint3'], *self.Joint_boundary[2])
            self.FK['Joint4'] = np.clip(self.FK['Joint4'], *self.Joint_boundary[3])
            self.FK['Joint5'] = np.clip(self.FK['Joint5'], *self.Joint_boundary[4])
            self.FK['Joint6'] = np.clip(self.FK['Joint6'], *self.Joint_boundary[5])

            # send signal
            vrep.simxSetFloatSignal(self.clientID, "Joint1",
                                    (self.FK['Joint1'] * np.pi / 180 - self.Joints[0][0]) / self.Joints[0][1] * 1000,
                                    vrep.simx_opmode_oneshot)
            vrep.simxSetFloatSignal(self.clientID, "Joint2",
                                    (self.FK['Joint2'] * np.pi / 180 - self.Joints[1][0]) / self.Joints[1][1] * 1000,
                                    vrep.simx_opmode_oneshot)
            vrep.simxSetFloatSignal(self.clientID, "Joint3",
                                    (self.FK['Joint3'] * np.pi / 180 - self.Joints[2][0]) / self.Joints[2][1] * 1000,
                                    vrep.simx_opmode_oneshot)
            vrep.simxSetFloatSignal(self.clientID, "Joint4",
                                    (self.FK['Joint4'] * np.pi / 180 - self.Joints[3][0]) / self.Joints[3][1] * 1000,
                                    vrep.simx_opmode_oneshot)
            vrep.simxSetFloatSignal(self.clientID, "Joint5",
                                    (self.FK['Joint5'] * np.pi / 180 - self.Joints[4][0]) / self.Joints[4][1] * 1000,
                                    vrep.simx_opmode_oneshot)
            vrep.simxSetFloatSignal(self.clientID, "Joint6",
                                    (self.FK['Joint6'] * np.pi / 180 - self.Joints[5][0]) / self.Joints[5][1] * 1000,
                                    vrep.simx_opmode_oneshot)
            # time.sleep(0.01)  # wait for action to finish

        # print(self.position)
        # state
        self.state = np.concatenate((self.position, self.forceVector, self.torqueVector))

        # done and reward
        r, done = cal.reword(self.state)

        # safety check
        safe = cal.safetycheck(self.state)

        return self.code_state(self.state), self.state, r, done, safe

    def reset(self):

        '''Need to try if this can be removed'''
        # read force sensor
        self.errorCode, self.forceState, self.forceVector, self.torqueVector = \
            vrep.simxReadForceSensor(self.clientID, self.force_sensor_handle, vrep.simx_opmode_buffer)
        self.errorCode, self.position = \
            vrep.simxGetObjectPosition(self.clientID, self.force_sensor_handle, self.target_handle, vrep.simx_opmode_buffer)

        # restart scene
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_oneshot)
        time.sleep(1)  # must wait until stop command is finished
        vrep.simxFinish(-1)  # end all communications
        vrep.c_Synchronous(self.clientID, True)
        self.clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)  # restart communication to the server
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_oneshot)  # start simulation
        # Setup the force sensor
        self.errorCode, self.force_sensor_handle = vrep.simxGetObjectHandle(self.clientID, 'IRB140_connection',
                                                                            vrep.simx_opmode_blocking)
        self.errorCode, self.forceState, self.forceVector, self.torqueVector = \
            vrep.simxReadForceSensor(self.clientID, self.force_sensor_handle, vrep.simx_opmode_streaming)
        self.errorCode, self.position = \
            vrep.simxGetObjectPosition(self.clientID, self.force_sensor_handle, self.target_handle, vrep.simx_opmode_streaming)
        print("***********************scene rested***********************")
        vrep.simxSetIntegerSignal(self.clientID, "Apimode", 1, vrep.simx_opmode_oneshot)

        # set random hole position
        new_position = self.init_position.copy()
        new_orientation = self.init_orientation.copy()
        new_position[0] += (np.random.rand(1) - 0.5) * 0.002
        new_position[1] += (np.random.rand(1) - 0.5) * 0.002
        new_position[0] += 0
        new_orientation[0] += (np.random.rand(1) - 0.5) * 0.04
        new_orientation[1] += (np.random.rand(1) - 0.5) * 0.04
        new_orientation[2] += (np.random.rand(1) - 0.5) * 0.04
        vrep.simxSetObjectPosition(self.clientID, self.hole_handle, -1, new_position, vrep.simx_opmode_oneshot)
        vrep.simxSetObjectOrientation(self.clientID, self.hole_handle, -1, new_orientation, vrep.simx_opmode_oneshot)

        # reset signals
        if self.movementMode:  # in IK mode
            self.IK['Pos_x'] = 0
            self.IK['Pos_y'] = 0
            self.IK['Pos_z'] = 0
            self.IK['Alpha'] = 0
            self.IK['Beta'] = 0
            self.IK['Gamma'] = 0
            # send signal
            vrep.simxSetFloatSignal(self.clientID, "pos_X", self.IK['Pos_x'], vrep.simx_opmode_oneshot)
            vrep.simxSetFloatSignal(self.clientID, "pos_Y", self.IK['Pos_y'], vrep.simx_opmode_oneshot)
            vrep.simxSetFloatSignal(self.clientID, "pos_Z", self.IK['Pos_z'], vrep.simx_opmode_oneshot)
            vrep.simxSetFloatSignal(self.clientID, "Alpha", self.IK['Alpha'], vrep.simx_opmode_oneshot)
            vrep.simxSetFloatSignal(self.clientID, "Beta", self.IK['Beta'], vrep.simx_opmode_oneshot)
            vrep.simxSetFloatSignal(self.clientID, "Gamma", self.IK['Gamma'], vrep.simx_opmode_oneshot)
        else:
            self.FK['Joint1'] = 0
            self.FK['Joint2'] = 0
            self.FK['Joint3'] = 0
            self.FK['Joint4'] = 0
            self.FK['Joint5'] = -90
            self.FK['Joint6'] = 0
            # send signal
            vrep.simxSetFloatSignal(self.clientID, "Joint1",
                                    (self.FK['Joint1'] * np.pi / 180 - self.Joints[0][0]) / self.Joints[0][1] * 1000,
                                    vrep.simx_opmode_oneshot)
            vrep.simxSetFloatSignal(self.clientID, "Joint2",
                                    (self.FK['Joint2'] * np.pi / 180 - self.Joints[1][0]) / self.Joints[1][1] * 1000,
                                    vrep.simx_opmode_oneshot)
            vrep.simxSetFloatSignal(self.clientID, "Joint3",
                                    (self.FK['Joint3'] * np.pi / 180 - self.Joints[2][0]) / self.Joints[2][1] * 1000,
                                    vrep.simx_opmode_oneshot)
            vrep.simxSetFloatSignal(self.clientID, "Joint4",
                                    (self.FK['Joint4'] * np.pi / 180 - self.Joints[3][0]) / self.Joints[3][1] * 1000,
                                    vrep.simx_opmode_oneshot)
            vrep.simxSetFloatSignal(self.clientID, "Joint5",
                                    (self.FK['Joint5'] * np.pi / 180 - self.Joints[4][0]) / self.Joints[4][1] * 1000,
                                    vrep.simx_opmode_oneshot)
            vrep.simxSetFloatSignal(self.clientID, "Joint6",
                                    (self.FK['Joint6'] * np.pi / 180 - self.Joints[5][0]) / self.Joints[5][1] * 1000,
                                    vrep.simx_opmode_oneshot)
        # state
        # wait for the environment to stabilize
        time.sleep(0.5)
        # read force sensor
        self.errorCode, self.forceState, self.forceVector, self.torqueVector = \
            vrep.simxReadForceSensor(self.clientID, self.force_sensor_handle, vrep.simx_opmode_buffer)
        self.errorCode, self.position = \
            vrep.simxGetObjectPosition(self.clientID, self.force_sensor_handle, self.target_handle, vrep.simx_opmode_buffer)
        self.init_state = np.concatenate((self.position, self.forceVector, self.torqueVector))
        print('initial state :::::', self.init_state)
        done = False
        return self.code_state(self.init_state), self.init_state, done

    @staticmethod
    def sample_action():
        return np.random.rand(6) - 0.5

    """ normalize state """
    def code_state(self, current_state):
        state = cp.deepcopy(current_state)

        """normalize the state"""
        scale = 1
        final_state = state / scale

        return final_state


# input random action to the robot
if __name__ == '__main__':
    env = ArmEnv()
    while True:
        for i in range(30):
            a = env.sample_action()
            env.step(a)
        env.reset()
