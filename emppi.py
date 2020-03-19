#!/usr/bin/env python3

import numpy as np

from minitaur_env import MinitaurEnv
from minitaur_multmodel import MinitaurMultModel
from scipy.signal import savgol_filter
import sys
sys.path.append('../')

import lcm
from exlcm import MinitaurState, MinitaurCommand
import select

class EMPPI(object):

    def __init__(self):
        self.lc = lcm.LCM()
        self.timeout = 0.01
        self.__base_model = MinitaurEnv()
        self.model = MinitaurMultModel(n_sims=10)
        self.subs = [
            self.lc.subscribe('minitaur_state', self.callback)
        ]
        self.subs[0].set_queue_capacity(1)
        self.minitaur_state = MinitaurState()
        self.minitaur_command = MinitaurCommand()
        self.__base_model.reset()
        self.__base_state = self.__base_model.get_state()
        self.horizon = 10
        self.u = [np.zeros(self.model.action_space.shape[0]) for _ in range(self.horizon)]
        self.__sample_size = (self.model.n_sims, self.model.action_space.shape[0])
        self.__lam = 0.2
        self.__sig = 0.2
    def callback(self, channel, data):
        self.minitaur_state = MinitaurState.decode(data)

    def get_action(self, measured_state):
        self.model.set_state(measured_state, self.__base_state)
        eps = []
        s = []
        for t in range(self.horizon):
            eps.append(np.random.normal(0., self.__sig, self.__sample_size))
            obs, rew, done, _ = self.model.step(self.u[t] + eps[-1])
            s.append(rew)
        s = np.cumsum(s[::-1], 0)[::-1, :]
        for t in range(self.horizon):
            s[t] -= np.max(s[t])
            w = np.exp(s[t]/self.__lam) + 1e-8
            w /= np.sum(w)
            self.u[t] = self.u[t] + np.dot(w, eps[t])
        return self.u[0]
    def loop(self):

        while True:
            rfds, wfds, efds = select.select([self.lc.fileno()], [], [], self.timeout)
            if rfds:
                self.lc.handle()
                self.__base_model.set_state_from_robot(self.minitaur_state)
                self.__base_state = self.__base_model.get_state()
                self.__base_model.step(self.__base_model.action_space.sample()*0.)
                self.__base_model.render()
                u = self.get_action(self.minitaur_state)
                u_converted = self.__base_model.act_mid + u * self.__base_model.act_rng
                self.minitaur_command.joint_swing = u_converted[:4]
                self.minitaur_command.joint_extension = u_converted[4:]
                self.lc.publish('minitaur_command', self.minitaur_command.encode())
if __name__=='__main__':
    emppi = EMPPI()
    emppi.loop()
