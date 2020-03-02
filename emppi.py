#!/usr/bin/env python3

import numpy as np

from minitaur_env import MinitaurEnv
from minitaur_multmodel import MinitaurMultModel
from scipy.signal import savgol_filter
import sys
sys.path.append('../')

import lcm
from exlcm import MinitaurState
import select

class EMPPI(object):

    def __init__(self):
        self.lc = lcm.LCM()
        self.timeout = 0.01
        self.model = MinitaurEnv()
        self.subs = [
            self.lc.subscribe('minitaur_state', self.callback)
        ]
        self.subs[0].set_queue_capacity(1)
        self.minitaur_state = MinitaurState()
        self.model.reset()
    def callback(self, channel, data):
        self.minitaur_state = MinitaurState.decode(data)

    def loop(self):
        while True:
            rfds, wfds, efds = select.select([self.lc.fileno()], [], [], self.timeout)
            if rfds:
                self.lc.handle()
                self.model.set_state_from_robot(self.minitaur_state)
                self.model.step(self.model.action_space.sample()*0)
                self.model.render()
if __name__=='__main__':
    emppi = EMPPI()
    emppi.loop()
