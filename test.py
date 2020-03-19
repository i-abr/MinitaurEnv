#!/usr/bin/env python3


import numpy as np

from minitaur_env import MinitaurEnv
from minitaur_multmodel import MinitaurMultModel
from scipy.signal import savgol_filter
import sys
sys.path.append('../')

import matplotlib.pyplot as plt

def mppi(state, model, u_seq, horizon, lam=0.2, sig=0.6):
    assert len(u_seq) == horizon

    model.set_state(state)

    s   = []
    eps = []
    eta = 0.
    alpha = 0.4
    for t in range(horizon):
        eta = alpha * eta + (1-alpha) * np.random.normal(0., sig, size=(model.n_sims, model.action_space.shape[0]))
        eps.append(eta)
        obs, rew, done, _ = model.step(u_seq[t] + eps[-1])
        s.append(rew)

    s = np.cumsum(s[::-1], 0)[::-1, :]

    for t in range(horizon):
        s[t] -= np.max(s[t])
        w = np.exp(s[t]/lam) + 1e-8 # offset
        w /= np.sum(w)
        u_seq[t] = u_seq[t] + np.dot(w, eps[t])
    # return savgol_filter(u_seq, horizon-1, 3, axis=0)
    return u_seq

env = MinitaurEnv()
model = MinitaurMultModel(n_sims=50)


env.reset()
model.reset()

horizon = 10

num_actions = env.action_space.shape[0]

u_seq = [np.zeros(model.action_space.shape[0]) for _ in range(horizon)]

t = 0
# plt.ion()

while True:

    state = env.get_state()
    u_seq = mppi(state, model, u_seq, horizon)

    # obs, rew, done, _ = env.step(u_seq[0]*0 + np.array([0.]*4 + [-0]*4) )
    # obs, rew, done, _ = env.step(u_seq[0]*0 + np.sin(np.pi*t/30))
    #obs, rew, done, _ = env.step(env.action_space.sample()*0)
    obs, rew, done, _ = env.step(u_seq[0])
    env.render()
    u_seq[:-1] = u_seq[1:]
    u_seq[-1]  = np.zeros(env.action_space.shape[0])
    # plt.clf()
    # plt.plot(u_seq)
    # plt.draw()
    # plt.pause(0.001)
    # print(env.sim.data.qpos[:3])

    t += 1

    env.print_status()
