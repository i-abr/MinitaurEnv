import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
from gym.spaces import Box

from quatmath import mat2euler, mat2quat, euler2quat

class MinitaurEnv(object):

    def __init__(self, frame_skip=5, viewer=True):

        self.frame_skip = frame_skip
        model_path = './minitaur_run.xml'

        self.model = load_model_from_path(model_path)
        self.sim   = MjSim(self.model, nsubsteps=frame_skip)
        self.data = self.sim.data
        if viewer:
            self.viewer = MjViewer(self.sim)

        self.chassis_bid = self.model.body_name2id('base_chassis_link')
        self.motor_bids  = [
                self.model.get_joint_qpos_addr('motor_front_leftL_joint'),
                self.model.get_joint_qpos_addr('motor_back_leftL_joint'),
                self.model.get_joint_qpos_addr('motor_front_rightR_joint'),
                self.model.get_joint_qpos_addr('motor_back_rightR_joint'),
            ]

        self.extension_bids  = [
                self.model.get_joint_qpos_addr('motor_front_leftL_extension'),
                self.model.get_joint_qpos_addr('motor_back_leftL_extension'),
                self.model.get_joint_qpos_addr('motor_front_rightR_extension'),
                self.model.get_joint_qpos_addr('motor_back_rightR_extension'),
            ]


        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (self.model.actuator_ctrlrange[:,1] - self.model.actuator_ctrlrange[:,0])
        nu = len(self.act_rng)
        self.action_space = Box(low=-1., high=1., shape=(nu,), dtype=np.float32)

        ob = self.get_obs(self.data)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(len(ob),), dtype=np.float32)

    def reset(self):
        self.sim.reset()
        ob = self.get_obs(self.data)
        return ob

    def get_obs(self, data):
        return np.concatenate([data.qpos.ravel(), data.qvel.ravel()])

    def print_status(self):
        # print('forward vel : {:.2f}'.format(self.data.qvel[0]))
        t_rpy = euler2quat(np.array([0., 0., 2.*np.pi]))
        rot = self.data.body_xquat[self.chassis_bid]

        print('rot : ', t_rpy, rot)

    def set_state_from_robot(self, measured_state):
        state = self.get_state()
        for i in range(4):
            state.qpos[self.motor_bids[i]] = measured_state.joint_swing[i]
            state.qpos[self.extension_bids[i]] = measured_state.joint_extension[i]
        state.qpos[:2] = measured_state.pose[:2]
        state.qpos[3:7] = measured_state.orientation
        self.sim.set_state(state)
        self.sim.forward()

    def get_rew(self, data):
        # rot = data.body_xmat[self.chassis_bid].reshape(3,3)
        rot = data.body_xquat[self.chassis_bid]
        # rpy = mat2quat(rot)
        rpy = rot
        # t_rpy = mat2quat(np.eye(3))
        t_rpy = np.array([1., 0., 0., 0.])
        # t_rpy = euler2quat(np.array([0., 2.*np.pi, 0.]))
        jnts = [data.qpos[jnt_addr] for jnt_addr in self.motor_bids]
        ext  = [data.qpos[jnt_addr] for jnt_addr in self.extension_bids]
        # return (data.qvel[0]-0.6)**2 + 10*np.sum((rpy-t_rpy)**2) #np.sum(data.qpos[:3]**2) #+ 10.*data.qpos[3]**2
        # return -data.qvel[2]# + 100*np.sum((rpy-t_rpy)**2)
        return -(data.qpos[0]-0.5)**2-np.sum((rpy-t_rpy)**2) #+ data.qvel[2]**2# + 1e-3*np.sum(np.square(jnts)) + 1e-3*np.sum(np.square(ext))

    def step(self, a):
        ctrl = np.clip(a, -1.0, 1.0)
        ctrl = self.act_mid + ctrl * self.act_rng
        self.data.ctrl[:] = ctrl
        self.sim.step()
        ob = self.get_obs(self.data)
        rew = self.get_rew(self.data)
        done = False
        return ob, rew, done, {}

    def render(self):
        self.viewer.render()

    def get_state(self):
        return self.sim.get_state()
