#!/usr/bin/env python

import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
import sys
sys.path.append('../')

from exlcm import MinitaurState
import lcm


class RobotListener(object):

    def __init__(self):
        self.lc = lcm.LCM()
        self.minitaur_state = MinitaurState()
        self.subs = [
            rospy.Subscriber('/robot0/state/pose', Pose, self.pose_callback),
            rospy.Subscriber('/robot0/state/jointURDF', JointState, self.joint_callback)
        ]
        rospy.init_node('robot_env_interface')
        self.rate = rospy.Rate(10)
    def pose_callback(self, data):
        self.minitaur_state.orientation[0] = data.orientation.x
        self.minitaur_state.orientation[1] = data.orientation.y
        self.minitaur_state.orientation[2] = data.orientation.z
        self.minitaur_state.orientation[3] = data.orientation.w

    def joint_callback(self, data):
        self.minitaur_state.joint_swing[0] = data.position[0]
        self.minitaur_state.joint_swing[1] = data.position[1]
        self.minitaur_state.joint_swing[2] = data.position[2]
        self.minitaur_state.joint_swing[3] = data.position[3]

        self.minitaur_state.joint_extension[0] = data.position[4]
        self.minitaur_state.joint_extension[1] = data.position[5]
        self.minitaur_state.joint_extension[2] = data.position[6]
        self.minitaur_state.joint_extension[3] = data.position[7]

    def loop(self):
        while not rospy.is_shutdown():
            self.lc.publish('minitaur_state', self.minitaur_state.encode())
            self.rate.sleep()

if __name__ == '__main__':
    robot_listener = RobotListener()
    robot_listener.loop()
