#!/usr/bin/env python

import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
import tf
from tf import transformations as trans
import sys
sys.path.append('../')

from exlcm import MinitaurState, MinitaurCommand
import lcm
import select


class RobotListener(object):

    def __init__(self):
        rospy.init_node('robot_env_interface')
        self.lc = lcm.LCM()
        self.timeout = 0.01
        self.minitaur_state = MinitaurState()
        self.minitaur_command = JointState()
        self.minitaur_command.position = [0,0,0,0,0.14,0.14,0.14,0.14]
        self.subs = [
            rospy.Subscriber('/robot0/state/pose', Pose, self.pose_callback),
            rospy.Subscriber('/robot0/state/jointURDF', JointState, self.joint_callback),
            self.lc.subscribe('minitaur_command', self.command_callback)
        ]
        self.subs[-1].set_queue_capacity(1)
        self.pubs = [
            rospy.Publisher('/robot0/joint_cmd', JointState, queue_size=0)
        ]
        self.tf_listener = tf.TransformListener()
        self.rate = rospy.Rate(10)
        self.__body_offset = 0.06
        self.beat = 0
    def pose_callback(self, data):
        self.minitaur_state.orientation[0] = data.orientation.w
        self.minitaur_state.orientation[1] = data.orientation.x
        self.minitaur_state.orientation[2] = data.orientation.y
        self.minitaur_state.orientation[3] = data.orientation.z

    def joint_callback(self, data):
        self.minitaur_state.joint_swing[0] = data.position[0]
        self.minitaur_state.joint_swing[1] = data.position[1]
        self.minitaur_state.joint_swing[2] = data.position[2]
        self.minitaur_state.joint_swing[3] = data.position[3]

        self.minitaur_state.joint_extension[0] = data.position[4]
        self.minitaur_state.joint_extension[1] = data.position[5]
        self.minitaur_state.joint_extension[2] = data.position[6]
        self.minitaur_state.joint_extension[3] = data.position[7]

    def command_callback(self, channel, data):
        cmd = MinitaurCommand.decode(data)
        for i in range(4):
            self.minitaur_command.position[i] = cmd.joint_swing[i]
            self.minitaur_command.position[i+4] = cmd.joint_extension[i]

    def loop(self):
        while not rospy.is_shutdown():
            try:
                (trans, quat) = self.tf_listener.lookupTransform('origin', 'robot', rospy.Time(0))
                self.minitaur_state.pose[0] = trans[0]
                self.minitaur_state.pose[1] = trans[1]
                self.minitaur_state.pose[2] = trans[2]-self.__body_offset
                self.minitaur_state.orientation[0] = quat[3]
                self.minitaur_state.orientation[1] = quat[0]
                self.minitaur_state.orientation[2] = quat[1]
                self.minitaur_state.orientation[3] = quat[2]
                self.lc.publish('minitaur_state', self.minitaur_state.encode())
                rfds, wfds, efds = select.select([self.lc.fileno()], [], [], self.timeout)
                if rfds:
                    self.lc.handle()
                self.pubs[0].publish(self.minitaur_command)
                self.beat += 1
                print(self.beat, ' heart beat')
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
            self.rate.sleep()

if __name__ == '__main__':
    robot_listener = RobotListener()
    robot_listener.loop()
