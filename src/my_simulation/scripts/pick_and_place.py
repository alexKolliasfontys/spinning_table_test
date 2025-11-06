import re
import threading
import time
import subprocess
import numpy as np
from collections import deque
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

#from moveit2 import MoveIt2
from pymoveit2 import MoveIt2
from pymoveit2.robots import lite6

from geometry_msgs.msg import Pose, PoseStamped, Quaternion


class PoseReceiver:
    """Class to receive and process pose data"""
    def __init__(self, model_name="20x20_Profile", hz=30):
        self.model_name = model_name
        self.hz = hz
        self.dt = 1.0 / hz

        #Data storage
        self.poses = deque(maxlen=1000)
        self.running = False

        #KF

        #Polynomial reg

        self.thread = None

    def parse_pose_output(self, output: str) -> Optional[Pose]:
        try:
            xyz_match = re.search(r'\[([0-9.-]+)\s+([0-9.-]+)\s+([0-9.-]+)\]', output)
            if not xyz_match:
                return None
            
            x, y, z = map(float, xyz_match.groups())

            #extract rpy coordinates
            lines = output.split('\n')
            rpy_line = None
            for i, line in enumerate(lines):
                if 'XYZ' in line and i + 1 < len(lines):
                    rpy_line = lines[i + 1]
                    break

            if not rpy_line:
                return None

            rpy_match = re.search(r'\[([0-9.-]+)\s+([0-9.-]+)\s+([0-9.-]+)\]', rpy_line)
            if not rpy_match:
                return None

            roll, pitch, yaw = map(float, rpy_match.groups())
            
            # return Pose(
            #     position=Point(x=x, y=y, z=z),
            #     orientation=Quaternion(roll=roll, pitch=pitch, yaw=yaw))
            return Pose(
                timestamps=time.time(),
                x=x, y=y, z=z,
                roll=roll, pitch=pitch, yaw=yaw
            )
        except Exception as e:
            self.get_logger().error(f"Failed to parse pose output: {e}")
            return None
        
    def get_pose(self) -> Optional[Pose]:

class ObjectPathPredictor:
    def __init__(self):
        #setup the filter, topic subscriptions, etc.
        pass

    def predict_picking_point(self):
        """
        Return a pose as geometry_msgs/Pose,
        based on the prediction output.
        """
        # Placeholders
        pose = Pose()
        pose.position.x = 0.3
        pose.position.y = 0.0
        pose.position.z = 0.0
        return pose

class SleekControl(Node):
    def __init__(self, predictor):
        super().__init__('pick_and_place_control_node')
        self.predictor = predictor

        self.callback_group = ReentrantCallbackGroup()

        self.moveit2 = MoveIt2(
            node=self,
            joint_names=lite6.joint_names(),#['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'],
            base_link_name=lite6.base_link_name(),#'link_base',
            end_effector_name=lite6.end_effector_name(),#'end_effector',
            group_name=lite6.MOVE_GROUP_ARM,#'lite6',
            planner_id='ompl',
            callback_group=self.callback_group
        )

    def set_workspace(self, min_corner, max_corner):
        """
        Set workspace limits for the planner.
        """
        self.moveit2.workspace_parameters = {
            'min_corner': min_corner,
            'max_corner': max_corner
        }

    def set_velocity_scaling(self, scaling_factor):
        """
        Set velocity scaling for the planner
        according to pre or post grasp scenario.
        """
        self.moveit2.velocity_scaling = scaling_factor

    def go_to_pose(self, pose):
        self.moveit2.move_to_pose(
            position=[pose.position.x, pose.position.y, pose.position.z],
            orientation=[pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
            cartesian=False
        )
        self.moveit2.wait_until_executed()
        # ----------Previous-----------
        # self.moveit2.move_to_pose(pose)
        # self.moveit2.wait_until_executed()

    def go_to_scanning_position(self):
        scan_pose = Pose()
        scan_pose.position.x = 0.2
        scan_pose.position.y = 0.0
        scan_pose.position.z = 0.3
        scan_pose.orientation.w = 1.0

        self.go_to_pose(scan_pose)

    def pick_object(self, object_pose):
        pre_grasp_min = [-0.4, -0.4, 0.0]
        pre_grasp_max = [0.4, 0.4, 0.4]
        self.set_workspace(pre_grasp_min, pre_grasp_max)
        self.set_velocity_scaling(0.11)

        self.go_to_pose(object_pose)
        #gripper closing is not implemented

    def place_object(self, place_pose):
        post_grasp_min = [-0.4, -0.4, 0.0]
        post_grasp_max = [0.4, 0.4, 0.4]
        self.set_workspace(post_grasp_min, post_grasp_max)
        self.set_velocity_scaling(0.18)

        self.go_to_pose(place_pose)
        # gripper opening is not implemented

    def get_place_pose(self):
        place_pose = Pose()
        place_pose.position.x = 0.4
        place_pose.position.y = 0.0
        place_pose.position.z = 0.1
        place_pose.orientation.w = 1.0
        return place_pose

    def run(self):
        # Main loop: scan, predict, pick, place, repeat

        while rclpy.ok():
            
            self.go_to_scanning_position()

            object_pose = self.predictor.predict_picking_point()
            if object_pose is None:
                self.get_logger().info("No object detected, rescanning...")
                continue

            self.pick_object(object_pose)
            
            place_pose = self.get_place_pose()
            self.place_object(place_pose)
            self.get_logger().info("PnP completed. Waiting for next object...")

def main(args=None):
    rclpy.init(args=args)
    predictor = ObjectPathPredictor()
    arm = SleekControl(predictor)
    arm.run()
    rclpy.shutdown()

if __name__ == '__main__':
    main()