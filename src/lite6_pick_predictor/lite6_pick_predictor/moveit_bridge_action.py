#!/usr/bin/env python3
# MoveIt2 bridge with predictor readiness gating and delayed pick trigger.

import time
import math
from typing import Optional, List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import PoseStamped, Pose, Quaternion
from nav_msgs.msg import Path
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import (
    Constraints, PositionConstraint, OrientationConstraint, BoundingVolume,
    MotionPlanRequest, RobotState
)
from moveit_msgs.srv import GetMotionPlan, GetPositionIK
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from control_msgs.action import FollowJointTrajectory
from std_msgs.msg import Float32, Bool, String
from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Time as TimeMsg

import tf2_ros
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose_stamped
from geometry_msgs.msg import Pose as GeoPose

from lite6_pick_predictor_interfaces.srv import GetPoseAt

def yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    half = 0.5 * yaw
    q.z = math.sin(half)
    q.w = math.cos(half)
    return q

def time_from_start_to_sec(tfs) -> float:
    return float(getattr(tfs, 'sec', 0)) + float(getattr(tfs, 'nanosec', 0)) * 1e-9

def sec_to_duration(sec: float):
    from builtin_interfaces.msg import Duration
    d = Duration()
    d.sec = int(sec)
    d.nanosec = int((sec - int(sec)) * 1e9)
    return d

class MoveItBridgeAction(Node):
    def __init__(self):
        super().__init__('moveit_bridge_action')

        # Planning basics
        self.declare_parameter('group_name', 'lite6')
        self.declare_parameter('end_effector_link', 'link_tcp')
        self.declare_parameter('planning_frame', 'world')
        self.declare_parameter('allowed_planning_time', 1.2)
        self.declare_parameter('goal_tolerance_pos', 0.003)
        self.declare_parameter('goal_tolerance_ori', 0.02)
        self.declare_parameter('max_velocity_scaling', 0.8)
        self.declare_parameter('max_acceleration_scaling', 0.6)
        self.declare_parameter('planning_service', '/plan_kinematic_path')

        # Execution
        self.declare_parameter('follow_action', '/lite6_traj_controller/follow_joint_trajectory')
        self.declare_parameter('joint_names', ['joint1','joint2','joint3','joint4','joint5','joint6'])

        # Predictor integration
        self.declare_parameter('path_topic', 'predicted_ee_path')
        self.declare_parameter('pick_time_s', 2.5)
        self.declare_parameter('pred_pose_topic', 'predicted_pick_pose')

        # Orientation policy
        self.declare_parameter('target_orientation_from_predictor', True)
        self.declare_parameter('ik_yaw_align_tangent', False)
        self.declare_parameter('table_center_xy', [0.0, 0.0])

        # Hover / grasp geometry
        self.declare_parameter('z_hover', 0.15)
        self.declare_parameter('z_grasp', 0.02)
        self.declare_parameter('t_hover_s', 3.0)
        self.declare_parameter('t_grasp_s', 4.5)

        # Gripper
        self.declare_parameter('grip_topic', 'gripper/close')
        self.declare_parameter('grip_fire_early_s', 0.05)

        # IK Gate
        self.declare_parameter('use_ik_reachability_gate', True)
        self.declare_parameter('ik_group_name', 'lite6')
        self.declare_parameter('ik_link_name', 'link_tcp')
        self.declare_parameter('ik_timeout_ms', 20)
        self.declare_parameter('ik_avoid_collisions', False)

        # Safety
        self.declare_parameter('contact_speed_max', 0.35)
        self.declare_parameter('require_contact_speed_check', False)
        self.declare_parameter('theta_replan_thresh_rad', 0.20)

        # Trigger control
        self.declare_parameter('path_trigger_decim', 20)  # increased default (was 5)
        self.declare_parameter('require_predictor_ready', True)
        self.declare_parameter('min_planning_delay_s', 1.5)  # time after predictor ready

        # Params -> members
        self.group_name = self.get_parameter('group_name').value
        self.eef_link = self.get_parameter('end_effector_link').value
        self.planning_frame = str(self.get_parameter('planning_frame').value)
        self.allowed_ptime = float(self.get_parameter('allowed_planning_time').value)
        self.tol_pos = float(self.get_parameter('goal_tolerance_pos').value)
        self.tol_ori = float(self.get_parameter('goal_tolerance_ori').value)
        self.vel_scale = float(self.get_parameter('max_velocity_scaling').value)
        self.acc_scale = float(self.get_parameter('max_acceleration_scaling').value)
        self.plan_srv_name = str(self.get_parameter('planning_service').value)

        self.follow_action_name = str(self.get_parameter('follow_action').value)
        self.joint_names: List[str] = list(self.get_parameter('joint_names').value)

        self.path_topic = str(self.get_parameter('path_topic').value)
        self.pick_time_s = float(self.get_parameter('pick_time_s').value)
        self.pred_pose_topic = str(self.get_parameter('pred_pose_topic').value)

        self.use_predictor_orientation = bool(self.get_parameter('target_orientation_from_predictor').value)
        self.align_tangent = bool(self.get_parameter('ik_yaw_align_tangent').value)
        self.table_cx, self.table_cy = [float(v) for v in self.get_parameter('table_center_xy').value]

        self.z_hover = float(self.get_parameter('z_hover').value)
        self.z_grasp = float(self.get_parameter('z_grasp').value)
        self.t_hover = float(self.get_parameter('t_hover_s').value)
        self.t_grasp = float(self.get_parameter('t_grasp_s').value)

        self.grip_topic = str(self.get_parameter('grip_topic').value)
        self.grip_early = float(self.get_parameter('grip_fire_early_s').value)

        self.v_contact_max = float(self.get_parameter('contact_speed_max').value)
        self.require_v_contact = bool(self.get_parameter('require_contact_speed_check').value)
        self.theta_thresh = float(self.get_parameter('theta_replan_thresh_rad').value)

        self.path_trigger_decim = int(self.get_parameter('path_trigger_decim').value or 1)
        if self.path_trigger_decim < 1:
            self.path_trigger_decim = 1
        self.require_predictor_ready = bool(self.get_parameter('require_predictor_ready').value)
        self.min_planning_delay_s = float(self.get_parameter('min_planning_delay_s').value)

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Services & action
        self.plan_srv = self.create_client(GetMotionPlan, self.plan_srv_name)
        self.follow_ac = ActionClient(self, FollowJointTrajectory, self.follow_action_name)

        # Publishers
        self.plan_latency_pub = self.create_publisher(Float32, 'latency/plan_s', 10)
        self.exec_latency_pub = self.create_publisher(Float32, 'latency/exec_s', 10)
        self.grip_pub = self.create_publisher(Bool, self.grip_topic, 10)
        self.pred_pose_pub = self.create_publisher(PoseStamped, self.pred_pose_topic, 10)
        self.marker_pub = self.create_publisher(Marker, 'pick_pose_marker', 10)
        self.pick_marker_pub = self.create_publisher(Marker, 'marker/predicted_pick', 10)
        self.reach_pub = self.create_publisher(String, 'reachability_status', 10)

        # Predictor diagnostics (subscribing)
        self.obj_omega = 0.0
        self.obj_radius = 0.0
        self.theta_now = None
        self.theta_target = None
        self.create_subscription(Float32, 'obj_omega_rad_s', lambda m: setattr(self, 'obj_omega', float(m.data)), 10)
        self.create_subscription(Float32, 'obj_radius_m',   lambda m: setattr(self, 'obj_radius', float(m.data)), 10)
        self.create_subscription(Float32, 'obj_theta_now',   lambda m: setattr(self, 'theta_now', float(m.data)), 10)
        self.create_subscription(Float32, 'obj_theta_target',lambda m: setattr(self, 'theta_target', float(m.data)), 10)

        # Predictor readiness
        self._predictor_ready = (not self.require_predictor_ready)
        self._predictor_ready_time: Optional[float] = None
        if self.require_predictor_ready:
            self.create_subscription(
                Bool, 'predictor_ready',
                self._ready_cb,
                10
            )

        # IK gate
        self.use_ik_gate = bool(self.get_parameter('use_ik_reachability_gate').value)
        self.ik_group = str(self.get_parameter('ik_group_name').value)
        self.ik_link = str(self.get_parameter('ik_link_name').value)
        self.ik_timeout_ms = int(self.get_parameter('ik_timeout_ms').value)
        self.ik_avoid_collisions = bool(self.get_parameter('ik_avoid_collisions').value)
        self.ik_cli = self.create_client(GetPositionIK, '/compute_ik')
        self.latest_joint_state = JointState()
        self.create_subscription(JointState, '/joint_states',
                                 lambda msg: setattr(self, 'latest_joint_state', msg), 10)

        qos = QoSProfile(depth=10)
        self.path_sub = self.create_subscription(Path, self.path_topic, self.path_cb, qos)

        self.cli_get_pose = self.create_client(GetPoseAt, 'get_predicted_pose_at')

        self.get_logger().info('Bridge ready (GetMotionPlan + FollowJT, pick from predictor).')
        self.get_logger().info(f'Planning group={self.group_name} frame={self.planning_frame}')
        self.get_logger().info(f'Pick time={self.pick_time_s:.2f}s hover_z={self.z_hover:.3f} grasp_z={self.z_grasp:.3f}')
        self.get_logger().info(f'IK gate: {"ENABLED" if self.use_ik_gate else "DISABLED"}')
        self.get_logger().info(f'Path trigger decimation={self.path_trigger_decim} require_predictor_ready={self.require_predictor_ready} delay={self.min_planning_delay_s}s')

        self.create_timer(2.0, self._diag_once)
        self._diag_ran = False
        self._path_msg_count = 0
        self._pending_pose_future = None

    def _ready_cb(self, msg: Bool):
        if msg.data and not self._predictor_ready:
            self._predictor_ready = True
            self._predictor_ready_time = time.time()
            self.get_logger().info('Predictor readiness received.')
        elif not msg.data:
            self._predictor_ready = False
            self._predictor_ready_time = None

    def _diag_once(self):
        if self._diag_ran:
            return
        self._diag_ran = True
        self.get_logger().info(f'Service available(plan)={self.plan_srv.wait_for_service(timeout_sec=0.2)}')
        self.get_logger().info(f'Action available(follow)={self.follow_ac.wait_for_server(timeout_sec=0.2)}')
        self.get_logger().info(f'Service available(GetPoseAt)={self.cli_get_pose.wait_for_service(timeout_sec=0.2)}')

    # ---------------- Service helper ----------------
    def request_pose_at_async(self, t_rel_s: float):
        if not self.cli_get_pose.wait_for_service(timeout_sec=0.1):
            self.get_logger().warn('GetPoseAt service not available.')
            return None
        req = GetPoseAt.Request()
        if hasattr(req, 't_rel_s'):
            req.t_rel_s = float(t_rel_s)
        else:
            if hasattr(req, 'use_relative'):
                req.use_relative = True
            if hasattr(req, 'query_time'):
                sec = int(t_rel_s); nsec = int((t_rel_s - sec)*1e9)
                req.query_time = TimeMsg(sec=sec, nanosec=nsec)
        future = self.cli_get_pose.call_async(req)
        return future

    # ---------------- TF helper ----------------
    def transform_to_planning_frame(self, pose: PoseStamped) -> Optional[PoseStamped]:
        if pose.header.frame_id == self.planning_frame:
            return pose
        try:
            tf = self.tf_buffer.lookup_transform(self.planning_frame,
                                                 pose.header.frame_id,
                                                 rclpy.time.Time(),
                                                 timeout=Duration(seconds=0.2))
            return do_transform_pose_stamped(pose, tf)
        except Exception as e:
            self.get_logger().warn(f'TF lookup failed: {e}')
            return None

    def make_pose_goal_constraints(self, pose: PoseStamped) -> Constraints:
        pc = PositionConstraint()
        pc.header = pose.header
        pc.link_name = self.eef_link
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [self.tol_pos]
        bv = BoundingVolume()
        bv.primitives.append(sphere)
        center_pose = GeoPose()
        center_pose.position = pose.pose.position
        center_pose.orientation.w = 1.0
        bv.primitive_poses.append(center_pose)
        pc.constraint_region = bv
        pc.weight = 1.0

        oc = OrientationConstraint()
        oc.header = pose.header
        oc.link_name = self.eef_link
        oc.orientation = pose.pose.orientation
        oc.absolute_x_axis_tolerance = self.tol_ori
        oc.absolute_y_axis_tolerance = self.tol_ori
        oc.absolute_z_axis_tolerance = self.tol_ori
        oc.weight = 1.0

        c = Constraints()
        c.position_constraints.append(pc)
        c.orientation_constraints.append(oc)
        return c

    def call_plan(self, goal_pose: PoseStamped, start_state: Optional[RobotState] = None) -> Optional[JointTrajectory]:
        if not self.plan_srv.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn('Plan service not available.')
            return None
        pos = goal_pose.pose.position
        self.get_logger().info(f'Planning to goal: x={pos.x:.3f} y={pos.y:.3f} z={pos.z:.3f}')
        req = GetMotionPlan.Request()
        mpr = MotionPlanRequest()
        mpr.group_name = self.group_name
        mpr.num_planning_attempts = 1
        mpr.allowed_planning_time = self.allowed_ptime
        mpr.goal_constraints.append(self.make_pose_goal_constraints(goal_pose))
        mpr.max_velocity_scaling_factor = self.vel_scale
        mpr.max_acceleration_scaling_factor = self.acc_scale
        if start_state:
            mpr.start_state = start_state
        req.motion_plan_request = mpr
        future = self.plan_srv.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.allowed_ptime + 1.0)
        if not future.done():
            self.get_logger().warn('Planning timed out.')
            return None
        resp = future.result()
        if not resp or resp.motion_plan_response.error_code.val != 1:
            self.get_logger().warn(f'Planning failed code={resp.motion_plan_response.error_code.val if resp else "none"}')
            return None
        return resp.motion_plan_response.trajectory.joint_trajectory

    def ik_reachable(self, grasp_pose: PoseStamped) -> bool:
        if not self.use_ik_gate:
            return True
        if not self.ik_cli.wait_for_service(timeout_sec=0.25):
            self.get_logger().warn('/compute_ik not available (skipping gate).')
            return True
        req = GetPositionIK.Request()
        ikr = req.ik_request
        ikr.group_name = self.ik_group
        ikr.ik_link_name = self.ik_link
        ikr.pose_stamped = grasp_pose
        ikr.avoid_collisions = self.ik_avoid_collisions
        ikr.timeout.sec = self.ik_timeout_ms // 500
        ikr.timeout.nanosec = (self.ik_timeout_ms % 1000) * 1_000_000
        ikr.robot_state.joint_state = self.latest_joint_state
        future = self.ik_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=0.5)
        if not future.done():
            self.get_logger().warn('IK request timeout.')
            return False
        res = future.result()
        ok = (res and res.error_code.val == res.error_code.SUCCESS)
        status = 'REACHABLE' if ok else 'UNREACHABLE'
        self.reach_pub.publish(String(data=status))
        self.get_logger().info(f'IK gate: {status}')
        return ok

    def path_cb(self, _msg: Path):
        self._path_msg_count += 1
        if self._path_msg_count % self.path_trigger_decim != 0:
            return
        if self.require_predictor_ready and not self._predictor_ready:
            return
        if self.require_predictor_ready and self._predictor_ready_time:
            if (time.time() - self._predictor_ready_time) < self.min_planning_delay_s:
                return
        if self._pending_pose_future and not self._pending_pose_future.done():
            return
        t_req = self.pick_time_s
        self._svc_call_start = time.time()
        self.get_logger().info(f'Path trigger: async query t={t_req:.2f}s (count={self._path_msg_count})')
        fut = self.request_pose_at_async(t_req)
        if fut is None:
            return
        self._pending_pose_future = fut
        fut.add_done_callback(self._on_pose_future)

    def _on_pose_future(self, fut):
        svc_dt = time.time() - getattr(self, '_svc_call_start', time.time())
        try:
            resp = fut.result()
        except Exception as e:
            self.get_logger().warn(f'GetPoseAt exception after {svc_dt:.3f}s: {e}')
            self._pending_pose_future = None
            return
        self._pending_pose_future = None
        ok_flag = getattr(resp, 'ok', getattr(resp, 'success', False))
        if not ok_flag:
            self.get_logger().warn(f'GetPoseAt failure flag after {svc_dt:.3f}s')
            return
        pick_pose = getattr(resp, 'pose', None)
        if pick_pose is None:
            self.get_logger().warn(f'GetPoseAt no pose after {svc_dt:.3f}s')
            return
        pos = pick_pose.pose.position
        dist = math.sqrt(pos.x*pos.x + pos.y*pos.y)
        self.get_logger().info(f'Pick pose received in {svc_dt:.3f}s: x={pos.x:.3f} y={pos.y:.3f} z={pos.z:.3f} (r={dist:.3f})')
        pick_pose_tf = self.transform_to_planning_frame(pick_pose)
        if pick_pose_tf is None:
            self.get_logger().warn('TF to planning frame failed.')
            return
        _, grasp_pose = self.build_hover_grasp_poses(pick_pose_tf)
        if not self.ik_reachable(grasp_pose):
            self.get_logger().info('IK unreachable. Skipping plan.')
            return
        marker = Marker()
        marker.header = pick_pose_tf.header
        marker.ns = 'pick'; marker.id = 0
        marker.type = Marker.SPHERE; marker.action = Marker.ADD
        marker.pose = pick_pose_tf.pose
        marker.scale.x = marker.scale.y = marker.scale.z = 0.03
        marker.color.r = 1.0; marker.color.g = 0.3; marker.color.b = 0.3; marker.color.a = 0.9
        self.marker_pub.publish(marker)
        self.get_logger().info('Reachable. Planning...')
        self.plan_and_execute_from_pick(pick_pose_tf)

    def build_hover_grasp_poses(self, pick_pose_in: PoseStamped) -> Tuple[Optional[PoseStamped], Optional[PoseStamped]]:
        hover = PoseStamped()
        hover.header = pick_pose_in.header
        hover.pose = pick_pose_in.pose
        hover.pose.position.z = self.z_hover
        grasp = PoseStamped()
        grasp.header = pick_pose_in.header
        grasp.pose = pick_pose_in.pose
        grasp.pose.position.z = self.z_grasp
        return hover, grasp

    def concat_traj(self, a: JointTrajectory, b: JointTrajectory) -> JointTrajectory:
        if not a.points:
            return b
        if not b.points:
            return a
        last_t = time_from_start_to_sec(a.points[-1].time_from_start)
        for p in b.points:
            t = time_from_start_to_sec(p.time_from_start)
            p.time_from_start = sec_to_duration(t + last_t)
        out = JointTrajectory()
        out.joint_names = a.joint_names
        out.points = a.points + b.points
        return out

    def hover_index(self, jt_hover_len: int) -> int:
        return jt_hover_len - 1

    def retime_to_milestones(self, jt: JointTrajectory, idx_hover: int, t_hover: float, t_grasp: float) -> JointTrajectory:
        if not jt.points:
            self.get_logger().warn('JointTrajectory empty.')
            return jt
        total_current = time_from_start_to_sec(jt.points[-1].time_from_start)
        if total_current <= 0:
            self.get_logger().warn('JointTrajectory duration invalid.')
            return jt
        scale = t_grasp / total_current
        for p in jt.points:
            t = time_from_start_to_sec(p.time_from_start)
            p.time_from_start = sec_to_duration(t * scale)
        return jt

    def plan_and_execute_from_pick(self, pick_pose: PoseStamped):
        hover_pose, grasp_pose = self.build_hover_grasp_poses(pick_pose)
        if hover_pose is None or grasp_pose is None:
            self.get_logger().warn('Could not build hover/grasp.')
            return
        t0 = time.time()
        jt1 = self.call_plan(hover_pose)
        if jt1 is None:
            self.get_logger().warn('Hover plan failed.')
            return
        jt2 = self.call_plan(grasp_pose)
        if jt2 is None:
            self.get_logger().warn('Grasp plan failed.')
            return
        full_jt = self.concat_traj(jt1, jt2)
        full_jt = self.retime_to_milestones(full_jt, self.hover_index(len(jt1.points)), self.t_hover, self.t_grasp)
        plan_dt = time.time() - t0
        self.plan_latency_pub.publish(Float32(data=float(plan_dt)))
        if not self.follow_ac.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn('FollowJointTrajectory action unavailable.')
            return
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = full_jt
        send_future = self.follow_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=2.0)
        if not send_future.done():
            self.get_logger().warn('Trajectory goal send timeout.')
            return
        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Trajectory goal rejected.')
            return
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=self.t_grasp + 5.0)
        exec_lat = float(time.time() - t0 - plan_dt)
        self.exec_latency_pub.publish(Float32(data=exec_lat))
        self.get_logger().info(f'Execution complete (plan {plan_dt:.3f}s, exec {exec_lat:.3f}s).')

def main(args=None):
    rclpy.init(args=args)
    node = MoveItBridgeAction()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()