#!/usr/bin/env python3
# Poll Gazebo Ignition for a model pose and publish as:
#  - geometry_msgs/PointStamped on "obj_1_coord" (backward compatible)
#  - (optional) geometry_msgs/PoseStamped with orientation on "obj_1_pose"
# Additionally:
#  - Transform to planning frame and call MoveIt's /compute_ik to gate reachability
#  - Publish reachability status as std_msgs/String on configurable topic
#
# Orientation quaternion derived from roll/pitch/yaw reported by `ign model -m <model> -p`.
# Frame transformation (world_frame -> output_frame) applied if requested.

import rclpy, subprocess, threading, shlex, time, math
from typing import Optional, List, Tuple
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from moveit_msgs.srv import GetPositionIK

from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point, do_transform_pose


def rpy_to_quat(r: float, p: float, y: float) -> Quaternion:
    # Intrinsic XYZ (roll, pitch, yaw)
    cr = math.cos(0.5*r); sr = math.sin(0.5*r)
    cp = math.cos(0.5*p); sp = math.sin(0.5*p)
    cy = math.cos(0.5*y); sy = math.sin(0.5*y)
    q = Quaternion()
    q.w = cr*cp*cy + sr*sp*sy
    q.x = sr*cp*cy - cr*sp*sy
    q.y = cr*sp*cy + sr*cp*sy
    q.z = cr*cp*sy - sr*sp*cy
    return q


class IgnObjBridge(Node):
    def __init__(self):
        super().__init__('ign_obj_bridge')

        # -------- Parameters: polling/publish --------
        self.declare_parameter('world_name', 'default')
        self.declare_parameter('model_name', '20x20_Profile')
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('output_frame', 'world')
        self.declare_parameter('poll_rate_hz', 40.0)
        self.declare_parameter('offset_x', 0.0)
        self.declare_parameter('offset_y', 0.0)
        self.declare_parameter('offset_z', 0.0)
        self.declare_parameter('startup_grace_s', 0.5)
        self.declare_parameter('topic', 'obj_1_coord')
        self.declare_parameter('use_gz_fallback', True)
        self.declare_parameter('verbose_poll_debug', True)
        self.declare_parameter('warn_every_fail', 30)
        self.declare_parameter('cmd_override', '')
        self.declare_parameter('publish_pose', True)
        self.declare_parameter('pose_topic', 'obj_1_pose')
        self.declare_parameter('output_topic', '/ik_gate/output')

        # -------- Parameters: IK gating --------
        self.declare_parameter('group_name', 'lite6')
        self.declare_parameter('ik_link_name', 'link_tcp')
        self.declare_parameter('planning_frame', 'link_base')
        self.declare_parameter('gate_rate_hz', 80.0)
        self.declare_parameter('ik_timeout_ms', 20)
        self.declare_parameter('avoid_collisions', False)
        self.declare_parameter('poll_startup_grace_sec', 0.5)
        self.declare_parameter('ik_service', '/compute_ik')

        # -------- Get values: polling/publish --------
        self.model_name  = self.get_parameter('model_name').value
        self.world_name  = self.get_parameter('world_name').value
        self.world_frame = self.get_parameter('world_frame').value
        self.output_frame = self.get_parameter('output_frame').value
        self.poll_rate_hz = float(self.get_parameter('poll_rate_hz').value)
        self.offset = (
            float(self.get_parameter('offset_x').value),
            float(self.get_parameter('offset_y').value),
            float(self.get_parameter('offset_z').value),
        )
        self.topic = self.get_parameter('topic').value
        self.startup_grace = float(self.get_parameter('startup_grace_s').value)
        self.use_gz_fallback = bool(self.get_parameter('use_gz_fallback').value)
        self.verbose = bool(self.get_parameter('verbose_poll_debug').value)
        self.warn_every = int(self.get_parameter('warn_every_fail').value)
        override = self.get_parameter('cmd_override').value.strip()
        self.publish_pose = bool(self.get_parameter('publish_pose').value)
        self.pose_topic = str(self.get_parameter('pose_topic').value)

        # -------- Get values: IK gating --------
        self.group_name     = self.get_parameter('group_name').value
        self.ik_link_name   = self.get_parameter('ik_link_name').value
        self.planning_frame = self.get_parameter('planning_frame').value
        self.gate_rate_hz   = float(self.get_parameter('gate_rate_hz').value)
        self.ik_timeout_ms  = int(self.get_parameter('ik_timeout_ms').value)
        self.avoid_collisions = bool(self.get_parameter('avoid_collisions').value)
        self.poll_startup_grace_sec = float(self.get_parameter('poll_startup_grace_sec').value)
        self.ik_service = str(self.get_parameter('ik_service').value)
        self.output_topic = str(self.get_parameter('output_topic').value)

        # -------- Build command list --------
        self._cmds: List[str] = []
        if override:
            self._cmds.append(override)
        else:
            base = f'-m {self.model_name} -p'
            if self.world_name and self.world_name != 'default':
                base = f'-m {self.model_name} -w {self.world_name} -p'
            self._cmds.append(f'ign model {base}')
            if self.use_gz_fallback:
                self._cmds.append(f'gz model {base}')
        self._cmd_index = 0

        # -------- TF --------
        self.tf_buffer = Buffer(cache_time=Duration(seconds=3.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # -------- Publishers --------
        self.pub = self.create_publisher(PointStamped, self.topic, 10)
        self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, 10) if self.publish_pose else None

        # IK gate status publisher (latched)
        status_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.status_pub = self.create_publisher(String, self.output_topic, status_qos)

        # -------- IK client --------
        self.ik_cli = self.create_client(GetPositionIK, self.ik_service)

        # -------- Joint state seed --------
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=15
        )
        self.latest_joint_state = JointState()
        self.create_subscription(JointState, '/joint_states', self._on_joint_state, qos)

        # -------- Internal state --------
        self._poll_start_time = time.time()
        self._latest_xyz: Optional[Tuple[float,float,float]] = None
        self._latest_pose: Optional[Tuple[float,float,float,float,float,float]] = None  # x,y,z,r,p,y
        self._fail = 0
        self._ok = 0
        self._first_logged = 0
        self._stop = False
        self._lock = threading.Lock()
        self._last_status: Optional[str] = None

        # -------- Polling thread --------
        self._thread = threading.Thread(target=self._poll_ign_loop, daemon=True)
        self._thread.start()

        # Publish initial status so new subscribers see something
        self._publish_status('STARTING')

        # -------- Timers --------
        # Publish latest sample to topics
        self.create_timer(1.0 / max(1e-3, self.poll_rate_hz), self._publish_latest)
        # IK gating tick
        self.create_timer(1.0 / max(1e-3, self.gate_rate_hz), self._gate_tick)

        self.get_logger().info(
            f'Ignition bridge started: model="{self.model_name}" topic="{self.topic}" cmds={self._cmds} '
            f'publish_pose={self.publish_pose} planning_frame="{self.planning_frame}" group="{self.group_name}"'
        )

    # -------- Poll loop --------
    def _poll_ign_loop(self):
        rate = self.poll_rate_hz
        while rclpy.ok() and not self._stop:
            cmd = self._cmds[self._cmd_index]
            try:
                out = subprocess.check_output(shlex.split(cmd), stderr=subprocess.DEVNULL).decode()
                tokens = out.replace('[', ' ').replace(']', ' ').split()
                floats: List[float] = []
                for t in tokens:
                    if not t:
                        continue
                    if t[0] not in '-+0123456789.':
                        continue
                    try:
                        v = float(t)
                        if math.isfinite(v):
                            floats.append(v)
                    except Exception:
                        continue
                if len(floats) >= 6:
                    x, y, z, r, p, yw = floats[-6:]
                    with self._lock:
                        self._latest_xyz = (x, y, z)
                        self._latest_pose = (x, y, z, r, p, yw)
                        self._ok += 1
                    if self.verbose:# and self._first_logged < 3:
                        self.get_logger().info(f'Pose sample via "{cmd}": {(x,y,z,r,p,yw)}')
                        #self._first_logged += 1
                else:
                    self._fail += 1
                    if (time.time() - self._poll_start_time) > self.startup_grace and self._fail % self.warn_every == 0:
                        self.get_logger().warn(f'Parse fail #{self._fail} (floats={len(floats)}). Raw: {out.strip()[:140]}')
            except FileNotFoundError:
                self._fail += 1
                if self._fail == 3 and self.use_gz_fallback and len(self._cmds) > 1:
                    self._cmd_index = (self._cmd_index + 1) % len(self._cmds)
                    self.get_logger().warn(f'Switching to fallback command: {self._cmds[self._cmd_index]}')
            except subprocess.CalledProcessError as e:
                self._fail += 1
                if self._fail % self.warn_every == 0:
                    self.get_logger().warn(f'Command error #{self._fail}: {e}')
            except Exception as e:
                self._fail += 1
                if self._fail % self.warn_every == 0:
                    self.get_logger().warn(f'Poll exception #{self._fail}: {e}')
            time.sleep(max(0.0, 1.0 / rate))

    # -------- Publish latest point/pose --------
    def _publish_latest(self):
        with self._lock:
            xyz = self._latest_xyz
            pose_tuple = self._latest_pose
            ok_count = self._ok
            fail_count = self._fail
        if xyz is None:
            return

        x, y, z = xyz
        x += self.offset[0]; y += self.offset[1]; z += self.offset[2]

        pt_msg = PointStamped()
        pt_msg.header.stamp = self.get_clock().now().to_msg()
        pt_msg.header.frame_id = self.world_frame
        pt_msg.point.x = x; pt_msg.point.y = y; pt_msg.point.z = z

        pose_msg: Optional[PoseStamped] = None
        if self.publish_pose and pose_tuple:
            px, py, pz, rr, pp, yy = pose_tuple
            px += self.offset[0]; py += self.offset[1]; pz += self.offset[2]
            pose_msg = PoseStamped()
            pose_msg.header = pt_msg.header
            pose_msg.pose.position.x = px
            pose_msg.pose.position.y = py
            pose_msg.pose.position.z = pz
            pose_msg.pose.orientation = rpy_to_quat(rr, pp, yy)

        # old
        # # Frame transform to output_frame for publishing
        # if self.output_frame != self.world_frame:
        #     try:
        #         pt_msg = self.tf_buffer.transform(
        #             pt_msg, self.output_frame,
        #             timeout=Duration(seconds=0.5)
        #         )
        #     except Exception as e:
        #         if ok_count < 5 or ok_count % self.warn_every == 0:
        #             self.get_logger().debug(f'TF fail: {e}')

        # self.pub.publish(pt_msg)
        # if pose_msg and self.pose_pub:
        #     self.pose_pub.publish(pose_msg)
        # new
        if self.output_frame != self.world_frame:
            try:
                pt_msg = self.tf_buffer.transform(
                    pt_msg, self.output_frame,
                    timeout=Duration(seconds=0.5)
                )
            except Exception as e:
                if ok_count < 5 or ok_count % self.warn_every == 0:
                    self.get_logger().debug(f'TF fail: {e}')
            if pose_msg and self.pose_pub:
                try:
                    pose_msg = self.tf_buffer.transform(
                        pose_msg, self.output_frame,
                        timeout=Duration(seconds=0.5)
                    )
                except Exception as e:
                    if ok_count < 5 or ok_count % self.warn_every == 0:
                        self.get_logger().debug(f'TF pose fail: {e}')

        self.pub.publish(pt_msg)
        if pose_msg and self.pose_pub:
            self.pose_pub.publish(pose_msg)

        # Stats
        if ok_count % 100 == 0 and ok_count > 0:
            self.get_logger().info(
                f'Stats: published={ok_count} fails={fail_count} last=({pt_msg.point.x:.3f},{pt_msg.point.y:.3f},{pt_msg.point.z:.3f})'
            )

    # -------- Joint State callback --------
    def _on_joint_state(self, msg: JointState):
        self.latest_joint_state = msg

    # -------- IK gating tick --------
    def _gate_tick(self):
        # Get latest world pose (without output-frame transform and without offset applied in world frame)
        with self._lock:
            pose_tuple = self._latest_pose

        if not pose_tuple:
            self._publish_status('NO_POSE')
            return

        x, y, z, rr, pp, yy = pose_tuple

        # Build PoseStamped in world_frame (no offset yet)
        ps_world = PoseStamped()
        ps_world.header.frame_id = self.world_frame
        ps_world.header.stamp = self.get_clock().now().to_msg()
        ps_world.pose.position.x = x
        ps_world.pose.position.y = y
        ps_world.pose.position.z = z
        ps_world.pose.orientation = rpy_to_quat(rr, pp, yy)

        # Transform to planning frame if needed
        ps_planning = ps_world
        if self.planning_frame and self.planning_frame != ps_world.header.frame_id:
            try:
                ps_planning = self.tf_buffer.transform(
                    ps_world, self.planning_frame,
                    timeout=Duration(seconds=0.5)
                )
            except Exception as e:
                self.get_logger().warn(
                    f'NO_TF: {self.planning_frame} <- {ps_world.header.frame_id}: {e}'
                )
                self._publish_status('NO_TF')
                return

        # Apply grasp offset in planning frame
        ps_planning.pose.position.x += self.offset[0]
        ps_planning.pose.position.y += self.offset[1]
        ps_planning.pose.position.z += self.offset[2]

        # Prepare IK request
        if not self.ik_cli.service_is_ready():
            # Try a short wait once in a while without blocking too much
            self.ik_cli.wait_for_service(timeout_sec=0.1)
            if not self.ik_cli.service_is_ready():
                self._publish_status('NO_IK_SERVICE')
                return

        req = GetPositionIK.Request()
        ik = req.ik_request
        ik.group_name = self.group_name
        ik.ik_link_name = self.ik_link_name
        ik.pose_stamped = ps_planning
        ik.avoid_collisions = self.avoid_collisions
        # Timeout
        ik.timeout.sec = int(self.ik_timeout_ms // 1000)
        ik.timeout.nanosec = int((self.ik_timeout_ms % 1000) * 1_000_000)
        # Seed joints
        if not self.latest_joint_state.name:
            self._publish_status('NO_JOINT_STATE')
            return
        ik.robot_state.joint_state = self.latest_joint_state

        future = self.ik_cli.call_async(req)
        future.add_done_callback(self._on_ik_result)

    def _publish_status(self, status: str):
        if status != self._last_status:
            msg = String()
            msg.data = status
            self.status_pub.publish(msg)
            self._last_status = status

    def _on_ik_result(self, future):
        try:
            res = future.result()
        except Exception as e:
            self.get_logger().debug(f'IK future exception: {e}')
            self._publish_status('IK_ERROR')
            return

        if res and res.error_code.val == res.error_code.SUCCESS:
            self._publish_status('REACHABLE')
        else:
            self._publish_status('UNREACHABLE')

    # -------- Shutdown --------
    def destroy_node(self):
        self._stop = True
        return super().destroy_node()


def main():
    rclpy.init()
    node = IgnObjBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()