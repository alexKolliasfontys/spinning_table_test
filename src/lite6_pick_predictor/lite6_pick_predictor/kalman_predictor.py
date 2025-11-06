#!/usr/bin/env python3
# EKF predictor with readiness gating and orientation support.

import math, time
from collections import deque
from typing import Deque, Tuple, Optional
import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion
from nav_msgs.msg import Path
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from std_msgs.msg import Float32, Bool, String
from builtin_interfaces.msg import Time as TimeMsg, Duration as DurationMsg
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


from lite6_pick_predictor_interfaces.srv import GetPoseAt

def ros_time_to_sec(t: TimeMsg) -> float:
    return float(t.sec) + 1e-9 * float(t.nanosec)

def sec_to_time_msg(t: float) -> TimeMsg:
    msg = TimeMsg()
    msg.sec = int(t); msg.nanosec = int((t - int(t)) * 1e9)
    return msg

def sec_to_duration(t: float) -> DurationMsg:
    d = DurationMsg(); d.sec = int(t); d.nanosec = int((t - int(t))*1e9); return d

def yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion(); half = 0.5 * yaw
    q.z = math.sin(half); q.w = math.cos(half)
    return q

class EKFPredictor(Node):
    def __init__(self):
        super().__init__('kalman_predictor')

        # ---- Parameters ----
        self.declare_parameter('planning_latency_s', 0.25)
        self.declare_parameter('execution_latency_s', 0.55)
        self.declare_parameter('bootstrap_min_samples', 30)
        self.declare_parameter('bootstrap_maxlen', 200)
        self.declare_parameter('q_cx', 1e-6)
        self.declare_parameter('q_cy', 1e-6)
        self.declare_parameter('q_r',  1e-6)
        self.declare_parameter('q_theta', 1e-4)
        self.declare_parameter('q_omega', 5e-4)
        self.declare_parameter('r_meas',  1e-4)
        self.declare_parameter('output_frame_id', '')
        self.declare_parameter('use_tangent_yaw', True)
        self.declare_parameter('use_last_z', True)
        self.declare_parameter('fixed_pick_z', 0.02)
        self.declare_parameter('path_horizon_s', 15.0)
        self.declare_parameter('path_rate_hz', 5.0)
        self.declare_parameter('verbose_debug', True)
        self.declare_parameter('log_innovation', True)
        self.declare_parameter('diagnostic_log_interval_s', 2.0)
        self.declare_parameter('input_pose_topic', '')
        self.declare_parameter('use_input_orientation', False)
        self.declare_parameter('publish_predicted_pose', True)
        self.declare_parameter('predicted_pose_rate_hz', 5.0)
        self.declare_parameter('service_eval_direct', True)
        self.declare_parameter('path_publish_rate_hz', 5.0)
        self.declare_parameter('oneshot_on_reachable', False)
        self.declare_parameter('ik_gate_topic', '/ik_gate/status')
        # Readiness / timing quality
        self.declare_parameter('bootstrap_min_angle_span_rad', 0.35)
        self.declare_parameter('bootstrap_min_time_span_s', 1.0)
        self.declare_parameter('max_dt_cap_s', 0.05)
        self.declare_parameter('planning_ready_min_age_s', 1.5)
        self.declare_parameter('planning_ready_min_radius_samples', 40)
        self.declare_parameter('planning_ready_radius_var_max', 5e-5)

        # Params -> members
        self.planning_latency = float(self.get_parameter('planning_latency_s').value)
        self.execution_latency = float(self.get_parameter('execution_latency_s').value)
        self.bootstrap_min = int(self.get_parameter('bootstrap_min_samples').value)
        self.bootstrap_maxlen = int(self.get_parameter('bootstrap_maxlen').value)
        self.q_vals = np.array([
            float(self.get_parameter('q_cx').value),
            float(self.get_parameter('q_cy').value),
            float(self.get_parameter('q_r').value),
            float(self.get_parameter('q_theta').value),
            float(self.get_parameter('q_omega').value),
        ], dtype=float)
        self.r_meas = float(self.get_parameter('r_meas').value)
        self.output_frame_id_param = str(self.get_parameter('output_frame_id').value)
        self.use_tangent_yaw = bool(self.get_parameter('use_tangent_yaw').value)
        self.use_last_z = bool(self.get_parameter('use_last_z').value)
        self.fixed_pick_z = float(self.get_parameter('fixed_pick_z').value)
        self.path_H = float(self.get_parameter('path_horizon_s').value)
        self.path_rate = float(self.get_parameter('path_rate_hz').value)
        self.verbose_debug = bool(self.get_parameter('verbose_debug').value)
        self.log_innovation = bool(self.get_parameter('log_innovation').value)
        self.diag_interval = float(self.get_parameter('diagnostic_log_interval_s').value)
        self.input_pose_topic = str(self.get_parameter('input_pose_topic').value)
        self.use_input_orientation = bool(self.get_parameter('use_input_orientation').value)
        self.publish_predicted_pose = bool(self.get_parameter('publish_predicted_pose').value)
        self.pred_rate = float(self.get_parameter('predicted_pose_rate_hz').value)
        self.service_eval_direct = bool(self.get_parameter('service_eval_direct').value)
        self.path_publish_rate = float(self.get_parameter('path_publish_rate_hz').value)
        self.bootstrap_min_angle_span = float(self.get_parameter('bootstrap_min_angle_span_rad').value)
        self.bootstrap_min_time_span = float(self.get_parameter('bootstrap_min_time_span_s').value)
        self.max_dt_cap = float(self.get_parameter('max_dt_cap_s').value)
        self.ready_min_age = float(self.get_parameter('planning_ready_min_age_s').value)
        self.ready_min_r_samples = int(self.get_parameter('planning_ready_min_radius_samples').value)
        self.ready_r_var_max = float(self.get_parameter('planning_ready_radius_var_max').value)

        # ---- State buffers ----
        self.buf: Deque[Tuple[float,float,float,float,str]] = deque(maxlen=self.bootstrap_maxlen)
        self.pose_cache: Deque[Tuple[float, PoseStamped]] = deque(
            maxlen=int(self.path_H * max(1.0, self.path_rate)) + 50)
        self.initialized = False
        self.x = np.zeros((5,1), dtype=float)  # cx, cy, r, theta, omega
        self.P = np.eye(5, dtype=float)
        self.last_t: Optional[float] = None

        # Readiness tracking
        self._bootstrap_time: Optional[float] = None
        self._r_history: Deque[float] = deque(maxlen=max(10, self.ready_min_r_samples))
        self._ready = False

        # Diagnostics counters
        self._last_diag_t = time.time()
        self._meas_count = 0
        self._service_calls = 0

        # I/O
        self.sub = self.create_subscription(PointStamped, 'obj_1_coord', self.obj_cb, 100)
        if self.input_pose_topic:
            self.pose_sub = self.create_subscription(PoseStamped, self.input_pose_topic,
                                                     self.obj_pose_cb, 50)
        else:
            self.pose_sub = None

        # Publishers: make predicted_pick_pose transient local (latched) if you want late subscribers to get the last one-shot
        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.pub_pose   = self.create_publisher(PoseStamped, 'predicted_pick_pose', latched_qos)
        self.pub_path   = self.create_publisher(Path, 'predicted_ee_path', latched_qos)
        self.pub_traj   = self.create_publisher(MultiDOFJointTrajectory, 'predicted_ee_traj', latched_qos)
        self.pub_omega  = self.create_publisher(Float32, 'obj_omega_rad_s', latched_qos)
        self.pub_radius = self.create_publisher(Float32, 'obj_radius_m', latched_qos)
        self.pub_theta_now    = self.create_publisher(Float32, 'obj_theta_now', latched_qos)
        self.pub_theta_target = self.create_publisher(Float32, 'obj_theta_target', latched_qos)
        self.ready_pub = self.create_publisher(Bool, 'predictor_ready', 10)

        self._last_pred_pub_wall = 0.0
        self._last_path_build_wall = 0.0

        # Latency feedback
        self.alpha = 0.2
        self.create_subscription(Float32, 'latency/plan_s', self._plan_cb, 10)
        self.create_subscription(Float32, 'latency/exec_s', self._exec_cb, 10)

        # Service
        self.srv = self.create_service(GetPoseAt, 'get_predicted_pose_at', self.handle_get_pose)

        self.last_input_yaw: Optional[float] = None

        # IK gate one-shot wiring
        self.oneshot_on_reachable = bool(self.get_parameter('oneshot_on_reachable').value)
        self.ik_gate_topic = str(self.get_parameter('ik_gate_topic').value)
        self._oneshot_ready = True
        if self.oneshot_on_reachable:
            self.ik_sub = self.create_subscription(String, self.ik_gate_topic, self._ik_gate_cb, 10)
        else:
            self.ik_sub = None

        self.get_logger().info('Kalman (EKF) predictor started.')
        if self.verbose_debug:
            self.get_logger().info(
                f'Config: bootstrap_min={self.bootstrap_min}, path_rate={self.path_rate}, '
                f'horizon={self.path_H}s q={self.q_vals.tolist()} r={self.r_meas} '
                f'use_input_orientation={self.use_input_orientation} tangent_yaw={self.use_tangent_yaw}'
            )

    def _plan_cb(self, msg: Float32):
        self.planning_latency = (1 - self.alpha)*self.planning_latency + self.alpha*float(msg.data)
    def _exec_cb(self, msg: Float32):
        self.execution_latency = (1 - self.alpha)*self.execution_latency + self.alpha*float(msg.data)

    # Evaluate pose at now + planning+execution latencies and publish once
    def _ik_gate_cb(self, msg: String):
        data = (msg.data or "").upper()
        if data == 'REACHABLE':
            if self._oneshot_ready and self.initialized:
                t_now = self.get_clock().now().nanoseconds * 1e-9
                t_pick = t_now + (self.planning_latency + self.execution_latency)
                pose = self._eval_pose_at_time(t_pick)
                self.pub_pose.publish(pose)
                self._oneshot_ready = False
        else:
            # Reset when gate flips back
            self._oneshot_ready = True

    @staticmethod
    def _kasa_circle_fit(x: np.ndarray, y: np.ndarray):
        A = np.column_stack([2*x, 2*y, np.ones_like(x)])
        b = x**2 + y**2
        try:
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            cx, cy, c = sol
            r = math.sqrt(max(c + cx*cx + cy*cy, 0.0))
            return float(cx), float(cy), float(r)
        except np.linalg.LinAlgError:
            return None
        
    def _bootstrap(self) -> bool:
        if len(self.buf) < self.bootstrap_min:
            if self.verbose_debug and len(self.buf) % 10 == 0:
                self.get_logger().debug(f'Bootstrap waiting: {len(self.buf)}/{self.bootstrap_min}')
            return False
        t_arr = np.array([b[0] for b in self.buf], dtype=float)
        span_time = float(t_arr[-1] - t_arr[0])
        if span_time < self.bootstrap_min_time_span:
            if self.verbose_debug and len(self.buf) % 10 == 0:
                self.get_logger().debug(f'Bootstrap waiting (time span): {span_time:.3f}/{self.bootstrap_min_time_span}')
            return False
        xs = np.array([b[1] for b in self.buf], dtype=float)
        ys = np.array([b[2] for b in self.buf], dtype=float)
        cf = self._kasa_circle_fit(xs, ys)
        if cf is None:
            if self.verbose_debug:
                self.get_logger().warn('Circle fit failed; need more samples.')
            return False
        cx, cy, r = cf
        theta_raw = np.arctan2(ys - cy, xs - cx)
        theta_unwrap = np.unwrap(theta_raw)
        ang_span = float(theta_unwrap.max() - theta_unwrap.min())
        if ang_span < self.bootstrap_min_angle_span:
            if self.verbose_debug and len(self.buf) % 10 == 0:
                self.get_logger().debug(
                    f'Bootstrap waiting (angle span): {ang_span:.3f}/{self.bootstrap_min_angle_span}'
                )
            return False
        A = np.column_stack([t_arr, np.ones_like(t_arr)])
        try:
            coeff, *_ = np.linalg.lstsq(A, theta_unwrap, rcond=None)
        except np.linalg.LinAlgError:
            return False
        omega = float(coeff[0]); theta0 = float(coeff[1])
        t_last = t_arr[-1]; th_last = omega * t_last + theta0
        self.x = np.array([[cx],[cy],[r],[th_last],[omega]], dtype=float)
        self.P = np.diag([1e-4,1e-4,1e-4,1e-3,1e-4])
        self.initialized = True
        self.last_t = t_last
        self._bootstrap_time = time.time()
        self._r_history.clear()
        self._r_history.append(r)
        if self.verbose_debug:
            self.get_logger().info(
                f'Bootstrap: cx={cx:.4f} cy={cy:.4f} r={r:.4f} omega={omega:.5f} '
                f'samples={len(self.buf)} span={span_time:.3f}s angle_span={ang_span:.3f}'
            )
        self.ready_pub.publish(Bool(data=False))
        return True

    def _predict_step(self, dt: float):
        cx,cy,r,th,om = (self.x[i,0] for i in range(5))
        self.x = np.array([[cx],[cy],[r],[th + om*dt],[om]], dtype=float)
        F = np.eye(5); F[3,4] = dt
        Q = np.diag(self.q_vals) * max(dt, 1e-3)
        self.P = F @ self.P @ F.T + Q
        if self.verbose_debug:
            self.get_logger().debug(f'Predict dt={dt:.4f} theta-> {self.x[3,0]:.4f}')

    def _update_step(self, z: np.ndarray):
        cx,cy,r,th,_ = (self.x[i,0] for i in range(5))
        c = math.cos(th); s = math.sin(th)
        hx = np.array([[cx + r*c],[cy + r*s]])
        H = np.zeros((2,5))
        H[0,0]=1; H[1,1]=1; H[0,2]=c; H[1,2]=s; H[0,3]=-r*s; H[1,3]= r*c
        R = np.eye(2) * self.r_meas
        y = z - hx
        if self.log_innovation:
            innov_norm = float(np.linalg.norm(y))
            self.get_logger().debug(
                f'Update z=({z[0,0]:.4f},{z[1,0]:.4f}) hx=({hx[0,0]:.4f},{hx[1,0]:.4f}) |y|={innov_norm:.6e}'
            )
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(5)
        self.P = (I - K @ H) @ self.P
        self.x[3,0] = math.atan2(math.sin(self.x[3,0]), math.cos(self.x[3,0]))
        if self.verbose_debug:
            cx,cy,r,th,om = (self.x[i,0] for i in range(5))
            self.get_logger().debug(
                f'State cx={cx:.4f} cy={cy:.4f} r={r:.4f} th={th:.4f} om={om:.5f}'
            )
    # Helper: evaluate pose at absolute time (no cache scan)
    def _eval_pose_at_time(self, t_abs: float) -> PoseStamped:
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = max(0.0, t_abs - now)
        cx, cy, r, th, om = (self.x[i,0] for i in range(5))
        th_q = th + om * dt
        x_pred = cx + r*math.cos(th_q)
        y_pred = cy + r*math.sin(th_q)
        # choose z and yaw like in process_measurement
        z_pred = self.fixed_pick_z if not self.use_last_z else (self.buf[-1][3] if self.buf else 0.0)
        if self.use_tangent_yaw:
            yaw = th_q + math.pi/2.0
        elif self.use_input_orientation and self.last_input_yaw is not None:
            yaw = self.last_input_yaw
        else:
            yaw = 0.0
        q = yaw_to_quat(yaw)
        ps = PoseStamped()
        ps.header.stamp = sec_to_time_msg(t_abs)
        ps.header.frame_id = self.output_frame_id_param or (self.buf[-1][4] if self.buf else 'world')
        ps.pose.position.x = x_pred
        ps.pose.position.y = y_pred
        ps.pose.position.z = z_pred
        ps.pose.orientation = q
        return ps

    def handle_get_pose(self, req, res):
        self._service_calls += 1
        # Direct evaluation (preferred; avoids heavy cache rebuilds)
        if self.service_eval_direct:
            t_now = self.get_clock().now().nanoseconds * 1e-9
            if hasattr(req, 'use_relative') and req.use_relative:
                dt = float(req.query_time.sec) + 1e-9*float(req.query_time.nanosec)
                t_target = t_now + max(0.0, dt)
            elif hasattr(req, 't_rel_s'):
                t_target = t_now + max(0.0, float(req.t_rel_s))
            else:
                t_target = ros_time_to_sec(getattr(req, 'query_time', TimeMsg()))
            res.ok = True
            res.pose = self._eval_pose_at_time(t_target)
            return res

    # old
    # def handle_get_pose(self, req, res):
    #     self._service_calls += 1
        if not self.pose_cache:
            res.ok = False
            if self.verbose_debug:
                self.get_logger().warn('Service: pose_cache empty.')
            return res
    #     if hasattr(req, 'use_relative') and req.use_relative:
    #         t_now = self.get_clock().now().nanoseconds * 1e-9
    #         dt = ros_time_to_sec(req.query_time)
    #         t_target = t_now + dt
    #     elif hasattr(req, 't_rel_s'):
    #         t_now = self.get_clock().now().nanoseconds * 1e-9
    #         t_target = t_now + float(req.t_rel_s)
    #     else:
    #         t_target = ros_time_to_sec(getattr(req, 'query_time', TimeMsg()))
    #     best = min(self.pose_cache, key=lambda kv: abs(kv[0]-t_target))
    #     res.ok = True
    #     res.pose = best[1]
    #     if self.verbose_debug:
    #         self.get_logger().debug(
    #             f'Service#{self._service_calls} target={t_target:.6f} chosen={best[0]:.6f} err={abs(best[0]-t_target):.6f}'
    #         )
    #     return res

    def obj_cb(self, msg: PointStamped):
        self.process_measurement(
            ros_time_to_sec(msg.header.stamp),
            float(msg.point.x),
            float(msg.point.y),
            float(msg.point.z),
            frame_in=msg.header.frame_id or 'world'
        )

    def obj_pose_cb(self, msg: PoseStamped):
        q = msg.pose.orientation
        siny_cosp = 2*(q.w*q.z + q.x*q.y)
        cosy_cosp = 1 - 2*(q.y*q.y + q.z*q.z)
        self.last_input_yaw = math.atan2(siny_cosp, cosy_cosp)
        self.process_measurement(
            ros_time_to_sec(msg.header.stamp),
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z),
            frame_in=msg.header.frame_id or 'world'
        )

    def process_measurement(self, t_meas: float, x_meas: float, y_meas: float, z_meas: float, frame_in: str):
        wall_now = time.time()
        self._meas_count += 1
        self.buf.append((t_meas, x_meas, y_meas, z_meas, frame_in))

        if not self.initialized:
            if not self._bootstrap():
                if self.verbose_debug and self._meas_count % 20 == 0:
                    self.get_logger().debug(f'Buffering measurements: {len(self.buf)}/{self.bootstrap_min}')
                return
            self.last_t = t_meas

        raw_dt = t_meas - (self.last_t if self.last_t is not None else t_meas)
        if raw_dt <= 0.0:
            dt = 1e-4
        else:
            dt = min(raw_dt, self.max_dt_cap)
        if self.last_t is None or t_meas > self.last_t:
            self.last_t = t_meas

        self._predict_step(dt)
        self._update_step(np.array([[x_meas],[y_meas]]))

        # Radius stability tracking
        self._r_history.append(float(self.x[2,0]))
        if not self._ready and self._bootstrap_time:
            age = wall_now - self._bootstrap_time
            if age >= self.ready_min_age and len(self._r_history) >= self.ready_min_r_samples:
                var_r = float(np.var(self._r_history))
                if var_r <= self.ready_r_var_max:
                    self._ready = True
                    self.ready_pub.publish(Bool(data=True))
                    self.get_logger().info(
                        f'Predictor READY (age={age:.2f}s r_var={var_r:.2e} samples={len(self._r_history)})'
                    )
                else:
                    if self.verbose_debug and self._meas_count % 50 == 0:
                        self.get_logger().debug(
                            f'Not ready: age={age:.2f} r_var={var_r:.2e} thr={self.ready_r_var_max}'
                        )

        # Pick prediction
        now_ros = self.get_clock().now().to_msg()
        now = ros_time_to_sec(now_ros)
        horizon = self.planning_latency + self.execution_latency
        t_pick = now + horizon
        dt_pick = max(0.0, t_pick - t_meas)
        cx,cy,r,th,om = (self.x[i,0] for i in range(5))
        th_pick = th + om * dt_pick
        x_pred = cx + r*math.cos(th_pick)
        y_pred = cy + r*math.sin(th_pick)
        z_pred = z_meas if self.use_last_z else self.fixed_pick_z

        if self.use_tangent_yaw:
            yaw = th_pick + math.pi/2.0
        elif self.use_input_orientation and self.last_input_yaw is not None:
            yaw = self.last_input_yaw
        else:
            yaw = 0.0

        q = yaw_to_quat(yaw)
        frame_out = self.output_frame_id_param or frame_in

        pick_pose = PoseStamped()
        pick_pose.header.frame_id = frame_out
        pick_pose.header.stamp = sec_to_time_msg(t_pick)
        pick_pose.pose.position.x = x_pred
        pick_pose.pose.position.y = y_pred
        pick_pose.pose.position.z = z_pred
        pick_pose.pose.orientation = q
        #self.pub_pose.publish(pick_pose)

        # # Path
        # H = self.path_H; rate = self.path_rate; N = max(1, int(H * rate))
        # path = Path(); path.header.frame_id = frame_out; path.header.stamp = now_ros
        # self.pose_cache.clear()
        # for i in range(N):
        #     t_rel = i / rate
        #     th_i = th + om * t_rel
        #     px = cx + r*math.cos(th_i); py = cy + r*math.sin(th_i); pz = z_pred
        #     if self.use_tangent_yaw:
        #         yaw_i = th_i + math.pi/2.0
        #     elif self.use_input_orientation and self.last_input_yaw is not None:
        #         yaw_i = self.last_input_yaw
        #     else:
        #         yaw_i = 0.0
        #     ps = PoseStamped()
        #     t_abs = now + t_rel
        #     ps.header.frame_id = frame_out
        #     ps.header.stamp = sec_to_time_msg(t_abs)
        #     ps.pose.position.x = px; ps.pose.position.y = py; ps.pose.position.z = pz
        #     ps.pose.orientation = yaw_to_quat(yaw_i)
        #     path.poses.append(ps)
        #     self.pose_cache.append((t_abs, ps))
        # self.pub_path.publish(path)
        # Throttle predicted_pick_pose topic
        if self.publish_predicted_pose and (wall_now - self._last_pred_pub_wall) >= (1.0 / max(1e-3, self.pred_rate)):
            self.pub_pose.publish(pick_pose)
            self._last_pred_pub_wall = wall_now

        # Throttle path (and cache) publication
        if (wall_now - self._last_path_build_wall) >= (1.0 / max(1e-3, self.path_publish_rate)):
            self._last_path_build_wall = wall_now
            H = self.path_H; rate = self.path_rate
            N = max(1, int(H * rate))
            path = Path(); path.header.frame_id = frame_out; path.header.stamp = now_ros
            self.pose_cache.clear()
            for i in range(N):
                t_rel = i / rate
                th_i = th + om * t_rel
                px = cx + r*math.cos(th_i); py = cy + r*math.sin(th_i); pz = z_pred
                if self.use_tangent_yaw:
                    yaw_i = th_i + math.pi/2.0
                elif self.use_input_orientation and self.last_input_yaw is not None:
                    yaw_i = self.last_input_yaw
                else:
                    yaw_i = 0.0
                ps = PoseStamped()
                t_abs = now + t_rel
                ps.header.frame_id = frame_out
                ps.header.stamp = sec_to_time_msg(t_abs)
                ps.pose.position.x = px; ps.pose.position.y = py; ps.pose.position.z = pz
                ps.pose.orientation = yaw_to_quat(yaw_i)
                path.poses.append(ps)
                self.pose_cache.append((t_abs, ps))
            self.pub_path.publish(path)

        # Diagnostics topics
        self.pub_omega.publish(Float32(data=float(om)))
        self.pub_radius.publish(Float32(data=float(r)))
        self.pub_theta_now.publish(Float32(data=float(th)))
        self.pub_theta_target.publish(Float32(data=float(th_pick)))

        if self.verbose_debug and self._meas_count % 25 == 0:
            self.get_logger().info(
                f'Meas#{self._meas_count} t={t_meas:.3f} dt={dt:.4f} cx={cx:.3f} cy={cy:.3f} r={r:.3f} om={om:.4f}'
            )
        if time.time() - self._last_diag_t > self.diag_interval:
            self._last_diag_t = time.time()
            self.get_logger().info(
                f'Heartbeat: init={self.initialized} ready={self._ready} meas={self._meas_count} cache={len(self.pose_cache)} omega={om:.4f}'
            )

def main(args=None):
    rclpy.init(args=args)
    node = EKFPredictor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
