#include <rclcpp/rclcpp.hpp>
#include <rclcpp/parameter_client.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <eigen3/Eigen/Geometry>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/robot_state/conversions.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit_msgs/msg/move_it_error_codes.h>
#include <moveit_visual_tools/moveit_visual_tools.h>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <moveit_msgs/msg/constraints.hpp>
#include <moveit_msgs/msg/orientation_constraint.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/planning_scene.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

#include <moveit/trajectory_processing/time_optimal_trajectory_generation.h>
#include <moveit/trajectory_processing/ruckig_traj_smoothing.h>

#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/time.h>

#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <lite6_pick_predictor_interfaces/srv/get_pose_at.hpp>

#include <algorithm>
#include <limits>
#include <cmath>
#include <atomic>
#include <string>
#include <vector>
#include <chrono>

namespace tp_control_cpp {

enum class TimingMethod { TOTG, RUCKIG, TOTG_THEN_RUCKIG };

struct TimingTargets
{
  double t_hover{0.0}; // min time to hover (0=disabled)
  double t_grasp{0.0}; // min time to grasp (0=disabled)
};

inline double durationFromStart(const robot_trajectory::RobotTrajectory& rt, size_t idx)
{
  if (rt.getWayPointCount() == 0 || idx == 0) return 0.0;
  idx = std::min(idx, rt.getWayPointCount() - 1);
  double sum = 0.0;
  for (size_t i = 1; i <= idx; ++i)
    sum += rt.getWayPointDurationFromPrevious(i);
  return sum;
}

inline bool applyTimeParameterization(robot_trajectory::RobotTrajectory& rt,
                                      TimingMethod method,
                                      double max_vel_scaling = 0.9,
                                      double max_acc_scaling = 0.7)
{
  using trajectory_processing::TimeOptimalTrajectoryGeneration;
  using trajectory_processing::RuckigSmoothing;

  switch (method)
  {
    case TimingMethod::TOTG:
    {
      TimeOptimalTrajectoryGeneration totg;
      return totg.computeTimeStamps(rt, max_vel_scaling, max_acc_scaling);
    }
    case TimingMethod::RUCKIG:
    {
      return RuckigSmoothing::applySmoothing(rt, max_vel_scaling, max_acc_scaling);
    }
    case TimingMethod::TOTG_THEN_RUCKIG:
    {
      TimeOptimalTrajectoryGeneration totg;
      if (!totg.computeTimeStamps(rt, max_vel_scaling, max_acc_scaling))
        return false;
      return RuckigSmoothing::applySmoothing(rt, max_vel_scaling, max_acc_scaling);
    }
    default:
      return false;
  }
}

// Old helper kept (min-time only) in case you need it elsewhere
inline bool enforceSegmentTimes(robot_trajectory::RobotTrajectory& rt,
                                size_t idx_hover,
                                size_t idx_grasp,
                                const TimingTargets& targets)
{
  const size_t N = rt.getWayPointCount();
  if (N < 2) return false;
  idx_hover = std::min(idx_hover, N - 1);
  idx_grasp = std::min(idx_grasp, N - 1);
  if (idx_hover >= idx_grasp) idx_hover = std::max<size_t>(1, std::min(idx_grasp, idx_hover));

  auto safe_get = [&](size_t i) { return rt.getWayPointDurationFromPrevious(i); };
  auto safe_set = [&](size_t i, double v) { rt.setWayPointDurationFromPrevious(i, std::max(1e-9, v)); };

  bool changed = false;

  double t_hover_cur = durationFromStart(rt, idx_hover);
  if (targets.t_hover > 0.0 && t_hover_cur < targets.t_hover && idx_hover >= 1)
  {
    const double scale1 = targets.t_hover / std::max(1e-9, t_hover_cur);
    for (size_t i = 1; i <= idx_hover; ++i)
      safe_set(i, safe_get(i) * scale1);
    changed = true;
  }

  double t_hover_new = durationFromStart(rt, idx_hover);
  const double t_total_cur = durationFromStart(rt, N - 1);
  const double tail_cur = std::max(0.0, t_total_cur - t_hover_new);

  if (targets.t_grasp > 0.0 && (t_hover_new + tail_cur) < targets.t_grasp && idx_grasp >= (idx_hover + 1))
  {
    const double needed_tail = targets.t_grasp - t_hover_new;
    if (tail_cur <= 1e-9)
    {
      double last = safe_get(N - 1);
      if (last < 1e-6) last = 1e-3;
      safe_set(N - 1, last + std::max(0.0, needed_tail));
    }
    else
    {
      const double scale2 = std::max(1.0, needed_tail / tail_cur);
      for (size_t i = idx_hover + 1; i <= idx_grasp; ++i)
        safe_set(i, safe_get(i) * scale2);
    }
    changed = true;
  }

  return changed;
}

// New: bound a segment duration to [min_time, max_time] by scaling its segment-only durations
inline bool enforceSegmentTimeBounded(robot_trajectory::RobotTrajectory& rt,
                                      size_t idx_start,
                                      size_t idx_end,
                                      double min_time,
                                      double max_time)
{
  const size_t N = rt.getWayPointCount();
  if (N < 2) return false;
  idx_start = std::min(idx_start, N - 1);
  idx_end = std::min(idx_end, N - 1);
  if (idx_end <= idx_start) return false;

  auto seg_time = durationFromStart(rt, idx_end) - durationFromStart(rt, idx_start);
  if (seg_time <= 1e-9) return false;

  double scale = 1.0;
  if (max_time > 0.0 && seg_time > max_time)
    scale = std::min(scale, std::max(1e-6, max_time / seg_time));
  if (min_time > 0.0 && seg_time < min_time)
    scale = std::max(scale, std::max(1e-6, min_time / seg_time));

  if (std::abs(scale - 1.0) < 1e-6) return false;

  for (size_t i = idx_start + 1; i <= idx_end; ++i) {
    double d = rt.getWayPointDurationFromPrevious(i);
    rt.setWayPointDurationFromPrevious(i, std::max(1e-9, d * scale));
  }
  return true;
}

} // namespace tp_control_cpp

using tp_control_cpp::TimingMethod;
using tp_control_cpp::TimingTargets;

class TpControlNode : public rclcpp::Node
{
public:
  TpControlNode() : Node("tp_control_node")
  {
    this->declare_parameter<std::string>("robot_description", "");
    this->declare_parameter<std::string>("robot_description_semantic", "");
    this->declare_parameter<std::string>("planning_group", "lite6");
    this->declare_parameter<std::string>("ee_link", "link_tcp");
    this->declare_parameter<std::string>("base_frame", "link_base");
    this->declare_parameter<std::string>("timing_method", "totg_then_ruckig");
    this->declare_parameter<double>("planning_time", 10.0);
    this->declare_parameter<std::string>("planner_id", "RRTConnect");
    this->declare_parameter<bool>("use_pose_target", false);
    this->declare_parameter<std::string>("target_mode", "absolute");
    this->declare_parameter<std::string>("target_frame", "");

    this->declare_parameter<double>("vel_scale", 0.9);
    this->declare_parameter<double>("acc_scale", 0.6);

    // New per-leg scaling (fallback to vel_scale/acc_scale if <=0)
    this->declare_parameter<double>("vel_scale_leg1", -1.0);
    this->declare_parameter<double>("acc_scale_leg1", -1.0);
    this->declare_parameter<double>("vel_scale_descent", -1.0);
    this->declare_parameter<double>("acc_scale_descent", -1.0);

    this->declare_parameter<double>("target_x", 0.29);
    this->declare_parameter<double>("target_y", -0.16);
    this->declare_parameter<double>("target_z", 0.14);
    this->declare_parameter<double>("z_hover", 0.29);
    this->declare_parameter<double>("z_grasp", 0.14);
    this->declare_parameter<double>("target_roll", 0.0);
    this->declare_parameter<double>("target_pitch", 0.0);
    this->declare_parameter<double>("target_yaw", 0.0);
    this->declare_parameter<bool>("keep_orientation", false);

    // Horizontal orientation helpers
    this->declare_parameter<bool>("enforce_horizontal_orientation", false);
    this->declare_parameter<std::string>("horizontal_yaw_mode", "predictor"); // predictor|current|fixed
    this->declare_parameter<double>("fixed_yaw", 0.0);
    this->declare_parameter<double>("horizontal_roll_offset", M_PI);
    this->declare_parameter<double>("constraint_rp_tolerance", 0.5);
    this->declare_parameter<double>("constraint_yaw_tolerance", M_PI);
    this->declare_parameter<bool>("use_path_constraints_for_hover", true);
    this->declare_parameter<std::string>("constraint_mode", "goal_only");

    // Rotating table collision object params
    this->declare_parameter<bool>("add_table_collision", true);
    this->declare_parameter<std::string>("table_frame", "link_base");
    this->declare_parameter<double>("table_radius", 0.75);
    this->declare_parameter<double>("table_height", 0.06);
    this->declare_parameter<double>("table_x", 0.9);
    this->declare_parameter<double>("table_y", 0.0);
    this->declare_parameter<double>("table_z", 0.1);
    this->declare_parameter<std::string>("table_object_id", "rotating_table");
    this->declare_parameter<double>("table_clearance", 0.005);

    // Timing targets (min) + new max caps
    this->declare_parameter<double>("t_hover", 0.0);
    this->declare_parameter<double>("t_grasp", 0.0);
    this->declare_parameter<double>("t_hover_max", 0.0);
    this->declare_parameter<double>("t_grasp_max", 0.0);

    // IK gate
    this->declare_parameter<bool>("require_ik_gate", true);
    this->declare_parameter<std::string>("ik_gate_topic", "/ik_gate/output");

    // Predictor
    this->declare_parameter<bool>("use_predictor", true);
    this->declare_parameter<bool>("require_predictor_ready", true);
    this->declare_parameter<std::string>("predictor_ready_topic", "predictor_ready");
    this->declare_parameter<std::string>("predictor_service", "get_predicted_pose_at");
    this->declare_parameter<double>("commit_pick_time_s", 10.0);

    // Cartesian descent execution guards and options
    this->declare_parameter<double>("min_cart_fraction", 0.15);
    this->declare_parameter<double>("min_descent_dz", 0.03);
    this->declare_parameter<bool>("cartesian_avoid_collisions", true);
    this->declare_parameter<bool>("allow_ee_table_touch", false);

    // Cooldown
    this->declare_parameter<double>("replan_cooldown_s", 2.0);

    // Gripper
    this->declare_parameter<std::string>("gripper_controller_topic", "/lite6_gripper_traj_controller/joint_trajectory");
    this->declare_parameter<std::vector<std::string>>("gripper_joints", {"jaw_left","jaw_right"});
    this->declare_parameter<std::vector<double>>("gripper_open", {0.02, 0.02});
    this->declare_parameter<std::vector<double>>("gripper_close", {0.0, 0.0});
    this->declare_parameter<double>("gripper_motion_time", 1.0);

    // Detect /clock
    auto topics = this->get_topic_names_and_types();
    bool has_clock = std::any_of(topics.begin(), topics.end(), [](const auto& t){ return t.first == "/clock"; });
    this->set_parameter(rclcpp::Parameter("use_sim_time", has_clock));
    RCLCPP_INFO(get_logger(), has_clock ? "Found /clock, use_sim_time=true" : "No /clock, use_sim_time=false");

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    display_pub_ = this->create_publisher<moveit_msgs::msg::DisplayTrajectory>("display_planned_path", rclcpp::QoS(5));
    planning_scene_pub_ = this->create_publisher<moveit_msgs::msg::PlanningScene>("/planning_scene", 10);

    get_parameters();

    RCLCPP_INFO(get_logger(), "TpControlNode initializing...");
  }

  void initialize()
  {
    try
    {
      if (!check_joint_states_topic()) {
        RCLCPP_ERROR(get_logger(), "Joint states topic not available, cannot proceed");
        return;
      }
      if (!wait_for_move_group_parameters()) {
        RCLCPP_ERROR(get_logger(), "Could not get robot description from move_group");
        return;
      }
      move_group_params_ready_ = true;

      move_group_ = std::make_unique<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), planning_group_);
      move_group_->setEndEffectorLink(ee_link_);
      move_group_->startStateMonitor(5.0);

      visual_tools_ = std::make_shared<moveit_visual_tools::MoveItVisualTools>(
          shared_from_this(), base_frame_, "/rviz_visual_tools", move_group_->getRobotModel());
      visual_tools_->deleteAllMarkers();
      visual_tools_->loadRemoteControl();

      RCLCPP_INFO(get_logger(), "TpControlNode ready. Method=%s", timing_method_str_.c_str());

      if (!wait_for_valid_joint_states_with_time()) {
        RCLCPP_ERROR(get_logger(), "Failed to receive valid joint states");
        return;
      }
      auto cs = move_group_->getCurrentState(0.0);
      if (!cs){
        RCLCPP_ERROR(get_logger(), "MoveGroup current state unavailable");
        return;
      }
      move_group_->setStartStateToCurrentState();

      // Predictor wiring
      if (use_predictor_) {
        predictor_ready_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            predictor_ready_topic_, 10,
            [this](const std_msgs::msg::Bool::SharedPtr msg){
              predictor_ready_.store(msg->data, std::memory_order_relaxed);
            });
        pick_client_ = this->create_client<lite6_pick_predictor_interfaces::srv::GetPoseAt>(predictor_service_);
      }

      // Add rotating table collision object
      if (add_table_collision_) {
        geometry_msgs::msg::Pose table_pose;
        table_pose.position.x = table_x_;
        table_pose.position.y = table_y_;
        table_pose.position.z = table_z_;
        table_pose.orientation.w = 1.0; // identity
        add_rotating_table_collision_(table_pose, table_radius_, table_height_);
        RCLCPP_INFO(get_logger(), "Applied rotating_table collision object at [%.2f, %.2f, %.2f] (r=%.2f, h=%.2f, frame=%s)",
                    table_x_, table_y_, table_z_, table_radius_, table_height_, table_frame_.c_str());
        rclcpp::sleep_for(std::chrono::milliseconds(200)); // allow planning scene sync

        if (allow_ee_table_touch_) {
          allow_ee_table_collision_(true);
          RCLCPP_WARN(get_logger(), "Allowed collisions between '%s' and table object '%s'",
                      ee_link_.c_str(), table_object_id_.c_str());
        }
      }

      // IK gate rising-edge trigger + cooldown
      if (require_ik_gate_) {
        ik_gate_sub_ = this->create_subscription<std_msgs::msg::String>(
            ik_gate_topic_, 10,
            [this](const std_msgs::msg::String::SharedPtr msg) {
              const bool reachable = (msg->data == "REACHABLE");
              if (reachable && !last_ik_reachable_) {
                const auto now = std::chrono::steady_clock::now();
                if (!planning_in_progress_ && (now - last_attempt_wall_) >= std::chrono::duration<double>(replan_cooldown_s_)) {
                  planning_in_progress_ = true;
                  delayed_plan_timer_ = this->create_wall_timer(
                      std::chrono::milliseconds(5),
                      [this]() {
                        if (delayed_plan_timer_) delayed_plan_timer_->cancel();
                        this->run_once_with_predictor_();
                      });
                } else {
                  RCLCPP_WARN(get_logger(), "Cooldown active; skipping trigger");
                }
              }
              last_ik_reachable_ = reachable;
            });
      } else {
        delayed_plan_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(150),
            [this]() {
              if (delayed_plan_timer_) delayed_plan_timer_->cancel();
              if (!planning_in_progress_) {
                planning_in_progress_ = true;
                this->run_once_with_predictor_();
              }
            });
      }

      debug_robot_model();
    }
    catch (const std::exception& e)
    {
      RCLCPP_ERROR(get_logger(), "Failed to initialize: %s", e.what());
    }
  }

private:

  bool wait_for_move_group_parameters(double timeout_seconds = 20.0)
  {
    if (move_group_params_ready_) return true;
    RCLCPP_INFO(get_logger(), "Looking for move_group parameters...");
    auto start_time = this->get_clock()->now();

    while (rclcpp::ok() && (this->get_clock()->now() - start_time).seconds() < timeout_seconds) {
      auto node_names = this->get_node_names();
      for (const auto& node_name : node_names) {
        if (node_name.find("move_group") != std::string::npos) {
          RCLCPP_INFO(get_logger(), "Found move_group node: %s", node_name.c_str());
          try {
            auto param_client = std::make_shared<rclcpp::AsyncParametersClient>(this, node_name);
            if (param_client->wait_for_service(std::chrono::seconds(2))) {
              auto future = param_client->get_parameters(
                {"robot_description", "robot_description_semantic", "robot_description_kinematics"});

              if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), future,
                  std::chrono::seconds(5)) == rclcpp::FutureReturnCode::SUCCESS) {

                auto params = future.get();
                for (const auto& param : params) {
                  if (param.get_name() == "robot_description" &&
                      param.get_type() == rclcpp::ParameterType::PARAMETER_STRING) {
                    this->set_parameter(rclcpp::Parameter("robot_description", param.as_string()));
                    RCLCPP_INFO(get_logger(), "✓ robot_description copied");
                  } else if (param.get_name() == "robot_description_semantic" &&
                             param.get_type() == rclcpp::ParameterType::PARAMETER_STRING) {
                    this->set_parameter(rclcpp::Parameter("robot_description_semantic", param.as_string()));
                    RCLCPP_INFO(get_logger(), "✓ robot_description_semantic copied");
                  } else if (param.get_name() == "robot_description_kinematics" &&
                             param.get_type() == rclcpp::ParameterType::PARAMETER_STRING) {
                    this->set_parameter(rclcpp::Parameter("robot_description_kinematics", param.as_string()));
                    RCLCPP_INFO(get_logger(), "✓ robot_description_kinematics copied");
                  }
                }
              }
              move_group_params_ready_ = true;
              return true;
            }
          } catch (const std::exception& e) {
            RCLCPP_WARN(get_logger(), "Failed to get parameters: %s", e.what());
          }
        }
      }
      RCLCPP_INFO_THROTTLE(get_logger(), *this->get_clock(), 3000,
                           "Waiting for move_group node to be available...");
      rclcpp::sleep_for(std::chrono::milliseconds(300));
    }
    RCLCPP_ERROR(get_logger(), "Timeout waiting for move_group parameters");
    return false;
  }

  void get_parameters()
  {
    planning_group_ = this->get_parameter("planning_group").as_string();
    ee_link_ = this->get_parameter("ee_link").as_string();
    base_frame_ = this->get_parameter("base_frame").as_string();
    z_hover_ = this->get_parameter("z_hover").as_double();
    z_grasp_ = this->get_parameter("z_grasp").as_double();
    targets_.t_hover = this->get_parameter("t_hover").as_double();
    targets_.t_grasp = this->get_parameter("t_grasp").as_double();
    t_hover_max_ = this->get_parameter("t_hover_max").as_double();
    t_grasp_max_ = this->get_parameter("t_grasp_max").as_double();

    timing_method_str_ = this->get_parameter("timing_method").as_string();
    vel_scale_ = this->get_parameter("vel_scale").as_double();
    acc_scale_ = this->get_parameter("acc_scale").as_double();

    // Per-leg scaling (fallbacks)
    vel_scale_leg1_ = this->get_parameter("vel_scale_leg1").as_double();
    acc_scale_leg1_ = this->get_parameter("acc_scale_leg1").as_double();
    vel_scale_descent_ = this->get_parameter("vel_scale_descent").as_double();
    acc_scale_descent_ = this->get_parameter("acc_scale_descent").as_double();
    if (vel_scale_leg1_ <= 0.0) vel_scale_leg1_ = vel_scale_;
    if (acc_scale_leg1_ <= 0.0) acc_scale_leg1_ = acc_scale_;
    if (vel_scale_descent_ <= 0.0) vel_scale_descent_ = vel_scale_;
    if (acc_scale_descent_ <= 0.0) acc_scale_descent_ = acc_scale_;

    if (timing_method_str_ == "totg") {
      method_ = TimingMethod::TOTG;
    } else if (timing_method_str_ == "ruckig") {
      method_ = TimingMethod::RUCKIG;
    } else {
      method_ = TimingMethod::TOTG_THEN_RUCKIG;
      timing_method_str_ = "totg_then_ruckig";
    }

    use_pose_target_ = this->get_parameter("use_pose_target").as_bool();
    target_mode_ = this->get_parameter("target_mode").as_string();
    target_frame_ = this->get_parameter("target_frame").as_string();
    if (target_frame_.empty()) target_frame_ = base_frame_;
    target_x_ = this->get_parameter("target_x").as_double();
    target_y_ = this->get_parameter("target_y").as_double();
    target_z_ = this->get_parameter("target_z").as_double();
    keep_orientation_ = this->get_parameter("keep_orientation").as_bool();
    target_roll_ = this->get_parameter("target_roll").as_double();
    target_pitch_ = this->get_parameter("target_pitch").as_double();
    target_yaw_ = this->get_parameter("target_yaw").as_double();

    enforce_horizontal_ = this->get_parameter("enforce_horizontal_orientation").as_bool();
    horizontal_yaw_mode_ = this->get_parameter("horizontal_yaw_mode").as_string();
    fixed_yaw_ = this->get_parameter("fixed_yaw").as_double();
    horizontal_roll_offset_ = this->get_parameter("horizontal_roll_offset").as_double();
    rp_tol_ = this->get_parameter("constraint_rp_tolerance").as_double();
    yaw_tol_ = this->get_parameter("constraint_yaw_tolerance").as_double();
    use_path_constraints_for_hover_ = this->get_parameter("use_path_constraints_for_hover").as_bool();
    constraint_mode_ = this->get_parameter("constraint_mode").as_string();
    if (constraint_mode_ != "goal_only" && constraint_mode_ != "path_only" && constraint_mode_ != "both") {
      RCLCPP_WARN(get_logger(), "Invalid constraint_mode '%s', defaulting to 'path_only'", constraint_mode_.c_str());
      constraint_mode_ = "path_only";
    }

    // Predictor params
    use_predictor_ = this->get_parameter("use_predictor").as_bool();
    require_predictor_ready_ = this->get_parameter("require_predictor_ready").as_bool();
    predictor_ready_topic_ = this->get_parameter("predictor_ready_topic").as_string();
    predictor_service_ = this->get_parameter("predictor_service").as_string();
    commit_pick_time_s_ = this->get_parameter("commit_pick_time_s").as_double();

    replan_cooldown_s_ = this->get_parameter("replan_cooldown_s").as_double();

    // Planner params
    planning_time_ = this->get_parameter("planning_time").as_double();
    planner_id_ = this->get_parameter("planner_id").as_string();

    // IK gate params
    require_ik_gate_ = this->get_parameter("require_ik_gate").as_bool();
    ik_gate_topic_ = this->get_parameter("ik_gate_topic").as_string();

    // Gripper params
    gripper_controller_topic_ = this->get_parameter("gripper_controller_topic").as_string();
    gripper_joints_ = this->get_parameter("gripper_joints").as_string_array();
    gripper_open_pos_ = this->get_parameter("gripper_open").as_double_array();
    gripper_close_pos_ = this->get_parameter("gripper_close").as_double_array();
    gripper_motion_time_ = this->get_parameter("gripper_motion_time").as_double();
    gripper_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(gripper_controller_topic_, 10);

    // Rotating table collision params
    add_table_collision_ = this->get_parameter("add_table_collision").as_bool();
    table_frame_ = this->get_parameter("table_frame").as_string();
    if (table_frame_.empty()) table_frame_ = base_frame_;
    table_radius_ = this->get_parameter("table_radius").as_double();
    table_height_ = this->get_parameter("table_height").as_double();
    table_x_ = this->get_parameter("table_x").as_double();
    table_y_ = this->get_parameter("table_y").as_double();
    table_z_ = this->get_parameter("table_z").as_double();
    table_object_id_ = this->get_parameter("table_object_id").as_string();
    table_clearance_ = this->get_parameter("table_clearance").as_double();

    // Cartesian execution guards and options
    min_cart_fraction_ = this->get_parameter("min_cart_fraction").as_double();
    min_descent_dz_ = this->get_parameter("min_descent_dz").as_double();
    cartesian_avoid_collisions_ = this->get_parameter("cartesian_avoid_collisions").as_bool();
    allow_ee_table_touch_ = this->get_parameter("allow_ee_table_touch").as_bool();
  }

  // Gripper
  bool command_gripper_(const std::vector<double>& pos, double seconds, double start_delay_s = 0.0) {
    if (!gripper_pub_) return false;
    if (pos.size() != gripper_joints_.size()) {
      RCLCPP_ERROR(get_logger(), "Gripper command size mismatch");
      return false;
    }
    trajectory_msgs::msg::JointTrajectory traj;
    traj.header.stamp = this->get_clock()->now() + rclcpp::Duration::from_seconds(std::max(0.0, start_delay_s));
    traj.joint_names = gripper_joints_;
    trajectory_msgs::msg::JointTrajectoryPoint pt;
    pt.positions = pos;
    pt.time_from_start = rclcpp::Duration::from_seconds(seconds);
    traj.points.push_back(pt);
    gripper_pub_->publish(traj);
    return true;
  }
  bool open_gripper_(double start_delay_s = 0.0) { return command_gripper_(gripper_open_pos_, gripper_motion_time_, start_delay_s); }
  bool close_gripper_(double start_delay_s = 0.0) { return command_gripper_(gripper_close_pos_, gripper_motion_time_, start_delay_s); }

  void schedule_gripper_close_for_plan_(const moveit::planning_interface::MoveGroupInterface::Plan& plan) {
    if (plan.trajectory_.joint_trajectory.points.empty()) return;
    const auto& last_tfs = plan.trajectory_.joint_trajectory.points.back().time_from_start;
    const double traj_total = rclcpp::Duration(last_tfs).seconds();
    double desired_start = targets_.t_grasp - gripper_motion_time_;
    double latest_start = std::max(0.0, traj_total - gripper_motion_time_);
    double start_delay = std::clamp(desired_start, 0.0, latest_start);
    (void)close_gripper_(start_delay);
  }

  geometry_msgs::msg::Quaternion quatFromRPY(double r, double p, double y) {
    Eigen::AngleAxisd Rx(r, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd Ry(p, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd Rz(y, Eigen::Vector3d::UnitZ());
    Eigen::Quaterniond q = Rz * Ry * Rx;
    geometry_msgs::msg::Quaternion qmsg;
    qmsg.x = q.x(); qmsg.y = q.y(); qmsg.z = q.z(); qmsg.w = q.w();
    return qmsg;
  }
  static double yawFromQuat(const geometry_msgs::msg::Quaternion& q) {
    Eigen::Quaterniond qe(q.w, q.x, q.y, q.z);
    Eigen::Vector3d eul = qe.toRotationMatrix().eulerAngles(2, 1, 0);
    return eul[0];
  }

  geometry_msgs::msg::PoseStamped build_target_pose_(const geometry_msgs::msg::PoseStamped& current) {
    geometry_msgs::msg::PoseStamped tgt = current;
    tgt.header.frame_id = current.header.frame_id.empty() ? base_frame_ : current.header.frame_id;
    if (target_mode_ == "relative") {
      tgt.pose.position.x = current.pose.position.x + target_x_;
      tgt.pose.position.y = current.pose.position.y + target_y_;
      tgt.pose.position.z = current.pose.position.z + target_z_;
    } else {
      tgt.pose.position.x = target_x_;
      tgt.pose.position.y = target_y_;
      tgt.pose.position.z = target_z_;
    }
    if (!keep_orientation_) {
      tgt.pose.orientation = quatFromRPY(target_roll_, target_pitch_, target_yaw_);
    }
    return tgt;
  }

  bool check_joint_states_topic()
  {
    auto topic_names = this->get_topic_names_and_types();
    for (const auto& topic : topic_names) {
      if (topic.first == "/joint_states") {
        RCLCPP_INFO(get_logger(), "Found /joint_states topic");
        return true;
      }
    }
    RCLCPP_ERROR(get_logger(), "Joint states topic /joint_states not found!");
    return false;
  }

  void debug_robot_model() {
    if (!move_group_) return;
    auto robot_model = move_group_->getRobotModel();
    RCLCPP_INFO(get_logger(), "Robot model: %s", robot_model->getName().c_str());
    const auto& link_names = robot_model->getLinkModelNames();
    RCLCPP_INFO(get_logger(), "Available links (%zu):", link_names.size());
    for (const auto& link : link_names) RCLCPP_INFO(get_logger(), "  - %s", link.c_str());
    if (robot_model->hasLinkModel(ee_link_)) {
      RCLCPP_INFO(get_logger(), "✓ End effector link '%s' found", ee_link_.c_str());
    } else {
      RCLCPP_ERROR(get_logger(), "✗ End effector link '%s' NOT found", ee_link_.c_str());
    }
    auto current_state = move_group_->getCurrentState(0.0);
    if (!current_state) return;
    const auto* group = current_state->getJointModelGroup(planning_group_);
    if (group) {
      const auto& ee_links = group->getLinkModelNames();
      RCLCPP_INFO(get_logger(), "Links in planning group '%s':", planning_group_.c_str());
      for (const auto& link : ee_links) RCLCPP_INFO(get_logger(), "  - %s", link.c_str());
    }
  }

  bool wait_for_valid_joint_states_with_time(double timeout_seconds = 20.0) {
    RCLCPP_INFO(get_logger(), "Waiting for joint states with valid timestamps...");
    bool received_joint_states = false;

    auto joint_state_sub = this->create_subscription<sensor_msgs::msg::JointState>(
        "joint_states", 10,
        [&received_joint_states, this](const sensor_msgs::msg::JointState::SharedPtr msg) {
          RCLCPP_INFO_ONCE(get_logger(), "Received joint state with timestamp %.3f and %zu joints",
                           rclcpp::Time(msg->header.stamp).seconds(), msg->name.size());
          received_joint_states = true;
        });

    auto start_time = this->get_clock()->now();
    while (rclcpp::ok()) {
      rclcpp::spin_some(this->get_node_base_interface());
      if (received_joint_states) {
        if (move_group_) {
          try {
            auto state = move_group_->getCurrentState(0.1);
            if (state) {
              RCLCPP_INFO(get_logger(), "Successfully received robot state from MoveGroup!");
              return true;
            }
          } catch (const std::exception& e) {
            RCLCPP_WARN_THROTTLE(get_logger(), *this->get_clock(), 3000,
                                 "Exception getting current state: %s", e.what());
          }
        }
      }
      if ((this->get_clock()->now() - start_time).seconds() > timeout_seconds) {
        RCLCPP_ERROR(get_logger(), "Timeout waiting for valid joint states");
        return false;
      }
      rclcpp::sleep_for(std::chrono::milliseconds(100));
    }
    return false;
  }

  geometry_msgs::msg::PoseStamped get_current_pose_robust() {
    if (!move_group_) return geometry_msgs::msg::PoseStamped();
    for (int attempt = 0; attempt < 3; ++attempt) {
      try {
        rclcpp::sleep_for(std::chrono::milliseconds(80));
        auto pose = move_group_->getCurrentPose(ee_link_);
        if (std::abs(pose.pose.position.x) > 0.001 ||
            std::abs(pose.pose.position.y) > 0.001 ||
            std::abs(pose.pose.position.z) > 0.001) {
          return pose;
        }
      } catch (...) {}
    }
    return geometry_msgs::msg::PoseStamped();
  }

  size_t findNearestIndexToPose(const robot_trajectory::RobotTrajectory& rt,
                                const std::string& link,
                                const geometry_msgs::msg::Pose& target) {
    size_t best_idx = 0;
    double best_dist = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < rt.getWayPointCount(); ++i) {
      const auto& st = rt.getWayPoint(i);
      const auto& T = st.getGlobalLinkTransform(link);
      Eigen::Vector3d p = T.translation();
      const double dx = p.x() - target.position.x;
      const double dy = p.y() - target.position.y;
      const double dz = p.z() - target.position.z;
      const double d = std::sqrt(dx*dx + dy*dy + dz*dz);
      if (d < best_dist) { best_dist = d; best_idx = i; }
    }
    return best_idx;
  }

  // Predictor async flow (single request per attempt)
  void request_pick_pose_async_(double t_rel_s)
  {
    if (!pick_client_) { RCLCPP_ERROR(get_logger(), "Predictor client not created"); end_planning_session_(true); return; }
    if (!predictor_ready_.load(std::memory_order_relaxed) && require_predictor_ready_) {
      RCLCPP_WARN(get_logger(), "Predictor not ready yet, skipping request");
      end_planning_session_(true);
      return;
    }
    if (t_rel_s < 0.0) t_rel_s = 0.0;

    auto req = std::make_shared<lite6_pick_predictor_interfaces::srv::GetPoseAt::Request>();
    req->use_relative = true;
    const double si = std::floor(t_rel_s);
    req->query_time.sec = static_cast<int32_t>(si);
    req->query_time.nanosec = static_cast<uint32_t>((t_rel_s - si) * 1e9);

    auto future = pick_client_->async_send_request(req,
      [this](rclcpp::Client<lite6_pick_predictor_interfaces::srv::GetPoseAt>::SharedFuture fut)
      {
        try {
          auto res = fut.get();
          if (!res->ok) {
            RCLCPP_ERROR(get_logger(), "GetPoseAt returned ok=false");
            end_planning_session_(true);
            return;
          }
          const auto& pick_pose = res->pose;

          // Detailed logging of predictor timing
          const double t_now = this->get_clock()->now().seconds();
          const double t_pose = rclcpp::Time(pick_pose.header.stamp).seconds();
          RCLCPP_INFO(get_logger(),
                      "Received pick pose [%s] (%.3f, %.3f, %.3f) t_now=%.3f t_pose=%.3f dt=%.3f",
                      pick_pose.header.frame_id.c_str(),
                      pick_pose.pose.position.x, pick_pose.pose.position.y, pick_pose.pose.position.z,
                      t_now, t_pose, t_pose - t_now);

          // Plan hover + descent
          if (!plan_and_execute_from_pick_(pick_pose)) {
            RCLCPP_ERROR(get_logger(), "Planning/execution failed");
            end_planning_session_(true);
            return;
          }
          end_planning_session_(false);
        }
        catch (const std::exception& e) {
          RCLCPP_ERROR(get_logger(), "Predictor future exception: %s", e.what());
          end_planning_session_(true);
        }
      });

    (void)future;
  }

  void end_planning_session_(bool cooldown)
  {
    planning_in_progress_ = false;
    if (cooldown) last_attempt_wall_ = std::chrono::steady_clock::now();
  }

  bool transform_to_frame_(const geometry_msgs::msg::PoseStamped& in,
                           const std::string& to_frame,
                           geometry_msgs::msg::PoseStamped& out)
  {
    if (in.header.frame_id.empty() || in.header.frame_id == to_frame) {
      out = in;
      out.header.frame_id = to_frame;
      return true;
    }
    try {
      geometry_msgs::msg::TransformStamped tf =
          tf_buffer_->lookupTransform(to_frame, in.header.frame_id, tf2::TimePointZero, tf2::durationFromSec(0.3));
      tf2::doTransform(in, out, tf);
      out.header.frame_id = to_frame;
      return true;
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "TF transform %s -> %s failed: %s",
                   in.header.frame_id.c_str(), to_frame.c_str(), e.what());
      return false;
    }
  }

  bool plan_and_execute_from_pick_(const geometry_msgs::msg::PoseStamped& pick_pose_in)
  {
    if (!move_group_) return false;

    const std::string planning_frame = move_group_->getPlanningFrame();
    geometry_msgs::msg::PoseStamped pick_pose_tf;
    if (!transform_to_frame_(pick_pose_in, planning_frame, pick_pose_tf)) return false;

    geometry_msgs::msg::PoseStamped pose_hover = pick_pose_tf;
    geometry_msgs::msg::PoseStamped pose_grasp = pick_pose_tf;

    double yaw = 0.0;
    if (horizontal_yaw_mode_ == "current") {
      auto cur = get_current_pose_robust();
      yaw = cur.header.frame_id.empty() ? yawFromQuat(pick_pose_tf.pose.orientation)
                                        : yawFromQuat(cur.pose.orientation);
    } else if (horizontal_yaw_mode_ == "fixed") {
      yaw = fixed_yaw_;
    } else {
      yaw = yawFromQuat(pick_pose_tf.pose.orientation);
    }
    const auto horiz = quatFromRPY(horizontal_roll_offset_, 0.0, yaw);

    // Targets
    pose_hover.pose.position.z = z_hover_;
    pose_grasp.pose.position.z = z_grasp_;

    // Apply horizontal orientation for descent
    if (enforce_horizontal_) {
      // Keep current orientation for leg-1 translation; apply horizontal on descent
    }

    // Compute safe clearance above table top (in planning frame)
    const double table_top_z = table_z_;// + 0.5 + table_height_;
    const double lift_margin = 0.01; // tune if needed

    // Clamp grasp Z above table
    const double min_grasp_z = table_top_z + table_clearance_;
    if (pose_grasp.pose.position.z < min_grasp_z) {
      RCLCPP_WARN(get_logger(),
                  "Requested grasp Z=%.3f is below/too close to table top=%.3f; clamping to %.3f (clearance=%.3f)",
                  pose_grasp.pose.position.z, table_top_z, min_grasp_z, table_clearance_);
      pose_grasp.pose.position.z = min_grasp_z;
    }

    // Stage 1: Lift -> Rotate (at safe Z) -> Slide XY -> Down-to-hover
    move_group_->setStartStateToCurrentState();
    const auto pose_current = get_current_pose_robust();
    if (pose_current.header.frame_id.empty()) {
      RCLCPP_ERROR(get_logger(), "No current pose for Cartesian current->hover");
      return false;
    }

    double safe_z = std::max({pose_current.pose.position.z + 0.05, z_hover_, table_top_z + lift_margin});

    // Waypoints
    geometry_msgs::msg::Pose p_up = pose_current.pose;     // straight up, keep current orientation
    p_up.position.z = safe_z;

    geometry_msgs::msg::Pose p_rotate = p_up;              // rotate in place at safe Z
    if (enforce_horizontal_) p_rotate.orientation = horiz;

    geometry_msgs::msg::Pose p_xy = p_rotate;              // translate XY at safe Z
    p_xy.position.x = pose_hover.pose.position.x;
    p_xy.position.y = pose_hover.pose.position.y;

    geometry_msgs::msg::Pose p_down = p_xy;                // descend to hover Z
    p_down.position.z = pose_hover.pose.position.z;

    std::vector<geometry_msgs::msg::Pose> leg1_waypoints;
    leg1_waypoints.push_back(p_up);
    if (enforce_horizontal_) leg1_waypoints.push_back(p_rotate);
    leg1_waypoints.push_back(p_xy);
    leg1_waypoints.push_back(p_down);

    moveit_msgs::msg::RobotTrajectory leg1_msg;
    const double eef_step1 = 0.03;
    const double jump_threshold1 = 0.0;
    double fraction1 = 0.0;
    try {
      fraction1 = move_group_->computeCartesianPath(leg1_waypoints, eef_step1, jump_threshold1, leg1_msg, true);
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "Cartesian leg1 exception: %s", e.what());
      return false;
    }
    if (fraction1 < 0.8) {
      moveit_msgs::msg::RobotTrajectory dbg_msg;
      double fraction_nc = 0.0;
      try { fraction_nc = move_group_->computeCartesianPath(leg1_waypoints, eef_step1, jump_threshold1, dbg_msg, false); } catch (...) {}
      RCLCPP_WARN(get_logger(), "Leg-1 low fraction=%.2f (collisions). No-collision fraction=%.2f", fraction1, fraction_nc);
      if (fraction1 < 0.5) {
        RCLCPP_ERROR(get_logger(), "Cartesian leg1 failed (fraction=%.2f)", fraction1);
        return false;
      }
    }

    // Time-parameterize leg 1 (use per-leg scaling), then bound to [t_hover, t_hover_max]
    moveit::core::RobotModelConstPtr model1 = move_group_->getRobotModel();
    auto start_state1 = move_group_->getCurrentState(2.0);
    if (!start_state1) {
      RCLCPP_ERROR(get_logger(), "No current robot state before leg1");
      return false;
    }
    robot_trajectory::RobotTrajectory rt1(model1, planning_group_);
    rt1.setRobotTrajectoryMsg(*start_state1, leg1_msg);
    if (!tp_control_cpp::applyTimeParameterization(rt1, method_, vel_scale_leg1_, acc_scale_leg1_)) {
      RCLCPP_ERROR(get_logger(), "Time parameterization failed for leg1");
      return false;
    }
    // Compress/expand leg-1 total duration to fit bounds
    (void)tp_control_cpp::enforceSegmentTimeBounded(rt1, 0, rt1.getWayPointCount() ? (rt1.getWayPointCount()-1) : 0,
                                                    targets_.t_hover, t_hover_max_);
    if (method_ == TimingMethod::TOTG_THEN_RUCKIG) {
      (void)tp_control_cpp::applyTimeParameterization(rt1, TimingMethod::RUCKIG, vel_scale_leg1_, acc_scale_leg1_);
    }
    rt1.getRobotTrajectoryMsg(leg1_msg);

    moveit::planning_interface::MoveGroupInterface::Plan plan_to_hover;
    plan_to_hover.trajectory_ = leg1_msg;
    moveit::core::robotStateToRobotStateMsg(*start_state1, plan_to_hover.start_state_);
    plan_to_hover.planning_time_ = 0.0;

    if (display_pub_) {
      moveit_msgs::msg::DisplayTrajectory msg;
      msg.model_id = move_group_->getRobotModel()->getName();
      if (auto cs0 = move_group_->getCurrentState(0.0)) {
        moveit_msgs::msg::RobotState rs;
        moveit::core::robotStateToRobotStateMsg(*cs0, rs);
        msg.trajectory_start = rs;
      }
      msg.trajectory.push_back(plan_to_hover.trajectory_);
      display_pub_->publish(msg);
    }
    auto exec_code = move_group_->execute(plan_to_hover);
    if (exec_code != moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
      RCLCPP_ERROR(get_logger(), "Execution to hover failed (code=%d)", exec_code.val);
      return false;
    }

    // Stage 2: hover -> grasp (keep horizontal)
    move_group_->setStartStateToCurrentState();
    geometry_msgs::msg::PoseStamped hover_for_descent = pose_hover;
    if (enforce_horizontal_) hover_for_descent.pose.orientation = horiz;
    if (enforce_horizontal_) pose_grasp.pose.orientation = horiz;

    std::vector<geometry_msgs::msg::Pose> waypoints{hover_for_descent.pose, pose_grasp.pose};
    moveit_msgs::msg::RobotTrajectory cart_traj_msg;
    const double eef_step = 0.02;
    const double jump_threshold = 0.0;
    double fraction = 0.0;
    try {
      fraction = move_group_->computeCartesianPath(waypoints, eef_step, jump_threshold, cart_traj_msg, cartesian_avoid_collisions_);
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "Cartesian descent exception: %s", e.what());
      return false;
    }

    const double dz_req = std::abs(hover_for_descent.pose.position.z - pose_grasp.pose.position.z);
    const double dz_achieved = dz_req * std::clamp(fraction, 0.0, 1.0);

    if (fraction < min_cart_fraction_ || dz_achieved < min_descent_dz_) {
      RCLCPP_WARN(get_logger(),
                  "Aborting descent: fraction=%.2f (min=%.2f), dz_req=%.3f, dz_achieved=%.3f (min=%.3f), avoid_collisions=%s",
                  fraction, min_cart_fraction_, dz_req, dz_achieved, min_descent_dz_,
                  cartesian_avoid_collisions_ ? "true" : "false");
      return false;
    }

    moveit::core::RobotModelConstPtr model = move_group_->getRobotModel();
    auto start_state = move_group_->getCurrentState(2.0);
    if (!start_state) {
      RCLCPP_ERROR(get_logger(), "No current robot state before descent");
      return false;
    }
    robot_trajectory::RobotTrajectory rt(model, planning_group_);
    rt.setRobotTrajectoryMsg(*start_state, cart_traj_msg);
    if (!tp_control_cpp::applyTimeParameterization(rt, method_, vel_scale_descent_, acc_scale_descent_)) {
      RCLCPP_ERROR(get_logger(), "Time parameterization failed for descent");
      return false;
    }
    // Bound descent duration to [t_grasp, t_grasp_max]
    const size_t i_hover = 0;
    const size_t i_grasp = rt.getWayPointCount() ? (rt.getWayPointCount() - 1) : 0;
    (void)tp_control_cpp::enforceSegmentTimeBounded(rt, i_hover, i_grasp, targets_.t_grasp, t_grasp_max_);
    if (method_ == TimingMethod::TOTG_THEN_RUCKIG) {
      (void)tp_control_cpp::applyTimeParameterization(rt, TimingMethod::RUCKIG, vel_scale_descent_, acc_scale_descent_);
    }
    rt.getRobotTrajectoryMsg(cart_traj_msg);

    moveit::planning_interface::MoveGroupInterface::Plan plan_descent;
    plan_descent.trajectory_ = cart_traj_msg;
    moveit::core::robotStateToRobotStateMsg(*start_state, plan_descent.start_state_);
    plan_descent.planning_time_ = 0.0;

    if (display_pub_) {
      moveit_msgs::msg::DisplayTrajectory msg;
      msg.model_id = move_group_->getRobotModel()->getName();
      if (auto cs = move_group_->getCurrentState(0.0)) {
        moveit_msgs::msg::RobotState rs;
        moveit::core::robotStateToRobotStateMsg(*cs, rs);
        msg.trajectory_start = rs;
      }
      msg.trajectory.push_back(plan_descent.trajectory_);
      display_pub_->publish(msg);
    }
    schedule_gripper_close_for_plan_(plan_descent);
    (void)move_group_->execute(plan_descent);
    return true;
  }

  void run_once_with_predictor_()
  {
    if (!move_group_) { RCLCPP_ERROR(get_logger(), "MoveGroup not initialized"); end_planning_session_(true); return; }
    if (!pick_client_ || !pick_client_->wait_for_service(std::chrono::seconds(0))) {
      RCLCPP_WARN(get_logger(), "Predictor service '%s' not ready", predictor_service_.c_str());
      end_planning_session_(true);
      return;
    }
    move_group_->setStartStateToCurrentState();
    request_pick_pose_async_(commit_pick_time_s_);
  }

  // Demo without predictor
  void run_once()
  {
    if (!move_group_) return;
    auto pose0 = get_current_pose_robust();
    if (pose0.header.frame_id.empty()) return;
    geometry_msgs::msg::PoseStamped pose_hover = build_target_pose_(pose0);
    pose_hover.pose.position.z = z_hover_;
    geometry_msgs::msg::PoseStamped pose_grasp = pose_hover;
    pose_grasp.pose.position.z = z_grasp_;
    std::vector<geometry_msgs::msg::Pose> waypoints{pose_hover.pose, pose_grasp.pose};

    moveit_msgs::msg::RobotTrajectory cart_traj_msg;
    (void)move_group_->computeCartesianPath(waypoints, 0.005, 0.0, cart_traj_msg, true);
    moveit::core::RobotModelConstPtr model = move_group_->getRobotModel();
    auto start_state = move_group_->getCurrentState(2.0);
    if (!start_state) return;
    robot_trajectory::RobotTrajectory rt(model, planning_group_);
    rt.setRobotTrajectoryMsg(*start_state, cart_traj_msg);
    (void)tp_control_cpp::applyTimeParameterization(rt, method_, vel_scale_descent_, acc_scale_descent_);
    (void)tp_control_cpp::enforceSegmentTimeBounded(rt, 0, rt.getWayPointCount()? (rt.getWayPointCount()-1):0,
                                                    targets_.t_grasp, t_grasp_max_);
    rt.getRobotTrajectoryMsg(cart_traj_msg);
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    plan.trajectory_ = cart_traj_msg;
    moveit::core::robotStateToRobotStateMsg(*start_state, plan.start_state_);
    if (display_pub_) {
      moveit_msgs::msg::DisplayTrajectory msg;
      msg.model_id = move_group_->getRobotModel()->getName();
      msg.trajectory.push_back(plan.trajectory_);
      display_pub_->publish(msg);
    }
    (void)move_group_->execute(plan);
  }

  // Add a cylinder as the rotating table collision object (matches tp_control.py)
  void add_rotating_table_collision_(const geometry_msgs::msg::Pose& table_pose,
                                     double table_radius,
                                     double table_height)
  {
    moveit_msgs::msg::CollisionObject obj;
    obj.header.frame_id = table_frame_.empty() ? base_frame_ : table_frame_;
    obj.header.stamp = this->get_clock()->now();
    obj.id = table_object_id_;

    shape_msgs::msg::SolidPrimitive cylinder;
    cylinder.type = shape_msgs::msg::SolidPrimitive::CYLINDER;
    cylinder.dimensions.resize(2);
    cylinder.dimensions[shape_msgs::msg::SolidPrimitive::CYLINDER_HEIGHT] = table_height;
    cylinder.dimensions[shape_msgs::msg::SolidPrimitive::CYLINDER_RADIUS] = table_radius;

    obj.primitives.push_back(cylinder);
    obj.primitive_poses.push_back(table_pose);
    obj.operation = moveit_msgs::msg::CollisionObject::ADD;

    // Apply via PlanningSceneInterface
    planning_scene_interface_.applyCollisionObject(obj);

    // Optional: publish as diff
    moveit_msgs::msg::PlanningScene scene_msg;
    scene_msg.is_diff = true;
    scene_msg.world.collision_objects.push_back(obj);
    if (planning_scene_pub_) planning_scene_pub_->publish(scene_msg);
  }

  // Allow/disallow collisions between EE and table object
  void allow_ee_table_collision_(bool allow)
  {
    if (!planning_scene_pub_) return;
    moveit_msgs::msg::PlanningScene scene_msg;
    scene_msg.is_diff = true;

    auto& acm = scene_msg.allowed_collision_matrix;
    acm.entry_names = {ee_link_, table_object_id_};
    acm.entry_values.resize(2);
    for (auto& row : acm.entry_values) row.enabled.resize(2, false);
    acm.entry_values[0].enabled[1] = allow; // ee vs table
    acm.entry_values[1].enabled[0] = allow; // table vs ee

    planning_scene_pub_->publish(scene_msg);
  }

private:
  std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
  moveit_visual_tools::MoveItVisualToolsPtr visual_tools_;
  rclcpp::Publisher<moveit_msgs::msg::DisplayTrajectory>::SharedPtr display_pub_;

  std::string planning_group_, ee_link_, base_frame_;
  double z_hover_{0.20}, z_grasp_{0.05};
  double vel_scale_{0.9}, acc_scale_{0.6};
  // Per-leg scaling
  double vel_scale_leg1_{0.9}, acc_scale_leg1_{0.6};
  double vel_scale_descent_{0.9}, acc_scale_descent_{0.6};

  TimingMethod method_{TimingMethod::TOTG_THEN_RUCKIG};
  std::string timing_method_str_;
  TimingTargets targets_;
  double t_hover_max_{0.0};
  double t_grasp_max_{0.0};

  bool use_pose_target_{false};
  std::string target_mode_;
  std::string target_frame_;
  double target_x_{0.0}, target_y_{0.0}, target_z_{0.0};
  bool keep_orientation_{true};
  double target_roll_{0.0}, target_pitch_{0.0}, target_yaw_{0.0};

  // Horizontal control
  bool enforce_horizontal_{false};
  std::string horizontal_yaw_mode_{"predictor"}; // predictor|current|fixed
  double fixed_yaw_{0.0};
  double horizontal_roll_offset_{M_PI};
  double rp_tol_{0.10};
  double yaw_tol_{M_PI};
  bool use_path_constraints_for_hover_{false};
  std::string constraint_mode_{"goal_only"}; // goal_only|path_only|both

  // Flow control
  std::atomic_bool planning_in_progress_{false};
  bool move_group_params_ready_{false};
  rclcpp::TimerBase::SharedPtr delayed_plan_timer_;
  double replan_cooldown_s_{2.0};
  std::chrono::steady_clock::time_point last_attempt_wall_{};
  bool last_ik_reachable_{false};

  // TF
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // Predictor
  bool use_predictor_{true};
  bool require_predictor_ready_{true};
  double commit_pick_time_s_{1.0};
  std::string predictor_ready_topic_{"predictor_ready"};
  std::string predictor_service_{"get_predicted_pose_at"};
  std::atomic<bool> predictor_ready_{false};
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr predictor_ready_sub_;
  rclcpp::Client<lite6_pick_predictor_interfaces::srv::GetPoseAt>::SharedPtr pick_client_;

  // Planning params
  double planning_time_{0.5};
  std::string planner_id_{"RRTConnect"};

  // IK gate
  bool require_ik_gate_{true};
  std::string ik_gate_topic_{"/ik_gate/output"};
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr ik_gate_sub_;

  // Gripper
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr gripper_pub_;
  std::string gripper_controller_topic_;
  std::vector<std::string> gripper_joints_;
  std::vector<double> gripper_open_pos_;
  std::vector<double> gripper_close_pos_;
  double gripper_motion_time_{0.7};

  // Planning scene interface and optional publisher
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;
  rclcpp::Publisher<moveit_msgs::msg::PlanningScene>::SharedPtr planning_scene_pub_;

  // Rotating table config
  bool add_table_collision_{true};
  std::string table_frame_{"link_base"};
  double table_radius_{0.75};
  double table_height_{0.06};
  double table_x_{0.9};
  double table_y_{0.0};
  double table_z_{0.1};
  std::string table_object_id_{"rotating_table"};
  double table_clearance_{0.02};

  // Cartesian execution guards/options
  double min_cart_fraction_{0.95};
  double min_descent_dz_{0.03};
  bool cartesian_avoid_collisions_{true};
  bool allow_ee_table_touch_{false};
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<TpControlNode>();
  node->initialize();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
//==================================================
// 
//===================================================
// #include <rclcpp/rclcpp.hpp>
// #include <rclcpp/parameter_client.hpp>

// #include <geometry_msgs/msg/pose_stamped.hpp>
// #include <sensor_msgs/msg/joint_state.hpp>
// #include <eigen3/Eigen/Geometry>

// #include <moveit/move_group_interface/move_group_interface.h>
// #include <moveit/planning_scene_monitor/planning_scene_monitor.h>
// #include <moveit/planning_scene_interface/planning_scene_interface.h>
// #include <moveit/robot_state/conversions.h>
// #include <moveit/robot_trajectory/robot_trajectory.h>
// #include <moveit_msgs/msg/move_it_error_codes.h>
// #include <moveit_visual_tools/moveit_visual_tools.h>
// #include <moveit_msgs/msg/display_trajectory.hpp>
// #include <moveit_msgs/msg/constraints.hpp>
// #include <moveit_msgs/msg/orientation_constraint.hpp>
// #include <moveit_msgs/msg/collision_object.hpp>
// #include <moveit_msgs/msg/planning_scene.hpp>
// #include <shape_msgs/msg/solid_primitive.hpp>

// #include <moveit/trajectory_processing/time_optimal_trajectory_generation.h>
// #include <moveit/trajectory_processing/ruckig_traj_smoothing.h>

// #include <trajectory_msgs/msg/joint_trajectory.hpp>
// #include <trajectory_msgs/msg/joint_trajectory_point.hpp>

// #include <tf2_eigen/tf2_eigen.hpp>
// #include <tf2_ros/transform_listener.h>
// #include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
// #include <tf2/time.h>

// #include <std_msgs/msg/bool.hpp>
// #include <std_msgs/msg/string.hpp>
// #include <lite6_pick_predictor_interfaces/srv/get_pose_at.hpp>

// #include <algorithm>
// #include <limits>
// #include <cmath>
// #include <atomic>
// #include <string>
// #include <vector>
// #include <chrono>

// namespace tp_control_cpp {

// enum class TimingMethod { TOTG, RUCKIG, TOTG_THEN_RUCKIG };

// struct TimingTargets
// {
//   double t_hover{0.0};
//   double t_grasp{0.0};
// };

// inline double durationFromStart(const robot_trajectory::RobotTrajectory& rt, size_t idx)
// {
//   if (rt.getWayPointCount() == 0 || idx == 0) return 0.0;
//   idx = std::min(idx, rt.getWayPointCount() - 1);
//   double sum = 0.0;
//   for (size_t i = 1; i <= idx; ++i)
//     sum += rt.getWayPointDurationFromPrevious(i);
//   return sum;
// }

// inline bool applyTimeParameterization(robot_trajectory::RobotTrajectory& rt,
//                                       TimingMethod method,
//                                       double max_vel_scaling = 0.9,
//                                       double max_acc_scaling = 0.7)
// {
//   using trajectory_processing::TimeOptimalTrajectoryGeneration;
//   using trajectory_processing::RuckigSmoothing;

//   switch (method)
//   {
//     case TimingMethod::TOTG:
//     {
//       TimeOptimalTrajectoryGeneration totg;
//       return totg.computeTimeStamps(rt, max_vel_scaling, max_acc_scaling);
//     }
//     case TimingMethod::RUCKIG:
//     {
//       return RuckigSmoothing::applySmoothing(rt, max_vel_scaling, max_acc_scaling);
//     }
//     case TimingMethod::TOTG_THEN_RUCKIG:
//     {
//       TimeOptimalTrajectoryGeneration totg;
//       if (!totg.computeTimeStamps(rt, max_vel_scaling, max_acc_scaling))
//         return false;
//       return RuckigSmoothing::applySmoothing(rt, max_vel_scaling, max_acc_scaling);
//     }
//     default:
//       return false;
//   }
// }

// inline bool enforceSegmentTimes(robot_trajectory::RobotTrajectory& rt,
//                                 size_t idx_hover,
//                                 size_t idx_grasp,
//                                 const TimingTargets& targets)
// {
//   const size_t N = rt.getWayPointCount();
//   if (N < 2) return false;
//   idx_hover = std::min(idx_hover, N - 1);
//   idx_grasp = std::min(idx_grasp, N - 1);
//   if (idx_hover >= idx_grasp) idx_hover = std::max<size_t>(1, std::min(idx_grasp, idx_hover));

//   auto safe_get = [&](size_t i) { return rt.getWayPointDurationFromPrevious(i); };
//   auto safe_set = [&](size_t i, double v) { rt.setWayPointDurationFromPrevious(i, std::max(1e-9, v)); };

//   bool changed = false;

//   double t_hover_cur = durationFromStart(rt, idx_hover);
//   if (targets.t_hover > 0.0 && t_hover_cur < targets.t_hover && idx_hover >= 1)
//   {
//     const double scale1 = targets.t_hover / std::max(1e-9, t_hover_cur);
//     for (size_t i = 1; i <= idx_hover; ++i)
//       safe_set(i, safe_get(i) * scale1);
//     changed = true;
//   }

//   double t_hover_new = durationFromStart(rt, idx_hover);
//   const double t_total_cur = durationFromStart(rt, N - 1);
//   const double tail_cur = std::max(0.0, t_total_cur - t_hover_new);

//   if (targets.t_grasp > 0.0 && (t_hover_new + tail_cur) < targets.t_grasp && idx_grasp >= (idx_hover + 1))
//   {
//     const double needed_tail = targets.t_grasp - t_hover_new;
//     if (tail_cur <= 1e-9)
//     {
//       double last = safe_get(N - 1);
//       if (last < 1e-6) last = 1e-3;
//       safe_set(N - 1, last + std::max(0.0, needed_tail));
//     }
//     else
//     {
//       const double scale2 = std::max(1.0, needed_tail / tail_cur);
//       for (size_t i = idx_hover + 1; i <= idx_grasp; ++i)
//         safe_set(i, safe_get(i) * scale2);
//     }
//     changed = true;
//   }

//   return changed;
// }

// } // namespace tp_control_cpp

// using tp_control_cpp::TimingMethod;
// using tp_control_cpp::TimingTargets;

// class TpControlNode : public rclcpp::Node
// {
// public:
//   TpControlNode() : Node("tp_control_node")
//   {
//     this->declare_parameter<std::string>("robot_description", "");
//     this->declare_parameter<std::string>("robot_description_semantic", "");
//     this->declare_parameter<std::string>("planning_group", "lite6");
//     this->declare_parameter<std::string>("ee_link", "link_tcp");
//     this->declare_parameter<std::string>("base_frame", "link_base");
//     this->declare_parameter<std::string>("timing_method", "totg_then_ruckig");
//     this->declare_parameter<double>("planning_time", 10.0);
//     this->declare_parameter<std::string>("planner_id", "RRTConnect");
//     this->declare_parameter<bool>("use_pose_target", false);
//     this->declare_parameter<std::string>("target_mode", "absolute");
//     this->declare_parameter<std::string>("target_frame", "");

//     this->declare_parameter<double>("vel_scale", 0.9);
//     this->declare_parameter<double>("acc_scale", 0.6);

//     this->declare_parameter<double>("target_x", 0.29);
//     this->declare_parameter<double>("target_y", -0.16);
//     this->declare_parameter<double>("target_z", 0.14);
//     this->declare_parameter<double>("z_hover", 0.29);
//     this->declare_parameter<double>("z_grasp", 0.14);
//     this->declare_parameter<double>("target_roll", 0.0);
//     this->declare_parameter<double>("target_pitch", 0.0);
//     this->declare_parameter<double>("target_yaw", 0.0);
//     this->declare_parameter<bool>("keep_orientation", false);

//     // Horizontal orientation helpers
//     this->declare_parameter<bool>("enforce_horizontal_orientation", false);
//     this->declare_parameter<std::string>("horizontal_yaw_mode", "predictor"); // predictor|current|fixed
//     this->declare_parameter<double>("fixed_yaw", 0.0);
//     this->declare_parameter<double>("horizontal_roll_offset", M_PI);          // many TCPs need a pi flip
//     this->declare_parameter<double>("constraint_rp_tolerance", 0.5);          // roll/pitch tolerance (rad)
//     this->declare_parameter<double>("constraint_yaw_tolerance", M_PI);        // yaw free
//     this->declare_parameter<bool>("use_path_constraints_for_hover", true);
//     // Constraint selection: goal_only | path_only | both
//     this->declare_parameter<std::string>("constraint_mode", "goal_only");

//     // Rotating table collision object params (like tp_control.py)
//     this->declare_parameter<bool>("add_table_collision", true);
//     this->declare_parameter<std::string>("table_frame", "link_base");
//     this->declare_parameter<double>("table_radius", 0.75);
//     this->declare_parameter<double>("table_height", 0.06);
//     this->declare_parameter<double>("table_x", 0.9);
//     this->declare_parameter<double>("table_y", 0.0);
//     this->declare_parameter<double>("table_z", 0.1);
//     this->declare_parameter<std::string>("table_object_id", "rotating_table");
//     this->declare_parameter<double>("table_clearance", 0.02); // min clearance above table top

//     this->declare_parameter<double>("t_hover", 0.0);
//     this->declare_parameter<double>("t_grasp", 0.0);

//     // IK gate
//     this->declare_parameter<bool>("require_ik_gate", true);
//     this->declare_parameter<std::string>("ik_gate_topic", "/ik_gate/output");

//     // Predictor
//     this->declare_parameter<bool>("use_predictor", true);
//     this->declare_parameter<bool>("require_predictor_ready", true);
//     this->declare_parameter<std::string>("predictor_ready_topic", "predictor_ready");
//     this->declare_parameter<std::string>("predictor_service", "get_predicted_pose_at");
//     this->declare_parameter<double>("commit_pick_time_s", 10.0);

//     // Cartesian descent execution guards and options
//     this->declare_parameter<double>("min_cart_fraction", 0.95);
//     this->declare_parameter<double>("min_descent_dz", 0.03);
//     this->declare_parameter<bool>("cartesian_avoid_collisions", true);
//     this->declare_parameter<bool>("allow_ee_table_touch", false);

//     // Cooldown to prevent spam
//     this->declare_parameter<double>("replan_cooldown_s", 2.0);

//     // Gripper
//     this->declare_parameter<std::string>("gripper_controller_topic", "/lite6_gripper_traj_controller/joint_trajectory");
//     this->declare_parameter<std::vector<std::string>>("gripper_joints", {"jaw_left","jaw_right"});
//     this->declare_parameter<std::vector<double>>("gripper_open", {0.02, 0.02});
//     this->declare_parameter<std::vector<double>>("gripper_close", {0.0, 0.0});
//     this->declare_parameter<double>("gripper_motion_time", 1.0);

//     // Detect /clock
//     auto topics = this->get_topic_names_and_types();
//     bool has_clock = std::any_of(topics.begin(), topics.end(), [](const auto& t){ return t.first == "/clock"; });
//     this->set_parameter(rclcpp::Parameter("use_sim_time", has_clock));
//     RCLCPP_INFO(get_logger(), has_clock ? "Found /clock, use_sim_time=true" : "No /clock, use_sim_time=false");

//     tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
//     tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

//     display_pub_ = this->create_publisher<moveit_msgs::msg::DisplayTrajectory>("display_planned_path", rclcpp::QoS(5));
//     planning_scene_pub_ = this->create_publisher<moveit_msgs::msg::PlanningScene>("/planning_scene", 10);

//     get_parameters();

//     RCLCPP_INFO(get_logger(), "TpControlNode initializing...");
//   }

//   void initialize()
//   {
//     try
//     {
//       if (!check_joint_states_topic()) {
//         RCLCPP_ERROR(get_logger(), "Joint states topic not available, cannot proceed");
//         return;
//       }
//       if (!wait_for_move_group_parameters()) {
//         RCLCPP_ERROR(get_logger(), "Could not get robot description from move_group");
//         return;
//       }
//       move_group_params_ready_ = true;

//       move_group_ = std::make_unique<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), planning_group_);
//       move_group_->setEndEffectorLink(ee_link_);
//       move_group_->startStateMonitor(5.0);

//       visual_tools_ = std::make_shared<moveit_visual_tools::MoveItVisualTools>(
//           shared_from_this(), base_frame_, "/rviz_visual_tools", move_group_->getRobotModel());
//       visual_tools_->deleteAllMarkers();
//       visual_tools_->loadRemoteControl();

//       RCLCPP_INFO(get_logger(), "TpControlNode ready. Method=%s", timing_method_str_.c_str());

//       if (!wait_for_valid_joint_states_with_time()) {
//         RCLCPP_ERROR(get_logger(), "Failed to receive valid joint states");
//         return;
//       }
//       auto cs = move_group_->getCurrentState(0.0);
//       if (!cs){
//         RCLCPP_ERROR(get_logger(), "MoveGroup current state unavailable");
//         return;
//       }
//       move_group_->setStartStateToCurrentState();

//       // Predictor wiring
//       if (use_predictor_) {
//         predictor_ready_sub_ = this->create_subscription<std_msgs::msg::Bool>(
//             predictor_ready_topic_, 10,
//             [this](const std_msgs::msg::Bool::SharedPtr msg){
//               predictor_ready_.store(msg->data, std::memory_order_relaxed);
//             });
//         pick_client_ = this->create_client<lite6_pick_predictor_interfaces::srv::GetPoseAt>(predictor_service_);
//       }

//       // Add rotating table collision object
//       if (add_table_collision_) {
//         geometry_msgs::msg::Pose table_pose;
//         table_pose.position.x = table_x_;
//         table_pose.position.y = table_y_;
//         table_pose.position.z = table_z_;
//         table_pose.orientation.w = 1.0; // identity
//         add_rotating_table_collision_(table_pose, table_radius_, table_height_);
//         RCLCPP_INFO(get_logger(), "Applied rotating_table collision object at [%.2f, %.2f, %.2f] (r=%.2f, h=%.2f, frame=%s)",
//                     table_x_, table_y_, table_z_, table_radius_, table_height_, table_frame_.c_str());
//         rclcpp::sleep_for(std::chrono::milliseconds(200)); // allow planning scene sync

//         if (allow_ee_table_touch_) {
//           allow_ee_table_collision_(true);
//           RCLCPP_WARN(get_logger(), "Allowed collisions between '%s' and table object '%s'",
//                       ee_link_.c_str(), table_object_id_.c_str());
//         }
//       }

//       // IK gate rising-edge trigger + cooldown
//       if (require_ik_gate_) {
//         ik_gate_sub_ = this->create_subscription<std_msgs::msg::String>(
//             ik_gate_topic_, 10,
//             [this](const std_msgs::msg::String::SharedPtr msg) {
//               const bool reachable = (msg->data == "REACHABLE");
//               if (reachable && !last_ik_reachable_) {
//                 const auto now = std::chrono::steady_clock::now();
//                 if (!planning_in_progress_ && (now - last_attempt_wall_) >= std::chrono::duration<double>(replan_cooldown_s_)) {
//                   planning_in_progress_ = true;
//                   delayed_plan_timer_ = this->create_wall_timer(
//                       std::chrono::milliseconds(5),
//                       [this]() {
//                         if (delayed_plan_timer_) delayed_plan_timer_->cancel();
//                         this->run_once_with_predictor_();
//                       });
//                 } else {
//                   RCLCPP_WARN(get_logger(), "Cooldown active; skipping trigger");
//                 }
//               }
//               last_ik_reachable_ = reachable;
//             });
//       } else {
//         delayed_plan_timer_ = this->create_wall_timer(
//             std::chrono::milliseconds(150),
//             [this]() {
//               if (delayed_plan_timer_) delayed_plan_timer_->cancel();
//               if (!planning_in_progress_) {
//                 planning_in_progress_ = true;
//                 this->run_once_with_predictor_();
//               }
//             });
//       }

//       debug_robot_model();
//     }
//     catch (const std::exception& e)
//     {
//       RCLCPP_ERROR(get_logger(), "Failed to initialize: %s", e.what());
//     }
//   }

// private:

//   bool wait_for_move_group_parameters(double timeout_seconds = 20.0)
//   {
//     if (move_group_params_ready_) return true;
//     RCLCPP_INFO(get_logger(), "Looking for move_group parameters...");
//     auto start_time = this->get_clock()->now();

//     while (rclcpp::ok() && (this->get_clock()->now() - start_time).seconds() < timeout_seconds) {
//       auto node_names = this->get_node_names();
//       for (const auto& node_name : node_names) {
//         if (node_name.find("move_group") != std::string::npos) {
//           RCLCPP_INFO(get_logger(), "Found move_group node: %s", node_name.c_str());
//           try {
//             auto param_client = std::make_shared<rclcpp::AsyncParametersClient>(this, node_name);
//             if (param_client->wait_for_service(std::chrono::seconds(2))) {
//               auto future = param_client->get_parameters(
//                 {"robot_description", "robot_description_semantic", "robot_description_kinematics"});

//               if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), future,
//                   std::chrono::seconds(5)) == rclcpp::FutureReturnCode::SUCCESS) {

//                 auto params = future.get();
//                 for (const auto& param : params) {
//                   if (param.get_name() == "robot_description" &&
//                       param.get_type() == rclcpp::ParameterType::PARAMETER_STRING) {
//                     this->set_parameter(rclcpp::Parameter("robot_description", param.as_string()));
//                     RCLCPP_INFO(get_logger(), "✓ robot_description copied");
//                   } else if (param.get_name() == "robot_description_semantic" &&
//                              param.get_type() == rclcpp::ParameterType::PARAMETER_STRING) {
//                     this->set_parameter(rclcpp::Parameter("robot_description_semantic", param.as_string()));
//                     RCLCPP_INFO(get_logger(), "✓ robot_description_semantic copied");
//                   } else if (param.get_name() == "robot_description_kinematics" &&
//                              param.get_type() == rclcpp::ParameterType::PARAMETER_STRING) {
//                     this->set_parameter(rclcpp::Parameter("robot_description_kinematics", param.as_string()));
//                     RCLCPP_INFO(get_logger(), "✓ robot_description_kinematics copied");
//                   }
//                 }
//               }
//               move_group_params_ready_ = true;
//               return true;
//             }
//           } catch (const std::exception& e) {
//             RCLCPP_WARN(get_logger(), "Failed to get parameters: %s", e.what());
//           }
//         }
//       }
//       RCLCPP_INFO_THROTTLE(get_logger(), *this->get_clock(), 3000,
//                            "Waiting for move_group node to be available...");
//       rclcpp::sleep_for(std::chrono::milliseconds(300));
//     }
//     RCLCPP_ERROR(get_logger(), "Timeout waiting for move_group parameters");
//     return false;
//   }

//   void get_parameters()
//   {
//     planning_group_ = this->get_parameter("planning_group").as_string();
//     ee_link_ = this->get_parameter("ee_link").as_string();
//     base_frame_ = this->get_parameter("base_frame").as_string();
//     z_hover_ = this->get_parameter("z_hover").as_double();
//     z_grasp_ = this->get_parameter("z_grasp").as_double();
//     targets_.t_hover = this->get_parameter("t_hover").as_double();
//     targets_.t_grasp = this->get_parameter("t_grasp").as_double();
//     timing_method_str_ = this->get_parameter("timing_method").as_string();
//     vel_scale_ = this->get_parameter("vel_scale").as_double();
//     acc_scale_ = this->get_parameter("acc_scale").as_double();

//     if (timing_method_str_ == "totg") {
//       method_ = TimingMethod::TOTG;
//     } else if (timing_method_str_ == "ruckig") {
//       method_ = TimingMethod::RUCKIG;
//     } else {
//       method_ = TimingMethod::TOTG_THEN_RUCKIG;
//       timing_method_str_ = "totg_then_ruckig";
//     }

//     use_pose_target_ = this->get_parameter("use_pose_target").as_bool();
//     target_mode_ = this->get_parameter("target_mode").as_string();
//     target_frame_ = this->get_parameter("target_frame").as_string();
//     if (target_frame_.empty()) target_frame_ = base_frame_;
//     target_x_ = this->get_parameter("target_x").as_double();
//     target_y_ = this->get_parameter("target_y").as_double();
//     target_z_ = this->get_parameter("target_z").as_double();
//     keep_orientation_ = this->get_parameter("keep_orientation").as_bool();
//     target_roll_ = this->get_parameter("target_roll").as_double();
//     target_pitch_ = this->get_parameter("target_pitch").as_double();
//     target_yaw_ = this->get_parameter("target_yaw").as_double();

//     enforce_horizontal_ = this->get_parameter("enforce_horizontal_orientation").as_bool();
//     horizontal_yaw_mode_ = this->get_parameter("horizontal_yaw_mode").as_string();
//     fixed_yaw_ = this->get_parameter("fixed_yaw").as_double();
//     horizontal_roll_offset_ = this->get_parameter("horizontal_roll_offset").as_double();
//     rp_tol_ = this->get_parameter("constraint_rp_tolerance").as_double();
//     yaw_tol_ = this->get_parameter("constraint_yaw_tolerance").as_double();
//     use_path_constraints_for_hover_ = this->get_parameter("use_path_constraints_for_hover").as_bool();
//     constraint_mode_ = this->get_parameter("constraint_mode").as_string();
//     if (constraint_mode_ != "goal_only" && constraint_mode_ != "path_only" && constraint_mode_ != "both") {
//       RCLCPP_WARN(get_logger(), "Invalid constraint_mode '%s', defaulting to 'path_only'", constraint_mode_.c_str());
//       constraint_mode_ = "path_only";
//     }

//     // Predictor params
//     use_predictor_ = this->get_parameter("use_predictor").as_bool();
//     require_predictor_ready_ = this->get_parameter("require_predictor_ready").as_bool();
//     predictor_ready_topic_ = this->get_parameter("predictor_ready_topic").as_string();
//     predictor_service_ = this->get_parameter("predictor_service").as_string();
//     commit_pick_time_s_ = this->get_parameter("commit_pick_time_s").as_double();

//     replan_cooldown_s_ = this->get_parameter("replan_cooldown_s").as_double();

//     // Planner params
//     planning_time_ = this->get_parameter("planning_time").as_double();
//     planner_id_ = this->get_parameter("planner_id").as_string();

//     // IK gate params
//     require_ik_gate_ = this->get_parameter("require_ik_gate").as_bool();
//     ik_gate_topic_ = this->get_parameter("ik_gate_topic").as_string();

//     // Gripper params
//     gripper_controller_topic_ = this->get_parameter("gripper_controller_topic").as_string();
//     gripper_joints_ = this->get_parameter("gripper_joints").as_string_array();
//     gripper_open_pos_ = this->get_parameter("gripper_open").as_double_array();
//     gripper_close_pos_ = this->get_parameter("gripper_close").as_double_array();
//     gripper_motion_time_ = this->get_parameter("gripper_motion_time").as_double();
//     gripper_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(gripper_controller_topic_, 10);

//     // Rotating table collision params
//     add_table_collision_ = this->get_parameter("add_table_collision").as_bool();
//     table_frame_ = this->get_parameter("table_frame").as_string();
//     if (table_frame_.empty()) table_frame_ = base_frame_;
//     table_radius_ = this->get_parameter("table_radius").as_double();
//     table_height_ = this->get_parameter("table_height").as_double();
//     table_x_ = this->get_parameter("table_x").as_double();
//     table_y_ = this->get_parameter("table_y").as_double();
//     table_z_ = this->get_parameter("table_z").as_double();
//     table_object_id_ = this->get_parameter("table_object_id").as_string();
//     table_clearance_ = this->get_parameter("table_clearance").as_double();

//     // Cartesian execution guards and options
//     min_cart_fraction_ = this->get_parameter("min_cart_fraction").as_double();
//     min_descent_dz_ = this->get_parameter("min_descent_dz").as_double();
//     cartesian_avoid_collisions_ = this->get_parameter("cartesian_avoid_collisions").as_bool();
//     allow_ee_table_touch_ = this->get_parameter("allow_ee_table_touch").as_bool();
//   }

//   // Gripper
//   bool command_gripper_(const std::vector<double>& pos, double seconds, double start_delay_s = 0.0) {
//     if (!gripper_pub_) return false;
//     if (pos.size() != gripper_joints_.size()) {
//       RCLCPP_ERROR(get_logger(), "Gripper command size mismatch");
//       return false;
//     }
//     trajectory_msgs::msg::JointTrajectory traj;
//     traj.header.stamp = this->get_clock()->now() + rclcpp::Duration::from_seconds(std::max(0.0, start_delay_s));
//     traj.joint_names = gripper_joints_;
//     trajectory_msgs::msg::JointTrajectoryPoint pt;
//     pt.positions = pos;
//     pt.time_from_start = rclcpp::Duration::from_seconds(seconds);
//     traj.points.push_back(pt);
//     gripper_pub_->publish(traj);
//     return true;
//   }
//   bool open_gripper_(double start_delay_s = 0.0) { return command_gripper_(gripper_open_pos_, gripper_motion_time_, start_delay_s); }
//   bool close_gripper_(double start_delay_s = 0.0) { return command_gripper_(gripper_close_pos_, gripper_motion_time_, start_delay_s); }

//   void schedule_gripper_close_for_plan_(const moveit::planning_interface::MoveGroupInterface::Plan& plan) {
//     if (plan.trajectory_.joint_trajectory.points.empty()) return;
//     const auto& last_tfs = plan.trajectory_.joint_trajectory.points.back().time_from_start;
//     const double traj_total = rclcpp::Duration(last_tfs).seconds();
//     double desired_start = targets_.t_grasp - gripper_motion_time_;
//     double latest_start = std::max(0.0, traj_total - gripper_motion_time_);
//     double start_delay = std::clamp(desired_start, 0.0, latest_start);
//     (void)close_gripper_(start_delay);
//   }

//   geometry_msgs::msg::Quaternion quatFromRPY(double r, double p, double y) {
//     Eigen::AngleAxisd Rx(r, Eigen::Vector3d::UnitX());
//     Eigen::AngleAxisd Ry(p, Eigen::Vector3d::UnitY());
//     Eigen::AngleAxisd Rz(y, Eigen::Vector3d::UnitZ());
//     Eigen::Quaterniond q = Rz * Ry * Rx;
//     geometry_msgs::msg::Quaternion qmsg;
//     qmsg.x = q.x(); qmsg.y = q.y(); qmsg.z = q.z(); qmsg.w = q.w();
//     return qmsg;
//   }
//   static double yawFromQuat(const geometry_msgs::msg::Quaternion& q) {
//     Eigen::Quaterniond qe(q.w, q.x, q.y, q.z);
//     Eigen::Vector3d eul = qe.toRotationMatrix().eulerAngles(2, 1, 0);
//     return eul[0];
//   }

//   geometry_msgs::msg::PoseStamped build_target_pose_(const geometry_msgs::msg::PoseStamped& current) {
//     geometry_msgs::msg::PoseStamped tgt = current;
//     tgt.header.frame_id = current.header.frame_id.empty() ? base_frame_ : current.header.frame_id;
//     if (target_mode_ == "relative") {
//       tgt.pose.position.x = current.pose.position.x + target_x_;
//       tgt.pose.position.y = current.pose.position.y + target_y_;
//       tgt.pose.position.z = current.pose.position.z + target_z_;
//     } else {
//       tgt.pose.position.x = target_x_;
//       tgt.pose.position.y = target_y_;
//       tgt.pose.position.z = target_z_;
//     }
//     if (!keep_orientation_) {
//       tgt.pose.orientation = quatFromRPY(target_roll_, target_pitch_, target_yaw_);
//     }
//     return tgt;
//   }

//   bool check_joint_states_topic()
//   {
//     auto topic_names = this->get_topic_names_and_types();
//     for (const auto& topic : topic_names) {
//       if (topic.first == "/joint_states") {
//         RCLCPP_INFO(get_logger(), "Found /joint_states topic");
//         return true;
//       }
//     }
//     RCLCPP_ERROR(get_logger(), "Joint states topic /joint_states not found!");
//     return false;
//   }

//   void debug_robot_model() {
//     if (!move_group_) return;
//     auto robot_model = move_group_->getRobotModel();
//     RCLCPP_INFO(get_logger(), "Robot model: %s", robot_model->getName().c_str());
//     const auto& link_names = robot_model->getLinkModelNames();
//     RCLCPP_INFO(get_logger(), "Available links (%zu):", link_names.size());
//     for (const auto& link : link_names) RCLCPP_INFO(get_logger(), "  - %s", link.c_str());
//     if (robot_model->hasLinkModel(ee_link_)) {
//       RCLCPP_INFO(get_logger(), "✓ End effector link '%s' found", ee_link_.c_str());
//     } else {
//       RCLCPP_ERROR(get_logger(), "✗ End effector link '%s' NOT found", ee_link_.c_str());
//     }
//     auto current_state = move_group_->getCurrentState(0.0);
//     if (!current_state) return;
//     const auto* group = current_state->getJointModelGroup(planning_group_);
//     if (group) {
//       const auto& ee_links = group->getLinkModelNames();
//       RCLCPP_INFO(get_logger(), "Links in planning group '%s':", planning_group_.c_str());
//       for (const auto& link : ee_links) RCLCPP_INFO(get_logger(), "  - %s", link.c_str());
//     }
//   }

//   bool wait_for_valid_joint_states_with_time(double timeout_seconds = 20.0) {
//     RCLCPP_INFO(get_logger(), "Waiting for joint states with valid timestamps...");
//     bool received_joint_states = false;

//     auto joint_state_sub = this->create_subscription<sensor_msgs::msg::JointState>(
//         "joint_states", 10,
//         [&received_joint_states, this](const sensor_msgs::msg::JointState::SharedPtr msg) {
//           RCLCPP_INFO_ONCE(get_logger(), "Received joint state with timestamp %.3f and %zu joints",
//                            rclcpp::Time(msg->header.stamp).seconds(), msg->name.size());
//           received_joint_states = true;
//         });

//     auto start_time = this->get_clock()->now();
//     while (rclcpp::ok()) {
//       rclcpp::spin_some(this->get_node_base_interface());
//       if (received_joint_states) {
//         if (move_group_) {
//           try {
//             auto state = move_group_->getCurrentState(0.1);
//             if (state) {
//               RCLCPP_INFO(get_logger(), "Successfully received robot state from MoveGroup!");
//               return true;
//             }
//           } catch (const std::exception& e) {
//             RCLCPP_WARN_THROTTLE(get_logger(), *this->get_clock(), 3000,
//                                  "Exception getting current state: %s", e.what());
//           }
//         }
//       }
//       if ((this->get_clock()->now() - start_time).seconds() > timeout_seconds) {
//         RCLCPP_ERROR(get_logger(), "Timeout waiting for valid joint states");
//         return false;
//       }
//       rclcpp::sleep_for(std::chrono::milliseconds(100));
//     }
//     return false;
//   }

//   geometry_msgs::msg::PoseStamped get_current_pose_robust() {
//     if (!move_group_) return geometry_msgs::msg::PoseStamped();
//     for (int attempt = 0; attempt < 3; ++attempt) {
//       try {
//         rclcpp::sleep_for(std::chrono::milliseconds(80));
//         auto pose = move_group_->getCurrentPose(ee_link_);
//         if (std::abs(pose.pose.position.x) > 0.001 ||
//             std::abs(pose.pose.position.y) > 0.001 ||
//             std::abs(pose.pose.position.z) > 0.001) {
//           return pose;
//         }
//       } catch (...) {}
//     }
//     return geometry_msgs::msg::PoseStamped();
//   }

//   size_t findNearestIndexToPose(const robot_trajectory::RobotTrajectory& rt,
//                                 const std::string& link,
//                                 const geometry_msgs::msg::Pose& target) {
//     size_t best_idx = 0;
//     double best_dist = std::numeric_limits<double>::infinity();
//     for (size_t i = 0; i < rt.getWayPointCount(); ++i) {
//       const auto& st = rt.getWayPoint(i);
//       const auto& T = st.getGlobalLinkTransform(link);
//       Eigen::Vector3d p = T.translation();
//       const double dx = p.x() - target.position.x;
//       const double dy = p.y() - target.position.y;
//       const double dz = p.z() - target.position.z;
//       const double d = std::sqrt(dx*dx + dy*dy + dz*dz);
//       if (d < best_dist) { best_dist = d; best_idx = i; }
//     }
//     return best_idx;
//   }

//   // Predictor async flow (single request per attempt)
//   void request_pick_pose_async_(double t_rel_s)
//   {
//     if (!pick_client_) { RCLCPP_ERROR(get_logger(), "Predictor client not created"); end_planning_session_(true); return; }
//     if (!predictor_ready_.load(std::memory_order_relaxed) && require_predictor_ready_) {
//       RCLCPP_WARN(get_logger(), "Predictor not ready yet, skipping request");
//       end_planning_session_(true);
//       return;
//     }
//     if (t_rel_s < 0.0) t_rel_s = 0.0;

//     auto req = std::make_shared<lite6_pick_predictor_interfaces::srv::GetPoseAt::Request>();
//     req->use_relative = true;
//     const double si = std::floor(t_rel_s);
//     req->query_time.sec = static_cast<int32_t>(si);
//     req->query_time.nanosec = static_cast<uint32_t>((t_rel_s - si) * 1e9);

//     auto future = pick_client_->async_send_request(req,
//       [this](rclcpp::Client<lite6_pick_predictor_interfaces::srv::GetPoseAt>::SharedFuture fut)
//       {
//         try {
//           auto res = fut.get();
//           if (!res->ok) {
//             RCLCPP_ERROR(get_logger(), "GetPoseAt returned ok=false");
//             end_planning_session_(true);
//             return;
//           }
//           const auto& pick_pose = res->pose;

//           // Detailed logging of predictor timing
//           const double t_now = this->get_clock()->now().seconds();
//           const double t_pose = rclcpp::Time(pick_pose.header.stamp).seconds();
//           RCLCPP_INFO(get_logger(),
//                       "Received pick pose [%s] (%.3f, %.3f, %.3f) t_now=%.3f t_pose=%.3f dt=%.3f",
//                       pick_pose.header.frame_id.c_str(),
//                       pick_pose.pose.position.x, pick_pose.pose.position.y, pick_pose.pose.position.z,
//                       t_now, t_pose, t_pose - t_now);

//           // Plan hover + descent
//           if (!plan_and_execute_from_pick_(pick_pose)) {
//             RCLCPP_ERROR(get_logger(), "Planning/execution failed");
//             end_planning_session_(true);
//             return;
//           }
//           end_planning_session_(false);
//         }
//         catch (const std::exception& e) {
//           RCLCPP_ERROR(get_logger(), "Predictor future exception: %s", e.what());
//           end_planning_session_(true);
//         }
//       });

//     (void)future;
//   }

//   void end_planning_session_(bool cooldown)
//   {
//     planning_in_progress_ = false;
//     if (cooldown) last_attempt_wall_ = std::chrono::steady_clock::now();
//   }

//   bool transform_to_frame_(const geometry_msgs::msg::PoseStamped& in,
//                            const std::string& to_frame,
//                            geometry_msgs::msg::PoseStamped& out)
//   {
//     if (in.header.frame_id.empty() || in.header.frame_id == to_frame) {
//       out = in;
//       out.header.frame_id = to_frame;
//       return true;
//     }
//     try {
//       geometry_msgs::msg::TransformStamped tf =
//           tf_buffer_->lookupTransform(to_frame, in.header.frame_id, tf2::TimePointZero, tf2::durationFromSec(0.3));
//       tf2::doTransform(in, out, tf);
//       out.header.frame_id = to_frame;
//       return true;
//     } catch (const std::exception& e) {
//       RCLCPP_ERROR(get_logger(), "TF transform %s -> %s failed: %s",
//                    in.header.frame_id.c_str(), to_frame.c_str(), e.what());
//       return false;
//     }
//   }

//   bool plan_and_execute_from_pick_(const geometry_msgs::msg::PoseStamped& pick_pose_in)
//   {
//     if (!move_group_) return false;

//     const std::string planning_frame = move_group_->getPlanningFrame();
//     geometry_msgs::msg::PoseStamped pick_pose_tf;
//     if (!transform_to_frame_(pick_pose_in, planning_frame, pick_pose_tf)) return false;

//     geometry_msgs::msg::PoseStamped pose_hover = pick_pose_tf;
//     geometry_msgs::msg::PoseStamped pose_grasp = pick_pose_tf;

//     double yaw = 0.0;
//     if (horizontal_yaw_mode_ == "current") {
//       auto cur = get_current_pose_robust();
//       yaw = cur.header.frame_id.empty() ? yawFromQuat(pick_pose_tf.pose.orientation)
//                                         : yawFromQuat(cur.pose.orientation);
//     } else if (horizontal_yaw_mode_ == "fixed") {
//       yaw = fixed_yaw_;
//     } else {
//       yaw = yawFromQuat(pick_pose_tf.pose.orientation);
//     }
//     const auto horiz = quatFromRPY(horizontal_roll_offset_, 0.0, yaw);

//     // Targets
//     pose_hover.pose.position.z = z_hover_;
//     pose_grasp.pose.position.z = z_grasp_;

//     // Apply horizontal orientation for descent
//     if (enforce_horizontal_) {
//       // Keep current orientation for leg-1 translation; apply horizontal on descent
//       // so we don't rotate close to the table
//     }

//     // Compute safe clearance above table top (in planning frame)
//     const double table_top_z = table_z_ + 0.5 * table_height_;
//     const double lift_margin = 0.05; // tune if needed

//     // Clamp grasp Z above table
//     const double min_grasp_z = table_top_z + table_clearance_;
//     if (pose_grasp.pose.position.z < min_grasp_z) {
//       RCLCPP_WARN(get_logger(),
//                   "Requested grasp Z=%.3f is below/too close to table top=%.3f; clamping to %.3f (clearance=%.3f)",
//                   pose_grasp.pose.position.z, table_top_z, min_grasp_z, table_clearance_);
//       pose_grasp.pose.position.z = min_grasp_z;
//     }

//     // Stage 1: Lift -> Rotate (at safe Z) -> Slide XY -> Down-to-hover
//     move_group_->setStartStateToCurrentState();
//     const auto pose_current = get_current_pose_robust();
//     if (pose_current.header.frame_id.empty()) {
//       RCLCPP_ERROR(get_logger(), "No current pose for Cartesian current->hover");
//       return false;
//     }

//     double safe_z = std::max({pose_current.pose.position.z + 0.05, z_hover_, table_top_z + lift_margin});

//     // Waypoints
//     geometry_msgs::msg::Pose p_up = pose_current.pose;     // straight up, keep current orientation
//     p_up.position.z = safe_z;

//     geometry_msgs::msg::Pose p_rotate = p_up;              // rotate in place at safe Z
//     if (enforce_horizontal_) p_rotate.orientation = horiz;

//     geometry_msgs::msg::Pose p_xy = p_rotate;              // translate XY at safe Z
//     p_xy.position.x = pose_hover.pose.position.x;
//     p_xy.position.y = pose_hover.pose.position.y;

//     geometry_msgs::msg::Pose p_down = p_xy;                // descend to hover Z
//     p_down.position.z = pose_hover.pose.position.z;

//     std::vector<geometry_msgs::msg::Pose> leg1_waypoints;
//     leg1_waypoints.push_back(p_up);
//     if (enforce_horizontal_) leg1_waypoints.push_back(p_rotate);
//     leg1_waypoints.push_back(p_xy);
//     leg1_waypoints.push_back(p_down);

//     moveit_msgs::msg::RobotTrajectory leg1_msg;
//     const double eef_step1 = 0.01;
//     const double jump_threshold1 = 0.0;
//     double fraction1 = 0.0;
//     try {
//       fraction1 = move_group_->computeCartesianPath(leg1_waypoints, eef_step1, jump_threshold1, leg1_msg, true);
//     } catch (const std::exception& e) {
//       RCLCPP_ERROR(get_logger(), "Cartesian leg1 exception: %s", e.what());
//       return false;
//     }
//     if (fraction1 < 0.8) {
//       moveit_msgs::msg::RobotTrajectory dbg_msg;
//       double fraction_nc = 0.0;
//       try { fraction_nc = move_group_->computeCartesianPath(leg1_waypoints, eef_step1, jump_threshold1, dbg_msg, false); } catch (...) {}
//       RCLCPP_WARN(get_logger(), "Leg-1 low fraction=%.2f (collisions). No-collision fraction=%.2f", fraction1, fraction_nc);
//       if (fraction1 < 0.5) {
//         RCLCPP_ERROR(get_logger(), "Cartesian leg1 failed (fraction=%.2f)", fraction1);
//         return false;
//       }
//     }

//     // Time-parameterize leg 1
//     moveit::core::RobotModelConstPtr model1 = move_group_->getRobotModel();
//     auto start_state1 = move_group_->getCurrentState(2.0);
//     if (!start_state1) {
//       RCLCPP_ERROR(get_logger(), "No current robot state before leg1");
//       return false;
//     }
//     robot_trajectory::RobotTrajectory rt1(model1, planning_group_);
//     rt1.setRobotTrajectoryMsg(*start_state1, leg1_msg);
//     if (!tp_control_cpp::applyTimeParameterization(rt1, method_, vel_scale_, acc_scale_)) {
//       RCLCPP_ERROR(get_logger(), "Time parameterization failed for leg1");
//       return false;
//     }
//     if (method_ == TimingMethod::TOTG_THEN_RUCKIG) {
//       (void)tp_control_cpp::applyTimeParameterization(rt1, TimingMethod::RUCKIG, vel_scale_, acc_scale_);
//     }
//     rt1.getRobotTrajectoryMsg(leg1_msg);

//     moveit::planning_interface::MoveGroupInterface::Plan plan_to_hover;
//     plan_to_hover.trajectory_ = leg1_msg;
//     moveit::core::robotStateToRobotStateMsg(*start_state1, plan_to_hover.start_state_);
//     plan_to_hover.planning_time_ = 0.0;

//     if (display_pub_) {
//       moveit_msgs::msg::DisplayTrajectory msg;
//       msg.model_id = move_group_->getRobotModel()->getName();
//       if (auto cs0 = move_group_->getCurrentState(0.0)) {
//         moveit_msgs::msg::RobotState rs;
//         moveit::core::robotStateToRobotStateMsg(*cs0, rs);
//         msg.trajectory_start = rs;
//       }
//       msg.trajectory.push_back(plan_to_hover.trajectory_);
//       display_pub_->publish(msg);
//     }
//     auto exec_code = move_group_->execute(plan_to_hover);
//     if (exec_code != moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
//       RCLCPP_ERROR(get_logger(), "Execution to hover failed (code=%d)", exec_code.val);
//       return false;
//     }

//     // Stage 2: hover -> grasp (keep horizontal)
//     move_group_->setStartStateToCurrentState();
//     geometry_msgs::msg::PoseStamped hover_for_descent = pose_hover;
//     if (enforce_horizontal_) hover_for_descent.pose.orientation = horiz;
//     if (enforce_horizontal_) pose_grasp.pose.orientation = horiz;

//     std::vector<geometry_msgs::msg::Pose> waypoints{hover_for_descent.pose, pose_grasp.pose};
//     moveit_msgs::msg::RobotTrajectory cart_traj_msg;
//     const double eef_step = 0.005;
//     const double jump_threshold = 0.0;
//     double fraction = 0.0;
//     try {
//       fraction = move_group_->computeCartesianPath(waypoints, eef_step, jump_threshold, cart_traj_msg, cartesian_avoid_collisions_);
//     } catch (const std::exception& e) {
//       RCLCPP_ERROR(get_logger(), "Cartesian descent exception: %s", e.what());
//       return false;
//     }

//     const double dz_req = std::abs(hover_for_descent.pose.position.z - pose_grasp.pose.position.z);
//     const double dz_achieved = dz_req * std::clamp(fraction, 0.0, 1.0);

//     if (fraction < min_cart_fraction_ || dz_achieved < min_descent_dz_) {
//       RCLCPP_WARN(get_logger(),
//                   "Aborting descent: fraction=%.2f (min=%.2f), dz_req=%.3f, dz_achieved=%.3f (min=%.3f), avoid_collisions=%s",
//                   fraction, min_cart_fraction_, dz_req, dz_achieved, min_descent_dz_,
//                   cartesian_avoid_collisions_ ? "true" : "false");
//       return false;
//     }

//     moveit::core::RobotModelConstPtr model = move_group_->getRobotModel();
//     auto start_state = move_group_->getCurrentState(2.0);
//     if (!start_state) {
//       RCLCPP_ERROR(get_logger(), "No current robot state before descent");
//       return false;
//     }
//     robot_trajectory::RobotTrajectory rt(model, planning_group_);
//     rt.setRobotTrajectoryMsg(*start_state, cart_traj_msg);
//     if (!tp_control_cpp::applyTimeParameterization(rt, method_, vel_scale_, acc_scale_)) {
//       RCLCPP_ERROR(get_logger(), "Time parameterization failed for descent");
//       return false;
//     }
//     const size_t i_hover = 0;
//     const size_t i_grasp = rt.getWayPointCount() ? (rt.getWayPointCount() - 1) : 0;
//     (void)tp_control_cpp::enforceSegmentTimes(rt, i_hover, i_grasp, targets_);
//     if (method_ == TimingMethod::TOTG_THEN_RUCKIG) {
//       (void)tp_control_cpp::applyTimeParameterization(rt, TimingMethod::RUCKIG, vel_scale_, acc_scale_);
//     }
//     rt.getRobotTrajectoryMsg(cart_traj_msg);

//     moveit::planning_interface::MoveGroupInterface::Plan plan_descent;
//     plan_descent.trajectory_ = cart_traj_msg;
//     moveit::core::robotStateToRobotStateMsg(*start_state, plan_descent.start_state_);
//     plan_descent.planning_time_ = 0.0;

//     if (display_pub_) {
//       moveit_msgs::msg::DisplayTrajectory msg;
//       msg.model_id = move_group_->getRobotModel()->getName();
//       if (auto cs = move_group_->getCurrentState(0.0)) {
//         moveit_msgs::msg::RobotState rs;
//         moveit::core::robotStateToRobotStateMsg(*cs, rs);
//         msg.trajectory_start = rs;
//       }
//       msg.trajectory.push_back(plan_descent.trajectory_);
//       display_pub_->publish(msg);
//     }
//     schedule_gripper_close_for_plan_(plan_descent);
//     (void)move_group_->execute(plan_descent);
//     return true;
//   }

//   void run_once_with_predictor_()
//   {
//     if (!move_group_) { RCLCPP_ERROR(get_logger(), "MoveGroup not initialized"); end_planning_session_(true); return; }
//     if (!pick_client_ || !pick_client_->wait_for_service(std::chrono::seconds(0))) {
//       RCLCPP_WARN(get_logger(), "Predictor service '%s' not ready", predictor_service_.c_str());
//       end_planning_session_(true);
//       return;
//     }
//     move_group_->setStartStateToCurrentState();
//     request_pick_pose_async_(commit_pick_time_s_);
//   }

//   // Demo without predictor (unused)
//   void run_once()
//   {
//     if (!move_group_) return;
//     auto pose0 = get_current_pose_robust();
//     if (pose0.header.frame_id.empty()) return;
//     geometry_msgs::msg::PoseStamped pose_hover = build_target_pose_(pose0);
//     pose_hover.pose.position.z = z_hover_;
//     geometry_msgs::msg::PoseStamped pose_grasp = pose_hover;
//     pose_grasp.pose.position.z = z_grasp_;
//     std::vector<geometry_msgs::msg::Pose> waypoints{pose_hover.pose, pose_grasp.pose};

//     moveit_msgs::msg::RobotTrajectory cart_traj_msg;
//     (void)move_group_->computeCartesianPath(waypoints, 0.005, 0.0, cart_traj_msg, true);
//     moveit::core::RobotModelConstPtr model = move_group_->getRobotModel();
//     auto start_state = move_group_->getCurrentState(2.0);
//     if (!start_state) return;
//     robot_trajectory::RobotTrajectory rt(model, planning_group_);
//     rt.setRobotTrajectoryMsg(*start_state, cart_traj_msg);
//     (void)tp_control_cpp::applyTimeParameterization(rt, method_, vel_scale_, acc_scale_);
//     rt.getRobotTrajectoryMsg(cart_traj_msg);
//     moveit::planning_interface::MoveGroupInterface::Plan plan;
//     plan.trajectory_ = cart_traj_msg;
//     moveit::core::robotStateToRobotStateMsg(*start_state, plan.start_state_);
//     if (display_pub_) {
//       moveit_msgs::msg::DisplayTrajectory msg;
//       msg.model_id = move_group_->getRobotModel()->getName();
//       msg.trajectory.push_back(plan.trajectory_);
//       display_pub_->publish(msg);
//     }
//     (void)move_group_->execute(plan);
//   }

//   // Add a cylinder as the rotating table collision object (matches tp_control.py)
//   void add_rotating_table_collision_(const geometry_msgs::msg::Pose& table_pose,
//                                      double table_radius,
//                                      double table_height)
//   {
//     moveit_msgs::msg::CollisionObject obj;
//     obj.header.frame_id = table_frame_.empty() ? base_frame_ : table_frame_;
//     obj.header.stamp = this->get_clock()->now();
//     obj.id = table_object_id_;

//     shape_msgs::msg::SolidPrimitive cylinder;
//     cylinder.type = shape_msgs::msg::SolidPrimitive::CYLINDER;
//     cylinder.dimensions.resize(2);
//     cylinder.dimensions[shape_msgs::msg::SolidPrimitive::CYLINDER_HEIGHT] = table_height;
//     cylinder.dimensions[shape_msgs::msg::SolidPrimitive::CYLINDER_RADIUS] = table_radius;

//     obj.primitives.push_back(cylinder);
//     obj.primitive_poses.push_back(table_pose);
//     obj.operation = moveit_msgs::msg::CollisionObject::ADD;

//     // Apply via PlanningSceneInterface
//     planning_scene_interface_.applyCollisionObject(obj);

//     // Optional: publish as diff
//     moveit_msgs::msg::PlanningScene scene_msg;
//     scene_msg.is_diff = true;
//     scene_msg.world.collision_objects.push_back(obj);
//     if (planning_scene_pub_) planning_scene_pub_->publish(scene_msg);
//   }

//   // Allow/disallow collisions between EE and table object
//   void allow_ee_table_collision_(bool allow)
//   {
//     if (!planning_scene_pub_) return;
//     moveit_msgs::msg::PlanningScene scene_msg;
//     scene_msg.is_diff = true;

//     auto& acm = scene_msg.allowed_collision_matrix;
//     // Build a 2x2 matrix [ee, table] with symmetric allowed flag
//     acm.entry_names = {ee_link_, table_object_id_};
//     acm.entry_values.resize(2);
//     for (auto& row : acm.entry_values) row.enabled.resize(2, false);
//     acm.entry_values[0].enabled[1] = allow; // ee vs table
//     acm.entry_values[1].enabled[0] = allow; // table vs ee

//     planning_scene_pub_->publish(scene_msg);
//   }

// private:
//   std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
//   moveit_visual_tools::MoveItVisualToolsPtr visual_tools_;
//   rclcpp::Publisher<moveit_msgs::msg::DisplayTrajectory>::SharedPtr display_pub_;

//   std::string planning_group_, ee_link_, base_frame_;
//   double z_hover_{0.20}, z_grasp_{0.05};
//   double vel_scale_{0.9}, acc_scale_{0.6};
//   TimingMethod method_{TimingMethod::TOTG_THEN_RUCKIG};
//   std::string timing_method_str_;
//   TimingTargets targets_;
//   bool use_pose_target_{false};
//   std::string target_mode_;
//   std::string target_frame_;
//   double target_x_{0.0}, target_y_{0.0}, target_z_{0.0};
//   bool keep_orientation_{true};
//   double target_roll_{0.0}, target_pitch_{0.0}, target_yaw_{0.0};

//   // Horizontal control
//   bool enforce_horizontal_{false};
//   std::string horizontal_yaw_mode_{"predictor"}; // predictor|current|fixed
//   double fixed_yaw_{0.0};
//   double horizontal_roll_offset_{M_PI};
//   double rp_tol_{0.10};
//   double yaw_tol_{M_PI};
//   bool use_path_constraints_for_hover_{false};
//   std::string constraint_mode_{"goal_only"}; // goal_only|path_only|both

//   // Flow control
//   std::atomic_bool planning_in_progress_{false};
//   bool move_group_params_ready_{false};
//   rclcpp::TimerBase::SharedPtr delayed_plan_timer_;
//   double replan_cooldown_s_{2.0};
//   std::chrono::steady_clock::time_point last_attempt_wall_{};
//   bool last_ik_reachable_{false};

//   // TF
//   std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
//   std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

//   // Predictor
//   bool use_predictor_{true};
//   bool require_predictor_ready_{true};
//   double commit_pick_time_s_{1.0};
//   std::string predictor_ready_topic_{"predictor_ready"};
//   std::string predictor_service_{"get_predicted_pose_at"};
//   std::atomic<bool> predictor_ready_{false};
//   rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr predictor_ready_sub_;
//   rclcpp::Client<lite6_pick_predictor_interfaces::srv::GetPoseAt>::SharedPtr pick_client_;

//   // Planning params
//   double planning_time_{0.5};
//   std::string planner_id_{"RRTConnect"};

//   // IK gate
//   bool require_ik_gate_{true};
//   std::string ik_gate_topic_{"/ik_gate/output"};
//   rclcpp::Subscription<std_msgs::msg::String>::SharedPtr ik_gate_sub_;

//   // Gripper
//   rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr gripper_pub_;
//   std::string gripper_controller_topic_;
//   std::vector<std::string> gripper_joints_;
//   std::vector<double> gripper_open_pos_;
//   std::vector<double> gripper_close_pos_;
//   double gripper_motion_time_{0.7};

//   // Planning scene interface and optional publisher
//   moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;
//   rclcpp::Publisher<moveit_msgs::msg::PlanningScene>::SharedPtr planning_scene_pub_;

//   // Rotating table config
//   bool add_table_collision_{true};
//   std::string table_frame_{"link_base"};
//   double table_radius_{0.75};
//   double table_height_{0.1};
//   double table_x_{0.9};
//   double table_y_{0.0};
//   double table_z_{0.1};
//   std::string table_object_id_{"rotating_table"};
//   double table_clearance_{0.02};

//   // Cartesian execution guards/options
//   double min_cart_fraction_{0.95};
//   double min_descent_dz_{0.03};
//   bool cartesian_avoid_collisions_{true};
//   bool allow_ee_table_touch_{false};
// };

// int main(int argc, char** argv)
// {
//   rclcpp::init(argc, argv);
//   auto node = std::make_shared<TpControlNode>();
//   node->initialize();
//   rclcpp::spin(node);
//   rclcpp::shutdown();
//   return 0;
// }
//========================================
// Planner is engaged after ik_gate, but 
// descending motion is problematic
//========================================
// #include <rclcpp/rclcpp.hpp>
// #include <rclcpp/parameter_client.hpp>

// #include <geometry_msgs/msg/pose_stamped.hpp>
// #include <sensor_msgs/msg/joint_state.hpp>
// #include <eigen3/Eigen/Geometry>

// #include <moveit/move_group_interface/move_group_interface.h>
// #include <moveit/planning_scene_monitor/planning_scene_monitor.h>
// #include <moveit/planning_scene_interface/planning_scene_interface.h>
// #include <moveit/robot_state/conversions.h>
// #include <moveit/robot_trajectory/robot_trajectory.h>
// #include <moveit_msgs/msg/move_it_error_codes.h>
// #include <moveit_visual_tools/moveit_visual_tools.h>
// #include <moveit_msgs/msg/display_trajectory.hpp>
// #include <moveit_msgs/msg/constraints.hpp>
// #include <moveit_msgs/msg/orientation_constraint.hpp>
// #include <moveit_msgs/msg/collision_object.hpp>
// #include <moveit_msgs/msg/planning_scene.hpp>
// #include <shape_msgs/msg/solid_primitive.hpp>

// #include <moveit/trajectory_processing/time_optimal_trajectory_generation.h>
// #include <moveit/trajectory_processing/ruckig_traj_smoothing.h>

// #include <trajectory_msgs/msg/joint_trajectory.hpp>
// #include <trajectory_msgs/msg/joint_trajectory_point.hpp>

// #include <tf2_eigen/tf2_eigen.hpp>
// #include <tf2_ros/transform_listener.h>
// #include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
// #include <tf2/time.h>

// #include <std_msgs/msg/bool.hpp>
// #include <std_msgs/msg/string.hpp>
// #include <lite6_pick_predictor_interfaces/srv/get_pose_at.hpp>

// #include <algorithm>
// #include <limits>
// #include <cmath>
// #include <atomic>
// #include <string>
// #include <vector>
// #include <chrono>

// namespace tp_control_cpp {

// enum class TimingMethod { TOTG, RUCKIG, TOTG_THEN_RUCKIG };

// struct TimingTargets
// {
//   double t_hover{0.0};
//   double t_grasp{0.0};
// };

// inline double durationFromStart(const robot_trajectory::RobotTrajectory& rt, size_t idx)
// {
//   if (rt.getWayPointCount() == 0 || idx == 0) return 0.0;
//   idx = std::min(idx, rt.getWayPointCount() - 1);
//   double sum = 0.0;
//   for (size_t i = 1; i <= idx; ++i)
//     sum += rt.getWayPointDurationFromPrevious(i);
//   return sum;
// }

// inline bool applyTimeParameterization(robot_trajectory::RobotTrajectory& rt,
//                                       TimingMethod method,
//                                       double max_vel_scaling = 0.9,
//                                       double max_acc_scaling = 0.7)
// {
//   using trajectory_processing::TimeOptimalTrajectoryGeneration;
//   using trajectory_processing::RuckigSmoothing;

//   switch (method)
//   {
//     case TimingMethod::TOTG:
//     {
//       TimeOptimalTrajectoryGeneration totg;
//       return totg.computeTimeStamps(rt, max_vel_scaling, max_acc_scaling);
//     }
//     case TimingMethod::RUCKIG:
//     {
//       return RuckigSmoothing::applySmoothing(rt, max_vel_scaling, max_acc_scaling);
//     }
//     case TimingMethod::TOTG_THEN_RUCKIG:
//     {
//       TimeOptimalTrajectoryGeneration totg;
//       if (!totg.computeTimeStamps(rt, max_vel_scaling, max_acc_scaling))
//         return false;
//       return RuckigSmoothing::applySmoothing(rt, max_vel_scaling, max_acc_scaling);
//     }
//     default:
//       return false;
//   }
// }

// inline bool enforceSegmentTimes(robot_trajectory::RobotTrajectory& rt,
//                                 size_t idx_hover,
//                                 size_t idx_grasp,
//                                 const TimingTargets& targets)
// {
//   const size_t N = rt.getWayPointCount();
//   if (N < 2) return false;
//   idx_hover = std::min(idx_hover, N - 1);
//   idx_grasp = std::min(idx_grasp, N - 1);
//   if (idx_hover >= idx_grasp) idx_hover = std::max<size_t>(1, std::min(idx_grasp, idx_hover));

//   auto safe_get = [&](size_t i) { return rt.getWayPointDurationFromPrevious(i); };
//   auto safe_set = [&](size_t i, double v) { rt.setWayPointDurationFromPrevious(i, std::max(1e-9, v)); };

//   bool changed = false;

//   double t_hover_cur = durationFromStart(rt, idx_hover);
//   if (targets.t_hover > 0.0 && t_hover_cur < targets.t_hover && idx_hover >= 1)
//   {
//     const double scale1 = targets.t_hover / std::max(1e-9, t_hover_cur);
//     for (size_t i = 1; i <= idx_hover; ++i)
//       safe_set(i, safe_get(i) * scale1);
//     changed = true;
//   }

//   double t_hover_new = durationFromStart(rt, idx_hover);
//   const double t_total_cur = durationFromStart(rt, N - 1);
//   const double tail_cur = std::max(0.0, t_total_cur - t_hover_new);

//   if (targets.t_grasp > 0.0 && (t_hover_new + tail_cur) < targets.t_grasp && idx_grasp >= (idx_hover + 1))
//   {
//     const double needed_tail = targets.t_grasp - t_hover_new;
//     if (tail_cur <= 1e-9)
//     {
//       double last = safe_get(N - 1);
//       if (last < 1e-6) last = 1e-3;
//       safe_set(N - 1, last + std::max(0.0, needed_tail));
//     }
//     else
//     {
//       const double scale2 = std::max(1.0, needed_tail / tail_cur);
//       for (size_t i = idx_hover + 1; i <= idx_grasp; ++i)
//         safe_set(i, safe_get(i) * scale2);
//     }
//     changed = true;
//   }

//   return changed;
// }

// } // namespace tp_control_cpp

// using tp_control_cpp::TimingMethod;
// using tp_control_cpp::TimingTargets;

// class TpControlNode : public rclcpp::Node
// {
// public:
//   TpControlNode() : Node("tp_control_node")
//   {
//     this->declare_parameter<std::string>("robot_description", "");
//     this->declare_parameter<std::string>("robot_description_semantic", "");
//     this->declare_parameter<std::string>("planning_group", "lite6");
//     this->declare_parameter<std::string>("ee_link", "link_tcp");
//     this->declare_parameter<std::string>("base_frame", "link_base");
//     this->declare_parameter<std::string>("timing_method", "totg_then_ruckig");
//     this->declare_parameter<double>("planning_time", 10.0);
//     this->declare_parameter<std::string>("planner_id", "RRTConnect");
//     this->declare_parameter<bool>("use_pose_target", false);
//     this->declare_parameter<std::string>("target_mode", "absolute");
//     this->declare_parameter<std::string>("target_frame", "");

//     this->declare_parameter<double>("vel_scale", 0.9);
//     this->declare_parameter<double>("acc_scale", 0.6);

//     this->declare_parameter<double>("target_x", 0.29);
//     this->declare_parameter<double>("target_y", -0.16);
//     this->declare_parameter<double>("target_z", 0.14);
//     this->declare_parameter<double>("z_hover", 0.29);
//     this->declare_parameter<double>("z_grasp", 0.14);
//     this->declare_parameter<double>("target_roll", 0.0);
//     this->declare_parameter<double>("target_pitch", 0.0);
//     this->declare_parameter<double>("target_yaw", 0.0);
//     this->declare_parameter<bool>("keep_orientation", false);

//     // Horizontal orientation helpers
//     this->declare_parameter<bool>("enforce_horizontal_orientation", false);
//     this->declare_parameter<std::string>("horizontal_yaw_mode", "predictor"); // predictor|current|fixed
//     this->declare_parameter<double>("fixed_yaw", 0.0);
//     this->declare_parameter<double>("horizontal_roll_offset", M_PI);          // many TCPs need a pi flip
//     this->declare_parameter<double>("constraint_rp_tolerance", 0.5);          // roll/pitch tolerance (rad)
//     this->declare_parameter<double>("constraint_yaw_tolerance", M_PI);        // yaw free
//     this->declare_parameter<bool>("use_path_constraints_for_hover", true);
//     // Constraint selection: goal_only | path_only | both
//     this->declare_parameter<std::string>("constraint_mode", "goal_only");

//     // Rotating table collision object params (like tp_control.py)
//     this->declare_parameter<bool>("add_table_collision", true);
//     this->declare_parameter<std::string>("table_frame", "link_base");
//     this->declare_parameter<double>("table_radius", 0.75);
//     this->declare_parameter<double>("table_height", 0.06);
//     this->declare_parameter<double>("table_x", 0.9);
//     this->declare_parameter<double>("table_y", 0.0);
//     this->declare_parameter<double>("table_z", 0.1);

//     this->declare_parameter<double>("t_hover", 0.0);
//     this->declare_parameter<double>("t_grasp", 0.0);

//     // IK gate
//     this->declare_parameter<bool>("require_ik_gate", true);
//     this->declare_parameter<std::string>("ik_gate_topic", "/ik_gate/output");

//     // Predictor
//     this->declare_parameter<bool>("use_predictor", true);
//     this->declare_parameter<bool>("require_predictor_ready", false);
//     this->declare_parameter<std::string>("predictor_ready_topic", "predictor_ready");
//     this->declare_parameter<std::string>("predictor_service", "get_predicted_pose_at");
//     this->declare_parameter<double>("commit_pick_time_s", 10.0);

//     // Cooldown to prevent spam
//     this->declare_parameter<double>("replan_cooldown_s", 2.0);

//     // Gripper
//     this->declare_parameter<std::string>("gripper_controller_topic", "/lite6_gripper_traj_controller/joint_trajectory");
//     this->declare_parameter<std::vector<std::string>>("gripper_joints", {"jaw_left","jaw_right"});
//     this->declare_parameter<std::vector<double>>("gripper_open", {0.02, 0.02});
//     this->declare_parameter<std::vector<double>>("gripper_close", {0.0, 0.0});
//     this->declare_parameter<double>("gripper_motion_time", 1.0);

//     // Detect /clock
//     auto topics = this->get_topic_names_and_types();
//     bool has_clock = std::any_of(topics.begin(), topics.end(), [](const auto& t){ return t.first == "/clock"; });
//     this->set_parameter(rclcpp::Parameter("use_sim_time", has_clock));
//     RCLCPP_INFO(get_logger(), has_clock ? "Found /clock, use_sim_time=true" : "No /clock, use_sim_time=false");

//     tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
//     tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

//     display_pub_ = this->create_publisher<moveit_msgs::msg::DisplayTrajectory>("display_planned_path", rclcpp::QoS(5));
//     planning_scene_pub_ = this->create_publisher<moveit_msgs::msg::PlanningScene>("/planning_scene", 10);

//     get_parameters();

//     RCLCPP_INFO(get_logger(), "TpControlNode initializing...");
//   }

//   void initialize()
//   {
//     try
//     {
//       if (!check_joint_states_topic()) {
//         RCLCPP_ERROR(get_logger(), "Joint states topic not available, cannot proceed");
//         return;
//       }
//       if (!wait_for_move_group_parameters()) {
//         RCLCPP_ERROR(get_logger(), "Could not get robot description from move_group");
//         return;
//       }
//       move_group_params_ready_ = true;

//       move_group_ = std::make_unique<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), planning_group_);
//       move_group_->setEndEffectorLink(ee_link_);
//       move_group_->startStateMonitor(5.0);

//       visual_tools_ = std::make_shared<moveit_visual_tools::MoveItVisualTools>(
//           shared_from_this(), base_frame_, "/rviz_visual_tools", move_group_->getRobotModel());
//       visual_tools_->deleteAllMarkers();
//       visual_tools_->loadRemoteControl();

//       RCLCPP_INFO(get_logger(), "TpControlNode ready. Method=%s", timing_method_str_.c_str());

//       if (!wait_for_valid_joint_states_with_time()) {
//         RCLCPP_ERROR(get_logger(), "Failed to receive valid joint states");
//         return;
//       }
//       auto cs = move_group_->getCurrentState(0.0);
//       if (!cs){
//         RCLCPP_ERROR(get_logger(), "MoveGroup current state unavailable");
//         return;
//       }
//       move_group_->setStartStateToCurrentState();

//       // Predictor wiring
//       if (use_predictor_) {
//         predictor_ready_sub_ = this->create_subscription<std_msgs::msg::Bool>(
//             predictor_ready_topic_, 10,
//             [this](const std_msgs::msg::Bool::SharedPtr msg){
//               predictor_ready_.store(msg->data, std::memory_order_relaxed);
//             });
//         pick_client_ = this->create_client<lite6_pick_predictor_interfaces::srv::GetPoseAt>(predictor_service_);
//       }

//       // Add rotating table collision object (like tp_control.py)
//       if (add_table_collision_) {
//         geometry_msgs::msg::Pose table_pose;
//         table_pose.position.x = table_x_;
//         table_pose.position.y = table_y_;
//         table_pose.position.z = table_z_;
//         table_pose.orientation.w = 1.0; // identity
//         add_rotating_table_collision_(table_pose, table_radius_, table_height_);
//         RCLCPP_INFO(get_logger(), "Applied rotating_table collision object at [%.2f, %.2f, %.2f] (r=%.2f, h=%.2f, frame=%s)",
//                     table_x_, table_y_, table_z_, table_radius_, table_height_, table_frame_.c_str());
//         rclcpp::sleep_for(std::chrono::milliseconds(200)); // allow planning scene sync
//       }

//       // IK gate rising-edge trigger + cooldown
//       if (require_ik_gate_) {
//         ik_gate_sub_ = this->create_subscription<std_msgs::msg::String>(
//             ik_gate_topic_, 10,
//             [this](const std_msgs::msg::String::SharedPtr msg) {
//               const bool reachable = (msg->data == "REACHABLE");
//               if (reachable && !last_ik_reachable_) {
//                 const auto now = std::chrono::steady_clock::now();
//                 if (!planning_in_progress_ && (now - last_attempt_wall_) >= std::chrono::duration<double>(replan_cooldown_s_)) {
//                   planning_in_progress_ = true;
//                   delayed_plan_timer_ = this->create_wall_timer(
//                       std::chrono::milliseconds(5),
//                       [this]() {
//                         if (delayed_plan_timer_) delayed_plan_timer_->cancel();
//                         this->run_once_with_predictor_();
//                       });
//                 } else {
//                   RCLCPP_WARN(get_logger(), "Cooldown active; skipping trigger");
//                 }
//               }
//               last_ik_reachable_ = reachable;
//             });
//       } else {
//         delayed_plan_timer_ = this->create_wall_timer(
//             std::chrono::milliseconds(150),
//             [this]() {
//               if (delayed_plan_timer_) delayed_plan_timer_->cancel();
//               if (!planning_in_progress_) {
//                 planning_in_progress_ = true;
//                 this->run_once_with_predictor_();
//               }
//             });
//       }

//       debug_robot_model();
//     }
//     catch (const std::exception& e)
//     {
//       RCLCPP_ERROR(get_logger(), "Failed to initialize: %s", e.what());
//     }
//   }

// private:

//   bool wait_for_move_group_parameters(double timeout_seconds = 20.0)
//   {
//     if (move_group_params_ready_) return true;
//     RCLCPP_INFO(get_logger(), "Looking for move_group parameters...");
//     auto start_time = this->get_clock()->now();

//     while (rclcpp::ok() && (this->get_clock()->now() - start_time).seconds() < timeout_seconds) {
//       auto node_names = this->get_node_names();
//       for (const auto& node_name : node_names) {
//         if (node_name.find("move_group") != std::string::npos) {
//           RCLCPP_INFO(get_logger(), "Found move_group node: %s", node_name.c_str());
//           try {
//             auto param_client = std::make_shared<rclcpp::AsyncParametersClient>(this, node_name);
//             if (param_client->wait_for_service(std::chrono::seconds(2))) {
//               auto future = param_client->get_parameters(
//                 {"robot_description", "robot_description_semantic", "robot_description_kinematics"});

//               if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), future,
//                   std::chrono::seconds(5)) == rclcpp::FutureReturnCode::SUCCESS) {

//                 auto params = future.get();
//                 for (const auto& param : params) {
//                   if (param.get_name() == "robot_description" &&
//                       param.get_type() == rclcpp::ParameterType::PARAMETER_STRING) {
//                     this->set_parameter(rclcpp::Parameter("robot_description", param.as_string()));
//                     RCLCPP_INFO(get_logger(), "✓ robot_description copied");
//                   } else if (param.get_name() == "robot_description_semantic" &&
//                              param.get_type() == rclcpp::ParameterType::PARAMETER_STRING) {
//                     this->set_parameter(rclcpp::Parameter("robot_description_semantic", param.as_string()));
//                     RCLCPP_INFO(get_logger(), "✓ robot_description_semantic copied");
//                   } else if (param.get_name() == "robot_description_kinematics" &&
//                              param.get_type() == rclcpp::ParameterType::PARAMETER_STRING) {
//                     this->set_parameter(rclcpp::Parameter("robot_description_kinematics", param.as_string()));
//                     RCLCPP_INFO(get_logger(), "✓ robot_description_kinematics copied");
//                   }
//                 }
//               }
//               move_group_params_ready_ = true;
//               return true;
//             }
//           } catch (const std::exception& e) {
//             RCLCPP_WARN(get_logger(), "Failed to get parameters: %s", e.what());
//           }
//         }
//       }
//       RCLCPP_INFO_THROTTLE(get_logger(), *this->get_clock(), 3000,
//                            "Waiting for move_group node to be available...");
//       rclcpp::sleep_for(std::chrono::milliseconds(300));
//     }
//     RCLCPP_ERROR(get_logger(), "Timeout waiting for move_group parameters");
//     return false;
//   }

//   void get_parameters()
//   {
//     planning_group_ = this->get_parameter("planning_group").as_string();
//     ee_link_ = this->get_parameter("ee_link").as_string();
//     base_frame_ = this->get_parameter("base_frame").as_string();
//     z_hover_ = this->get_parameter("z_hover").as_double();
//     z_grasp_ = this->get_parameter("z_grasp").as_double();
//     targets_.t_hover = this->get_parameter("t_hover").as_double();
//     targets_.t_grasp = this->get_parameter("t_grasp").as_double();
//     timing_method_str_ = this->get_parameter("timing_method").as_string();
//     vel_scale_ = this->get_parameter("vel_scale").as_double();
//     acc_scale_ = this->get_parameter("acc_scale").as_double();

//     if (timing_method_str_ == "totg") {
//       method_ = TimingMethod::TOTG;
//     } else if (timing_method_str_ == "ruckig") {
//       method_ = TimingMethod::RUCKIG;
//     } else {
//       method_ = TimingMethod::TOTG_THEN_RUCKIG;
//       timing_method_str_ = "totg_then_ruckig";
//     }

//     use_pose_target_ = this->get_parameter("use_pose_target").as_bool();
//     target_mode_ = this->get_parameter("target_mode").as_string();
//     target_frame_ = this->get_parameter("target_frame").as_string();
//     if (target_frame_.empty()) target_frame_ = base_frame_;
//     target_x_ = this->get_parameter("target_x").as_double();
//     target_y_ = this->get_parameter("target_y").as_double();
//     target_z_ = this->get_parameter("target_z").as_double();
//     keep_orientation_ = this->get_parameter("keep_orientation").as_bool();
//     target_roll_ = this->get_parameter("target_roll").as_double();
//     target_pitch_ = this->get_parameter("target_pitch").as_double();
//     target_yaw_ = this->get_parameter("target_yaw").as_double();

//     enforce_horizontal_ = this->get_parameter("enforce_horizontal_orientation").as_bool();
//     horizontal_yaw_mode_ = this->get_parameter("horizontal_yaw_mode").as_string();
//     fixed_yaw_ = this->get_parameter("fixed_yaw").as_double();
//     horizontal_roll_offset_ = this->get_parameter("horizontal_roll_offset").as_double();
//     rp_tol_ = this->get_parameter("constraint_rp_tolerance").as_double();
//     yaw_tol_ = this->get_parameter("constraint_yaw_tolerance").as_double();
//     use_path_constraints_for_hover_ = this->get_parameter("use_path_constraints_for_hover").as_bool();
//     constraint_mode_ = this->get_parameter("constraint_mode").as_string();
//     if (constraint_mode_ != "goal_only" && constraint_mode_ != "path_only" && constraint_mode_ != "both") {
//       RCLCPP_WARN(get_logger(), "Invalid constraint_mode '%s', defaulting to 'path_only'", constraint_mode_.c_str());
//       constraint_mode_ = "path_only";
//     }

//     // Predictor params
//     use_predictor_ = this->get_parameter("use_predictor").as_bool();
//     require_predictor_ready_ = this->get_parameter("require_predictor_ready").as_bool();
//     predictor_ready_topic_ = this->get_parameter("predictor_ready_topic").as_string();
//     predictor_service_ = this->get_parameter("predictor_service").as_string();
//     commit_pick_time_s_ = this->get_parameter("commit_pick_time_s").as_double();

//     replan_cooldown_s_ = this->get_parameter("replan_cooldown_s").as_double();

//     // Planner params
//     planning_time_ = this->get_parameter("planning_time").as_double();
//     planner_id_ = this->get_parameter("planner_id").as_string();

//     // IK gate params
//     require_ik_gate_ = this->get_parameter("require_ik_gate").as_bool();
//     ik_gate_topic_ = this->get_parameter("ik_gate_topic").as_string();

//     // Gripper params
//     gripper_controller_topic_ = this->get_parameter("gripper_controller_topic").as_string();
//     gripper_joints_ = this->get_parameter("gripper_joints").as_string_array();
//     gripper_open_pos_ = this->get_parameter("gripper_open").as_double_array();
//     gripper_close_pos_ = this->get_parameter("gripper_close").as_double_array();
//     gripper_motion_time_ = this->get_parameter("gripper_motion_time").as_double();
//     gripper_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(gripper_controller_topic_, 10);

//     // Rotating table collision params
//     add_table_collision_ = this->get_parameter("add_table_collision").as_bool();
//     table_frame_ = this->get_parameter("table_frame").as_string();
//     if (table_frame_.empty()) table_frame_ = base_frame_;
//     table_radius_ = this->get_parameter("table_radius").as_double();
//     table_height_ = this->get_parameter("table_height").as_double();
//     table_x_ = this->get_parameter("table_x").as_double();
//     table_y_ = this->get_parameter("table_y").as_double();
//     table_z_ = this->get_parameter("table_z").as_double();
//   }

//   // Gripper
//   bool command_gripper_(const std::vector<double>& pos, double seconds, double start_delay_s = 0.0) {
//     if (!gripper_pub_) return false;
//     if (pos.size() != gripper_joints_.size()) {
//       RCLCPP_ERROR(get_logger(), "Gripper command size mismatch");
//       return false;
//     }
//     trajectory_msgs::msg::JointTrajectory traj;
//     traj.header.stamp = this->get_clock()->now() + rclcpp::Duration::from_seconds(std::max(0.0, start_delay_s));
//     traj.joint_names = gripper_joints_;
//     trajectory_msgs::msg::JointTrajectoryPoint pt;
//     pt.positions = pos;
//     pt.time_from_start = rclcpp::Duration::from_seconds(seconds);
//     traj.points.push_back(pt);
//     gripper_pub_->publish(traj);
//     return true;
//   }
//   bool open_gripper_(double start_delay_s = 0.0) { return command_gripper_(gripper_open_pos_, gripper_motion_time_, start_delay_s); }
//   bool close_gripper_(double start_delay_s = 0.0) { return command_gripper_(gripper_close_pos_, gripper_motion_time_, start_delay_s); }

//   void schedule_gripper_close_for_plan_(const moveit::planning_interface::MoveGroupInterface::Plan& plan) {
//     if (plan.trajectory_.joint_trajectory.points.empty()) return;
//     const auto& last_tfs = plan.trajectory_.joint_trajectory.points.back().time_from_start;
//     const double traj_total = rclcpp::Duration(last_tfs).seconds();
//     double desired_start = targets_.t_grasp - gripper_motion_time_;
//     double latest_start = std::max(0.0, traj_total - gripper_motion_time_);
//     double start_delay = std::clamp(desired_start, 0.0, latest_start);
//     (void)close_gripper_(start_delay);
//   }

//   geometry_msgs::msg::Quaternion quatFromRPY(double r, double p, double y) {
//     Eigen::AngleAxisd Rx(r, Eigen::Vector3d::UnitX());
//     Eigen::AngleAxisd Ry(p, Eigen::Vector3d::UnitY());
//     Eigen::AngleAxisd Rz(y, Eigen::Vector3d::UnitZ());
//     Eigen::Quaterniond q = Rz * Ry * Rx;
//     geometry_msgs::msg::Quaternion qmsg;
//     qmsg.x = q.x(); qmsg.y = q.y(); qmsg.z = q.z(); qmsg.w = q.w();
//     return qmsg;
//   }
//   static double yawFromQuat(const geometry_msgs::msg::Quaternion& q) {
//     Eigen::Quaterniond qe(q.w, q.x, q.y, q.z);
//     Eigen::Vector3d eul = qe.toRotationMatrix().eulerAngles(2, 1, 0);
//     return eul[0];
//   }

//   geometry_msgs::msg::PoseStamped build_target_pose_(const geometry_msgs::msg::PoseStamped& current) {
//     geometry_msgs::msg::PoseStamped tgt = current;
//     tgt.header.frame_id = current.header.frame_id.empty() ? base_frame_ : current.header.frame_id;
//     if (target_mode_ == "relative") {
//       tgt.pose.position.x = current.pose.position.x + target_x_;
//       tgt.pose.position.y = current.pose.position.y + target_y_;
//       tgt.pose.position.z = current.pose.position.z + target_z_;
//     } else {
//       tgt.pose.position.x = target_x_;
//       tgt.pose.position.y = target_y_;
//       tgt.pose.position.z = target_z_;
//     }
//     if (!keep_orientation_) {
//       tgt.pose.orientation = quatFromRPY(target_roll_, target_pitch_, target_yaw_);
//     }
//     return tgt;
//   }

//   bool check_joint_states_topic()
//   {
//     auto topic_names = this->get_topic_names_and_types();
//     for (const auto& topic : topic_names) {
//       if (topic.first == "/joint_states") {
//         RCLCPP_INFO(get_logger(), "Found /joint_states topic");
//         return true;
//       }
//     }
//     RCLCPP_ERROR(get_logger(), "Joint states topic /joint_states not found!");
//     return false;
//   }

//   void debug_robot_model() {
//     if (!move_group_) return;
//     auto robot_model = move_group_->getRobotModel();
//     RCLCPP_INFO(get_logger(), "Robot model: %s", robot_model->getName().c_str());
//     const auto& link_names = robot_model->getLinkModelNames();
//     RCLCPP_INFO(get_logger(), "Available links (%zu):", link_names.size());
//     for (const auto& link : link_names) RCLCPP_INFO(get_logger(), "  - %s", link.c_str());
//     if (robot_model->hasLinkModel(ee_link_)) {
//       RCLCPP_INFO(get_logger(), "✓ End effector link '%s' found", ee_link_.c_str());
//     } else {
//       RCLCPP_ERROR(get_logger(), "✗ End effector link '%s' NOT found", ee_link_.c_str());
//     }
//     auto current_state = move_group_->getCurrentState(0.0);
//     if (!current_state) return;
//     const auto* group = current_state->getJointModelGroup(planning_group_);
//     if (group) {
//       const auto& ee_links = group->getLinkModelNames();
//       RCLCPP_INFO(get_logger(), "Links in planning group '%s':", planning_group_.c_str());
//       for (const auto& link : ee_links) RCLCPP_INFO(get_logger(), "  - %s", link.c_str());
//     }
//   }

//   bool wait_for_valid_joint_states_with_time(double timeout_seconds = 20.0) {
//     RCLCPP_INFO(get_logger(), "Waiting for joint states with valid timestamps...");
//     bool received_joint_states = false;

//     auto joint_state_sub = this->create_subscription<sensor_msgs::msg::JointState>(
//         "joint_states", 10,
//         [&received_joint_states, this](const sensor_msgs::msg::JointState::SharedPtr msg) {
//           RCLCPP_INFO_ONCE(get_logger(), "Received joint state with timestamp %.3f and %zu joints",
//                            rclcpp::Time(msg->header.stamp).seconds(), msg->name.size());
//           received_joint_states = true;
//         });

//     auto start_time = this->get_clock()->now();
//     while (rclcpp::ok()) {
//       rclcpp::spin_some(this->get_node_base_interface());
//       if (received_joint_states) {
//         if (move_group_) {
//           try {
//             auto state = move_group_->getCurrentState(0.1);
//             if (state) {
//               RCLCPP_INFO(get_logger(), "Successfully received robot state from MoveGroup!");
//               return true;
//             }
//           } catch (const std::exception& e) {
//             RCLCPP_WARN_THROTTLE(get_logger(), *this->get_clock(), 3000,
//                                  "Exception getting current state: %s", e.what());
//           }
//         }
//       }
//       if ((this->get_clock()->now() - start_time).seconds() > timeout_seconds) {
//         RCLCPP_ERROR(get_logger(), "Timeout waiting for valid joint states");
//         return false;
//       }
//       rclcpp::sleep_for(std::chrono::milliseconds(100));
//     }
//     return false;
//   }

//   geometry_msgs::msg::PoseStamped get_current_pose_robust() {
//     if (!move_group_) return geometry_msgs::msg::PoseStamped();
//     for (int attempt = 0; attempt < 3; ++attempt) {
//       try {
//         rclcpp::sleep_for(std::chrono::milliseconds(80));
//         auto pose = move_group_->getCurrentPose(ee_link_);
//         if (std::abs(pose.pose.position.x) > 0.001 ||
//             std::abs(pose.pose.position.y) > 0.001 ||
//             std::abs(pose.pose.position.z) > 0.001) {
//           return pose;
//         }
//       } catch (...) {}
//     }
//     return geometry_msgs::msg::PoseStamped();
//   }

//   size_t findNearestIndexToPose(const robot_trajectory::RobotTrajectory& rt,
//                                 const std::string& link,
//                                 const geometry_msgs::msg::Pose& target) {
//     size_t best_idx = 0;
//     double best_dist = std::numeric_limits<double>::infinity();
//     for (size_t i = 0; i < rt.getWayPointCount(); ++i) {
//       const auto& st = rt.getWayPoint(i);
//       const auto& T = st.getGlobalLinkTransform(link);
//       Eigen::Vector3d p = T.translation();
//       const double dx = p.x() - target.position.x;
//       const double dy = p.y() - target.position.y;
//       const double dz = p.z() - target.position.z;
//       const double d = std::sqrt(dx*dx + dy*dy + dz*dz);
//       if (d < best_dist) { best_dist = d; best_idx = i; }
//     }
//     return best_idx;
//   }

//   // Predictor async flow (single request per attempt)
//   void request_pick_pose_async_(double t_rel_s)
//   {
//     if (!pick_client_) { RCLCPP_ERROR(get_logger(), "Predictor client not created"); end_planning_session_(true); return; }
//     if (!predictor_ready_.load(std::memory_order_relaxed) && require_predictor_ready_) {
//       RCLCPP_WARN(get_logger(), "Predictor not ready yet, skipping request");
//       end_planning_session_(true);
//       return;
//     }
//     if (t_rel_s < 0.0) t_rel_s = 0.0;

//     auto req = std::make_shared<lite6_pick_predictor_interfaces::srv::GetPoseAt::Request>();
//     req->use_relative = true;
//     const double si = std::floor(t_rel_s);
//     req->query_time.sec = static_cast<int32_t>(si);
//     req->query_time.nanosec = static_cast<uint32_t>((t_rel_s - si) * 1e9);

//     auto future = pick_client_->async_send_request(req,
//       [this](rclcpp::Client<lite6_pick_predictor_interfaces::srv::GetPoseAt>::SharedFuture fut)
//       {
//         try {
//           auto res = fut.get();
//           if (!res->ok) {
//             RCLCPP_ERROR(get_logger(), "GetPoseAt returned ok=false");
//             end_planning_session_(true);
//             return;
//           }
//           const auto& pick_pose = res->pose;
//           RCLCPP_INFO(get_logger(), "Received pick pose [%s] (%.3f, %.3f, %.3f)",
//                       pick_pose.header.frame_id.c_str(),
//                       pick_pose.pose.position.x, pick_pose.pose.position.y, pick_pose.pose.position.z);

//           // Plan hover + descent
//           if (!plan_and_execute_from_pick_(pick_pose)) {
//             RCLCPP_ERROR(get_logger(), "Planning/execution failed");
//             end_planning_session_(true);
//             return;
//           }
//           end_planning_session_(false);
//         }
//         catch (const std::exception& e) {
//           RCLCPP_ERROR(get_logger(), "Predictor future exception: %s", e.what());
//           end_planning_session_(true);
//         }
//       });

//     (void)future;
//   }

//   void end_planning_session_(bool cooldown)
//   {
//     planning_in_progress_ = false;
//     if (cooldown) last_attempt_wall_ = std::chrono::steady_clock::now();
//   }

//   bool transform_to_frame_(const geometry_msgs::msg::PoseStamped& in,
//                            const std::string& to_frame,
//                            geometry_msgs::msg::PoseStamped& out)
//   {
//     if (in.header.frame_id.empty() || in.header.frame_id == to_frame) {
//       out = in;
//       out.header.frame_id = to_frame;
//       return true;
//     }
//     try {
//       geometry_msgs::msg::TransformStamped tf =
//           tf_buffer_->lookupTransform(to_frame, in.header.frame_id, tf2::TimePointZero, tf2::durationFromSec(0.3));
//       tf2::doTransform(in, out, tf);
//       out.header.frame_id = to_frame;
//       return true;
//     } catch (const std::exception& e) {
//       RCLCPP_ERROR(get_logger(), "TF transform %s -> %s failed: %s",
//                    in.header.frame_id.c_str(), to_frame.c_str(), e.what());
//       return false;
//     }
//   }

//   bool plan_and_execute_from_pick_(const geometry_msgs::msg::PoseStamped& pick_pose_in)
//   {
//     if (!move_group_) return false;

//     const std::string planning_frame = move_group_->getPlanningFrame();
//     geometry_msgs::msg::PoseStamped pick_pose_tf;
//     if (!transform_to_frame_(pick_pose_in, planning_frame, pick_pose_tf)) return false;

//     geometry_msgs::msg::PoseStamped pose_hover = pick_pose_tf;
//     geometry_msgs::msg::PoseStamped pose_grasp = pick_pose_tf;

//     double yaw = 0.0;
//     if (horizontal_yaw_mode_ == "current") {
//       auto cur = get_current_pose_robust();
//       yaw = cur.header.frame_id.empty() ? yawFromQuat(pick_pose_tf.pose.orientation)
//                                         : yawFromQuat(cur.pose.orientation);
//     } else if (horizontal_yaw_mode_ == "fixed") {
//       yaw = fixed_yaw_;
//     } else {
//       yaw = yawFromQuat(pick_pose_tf.pose.orientation);
//     }
//     const auto horiz = quatFromRPY(horizontal_roll_offset_, 0.0, yaw);

//     // Targets
//     pose_hover.pose.position.z = z_hover_;
//     pose_grasp.pose.position.z = z_grasp_;

//     // Apply horizontal orientation for descent
//     if (enforce_horizontal_) {
//       pose_grasp.pose.orientation = horiz;
//     }

//     // Stage 1: Lift -> Rotate (at safe Z) -> Slide XY -> Down-to-hover
//     move_group_->setStartStateToCurrentState();
//     const auto pose_current = get_current_pose_robust();
//     if (pose_current.header.frame_id.empty()) {
//       RCLCPP_ERROR(get_logger(), "No current pose for Cartesian current->hover");
//       return false;
//     }

//     // Compute safe clearance above table top
//     const double table_top_z = table_z_ + 0.5 * table_height_;
//     const double lift_margin = 0.15; // tune if needed
//     double safe_z = std::max({pose_current.pose.position.z + 0.05, z_hover_, table_top_z + lift_margin});

//     // Waypoints
//     geometry_msgs::msg::Pose p_up = pose_current.pose;     // straight up, keep current orientation
//     p_up.position.z = safe_z;

//     geometry_msgs::msg::Pose p_rotate = p_up;              // rotate in place at safe Z
//     if (enforce_horizontal_) p_rotate.orientation = horiz;

//     geometry_msgs::msg::Pose p_xy = p_rotate;              // translate XY at safe Z
//     p_xy.position.x = pose_hover.pose.position.x;
//     p_xy.position.y = pose_hover.pose.position.y;

//     geometry_msgs::msg::Pose p_down = p_xy;                // descend to hover Z
//     p_down.position.z = pose_hover.pose.position.z;

//     std::vector<geometry_msgs::msg::Pose> leg1_waypoints;
//     leg1_waypoints.push_back(p_up);
//     // Only add p_rotate if we actually change orientation
//     if (enforce_horizontal_) leg1_waypoints.push_back(p_rotate);
//     leg1_waypoints.push_back(p_xy);
//     leg1_waypoints.push_back(p_down);

//     moveit_msgs::msg::RobotTrajectory leg1_msg;
//     const double eef_step1 = 0.01;
//     const double jump_threshold1 = 0.0;
//     double fraction1 = 0.0;
//     try {
//       fraction1 = move_group_->computeCartesianPath(leg1_waypoints, eef_step1, jump_threshold1, leg1_msg, true);
//     } catch (const std::exception& e) {
//       RCLCPP_ERROR(get_logger(), "Cartesian leg1 exception: %s", e.what());
//       return false;
//     }
//     if (fraction1 < 0.8) {
//       moveit_msgs::msg::RobotTrajectory dbg_msg;
//       double fraction_nc = 0.0;
//       try { fraction_nc = move_group_->computeCartesianPath(leg1_waypoints, eef_step1, jump_threshold1, dbg_msg, false); } catch (...) {}
//       RCLCPP_WARN(get_logger(), "Leg-1 low fraction=%.2f (collisions). No-collision fraction=%.2f", fraction1, fraction_nc);
//       if (fraction1 < 0.5) {
//         RCLCPP_ERROR(get_logger(), "Cartesian leg1 failed (fraction=%.2f)", fraction1);
//         return false;
//       }
//     }

//     // Time-parameterize leg 1
//     moveit::core::RobotModelConstPtr model1 = move_group_->getRobotModel();
//     auto start_state1 = move_group_->getCurrentState(2.0);
//     if (!start_state1) {
//       RCLCPP_ERROR(get_logger(), "No current robot state before leg1");
//       return false;
//     }
//     robot_trajectory::RobotTrajectory rt1(model1, planning_group_);
//     rt1.setRobotTrajectoryMsg(*start_state1, leg1_msg);
//     if (!tp_control_cpp::applyTimeParameterization(rt1, method_, vel_scale_, acc_scale_)) {
//       RCLCPP_ERROR(get_logger(), "Time parameterization failed for leg1");
//       return false;
//     }
//     if (method_ == TimingMethod::TOTG_THEN_RUCKIG) {
//       (void)tp_control_cpp::applyTimeParameterization(rt1, TimingMethod::RUCKIG, vel_scale_, acc_scale_);
//     }
//     rt1.getRobotTrajectoryMsg(leg1_msg);

//     moveit::planning_interface::MoveGroupInterface::Plan plan_to_hover;
//     plan_to_hover.trajectory_ = leg1_msg;
//     moveit::core::robotStateToRobotStateMsg(*start_state1, plan_to_hover.start_state_);
//     plan_to_hover.planning_time_ = 0.0;

//     if (display_pub_) {
//       moveit_msgs::msg::DisplayTrajectory msg;
//       msg.model_id = move_group_->getRobotModel()->getName();
//       if (auto cs0 = move_group_->getCurrentState(0.0)) {
//         moveit_msgs::msg::RobotState rs;
//         moveit::core::robotStateToRobotStateMsg(*cs0, rs);
//         msg.trajectory_start = rs;
//       }
//       msg.trajectory.push_back(plan_to_hover.trajectory_);
//       display_pub_->publish(msg);
//     }
//     auto exec_code = move_group_->execute(plan_to_hover);
//     if (exec_code != moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
//       RCLCPP_ERROR(get_logger(), "Execution to hover failed (code=%d)", exec_code.val);
//       return false;
//     }

//     // Stage 2: hover -> grasp (keep horizontal)
//     move_group_->setStartStateToCurrentState();
//     geometry_msgs::msg::PoseStamped hover_for_descent = pose_hover;
//     if (enforce_horizontal_) hover_for_descent.pose.orientation = horiz;

//     std::vector<geometry_msgs::msg::Pose> waypoints{hover_for_descent.pose, pose_grasp.pose};
//     moveit_msgs::msg::RobotTrajectory cart_traj_msg;
//     const double eef_step = 0.005;
//     const double jump_threshold = 0.0;
//     double fraction = 0.0;
//     try {
//       fraction = move_group_->computeCartesianPath(waypoints, eef_step, jump_threshold, cart_traj_msg, true);
//     } catch (const std::exception& e) {
//       RCLCPP_ERROR(get_logger(), "Cartesian descent exception: %s", e.what());
//       return false;
//     }
//     if (fraction < 0.1) {
//       RCLCPP_ERROR(get_logger(), "Cartesian hover->grasp failed (fraction=%.2f)", fraction);
//       return false;
//     }
//     if (fraction < 0.99) {
//       RCLCPP_WARN(get_logger(), "Cartesian hover->grasp fraction=%.2f; proceeding", fraction);
//     }

//     moveit::core::RobotModelConstPtr model = move_group_->getRobotModel();
//     auto start_state = move_group_->getCurrentState(2.0);
//     if (!start_state) {
//       RCLCPP_ERROR(get_logger(), "No current robot state before descent");
//       return false;
//     }
//     robot_trajectory::RobotTrajectory rt(model, planning_group_);
//     rt.setRobotTrajectoryMsg(*start_state, cart_traj_msg);
//     if (!tp_control_cpp::applyTimeParameterization(rt, method_, vel_scale_, acc_scale_)) {
//       RCLCPP_ERROR(get_logger(), "Time parameterization failed for descent");
//       return false;
//     }
//     const size_t i_hover = 0;
//     const size_t i_grasp = rt.getWayPointCount() ? (rt.getWayPointCount() - 1) : 0;
//     (void)tp_control_cpp::enforceSegmentTimes(rt, i_hover, i_grasp, targets_);
//     if (method_ == TimingMethod::TOTG_THEN_RUCKIG) {
//       (void)tp_control_cpp::applyTimeParameterization(rt, TimingMethod::RUCKIG, vel_scale_, acc_scale_);
//     }
//     rt.getRobotTrajectoryMsg(cart_traj_msg);

//     moveit::planning_interface::MoveGroupInterface::Plan plan_descent;
//     plan_descent.trajectory_ = cart_traj_msg;
//     moveit::core::robotStateToRobotStateMsg(*start_state, plan_descent.start_state_);
//     plan_descent.planning_time_ = 0.0;

//     if (display_pub_) {
//       moveit_msgs::msg::DisplayTrajectory msg;
//       msg.model_id = move_group_->getRobotModel()->getName();
//       if (auto cs = move_group_->getCurrentState(0.0)) {
//         moveit_msgs::msg::RobotState rs;
//         moveit::core::robotStateToRobotStateMsg(*cs, rs);
//         msg.trajectory_start = rs;
//       }
//       msg.trajectory.push_back(plan_descent.trajectory_);
//       display_pub_->publish(msg);
//     }
//     schedule_gripper_close_for_plan_(plan_descent);
//     (void)move_group_->execute(plan_descent);
//     return true;
//   }

//   void run_once_with_predictor_()
//   {
//     if (!move_group_) { RCLCPP_ERROR(get_logger(), "MoveGroup not initialized"); end_planning_session_(true); return; }
//     if (!pick_client_ || !pick_client_->wait_for_service(std::chrono::seconds(0))) {
//       RCLCPP_WARN(get_logger(), "Predictor service '%s' not ready", predictor_service_.c_str());
//       end_planning_session_(true);
//       return;
//     }
//     move_group_->setStartStateToCurrentState();
//     request_pick_pose_async_(commit_pick_time_s_);
//   }

//   // Demo without predictor (unused)
//   void run_once()
//   {
//     if (!move_group_) return;
//     auto pose0 = get_current_pose_robust();
//     if (pose0.header.frame_id.empty()) return;
//     geometry_msgs::msg::PoseStamped pose_hover = build_target_pose_(pose0);
//     pose_hover.pose.position.z = z_hover_;
//     geometry_msgs::msg::PoseStamped pose_grasp = pose_hover;
//     pose_grasp.pose.position.z = z_grasp_;
//     std::vector<geometry_msgs::msg::Pose> waypoints{pose_hover.pose, pose_grasp.pose};

//     moveit_msgs::msg::RobotTrajectory cart_traj_msg;
//     (void)move_group_->computeCartesianPath(waypoints, 0.005, 0.0, cart_traj_msg, true);
//     moveit::core::RobotModelConstPtr model = move_group_->getRobotModel();
//     auto start_state = move_group_->getCurrentState(2.0);
//     if (!start_state) return;
//     robot_trajectory::RobotTrajectory rt(model, planning_group_);
//     rt.setRobotTrajectoryMsg(*start_state, cart_traj_msg);
//     (void)tp_control_cpp::applyTimeParameterization(rt, method_, vel_scale_, acc_scale_);
//     rt.getRobotTrajectoryMsg(cart_traj_msg);
//     moveit::planning_interface::MoveGroupInterface::Plan plan;
//     plan.trajectory_ = cart_traj_msg;
//     moveit::core::robotStateToRobotStateMsg(*start_state, plan.start_state_);
//     if (display_pub_) {
//       moveit_msgs::msg::DisplayTrajectory msg;
//       msg.model_id = move_group_->getRobotModel()->getName();
//       msg.trajectory.push_back(plan.trajectory_);
//       display_pub_->publish(msg);
//     }
//     (void)move_group_->execute(plan);
//   }

//   // Add a cylinder as the rotating table collision object (matches tp_control.py)
//   void add_rotating_table_collision_(const geometry_msgs::msg::Pose& table_pose,
//                                      double table_radius,
//                                      double table_height)
//   {
//     moveit_msgs::msg::CollisionObject obj;
//     obj.header.frame_id = table_frame_.empty() ? base_frame_ : table_frame_;
//     obj.header.stamp = this->get_clock()->now();
//     obj.id = "rotating_table";

//     shape_msgs::msg::SolidPrimitive cylinder;
//     cylinder.type = shape_msgs::msg::SolidPrimitive::CYLINDER;
//     cylinder.dimensions.resize(2);
//     cylinder.dimensions[shape_msgs::msg::SolidPrimitive::CYLINDER_HEIGHT] = table_height;
//     cylinder.dimensions[shape_msgs::msg::SolidPrimitive::CYLINDER_RADIUS] = table_radius;

//     obj.primitives.push_back(cylinder);
//     obj.primitive_poses.push_back(table_pose);
//     obj.operation = moveit_msgs::msg::CollisionObject::ADD;

//     // Apply via PlanningSceneInterface
//     planning_scene_interface_.applyCollisionObject(obj);

//     // Optional: publish as diff
//     moveit_msgs::msg::PlanningScene scene_msg;
//     scene_msg.is_diff = true;
//     scene_msg.world.collision_objects.push_back(obj);
//     if (planning_scene_pub_) planning_scene_pub_->publish(scene_msg);
//   }

// private:
//   std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
//   moveit_visual_tools::MoveItVisualToolsPtr visual_tools_;
//   rclcpp::Publisher<moveit_msgs::msg::DisplayTrajectory>::SharedPtr display_pub_;

//   std::string planning_group_, ee_link_, base_frame_;
//   double z_hover_{0.20}, z_grasp_{0.05};
//   double vel_scale_{0.9}, acc_scale_{0.6};
//   TimingMethod method_{TimingMethod::TOTG_THEN_RUCKIG};
//   std::string timing_method_str_;
//   TimingTargets targets_;
//   bool use_pose_target_{false};
//   std::string target_mode_;
//   std::string target_frame_;
//   double target_x_{0.0}, target_y_{0.0}, target_z_{0.0};
//   bool keep_orientation_{true};
//   double target_roll_{0.0}, target_pitch_{0.0}, target_yaw_{0.0};

//   // Horizontal control
//   bool enforce_horizontal_{false};
//   std::string horizontal_yaw_mode_{"predictor"}; // predictor|current|fixed
//   double fixed_yaw_{0.0};
//   double horizontal_roll_offset_{M_PI};
//   double rp_tol_{0.10};
//   double yaw_tol_{M_PI};
//   bool use_path_constraints_for_hover_{false};
//   std::string constraint_mode_{"goal_only"}; // goal_only|path_only|both

//   // Flow control
//   std::atomic_bool planning_in_progress_{false};
//   bool move_group_params_ready_{false};
//   rclcpp::TimerBase::SharedPtr delayed_plan_timer_;
//   double replan_cooldown_s_{2.0};
//   std::chrono::steady_clock::time_point last_attempt_wall_{};
//   bool last_ik_reachable_{false};

//   // TF
//   std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
//   std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

//   // Predictor
//   bool use_predictor_{true};
//   bool require_predictor_ready_{true};
//   double commit_pick_time_s_{1.0};
//   std::string predictor_ready_topic_{"predictor_ready"};
//   std::string predictor_service_{"get_predicted_pose_at"};
//   std::atomic<bool> predictor_ready_{false};
//   rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr predictor_ready_sub_;
//   rclcpp::Client<lite6_pick_predictor_interfaces::srv::GetPoseAt>::SharedPtr pick_client_;

//   // Planning params
//   double planning_time_{0.5};
//   std::string planner_id_{"RRTConnect"};

//   // IK gate
//   bool require_ik_gate_{true};
//   std::string ik_gate_topic_{"/ik_gate/output"};
//   rclcpp::Subscription<std_msgs::msg::String>::SharedPtr ik_gate_sub_;

//   // Gripper
//   rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr gripper_pub_;
//   std::string gripper_controller_topic_;
//   std::vector<std::string> gripper_joints_;
//   std::vector<double> gripper_open_pos_;
//   std::vector<double> gripper_close_pos_;
//   double gripper_motion_time_{0.7};

//   // Planning scene interface and optional publisher
//   moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;
//   rclcpp::Publisher<moveit_msgs::msg::PlanningScene>::SharedPtr planning_scene_pub_;

//   // Rotating table config
//   bool add_table_collision_{true};
//   std::string table_frame_{"link_base"};
//   double table_radius_{0.75};
//   double table_height_{0.1};
//   double table_x_{0.9};
//   double table_y_{0.0};
//   double table_z_{0.1};
// };

// int main(int argc, char** argv)
// {
//   rclcpp::init(argc, argv);
//   auto node = std::make_shared<TpControlNode>();
//   node->initialize();
//   rclcpp::spin(node);
//   rclcpp::shutdown();
//   return 0;
// }
//==================================================================================================
// The code below is succesfull ~20 %: a full process is achieved, but the execution delay is large.
//==================================================================================================
// #include <rclcpp/rclcpp.hpp>
// #include <rclcpp/parameter_client.hpp>

// #include <geometry_msgs/msg/pose_stamped.hpp>
// #include <sensor_msgs/msg/joint_state.hpp>
// #include <eigen3/Eigen/Geometry>

// #include <moveit/move_group_interface/move_group_interface.h>
// #include <moveit/planning_scene_monitor/planning_scene_monitor.h>
// #include <moveit/planning_scene_interface/planning_scene_interface.h>
// #include <moveit/robot_state/conversions.h>
// #include <moveit/robot_trajectory/robot_trajectory.h>
// #include <moveit_msgs/msg/move_it_error_codes.h>
// #include <moveit_visual_tools/moveit_visual_tools.h>
// #include <moveit_msgs/msg/display_trajectory.hpp>
// #include <moveit_msgs/msg/constraints.hpp>
// #include <moveit_msgs/msg/orientation_constraint.hpp>
// #include <moveit_msgs/msg/collision_object.hpp>
// #include <moveit_msgs/msg/planning_scene.hpp>
// #include <shape_msgs/msg/solid_primitive.hpp>

// #include <moveit/trajectory_processing/time_optimal_trajectory_generation.h>
// #include <moveit/trajectory_processing/ruckig_traj_smoothing.h>

// #include <trajectory_msgs/msg/joint_trajectory.hpp>
// #include <trajectory_msgs/msg/joint_trajectory_point.hpp>

// #include <tf2_eigen/tf2_eigen.hpp>
// #include <tf2_ros/transform_listener.h>
// #include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
// #include <tf2/time.h>

// #include <std_msgs/msg/bool.hpp>
// #include <std_msgs/msg/string.hpp>
// #include <lite6_pick_predictor_interfaces/srv/get_pose_at.hpp>

// #include <algorithm>
// #include <limits>
// #include <cmath>
// #include <atomic>
// #include <string>
// #include <vector>
// #include <chrono>

// namespace tp_control_cpp {

// enum class TimingMethod { TOTG, RUCKIG, TOTG_THEN_RUCKIG };

// struct TimingTargets
// {
//   double t_hover{1.5};
//   double t_grasp{3.0};
// };

// inline double durationFromStart(const robot_trajectory::RobotTrajectory& rt, size_t idx)
// {
//   if (rt.getWayPointCount() == 0 || idx == 0) return 0.0;
//   idx = std::min(idx, rt.getWayPointCount() - 1);
//   double sum = 0.0;
//   for (size_t i = 1; i <= idx; ++i)
//     sum += rt.getWayPointDurationFromPrevious(i);
//   return sum;
// }

// inline bool applyTimeParameterization(robot_trajectory::RobotTrajectory& rt,
//                                       TimingMethod method,
//                                       double max_vel_scaling = 0.9,
//                                       double max_acc_scaling = 0.7)
// {
//   using trajectory_processing::TimeOptimalTrajectoryGeneration;
//   using trajectory_processing::RuckigSmoothing;

//   switch (method)
//   {
//     case TimingMethod::TOTG:
//     {
//       TimeOptimalTrajectoryGeneration totg;
//       return totg.computeTimeStamps(rt, max_vel_scaling, max_acc_scaling);
//     }
//     case TimingMethod::RUCKIG:
//     {
//       return RuckigSmoothing::applySmoothing(rt, max_vel_scaling, max_acc_scaling);
//     }
//     case TimingMethod::TOTG_THEN_RUCKIG:
//     {
//       TimeOptimalTrajectoryGeneration totg;
//       if (!totg.computeTimeStamps(rt, max_vel_scaling, max_acc_scaling))
//         return false;
//       return RuckigSmoothing::applySmoothing(rt, max_vel_scaling, max_acc_scaling);
//     }
//     default:
//       return false;
//   }
// }

// inline bool enforceSegmentTimes(robot_trajectory::RobotTrajectory& rt,
//                                 size_t idx_hover,
//                                 size_t idx_grasp,
//                                 const TimingTargets& targets)
// {
//   const size_t N = rt.getWayPointCount();
//   if (N < 2) return false;
//   idx_hover = std::min(idx_hover, N - 1);
//   idx_grasp = std::min(idx_grasp, N - 1);
//   if (idx_hover >= idx_grasp) idx_hover = std::max<size_t>(1, std::min(idx_grasp, idx_hover));

//   auto safe_get = [&](size_t i) { return rt.getWayPointDurationFromPrevious(i); };
//   auto safe_set = [&](size_t i, double v) { rt.setWayPointDurationFromPrevious(i, std::max(1e-9, v)); };

//   bool changed = false;

//   double t_hover_cur = durationFromStart(rt, idx_hover);
//   if (targets.t_hover > 0.0 && t_hover_cur < targets.t_hover && idx_hover >= 1)
//   {
//     const double scale1 = targets.t_hover / std::max(1e-9, t_hover_cur);
//     for (size_t i = 1; i <= idx_hover; ++i)
//       safe_set(i, safe_get(i) * scale1);
//     changed = true;
//   }

//   double t_hover_new = durationFromStart(rt, idx_hover);
//   const double t_total_cur = durationFromStart(rt, N - 1);
//   const double tail_cur = std::max(0.0, t_total_cur - t_hover_new);

//   if (targets.t_grasp > 0.0 && (t_hover_new + tail_cur) < targets.t_grasp && idx_grasp >= (idx_hover + 1))
//   {
//     const double needed_tail = targets.t_grasp - t_hover_new;
//     if (tail_cur <= 1e-9)
//     {
//       double last = safe_get(N - 1);
//       if (last < 1e-6) last = 1e-3;
//       safe_set(N - 1, last + std::max(0.0, needed_tail));
//     }
//     else
//     {
//       const double scale2 = std::max(1.0, needed_tail / tail_cur);
//       for (size_t i = idx_hover + 1; i <= idx_grasp; ++i)
//         safe_set(i, safe_get(i) * scale2);
//     }
//     changed = true;
//   }

//   return changed;
// }

// } // namespace tp_control_cpp

// using tp_control_cpp::TimingMethod;
// using tp_control_cpp::TimingTargets;

// class TpControlNode : public rclcpp::Node
// {
// public:
//   TpControlNode() : Node("tp_control_node")
//   {
//     this->declare_parameter<std::string>("robot_description", "");
//     this->declare_parameter<std::string>("robot_description_semantic", "");
//     this->declare_parameter<std::string>("planning_group", "lite6");
//     this->declare_parameter<std::string>("ee_link", "link_tcp");
//     this->declare_parameter<std::string>("base_frame", "link_base");
//     this->declare_parameter<std::string>("timing_method", "totg_then_ruckig");
//     this->declare_parameter<double>("planning_time", 10.0);
//     this->declare_parameter<std::string>("planner_id", "RRTConnect");
//     this->declare_parameter<bool>("use_pose_target", false);
//     this->declare_parameter<std::string>("target_mode", "absolute");
//     this->declare_parameter<std::string>("target_frame", "");

//     this->declare_parameter<double>("vel_scale", 0.9);
//     this->declare_parameter<double>("acc_scale", 0.6);

//     this->declare_parameter<double>("target_x", 0.29);
//     this->declare_parameter<double>("target_y", -0.16);
//     this->declare_parameter<double>("target_z", 0.14);
//     this->declare_parameter<double>("z_hover", 0.29);
//     this->declare_parameter<double>("z_grasp", 0.14);
//     this->declare_parameter<double>("target_roll", 0.0);
//     this->declare_parameter<double>("target_pitch", 0.0);
//     this->declare_parameter<double>("target_yaw", 0.0);
//     this->declare_parameter<bool>("keep_orientation", true);

//     // Horizontal orientation helpers
//     this->declare_parameter<bool>("enforce_horizontal_orientation", true);
//     this->declare_parameter<std::string>("horizontal_yaw_mode", "predictor"); // predictor|current|fixed
//     this->declare_parameter<double>("fixed_yaw", 0.0);
//     this->declare_parameter<double>("horizontal_roll_offset", M_PI);          // many TCPs need a pi flip
//     this->declare_parameter<double>("constraint_rp_tolerance", 0.5);          // roll/pitch tolerance (rad)
//     this->declare_parameter<double>("constraint_yaw_tolerance", M_PI);        // yaw free
//     this->declare_parameter<bool>("use_path_constraints_for_hover", true);
//     // Constraint selection: goal_only | path_only | both
//     this->declare_parameter<std::string>("constraint_mode", "goal_only");

//     // Rotating table collision object params (like tp_control.py)
//     this->declare_parameter<bool>("add_table_collision", true);
//     this->declare_parameter<std::string>("table_frame", "link_base");
//     this->declare_parameter<double>("table_radius", 0.75);
//     this->declare_parameter<double>("table_height", 0.06);
//     this->declare_parameter<double>("table_x", 0.9);
//     this->declare_parameter<double>("table_y", 0.0);
//     this->declare_parameter<double>("table_z", 0.1);

//     this->declare_parameter<double>("t_hover", 1.5);
//     this->declare_parameter<double>("t_grasp", 3.0);

//     // IK gate
//     this->declare_parameter<bool>("require_ik_gate", true);
//     this->declare_parameter<std::string>("ik_gate_topic", "/ik_gate/output");

//     // Predictor
//     this->declare_parameter<bool>("use_predictor", true);
//     this->declare_parameter<bool>("require_predictor_ready", true);
//     this->declare_parameter<std::string>("predictor_ready_topic", "predictor_ready");
//     this->declare_parameter<std::string>("predictor_service", "get_predicted_pose_at");
//     this->declare_parameter<double>("commit_pick_time_s", 3.0);

//     // Cooldown to prevent spam
//     this->declare_parameter<double>("replan_cooldown_s", 2.0);

//     // Gripper
//     this->declare_parameter<std::string>("gripper_controller_topic", "/lite6_gripper_traj_controller/joint_trajectory");
//     this->declare_parameter<std::vector<std::string>>("gripper_joints", {"jaw_left","jaw_right"});
//     this->declare_parameter<std::vector<double>>("gripper_open", {0.02, 0.02});
//     this->declare_parameter<std::vector<double>>("gripper_close", {0.0, 0.0});
//     this->declare_parameter<double>("gripper_motion_time", 1.0);

//     // Detect /clock
//     auto topics = this->get_topic_names_and_types();
//     bool has_clock = std::any_of(topics.begin(), topics.end(), [](const auto& t){ return t.first == "/clock"; });
//     this->set_parameter(rclcpp::Parameter("use_sim_time", has_clock));
//     RCLCPP_INFO(get_logger(), has_clock ? "Found /clock, use_sim_time=true" : "No /clock, use_sim_time=false");

//     tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
//     tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

//     display_pub_ = this->create_publisher<moveit_msgs::msg::DisplayTrajectory>("display_planned_path", rclcpp::QoS(5));
//     planning_scene_pub_ = this->create_publisher<moveit_msgs::msg::PlanningScene>("/planning_scene", 10);

//     get_parameters();

//     RCLCPP_INFO(get_logger(), "TpControlNode initializing...");
//   }

//   void initialize()
//   {
//     try
//     {
//       if (!check_joint_states_topic()) {
//         RCLCPP_ERROR(get_logger(), "Joint states topic not available, cannot proceed");
//         return;
//       }
//       if (!wait_for_move_group_parameters()) {
//         RCLCPP_ERROR(get_logger(), "Could not get robot description from move_group");
//         return;
//       }
//       move_group_params_ready_ = true;

//       move_group_ = std::make_unique<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), planning_group_);
//       move_group_->setEndEffectorLink(ee_link_);
//       move_group_->startStateMonitor(5.0);

//       visual_tools_ = std::make_shared<moveit_visual_tools::MoveItVisualTools>(
//           shared_from_this(), base_frame_, "/rviz_visual_tools", move_group_->getRobotModel());
//       visual_tools_->deleteAllMarkers();
//       visual_tools_->loadRemoteControl();

//       RCLCPP_INFO(get_logger(), "TpControlNode ready. Method=%s", timing_method_str_.c_str());

//       if (!wait_for_valid_joint_states_with_time()) {
//         RCLCPP_ERROR(get_logger(), "Failed to receive valid joint states");
//         return;
//       }
//       auto cs = move_group_->getCurrentState(0.0);
//       if (!cs){
//         RCLCPP_ERROR(get_logger(), "MoveGroup current state unavailable");
//         return;
//       }
//       move_group_->setStartStateToCurrentState();

//       // Predictor wiring
//       if (use_predictor_) {
//         predictor_ready_sub_ = this->create_subscription<std_msgs::msg::Bool>(
//             predictor_ready_topic_, 10,
//             [this](const std_msgs::msg::Bool::SharedPtr msg){
//               predictor_ready_.store(msg->data, std::memory_order_relaxed);
//             });
//         pick_client_ = this->create_client<lite6_pick_predictor_interfaces::srv::GetPoseAt>(predictor_service_);
//       }

//       // Add rotating table collision object (like tp_control.py)
//       if (add_table_collision_) {
//         geometry_msgs::msg::Pose table_pose;
//         table_pose.position.x = table_x_;
//         table_pose.position.y = table_y_;
//         table_pose.position.z = table_z_;
//         table_pose.orientation.w = 1.0; // identity
//         add_rotating_table_collision_(table_pose, table_radius_, table_height_);
//         RCLCPP_INFO(get_logger(), "Applied rotating_table collision object at [%.2f, %.2f, %.2f] (r=%.2f, h=%.2f, frame=%s)",
//                     table_x_, table_y_, table_z_, table_radius_, table_height_, table_frame_.c_str());
//         rclcpp::sleep_for(std::chrono::milliseconds(200)); // allow planning scene sync
//       }

//       // IK gate rising-edge trigger + cooldown
//       if (require_ik_gate_) {
//         ik_gate_sub_ = this->create_subscription<std_msgs::msg::String>(
//             ik_gate_topic_, 10,
//             [this](const std_msgs::msg::String::SharedPtr msg) {
//               const bool reachable = (msg->data == "REACHABLE");
//               if (reachable && !last_ik_reachable_) {
//                 const auto now = std::chrono::steady_clock::now();
//                 if (!planning_in_progress_ && (now - last_attempt_wall_) >= std::chrono::duration<double>(replan_cooldown_s_)) {
//                   planning_in_progress_ = true;
//                   delayed_plan_timer_ = this->create_wall_timer(
//                       std::chrono::milliseconds(5),
//                       [this]() {
//                         if (delayed_plan_timer_) delayed_plan_timer_->cancel();
//                         this->run_once_with_predictor_();
//                       });
//                 } else {
//                   RCLCPP_WARN(get_logger(), "Cooldown active; skipping trigger");
//                 }
//               }
//               last_ik_reachable_ = reachable;
//             });
//       } else {
//         delayed_plan_timer_ = this->create_wall_timer(
//             std::chrono::milliseconds(150),
//             [this]() {
//               if (delayed_plan_timer_) delayed_plan_timer_->cancel();
//               if (!planning_in_progress_) {
//                 planning_in_progress_ = true;
//                 this->run_once_with_predictor_();
//               }
//             });
//       }

//       debug_robot_model();
//     }
//     catch (const std::exception& e)
//     {
//       RCLCPP_ERROR(get_logger(), "Failed to initialize: %s", e.what());
//     }
//   }

// private:

//   bool wait_for_move_group_parameters(double timeout_seconds = 20.0)
//   {
//     if (move_group_params_ready_) return true;
//     RCLCPP_INFO(get_logger(), "Looking for move_group parameters...");
//     auto start_time = this->get_clock()->now();

//     while (rclcpp::ok() && (this->get_clock()->now() - start_time).seconds() < timeout_seconds) {
//       auto node_names = this->get_node_names();
//       for (const auto& node_name : node_names) {
//         if (node_name.find("move_group") != std::string::npos) {
//           RCLCPP_INFO(get_logger(), "Found move_group node: %s", node_name.c_str());
//           try {
//             auto param_client = std::make_shared<rclcpp::AsyncParametersClient>(this, node_name);
//             if (param_client->wait_for_service(std::chrono::seconds(2))) {
//               auto future = param_client->get_parameters(
//                 {"robot_description", "robot_description_semantic", "robot_description_kinematics"});

//               if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), future,
//                   std::chrono::seconds(5)) == rclcpp::FutureReturnCode::SUCCESS) {

//                 auto params = future.get();
//                 for (const auto& param : params) {
//                   if (param.get_name() == "robot_description" &&
//                       param.get_type() == rclcpp::ParameterType::PARAMETER_STRING) {
//                     this->set_parameter(rclcpp::Parameter("robot_description", param.as_string()));
//                     RCLCPP_INFO(get_logger(), "✓ robot_description copied");
//                   } else if (param.get_name() == "robot_description_semantic" &&
//                              param.get_type() == rclcpp::ParameterType::PARAMETER_STRING) {
//                     this->set_parameter(rclcpp::Parameter("robot_description_semantic", param.as_string()));
//                     RCLCPP_INFO(get_logger(), "✓ robot_description_semantic copied");
//                   } else if (param.get_name() == "robot_description_kinematics" &&
//                              param.get_type() == rclcpp::ParameterType::PARAMETER_STRING) {
//                     this->set_parameter(rclcpp::Parameter("robot_description_kinematics", param.as_string()));
//                     RCLCPP_INFO(get_logger(), "✓ robot_description_kinematics copied");
//                   }
//                 }
//               }
//               move_group_params_ready_ = true;
//               return true;
//             }
//           } catch (const std::exception& e) {
//             RCLCPP_WARN(get_logger(), "Failed to get parameters: %s", e.what());
//           }
//         }
//       }
//       RCLCPP_INFO_THROTTLE(get_logger(), *this->get_clock(), 3000,
//                            "Waiting for move_group node to be available...");
//       rclcpp::sleep_for(std::chrono::milliseconds(300));
//     }
//     RCLCPP_ERROR(get_logger(), "Timeout waiting for move_group parameters");
//     return false;
//   }

//   void get_parameters()
//   {
//     planning_group_ = this->get_parameter("planning_group").as_string();
//     ee_link_ = this->get_parameter("ee_link").as_string();
//     base_frame_ = this->get_parameter("base_frame").as_string();
//     z_hover_ = this->get_parameter("z_hover").as_double();
//     z_grasp_ = this->get_parameter("z_grasp").as_double();
//     targets_.t_hover = this->get_parameter("t_hover").as_double();
//     targets_.t_grasp = this->get_parameter("t_grasp").as_double();
//     timing_method_str_ = this->get_parameter("timing_method").as_string();
//     vel_scale_ = this->get_parameter("vel_scale").as_double();
//     acc_scale_ = this->get_parameter("acc_scale").as_double();

//     if (timing_method_str_ == "totg") {
//       method_ = TimingMethod::TOTG;
//     } else if (timing_method_str_ == "ruckig") {
//       method_ = TimingMethod::RUCKIG;
//     } else {
//       method_ = TimingMethod::TOTG_THEN_RUCKIG;
//       timing_method_str_ = "totg_then_ruckig";
//     }

//     use_pose_target_ = this->get_parameter("use_pose_target").as_bool();
//     target_mode_ = this->get_parameter("target_mode").as_string();
//     target_frame_ = this->get_parameter("target_frame").as_string();
//     if (target_frame_.empty()) target_frame_ = base_frame_;
//     target_x_ = this->get_parameter("target_x").as_double();
//     target_y_ = this->get_parameter("target_y").as_double();
//     target_z_ = this->get_parameter("target_z").as_double();
//     keep_orientation_ = this->get_parameter("keep_orientation").as_bool();
//     target_roll_ = this->get_parameter("target_roll").as_double();
//     target_pitch_ = this->get_parameter("target_pitch").as_double();
//     target_yaw_ = this->get_parameter("target_yaw").as_double();

//     enforce_horizontal_ = this->get_parameter("enforce_horizontal_orientation").as_bool();
//     horizontal_yaw_mode_ = this->get_parameter("horizontal_yaw_mode").as_string();
//     fixed_yaw_ = this->get_parameter("fixed_yaw").as_double();
//     horizontal_roll_offset_ = this->get_parameter("horizontal_roll_offset").as_double();
//     rp_tol_ = this->get_parameter("constraint_rp_tolerance").as_double();
//     yaw_tol_ = this->get_parameter("constraint_yaw_tolerance").as_double();
//     use_path_constraints_for_hover_ = this->get_parameter("use_path_constraints_for_hover").as_bool();
//     constraint_mode_ = this->get_parameter("constraint_mode").as_string();
//     if (constraint_mode_ != "goal_only" && constraint_mode_ != "path_only" && constraint_mode_ != "both") {
//       RCLCPP_WARN(get_logger(), "Invalid constraint_mode '%s', defaulting to 'path_only'", constraint_mode_.c_str());
//       constraint_mode_ = "path_only";
//     }

//     // Predictor params
//     use_predictor_ = this->get_parameter("use_predictor").as_bool();
//     require_predictor_ready_ = this->get_parameter("require_predictor_ready").as_bool();
//     predictor_ready_topic_ = this->get_parameter("predictor_ready_topic").as_string();
//     predictor_service_ = this->get_parameter("predictor_service").as_string();
//     commit_pick_time_s_ = this->get_parameter("commit_pick_time_s").as_double();

//     replan_cooldown_s_ = this->get_parameter("replan_cooldown_s").as_double();

//     // Planner params
//     planning_time_ = this->get_parameter("planning_time").as_double();
//     planner_id_ = this->get_parameter("planner_id").as_string();

//     // IK gate params
//     require_ik_gate_ = this->get_parameter("require_ik_gate").as_bool();
//     ik_gate_topic_ = this->get_parameter("ik_gate_topic").as_string();

//     // Gripper params
//     gripper_controller_topic_ = this->get_parameter("gripper_controller_topic").as_string();
//     gripper_joints_ = this->get_parameter("gripper_joints").as_string_array();
//     gripper_open_pos_ = this->get_parameter("gripper_open").as_double_array();
//     gripper_close_pos_ = this->get_parameter("gripper_close").as_double_array();
//     gripper_motion_time_ = this->get_parameter("gripper_motion_time").as_double();
//     gripper_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(gripper_controller_topic_, 10);

//     // Rotating table collision params
//     add_table_collision_ = this->get_parameter("add_table_collision").as_bool();
//     table_frame_ = this->get_parameter("table_frame").as_string();
//     if (table_frame_.empty()) table_frame_ = base_frame_;
//     table_radius_ = this->get_parameter("table_radius").as_double();
//     table_height_ = this->get_parameter("table_height").as_double();
//     table_x_ = this->get_parameter("table_x").as_double();
//     table_y_ = this->get_parameter("table_y").as_double();
//     table_z_ = this->get_parameter("table_z").as_double();
//   }

//   // Gripper
//   bool command_gripper_(const std::vector<double>& pos, double seconds, double start_delay_s = 0.0) {
//     if (!gripper_pub_) return false;
//     if (pos.size() != gripper_joints_.size()) {
//       RCLCPP_ERROR(get_logger(), "Gripper command size mismatch");
//       return false;
//     }
//     trajectory_msgs::msg::JointTrajectory traj;
//     traj.header.stamp = this->get_clock()->now() + rclcpp::Duration::from_seconds(std::max(0.0, start_delay_s));
//     traj.joint_names = gripper_joints_;
//     trajectory_msgs::msg::JointTrajectoryPoint pt;
//     pt.positions = pos;
//     pt.time_from_start = rclcpp::Duration::from_seconds(seconds);
//     traj.points.push_back(pt);
//     gripper_pub_->publish(traj);
//     return true;
//   }
//   bool open_gripper_(double start_delay_s = 0.0) { return command_gripper_(gripper_open_pos_, gripper_motion_time_, start_delay_s); }
//   bool close_gripper_(double start_delay_s = 0.0) { return command_gripper_(gripper_close_pos_, gripper_motion_time_, start_delay_s); }

//   void schedule_gripper_close_for_plan_(const moveit::planning_interface::MoveGroupInterface::Plan& plan) {
//     if (plan.trajectory_.joint_trajectory.points.empty()) return;
//     const auto& last_tfs = plan.trajectory_.joint_trajectory.points.back().time_from_start;
//     const double traj_total = rclcpp::Duration(last_tfs).seconds();
//     double desired_start = targets_.t_grasp - gripper_motion_time_;
//     double latest_start = std::max(0.0, traj_total - gripper_motion_time_);
//     double start_delay = std::clamp(desired_start, 0.0, latest_start);
//     (void)close_gripper_(start_delay);
//   }

//   geometry_msgs::msg::Quaternion quatFromRPY(double r, double p, double y) {
//     Eigen::AngleAxisd Rx(r, Eigen::Vector3d::UnitX());
//     Eigen::AngleAxisd Ry(p, Eigen::Vector3d::UnitY());
//     Eigen::AngleAxisd Rz(y, Eigen::Vector3d::UnitZ());
//     Eigen::Quaterniond q = Rz * Ry * Rx;
//     geometry_msgs::msg::Quaternion qmsg;
//     qmsg.x = q.x(); qmsg.y = q.y(); qmsg.z = q.z(); qmsg.w = q.w();
//     return qmsg;
//   }
//   static double yawFromQuat(const geometry_msgs::msg::Quaternion& q) {
//     Eigen::Quaterniond qe(q.w, q.x, q.y, q.z);
//     Eigen::Vector3d eul = qe.toRotationMatrix().eulerAngles(2, 1, 0);
//     return eul[0];
//   }

//   geometry_msgs::msg::PoseStamped build_target_pose_(const geometry_msgs::msg::PoseStamped& current) {
//     geometry_msgs::msg::PoseStamped tgt = current;
//     tgt.header.frame_id = current.header.frame_id.empty() ? base_frame_ : current.header.frame_id;
//     if (target_mode_ == "relative") {
//       tgt.pose.position.x = current.pose.position.x + target_x_;
//       tgt.pose.position.y = current.pose.position.y + target_y_;
//       tgt.pose.position.z = current.pose.position.z + target_z_;
//     } else {
//       tgt.pose.position.x = target_x_;
//       tgt.pose.position.y = target_y_;
//       tgt.pose.position.z = target_z_;
//     }
//     if (!keep_orientation_) {
//       tgt.pose.orientation = quatFromRPY(target_roll_, target_pitch_, target_yaw_);
//     }
//     return tgt;
//   }

//   bool check_joint_states_topic()
//   {
//     auto topic_names = this->get_topic_names_and_types();
//     for (const auto& topic : topic_names) {
//       if (topic.first == "/joint_states") {
//         RCLCPP_INFO(get_logger(), "Found /joint_states topic");
//         return true;
//       }
//     }
//     RCLCPP_ERROR(get_logger(), "Joint states topic /joint_states not found!");
//     return false;
//   }

//   void debug_robot_model() {
//     if (!move_group_) return;
//     auto robot_model = move_group_->getRobotModel();
//     RCLCPP_INFO(get_logger(), "Robot model: %s", robot_model->getName().c_str());
//     const auto& link_names = robot_model->getLinkModelNames();
//     RCLCPP_INFO(get_logger(), "Available links (%zu):", link_names.size());
//     for (const auto& link : link_names) RCLCPP_INFO(get_logger(), "  - %s", link.c_str());
//     if (robot_model->hasLinkModel(ee_link_)) {
//       RCLCPP_INFO(get_logger(), "✓ End effector link '%s' found", ee_link_.c_str());
//     } else {
//       RCLCPP_ERROR(get_logger(), "✗ End effector link '%s' NOT found", ee_link_.c_str());
//     }
//     auto current_state = move_group_->getCurrentState(0.0);
//     if (!current_state) return;
//     const auto* group = current_state->getJointModelGroup(planning_group_);
//     if (group) {
//       const auto& ee_links = group->getLinkModelNames();
//       RCLCPP_INFO(get_logger(), "Links in planning group '%s':", planning_group_.c_str());
//       for (const auto& link : ee_links) RCLCPP_INFO(get_logger(), "  - %s", link.c_str());
//     }
//   }

//   bool wait_for_valid_joint_states_with_time(double timeout_seconds = 20.0) {
//     RCLCPP_INFO(get_logger(), "Waiting for joint states with valid timestamps...");
//     bool received_joint_states = false;

//     auto joint_state_sub = this->create_subscription<sensor_msgs::msg::JointState>(
//         "joint_states", 10,
//         [&received_joint_states, this](const sensor_msgs::msg::JointState::SharedPtr msg) {
//           RCLCPP_INFO_ONCE(get_logger(), "Received joint state with timestamp %.3f and %zu joints",
//                            rclcpp::Time(msg->header.stamp).seconds(), msg->name.size());
//           received_joint_states = true;
//         });

//     auto start_time = this->get_clock()->now();
//     while (rclcpp::ok()) {
//       rclcpp::spin_some(this->get_node_base_interface());
//       if (received_joint_states) {
//         if (move_group_) {
//           try {
//             auto state = move_group_->getCurrentState(0.1);
//             if (state) {
//               RCLCPP_INFO(get_logger(), "Successfully received robot state from MoveGroup!");
//               return true;
//             }
//           } catch (const std::exception& e) {
//             RCLCPP_WARN_THROTTLE(get_logger(), *this->get_clock(), 3000,
//                                  "Exception getting current state: %s", e.what());
//           }
//         }
//       }
//       if ((this->get_clock()->now() - start_time).seconds() > timeout_seconds) {
//         RCLCPP_ERROR(get_logger(), "Timeout waiting for valid joint states");
//         return false;
//       }
//       rclcpp::sleep_for(std::chrono::milliseconds(100));
//     }
//     return false;
//   }

//   geometry_msgs::msg::PoseStamped get_current_pose_robust() {
//     if (!move_group_) return geometry_msgs::msg::PoseStamped();
//     for (int attempt = 0; attempt < 3; ++attempt) {
//       try {
//         rclcpp::sleep_for(std::chrono::milliseconds(80));
//         auto pose = move_group_->getCurrentPose(ee_link_);
//         if (std::abs(pose.pose.position.x) > 0.001 ||
//             std::abs(pose.pose.position.y) > 0.001 ||
//             std::abs(pose.pose.position.z) > 0.001) {
//           return pose;
//         }
//       } catch (...) {}
//     }
//     return geometry_msgs::msg::PoseStamped();
//   }

//   size_t findNearestIndexToPose(const robot_trajectory::RobotTrajectory& rt,
//                                 const std::string& link,
//                                 const geometry_msgs::msg::Pose& target) {
//     size_t best_idx = 0;
//     double best_dist = std::numeric_limits<double>::infinity();
//     for (size_t i = 0; i < rt.getWayPointCount(); ++i) {
//       const auto& st = rt.getWayPoint(i);
//       const auto& T = st.getGlobalLinkTransform(link);
//       Eigen::Vector3d p = T.translation();
//       const double dx = p.x() - target.position.x;
//       const double dy = p.y() - target.position.y;
//       const double dz = p.z() - target.position.z;
//       const double d = std::sqrt(dx*dx + dy*dy + dz*dz);
//       if (d < best_dist) { best_dist = d; best_idx = i; }
//     }
//     return best_idx;
//   }

//   // Predictor async flow (single request per attempt)
//   void request_pick_pose_async_(double t_rel_s)
//   {
//     if (!pick_client_) { RCLCPP_ERROR(get_logger(), "Predictor client not created"); end_planning_session_(true); return; }
//     if (!predictor_ready_.load(std::memory_order_relaxed) && require_predictor_ready_) {
//       RCLCPP_WARN(get_logger(), "Predictor not ready yet, skipping request");
//       end_planning_session_(true);
//       return;
//     }
//     if (t_rel_s < 0.0) t_rel_s = 0.0;

//     auto req = std::make_shared<lite6_pick_predictor_interfaces::srv::GetPoseAt::Request>();
//     req->use_relative = true;
//     const double si = std::floor(t_rel_s);
//     req->query_time.sec = static_cast<int32_t>(si);
//     req->query_time.nanosec = static_cast<uint32_t>((t_rel_s - si) * 1e9);

//     auto future = pick_client_->async_send_request(req,
//       [this](rclcpp::Client<lite6_pick_predictor_interfaces::srv::GetPoseAt>::SharedFuture fut)
//       {
//         try {
//           auto res = fut.get();
//           if (!res->ok) {
//             RCLCPP_ERROR(get_logger(), "GetPoseAt returned ok=false");
//             end_planning_session_(true);
//             return;
//           }
//           const auto& pick_pose = res->pose;
//           RCLCPP_INFO(get_logger(), "Received pick pose [%s] (%.3f, %.3f, %.3f)",
//                       pick_pose.header.frame_id.c_str(),
//                       pick_pose.pose.position.x, pick_pose.pose.position.y, pick_pose.pose.position.z);

//           // Plan hover + descent
//           if (!plan_and_execute_from_pick_(pick_pose)) {
//             RCLCPP_ERROR(get_logger(), "Planning/execution failed");
//             end_planning_session_(true);
//             return;
//           }
//           end_planning_session_(false);
//         }
//         catch (const std::exception& e) {
//           RCLCPP_ERROR(get_logger(), "Predictor future exception: %s", e.what());
//           end_planning_session_(true);
//         }
//       });

//     (void)future;
//   }

//   void end_planning_session_(bool cooldown)
//   {
//     planning_in_progress_ = false;
//     if (cooldown) last_attempt_wall_ = std::chrono::steady_clock::now();
//   }

//   bool transform_to_frame_(const geometry_msgs::msg::PoseStamped& in,
//                            const std::string& to_frame,
//                            geometry_msgs::msg::PoseStamped& out)
//   {
//     if (in.header.frame_id.empty() || in.header.frame_id == to_frame) {
//       out = in;
//       out.header.frame_id = to_frame;
//       return true;
//     }
//     try {
//       geometry_msgs::msg::TransformStamped tf =
//           tf_buffer_->lookupTransform(to_frame, in.header.frame_id, tf2::TimePointZero, tf2::durationFromSec(0.3));
//       tf2::doTransform(in, out, tf);
//       out.header.frame_id = to_frame;
//       return true;
//     } catch (const std::exception& e) {
//       RCLCPP_ERROR(get_logger(), "TF transform %s -> %s failed: %s",
//                    in.header.frame_id.c_str(), to_frame.c_str(), e.what());
//       return false;
//     }
//   }

//   bool plan_and_execute_from_pick_(const geometry_msgs::msg::PoseStamped& pick_pose_in)
//   {
//     if (!move_group_) return false;

//     // Transform predictor pose
//     const std::string planning_frame = move_group_->getPlanningFrame();
//     geometry_msgs::msg::PoseStamped pick_pose_tf;
//     if (!transform_to_frame_(pick_pose_in, planning_frame, pick_pose_tf)) return false;

//     // Build hover/grasp with horizontal orientation (roll offset + yaw policy)
//     geometry_msgs::msg::PoseStamped pose_hover = pick_pose_tf;
//     geometry_msgs::msg::PoseStamped pose_grasp = pick_pose_tf;

//     double yaw = 0.0;
//     if (horizontal_yaw_mode_ == "current") {
//       auto cur = get_current_pose_robust();
//       yaw = cur.header.frame_id.empty() ? yawFromQuat(pick_pose_tf.pose.orientation)
//                                         : yawFromQuat(cur.pose.orientation);
//     } else if (horizontal_yaw_mode_ == "fixed") {
//       yaw = fixed_yaw_;
//     } else {
//       yaw = yawFromQuat(pick_pose_tf.pose.orientation);
//     }
//     auto horiz = quatFromRPY(horizontal_roll_offset_, 0.0, yaw);

//     if (enforce_horizontal_) {
//       pose_hover.pose.orientation = horiz;
//       pose_grasp.pose.orientation = horiz;
//     }
//     pose_hover.pose.position.z = z_hover_;
//     pose_grasp.pose.position.z = z_grasp_;

//     // Stage 1: OMPL plan to hover
//     move_group_->setStartStateToCurrentState();
//     move_group_->setMaxVelocityScalingFactor(vel_scale_);
//     move_group_->setMaxAccelerationScalingFactor(acc_scale_);
//     move_group_->setPlanningTime(planning_time_);
//     move_group_->setNumPlanningAttempts(6);
//     move_group_->setPlannerId(planner_id_);
//     move_group_->setGoalPositionTolerance(0.01);
//     move_group_->setGoalOrientationTolerance(0.20);

//     bool constraints_set = false;
//     if (enforce_horizontal_ &&
//         (constraint_mode_ == "path_only" || (constraint_mode_ == "both" && use_path_constraints_for_hover_))) {
//       RCLCPP_INFO(get_logger(), "Using path orientation constraint (mode=%s)", constraint_mode_.c_str());
//       moveit_msgs::msg::OrientationConstraint oc;
//       oc.header.frame_id = planning_frame;
//       oc.link_name = ee_link_;
//       oc.orientation = horiz;
//       oc.absolute_x_axis_tolerance = std::max(1e-3, rp_tol_);
//       oc.absolute_y_axis_tolerance = std::max(1e-3, rp_tol_);
//       oc.absolute_z_axis_tolerance = std::max(0.5, yaw_tol_); // yaw mostly free
//       oc.weight = 1.0;
//       moveit_msgs::msg::Constraints cs; cs.orientation_constraints.push_back(oc);
//       move_group_->setPathConstraints(cs);
//       constraints_set = true;
//     }

//     if (constraint_mode_ == "path_only") {
//       move_group_->setPositionTarget(
//           pose_hover.pose.position.x, pose_hover.pose.position.y, pose_hover.pose.position.z, ee_link_);
//       move_group_->setGoalOrientationTolerance(M_PI); // do not constrain goal orientation
//     } else {
//       move_group_->setPoseTarget(pose_hover.pose, ee_link_);
//     }

//     moveit::planning_interface::MoveGroupInterface::Plan plan_to_hover;
//     auto plan_code = move_group_->plan(plan_to_hover);

//     if (constraints_set) move_group_->clearPathConstraints();

//     if (plan_code != moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
//       RCLCPP_ERROR(get_logger(), "OMPL plan to hover failed (code=%d)", plan_code.val);
//       return false;
//     }

//     // Visualize and execute hover
//     if (display_pub_) {
//       moveit_msgs::msg::DisplayTrajectory msg;
//       msg.model_id = move_group_->getRobotModel()->getName();
//       if (auto cs = move_group_->getCurrentState(0.0)) {
//         moveit_msgs::msg::RobotState rs;
//         moveit::core::robotStateToRobotStateMsg(*cs, rs);
//         msg.trajectory_start = rs;
//       }
//       msg.trajectory.push_back(plan_to_hover.trajectory_);
//       display_pub_->publish(msg);
//     }

//     auto exec_code = move_group_->execute(plan_to_hover);
//     if (exec_code != moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
//       RCLCPP_ERROR(get_logger(), "Execution to hover failed (code=%d)", exec_code.val);
//       return false;
//     }

//     // Stage 2: Cartesian descent hover -> grasp with same orientation
//     move_group_->setStartStateToCurrentState();

//     std::vector<geometry_msgs::msg::Pose> waypoints{pose_hover.pose, pose_grasp.pose};
//     moveit_msgs::msg::RobotTrajectory cart_traj_msg;
//     const double eef_step = 0.005;
//     const double jump_threshold = 0.0;

//     double fraction = 0.0;
//     try {
//       fraction = move_group_->computeCartesianPath(waypoints, eef_step, jump_threshold, cart_traj_msg, true);
//     } catch (const std::exception& e) {
//       RCLCPP_ERROR(get_logger(), "Cartesian path exception: %s", e.what());
//       return false;
//     }
//     if (fraction < 0.1) {
//       RCLCPP_ERROR(get_logger(), "Cartesian hover->grasp failed (fraction=%.2f)", fraction);
//       return false;
//     }
//     if (fraction < 0.99) {
//       RCLCPP_WARN(get_logger(), "Cartesian hover->grasp fraction=%.2f; proceeding", fraction);
//     }

//     // Time-parameterize and enforce timing
//     moveit::core::RobotModelConstPtr model = move_group_->getRobotModel();
//     auto start_state = move_group_->getCurrentState(2.0);
//     if (!start_state) {
//       RCLCPP_ERROR(get_logger(), "No current robot state before descent");
//       return false;
//     }

//     robot_trajectory::RobotTrajectory rt(model, planning_group_);
//     rt.setRobotTrajectoryMsg(*start_state, cart_traj_msg);

//     if (!tp_control_cpp::applyTimeParameterization(rt, method_, vel_scale_, acc_scale_)) {
//       RCLCPP_ERROR(get_logger(), "Time parameterization failed for descent");
//       return false;
//     }

//     const size_t i_hover = 0;
//     const size_t i_grasp = rt.getWayPointCount() ? (rt.getWayPointCount() - 1) : 0;
//     (void)tp_control_cpp::enforceSegmentTimes(rt, i_hover, i_grasp, targets_);
//     if (method_ == TimingMethod::TOTG_THEN_RUCKIG) {
//       (void)tp_control_cpp::applyTimeParameterization(rt, TimingMethod::RUCKIG, vel_scale_, acc_scale_);
//     }

//     rt.getRobotTrajectoryMsg(cart_traj_msg);
//     moveit::planning_interface::MoveGroupInterface::Plan plan_descent;
//     plan_descent.trajectory_ = cart_traj_msg;
//     moveit::core::robotStateToRobotStateMsg(*start_state, plan_descent.start_state_);
//     plan_descent.planning_time_ = 0.0;

//     if (display_pub_) {
//       moveit_msgs::msg::DisplayTrajectory msg;
//       msg.model_id = move_group_->getRobotModel()->getName();
//       if (auto cs = move_group_->getCurrentState(0.0)) {
//         moveit_msgs::msg::RobotState rs;
//         moveit::core::robotStateToRobotStateMsg(*cs, rs);
//         msg.trajectory_start = rs;
//       }
//       msg.trajectory.push_back(plan_descent.trajectory_);
//       display_pub_->publish(msg);
//     }

//     schedule_gripper_close_for_plan_(plan_descent);
//     (void)move_group_->execute(plan_descent);
//     return true;
//   }

//   void run_once_with_predictor_()
//   {
//     if (!move_group_) { RCLCPP_ERROR(get_logger(), "MoveGroup not initialized"); end_planning_session_(true); return; }
//     if (!pick_client_ || !pick_client_->wait_for_service(std::chrono::seconds(0))) {
//       RCLCPP_WARN(get_logger(), "Predictor service '%s' not ready", predictor_service_.c_str());
//       end_planning_session_(true);
//       return;
//     }
//     move_group_->setStartStateToCurrentState();
//     request_pick_pose_async_(commit_pick_time_s_);
//   }

//   // Demo without predictor (unused)
//   void run_once()
//   {
//     if (!move_group_) return;
//     auto pose0 = get_current_pose_robust();
//     if (pose0.header.frame_id.empty()) return;
//     geometry_msgs::msg::PoseStamped pose_hover = build_target_pose_(pose0);
//     pose_hover.pose.position.z = z_hover_;
//     geometry_msgs::msg::PoseStamped pose_grasp = pose_hover;
//     pose_grasp.pose.position.z = z_grasp_;
//     std::vector<geometry_msgs::msg::Pose> waypoints{pose_hover.pose, pose_grasp.pose};

//     moveit_msgs::msg::RobotTrajectory cart_traj_msg;
//     (void)move_group_->computeCartesianPath(waypoints, 0.005, 0.0, cart_traj_msg, true);
//     moveit::core::RobotModelConstPtr model = move_group_->getRobotModel();
//     auto start_state = move_group_->getCurrentState(2.0);
//     if (!start_state) return;
//     robot_trajectory::RobotTrajectory rt(model, planning_group_);
//     rt.setRobotTrajectoryMsg(*start_state, cart_traj_msg);
//     (void)tp_control_cpp::applyTimeParameterization(rt, method_, vel_scale_, acc_scale_);
//     rt.getRobotTrajectoryMsg(cart_traj_msg);
//     moveit::planning_interface::MoveGroupInterface::Plan plan;
//     plan.trajectory_ = cart_traj_msg;
//     moveit::core::robotStateToRobotStateMsg(*start_state, plan.start_state_);
//     if (display_pub_) {
//       moveit_msgs::msg::DisplayTrajectory msg;
//       msg.model_id = move_group_->getRobotModel()->getName();
//       msg.trajectory.push_back(plan.trajectory_);
//       display_pub_->publish(msg);
//     }
//     (void)move_group_->execute(plan);
//   }

//   // Add a cylinder as the rotating table collision object (matches tp_control.py)
//   void add_rotating_table_collision_(const geometry_msgs::msg::Pose& table_pose,
//                                      double table_radius,
//                                      double table_height)
//   {
//     moveit_msgs::msg::CollisionObject obj;
//     obj.header.frame_id = table_frame_.empty() ? base_frame_ : table_frame_;
//     obj.header.stamp = this->get_clock()->now();
//     obj.id = "rotating_table";

//     shape_msgs::msg::SolidPrimitive cylinder;
//     cylinder.type = shape_msgs::msg::SolidPrimitive::CYLINDER;
//     cylinder.dimensions.resize(2);
//     cylinder.dimensions[shape_msgs::msg::SolidPrimitive::CYLINDER_HEIGHT] = table_height;
//     cylinder.dimensions[shape_msgs::msg::SolidPrimitive::CYLINDER_RADIUS] = table_radius;

//     obj.primitives.push_back(cylinder);
//     obj.primitive_poses.push_back(table_pose);
//     obj.operation = moveit_msgs::msg::CollisionObject::ADD;

//     // Apply via PlanningSceneInterface
//     planning_scene_interface_.applyCollisionObject(obj);

//     // Optional: publish as diff
//     moveit_msgs::msg::PlanningScene scene_msg;
//     scene_msg.is_diff = true;
//     scene_msg.world.collision_objects.push_back(obj);
//     if (planning_scene_pub_) planning_scene_pub_->publish(scene_msg);
//   }

// private:
//   std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
//   moveit_visual_tools::MoveItVisualToolsPtr visual_tools_;
//   rclcpp::Publisher<moveit_msgs::msg::DisplayTrajectory>::SharedPtr display_pub_;

//   std::string planning_group_, ee_link_, base_frame_;
//   double z_hover_{0.20}, z_grasp_{0.05};
//   double vel_scale_{0.9}, acc_scale_{0.6};
//   TimingMethod method_{TimingMethod::TOTG_THEN_RUCKIG};
//   std::string timing_method_str_;
//   TimingTargets targets_;
//   bool use_pose_target_{false};
//   std::string target_mode_;
//   std::string target_frame_;
//   double target_x_{0.0}, target_y_{0.0}, target_z_{0.0};
//   bool keep_orientation_{true};
//   double target_roll_{0.0}, target_pitch_{0.0}, target_yaw_{0.0};

//   // Horizontal control
//   bool enforce_horizontal_{true};
//   std::string horizontal_yaw_mode_{"predictor"}; // predictor|current|fixed
//   double fixed_yaw_{0.0};
//   double horizontal_roll_offset_{M_PI};
//   double rp_tol_{0.10};
//   double yaw_tol_{M_PI};
//   bool use_path_constraints_for_hover_{true};
//   std::string constraint_mode_{"path_only"};

//   // Flow control
//   std::atomic_bool planning_in_progress_{false};
//   bool move_group_params_ready_{false};
//   rclcpp::TimerBase::SharedPtr delayed_plan_timer_;
//   double replan_cooldown_s_{2.0};
//   std::chrono::steady_clock::time_point last_attempt_wall_{};
//   bool last_ik_reachable_{false};

//   // TF
//   std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
//   std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

//   // Predictor
//   bool use_predictor_{true};
//   bool require_predictor_ready_{true};
//   double commit_pick_time_s_{3.0};
//   std::string predictor_ready_topic_{"predictor_ready"};
//   std::string predictor_service_{"get_predicted_pose_at"};
//   std::atomic<bool> predictor_ready_{false};
//   rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr predictor_ready_sub_;
//   rclcpp::Client<lite6_pick_predictor_interfaces::srv::GetPoseAt>::SharedPtr pick_client_;

//   // Planning params
//   double planning_time_{0.5};
//   std::string planner_id_{"RRTConnect"};

//   // IK gate
//   bool require_ik_gate_{true};
//   std::string ik_gate_topic_{"/ik_gate/output"};
//   rclcpp::Subscription<std_msgs::msg::String>::SharedPtr ik_gate_sub_;

//   // Gripper
//   rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr gripper_pub_;
//   std::string gripper_controller_topic_;
//   std::vector<std::string> gripper_joints_;
//   std::vector<double> gripper_open_pos_;
//   std::vector<double> gripper_close_pos_;
//   double gripper_motion_time_{0.7};

//   // Planning scene interface and optional publisher
//   moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;
//   rclcpp::Publisher<moveit_msgs::msg::PlanningScene>::SharedPtr planning_scene_pub_;

//   // Rotating table config
//   bool add_table_collision_{true};
//   std::string table_frame_{"link_base"};
//   double table_radius_{0.75};
//   double table_height_{0.06};
//   double table_x_{0.9};
//   double table_y_{0.0};
//   double table_z_{0.1};
// };

// int main(int argc, char** argv)
// {
//   rclcpp::init(argc, argv);
//   auto node = std::make_shared<TpControlNode>();
//   node->initialize();
//   rclcpp::spin(node);
//   rclcpp::shutdown();
//   return 0;
// }