#pragma once
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/trajectory_processing/time_optimal_trajectory_generation.h>
#include <moveit/trajectory_processing/ruckig_traj_smoothing.h>
#include <rclcpp/rclcpp.hpp>

namespace tp_control_cpp {

enum class TimingMethod { TOTG, RUCKIG };

struct TimingTargets {
  double t_hover = 3.0;   // seconds
  double t_grasp = 4.5;   // seconds
};

inline bool applyTimeParameterization(robot_trajectory::RobotTrajectory& rt,
                                      TimingMethod method,
                                      double max_vel_scaling = 1.0,
                                      double max_acc_scaling = 1.0)
{
  using namespace trajectory_processing;

  if (method == TimingMethod::TOTG)
  {
    trajectory_processing::TimeOptimalTrajectoryGeneration totg;
    return totg.computeTimeStamps(rt, max_vel_scaling, max_acc_scaling);
  }
  else
  {
    // Ruckig smoothing operates in-place on a RobotTrajectory
    trajectory_processing::RuckigSmoothing ruckig;
    return ruckig.applySmoothing(rt, max_vel_scaling, max_acc_scaling);
  }
}

inline bool enforceSegmentTimes(robot_trajectory::RobotTrajectory& rt,
                                size_t i_hover, size_t i_grasp,
                                const TimingTargets& targets)
{
  if (rt.getWayPointCount() < 2 || i_hover >= rt.getWayPointCount() || i_grasp >= rt.getWayPointCount() || i_hover >= i_grasp)
    return false;

  // Gather current times
  std::vector<double> t(rt.getWayPointCount());
  for (size_t i = 0; i < t.size(); ++i)
    t[i] = rt.getWayPointDurationFromStart(i);

  const double t_hover_now = t[i_hover];
  const double t_grasp_now = t[i_grasp];

  if (targets.t_hover < t_hover_now - 1e-6) return false;  // would need to speed up
  if (targets.t_grasp < t_grasp_now - 1e-6) return false;  // would need to speed up

  const double scale1 = (t_hover_now > 1e-9) ? (targets.t_hover / t_hover_now) : 1.0;
  // Recompute times for i >= i_hover using the second scale around the hover anchor
  const double seg2_now = t_grasp_now - t_hover_now;
  if (seg2_now <= 1e-9) return false;
  const double scale2 = (targets.t_grasp - targets.t_hover) / seg2_now;

    //create new trajextory with scaled times
    //get trajectory message, modify timings and recreate trajectory
    moveit_msgs::msg::RobotTrajectory traj_msg;
    rt.getRobotTrajectoryMsg(traj_msg);

     // Scale the time_from_start for each point
  for (size_t i = 0; i <= i_hover && i < traj_msg.joint_trajectory.points.size(); ++i)
  {
    const double new_t = t[i] * scale1;
    traj_msg.joint_trajectory.points[i].time_from_start = rclcpp::Duration::from_seconds(new_t);
  }
  
  const double t_hover_new = (i_hover < traj_msg.joint_trajectory.points.size()) ? 
    rclcpp::Duration(traj_msg.joint_trajectory.points[i_hover].time_from_start).seconds() : targets.t_hover;
  
  for (size_t i = i_hover; i < t.size() && i < traj_msg.joint_trajectory.points.size(); ++i)
  {
    const double dt = t[i] - t[i_hover];
    const double new_t = t_hover_new + dt * scale2;
    traj_msg.joint_trajectory.points[i].time_from_start = rclcpp::Duration::from_seconds(new_t);
  }

  // Recreate trajectory from modified message
  moveit::core::RobotStatePtr start_state = rt.getFirstWayPointPtr();
  rt.setRobotTrajectoryMsg(*start_state, traj_msg);
  
  return true;
}

} // namespace tp_control_cpp

// #pragma once
// #include <moveit/robot_trajectory/robot_trajectory.h>
// #include <moveit/trajectory_processing/time_optimal_trajectory_generation.h>
// // Ruckig smoother (available in MoveIt2)
// #include <moveit/trajectory_processing/ruckig_traj_smoothing.h>

// #include <rclcpp/rclcpp.hpp>
// #include <geometry_msgs/msg/pose_stamped.hpp>
// #include <sensor_msgs/msg/joint_state.hpp>

// #include <moveit/move_group_interface/move_group_interface.h>
// #include <moveit/planning_scene_monitor/planning_scene_monitor.h>
// #include <moveit/robot_state/conversions.h>

// #include <moveit_msgs/msg/move_it_error_codes.h>
// #include <moveit_visual_tools/moveit_visual_tools.h>

// #include <tf2_eigen/tf2_eigen.hpp>
// #include "tp_control_cpp/trajectory_utils.hpp"


// namespace tp_control_cpp {

// enum class TimingMethod { TOTG, RUCKIG };

// struct TimingTargets {
//   double t_hover = 3.0;   // seconds
//   double t_grasp = 4.5;   // seconds
// };

// inline bool applyTimeParameterization(moveit::robot_trajectory::RobotTrajectory& rt,
//                                       TimingMethod method,
//                                       double max_vel_scaling = 1.0,
//                                       double max_acc_scaling = 1.0)
// {
//   using namespace moveit::trajectory_processing;

//   if (method == TimingMethod::TOTG)
//   {
//     TimeOptimalTrajectoryGeneration totg;
//     return totg.computeTimeStamps(rt, max_vel_scaling, max_acc_scaling);
//   }
//   else
//   {
//     // Ruckig smoothing operates in-place on a RobotTrajectory
//     RuckigSmoothing ruckig;
//     return ruckig.applySmoothing(rt, max_vel_scaling, max_acc_scaling);
//   }
// }

// /**
//  * Scale the time_from_start of segments so that the last point at indices
//  * i_hover and i_grasp land at the requested absolute times. We only slow
//  * down (scale >= 1). If requested time is earlier than current segment
//  * time, we refuse (return false) to avoid violating dynamics.
//  */
// inline bool enforceSegmentTimes(moveit::robot_trajectory::RobotTrajectory& rt,
//                                 size_t i_hover, size_t i_grasp,
//                                 const TimingTargets& targets)
// {
//   if (rt.getWayPointCount() < 2 || i_hover >= rt.getWayPointCount() || i_grasp >= rt.getWayPointCount() || i_hover >= i_grasp)
//     return false;

//   // Gather current times
//   std::vector<double> t(rt.getWayPointCount());
//   for (size_t i = 0; i < t.size(); ++i)
//     t[i] = rt.getWayPointDurationFromStart(i);

//   const double t_hover_now = t[i_hover];
//   const double t_grasp_now = t[i_grasp];

//   if (targets.t_hover < t_hover_now - 1e-6) return false;  // would need to speed up
//   if (targets.t_grasp < t_grasp_now - 1e-6) return false;  // would need to speed up

//   const double scale1 = (t_hover_now > 1e-9) ? (targets.t_hover / t_hover_now) : 1.0;
//   // Recompute times for i >= i_hover using the second scale around the hover anchor
//   const double seg2_now = t_grasp_now - t_hover_now;
//   if (seg2_now <= 1e-9) return false;
//   const double scale2 = (targets.t_grasp - targets.t_hover) / seg2_now;

//   // Apply scaling piecewise: [0..i_hover], [i_hover..end]
//   for (size_t i = 0; i <= i_hover; ++i)
//   {
//     const double new_t = t[i] * scale1;
//     rt.setWayPointDurationFromStart(i, new_t);
//   }
//   const double t_hover_new = rt.getWayPointDurationFromStart(i_hover);
//   for (size_t i = i_hover; i < t.size(); ++i)
//   {
//     const double dt = t[i] - t[i_hover];
//     rt.setWayPointDurationFromStart(i, t_hover_new + dt * scale2);
//   }
//   return true;
// }

// } // namespace tp_control_cpp
