from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ign_src = Node(
        package='lite6_pick_predictor',
        executable='ign_obj_bridge',
        name='ign_obj_bridge',
        output='screen',
        parameters=[{
            'world_name': 'default',
            'model_name': '20x20_Profile',
            'world_frame': 'world',
            'output_frame': 'world',
            'poll_rate_hz': 40.0,
            'offset_x': 0.0, 'offset_y': 0.0, 'offset_z': 0.0,
            'topic': 'obj_1_coord',
            'publish_pose': True,
            'pose_topic': 'obj_1_pose',
        }]
    )

    predictor = Node(
        package='lite6_pick_predictor',
        executable='kalman_predictor',
        name='kalman_predictor',
        output='screen',
        parameters=[{
            'planning_latency_s': 0.25,
            'execution_latency_s': 0.55,
            'bootstrap_min_samples': 30,
            'bootstrap_maxlen': 200,
            'bootstrap_min_time_span_s': 1.0,
            'bootstrap_min_angle_span_rad': 0.35,
            'q_cx': 1e-6, 'q_cy': 1e-6, 'q_r': 1e-6, 'q_theta': 1e-5, 'q_omega': 1e-4,
            'r_meas': 1e-4,
            'output_frame_id': 'world',
            'use_tangent_yaw': False,
            'use_input_orientation': True,
            'input_pose_topic': 'obj_1_pose',
            'use_last_z': True,
            'fixed_pick_z': 0.02,
            'path_horizon_s': 4.0,
            'path_rate_hz': 40.0,
            'max_dt_cap_s': 0.05,
            'planning_ready_min_age_s': 1.5,
            'planning_ready_min_radius_samples': 40,
            'planning_ready_radius_var_max': 5e-5,
        }]
    )

    # bridge = Node(
    #     package='lite6_pick_predictor',
    #     executable='moveit_bridge_action',
    #     name='moveit_bridge_action',
    #     output='screen',
    #     parameters=[{
    #         'group_name': 'lite6',
    #         'end_effector_link': 'link_tcp',
    #         'planning_frame': 'world',
    #         'allowed_planning_time': 1.2,
    #         'goal_tolerance_pos': 0.003,
    #         'goal_tolerance_ori': 0.02,
    #         'max_velocity_scaling': 0.8,
    #         'max_acceleration_scaling': 0.6,
    #         'planning_service': '/plan_kinematic_path',

    #         'follow_action': '/lite6_traj_controller/follow_joint_trajectory',
    #         'joint_names': ['joint1','joint2','joint3','joint4','joint5','joint6'],

    #         'path_topic': 'predicted_ee_path',
    #         'pick_time_s': 3.0,

    #         'target_orientation_from_predictor': True,
    #         'ik_yaw_align_tangent': False,
    #         'table_center_xy': [0.9, 0.0],

    #         'z_hover': 0.15,
    #         'z_grasp': 0.02,
    #         't_hover_s': 1.4,
    #         't_grasp_s': 2.4,

    #         'grip_topic': 'gripper/close',
    #         'grip_fire_early_s': 0.05,

    #         'contact_speed_max': 0.35,
    #         'require_contact_speed_check': False,
    #         'theta_replan_thresh_rad': 0.20,

    #         'pred_pose_topic': 'predicted_pick_pose',

    #         'path_trigger_decim': 20,
    #         'require_predictor_ready': True,
    #         'min_planning_delay_s': 1.5
    #     }]
    # )

    return LaunchDescription([ign_src, predictor])#, bridge
