# from launch import LaunchDescription
# from launch_ros.actions import Node
# from launch.actions import DeclareLaunchArgument
# from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
# from launch_ros.substitutions import FindPackageShare
# import os
# import xacro
# from ament_index_python.packages import get_package_share_directory

# def generate_launch_description():
#     # Get the robot description files
#     xarm_description_path = get_package_share_directory('xarm_description')
#     xarm_moveit_config_path = get_package_share_directory('xarm_moveit_config')
    
#     # Load URDF
#     urdf_file = os.path.join(xarm_description_path, 'urdf', 'lite6', 'lite6.urdf.xacro')
    
#     # Load SRDF  
#     srdf_file = os.path.join(xarm_moveit_config_path, 'srdf', '_lite6_macro.srdf.xacro')
    
#     # Read the files
#     with open(urdf_file, 'r') as file:
#         robot_description = file.read()
    
#     with open(srdf_file, 'r') as file:
#         robot_description_semantic = file.read()

#     # Declare arguments
#     method_arg = DeclareLaunchArgument('method', default_value='totg')
#     z_hover_arg = DeclareLaunchArgument('z_hover', default_value='0.20')
#     z_grasp_arg = DeclareLaunchArgument('z_grasp', default_value='0.05')
#     t_hover_arg = DeclareLaunchArgument('t_hover', default_value='3.0')
#     t_grasp_arg = DeclareLaunchArgument('t_grasp', default_value='4.5')

#     # Process URDF using xacro
#     robot_description = Command([
#         'xacro ', 
#         PathJoinSubstitution([
#             FindPackageShare('xarm_description'),
#             'urdf',
#             'lite6',
#             'lite6.urdf.xacro'
#         ]),
#         ' robot_type:=lite6',
#         ' dof:=6',
#         ' add_gripper:=add',
#         ' add_vacuum_gripper:=false'
#     ])

#     # Process SRDF using xacro
#     robot_description_semantic = Command([
#         'xacro ',
#         PathJoinSubstitution([
#             FindPackageShare('xarm_moveit_config'),
#             'srdf',
#             '_lite6_macro.srdf.xacro'
#         ]),
#         ' robot_type:=lite6',
#         ' dof:=6',
#         ' add_gripper:=add',
#         ' add_vacuum_gripper:=false'
#     ])

#     # Your tp_control_node with robot description parameters
#     tp_control_node = Node(
#         package='tp_control_cpp',
#         executable='tp_control_node',
#         output='screen',
#         parameters=[{
#             'use_sim_time': True,
#             'robot_description': robot_description,
#             'robot_description_semantic': robot_description_semantic,
#             'planning_group': 'lite6',
#             'ee_link': 'link_tcp',
#             'base_frame': 'link_base',
#             'timing_method': LaunchConfiguration('method'),
#             'z_hover': LaunchConfiguration('z_hover'),
#             'z_grasp': LaunchConfiguration('z_grasp'),
#             't_hover': LaunchConfiguration('t_hover'),
#             't_grasp': LaunchConfiguration('t_grasp'),
#             'vel_scale': 1.0,
#             'acc_scale': 1.0
#         }]
#     )

#     return LaunchDescription([
#         method_arg,
#         z_hover_arg,
#         z_grasp_arg,
#         t_hover_arg,
#         t_grasp_arg,
#         tp_control_node
#     ])
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('method', default_value='totg'),
        DeclareLaunchArgument('z_hover', default_value='0.20'),
        DeclareLaunchArgument('z_grasp', default_value='0.05'),
        DeclareLaunchArgument('t_hover', default_value='3.0'),
        DeclareLaunchArgument('t_grasp', default_value='4.5'),
        Node(
            package='tp_control_cpp',
            executable='tp_control_node',
            output='screen',
            parameters=[{
                'planning_group': 'lite6',
                'ee_link': 'link_6',
                'base_frame': 'link_base',
                'timing_method': LaunchConfiguration('method'),
                'z_hover': LaunchConfiguration('z_hover'),
                'z_grasp': LaunchConfiguration('z_grasp'),
                't_hover': LaunchConfiguration('t_hover'),
                't_grasp': LaunchConfiguration('t_grasp'),
                'vel_scale': 1.0,
                'acc_scale': 1.0
            }]
        )
    ])
