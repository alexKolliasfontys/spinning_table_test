#!/usr/bin/env python3
# filepath: /home/linux/robo_workspace/src/my_simulation/launch/profile_spawn.launch.py

import os
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import TimerAction, RegisterEventHandler
from launch_ros.actions import Node
from launch.event_handlers import OnProcessStart

#removed for troubleshooting
# ---------------Consecutive numbering of objects------------------#
# def get_next_id():
#     #File for temporary storage
#     counter_file = f'/tmp/profile_spawn_counter_{os.getpid()}.txt'

#     # Read current value
#     try:
#         with open(counter_file, 'r') as f:
#             current_id = int(f.read().strip())
#     except (FileNotFoundError, ValueError):
#             current_id = 0

#     # Increment
#     next_id = current_id + 1
#     with open(counter_file, 'w') as f:
#         f.write(str(next_id))

#     return next_id

def generate_launch_description():
    
    # Path to the URDF file
    object_20x20_profile_urdf_path = os.path.join(
        get_package_share_directory('my_simulation'), 'objects', 'urdf', 'F20_20_B.urdf'
    )
    
    # Numeration
    # object_id = get_next_id()
    # object_name = f"Profile_20x20_{object_id}"

    # URDF Publisher Node
    urdf_publisher_node = Node(
        package='my_simulation',
        executable='urdf_publisher.py',
        name='profile_urdf_publisher',
        output='screen',
        arguments=[
            object_20x20_profile_urdf_path,
            'object_20x20_profile_description'
        ]
    )
    
    # Gazebo Spawn Node (delayed to ensure URDF is published first)
    gazebo_spawn_object_node = TimerAction(
        period=3.0,  # Wait 3 seconds for URDF to be published
        actions=[
            Node(
                package="ros_ign_gazebo",
                executable="create",
                name='profile_spawner',
                output='screen',
                arguments=[
                    '-topic', 'object_20x20_profile_description',
                    '-name', '20x20_Profile_additional',
                    '-x', '0.3',      # Within robot reach
                    '-y', '0.0',      # In front of robot
                    '-z', '0.4',     # Just above ground
                    '-Y', '0.0'       # No rotation
                ],
                parameters=[{'use_sim_time': True}],
            )
        ]
    )
    
    return LaunchDescription([
        urdf_publisher_node,
        gazebo_spawn_object_node
    ])