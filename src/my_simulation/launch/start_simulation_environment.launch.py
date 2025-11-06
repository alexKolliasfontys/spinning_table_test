#-----------------------------------Disclaimer-----------------------------------
# this code is copied from _robot_beside_table.launch.py
# and modified to launch the lite6 in a different world- rotating table + objects
#--------------------------------------------------------------------------------

import os
import sys
import yaml
from pathlib import Path
from ament_index_python import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, OpaqueFunction,\
                            TimerAction, SetEnvironmentVariable, RegisterEventHandler, ExecuteProcess, Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch.conditions import IfCondition
from launch_ros.substitutions import FindPackageShare
from launch.event_handlers import OnProcessExit, OnProcessStart

from uf_ros_lib.uf_robot_utils import get_xacro_content, generate_ros2_control_params_temp_file
from uf_ros_lib.moveit_configs_builder import MoveItConfigsBuilder

#------------------------------------------------------
# Function to set up the launch configuration and nodes  
def launch_setup(context, *args, **kwargs):
    prefix = LaunchConfiguration('prefix', default='')
    hw_ns = LaunchConfiguration('hw_ns', default='ufactory')  # lite6
    limited = LaunchConfiguration('limited', default=False)
    effort_control = LaunchConfiguration('effort_control', default=False)
    velocity_control = LaunchConfiguration('velocity_control', default=False)
    add_gripper = LaunchConfiguration('add_gripper', default=False)
    add_vacuum_gripper = LaunchConfiguration('add_vacuum_gripper', default=False)
    add_bio_gripper = LaunchConfiguration('add_bio_gripper', default=False)
    dof = LaunchConfiguration('dof', default=6)
    robot_type = LaunchConfiguration('robot_type', default='lite')
    ros2_control_plugin = LaunchConfiguration('ros2_control_plugin', default='ign_ros2_control/IgnitionSystem')
    #ros2_control_plugin = 'uf_robot_hardware/UFRobotFakeSystemHardware'
    controllers_name = 'fake_controllers'
    #controllers_name = 'controllers'

    add_realsense_d435i = LaunchConfiguration('add_realsense_d435i', default=False)
    add_d435i_links = LaunchConfiguration('add_d435i_links', default=True)
    model1300 = LaunchConfiguration('model1300', default=False)
    robot_sn = LaunchConfiguration('robot_sn', default='')
    attach_to = LaunchConfiguration('attach_to', default='world')
    attach_xyz = LaunchConfiguration('attach_xyz', default='"0 0 0"')
    attach_rpy = LaunchConfiguration('attach_rpy', default='"0 0 0"')

    add_other_geometry = LaunchConfiguration('add_other_geometry', default=False)
    geometry_type = LaunchConfiguration('geometry_type', default='box')
    geometry_mass = LaunchConfiguration('geometry_mass', default=0.1)
    geometry_height = LaunchConfiguration('geometry_height', default=0.1)
    geometry_radius = LaunchConfiguration('geometry_radius', default=0.1)
    geometry_length = LaunchConfiguration('geometry_length', default=0.1)
    geometry_width = LaunchConfiguration('geometry_width', default=0.1)
    geometry_mesh_filename = LaunchConfiguration('geometry_mesh_filename', default='')
    geometry_mesh_origin_xyz = LaunchConfiguration('geometry_mesh_origin_xyz', default='"0 0 0"')
    geometry_mesh_origin_rpy = LaunchConfiguration('geometry_mesh_origin_rpy', default='"0 0 0"')
    geometry_mesh_tcp_xyz = LaunchConfiguration('geometry_mesh_tcp_xyz', default='"0 0 0"')
    geometry_mesh_tcp_rpy = LaunchConfiguration('geometry_mesh_tcp_rpy', default='"0 0 0"')
    kinematics_suffix = LaunchConfiguration('kinematics_suffix', default='')
    
    load_controller = LaunchConfiguration('load_controller', default=False)
    show_rviz = LaunchConfiguration('show_rviz', default=True)
    no_gui_ctrl = LaunchConfiguration('no_gui_ctrl', default=False)

    ros_namespace = LaunchConfiguration('ros_namespace', default='').perform(context)
    moveit_config_dump = LaunchConfiguration('moveit_config_dump', default='')

    moveit_config_dump = moveit_config_dump.perform(context)
    moveit_config_dict = yaml.load(moveit_config_dump, Loader=yaml.FullLoader) if moveit_config_dump else {}
    moveit_config_package_name = 'xarm_moveit_config'
    xarm_type = '{}{}'.format(robot_type.perform(context), dof.perform(context) if robot_type.perform(context) in ('xarm', 'lite') else '')
    
    if not moveit_config_dict:
        # ros2 control params
        ros2_control_params = generate_ros2_control_params_temp_file(
            os.path.join(get_package_share_directory('my_simulation'),'config','lite6_controllers.yaml'),
            prefix=prefix.perform(context), 
            add_gripper=add_gripper.perform(context) in ('True', 'true'),
            add_bio_gripper=add_bio_gripper.perform(context) in ('True', 'true'),
            ros_namespace=ros_namespace,
            update_rate=1000,
            use_sim_time=True,
            robot_type=robot_type.perform(context)
        )
    #----------------------------Old
         # robot_description
        robot_description = {
            'robot_description': get_xacro_content(
                context,
                xacro_file=Path(get_package_share_directory('my_simulation'))/'robot'/'xarm_device.urdf.xacro', 
                dof=dof,
                robot_type=robot_type,
                prefix=prefix,
                hw_ns=hw_ns,
                limited=limited,
                effort_control=effort_control,
                velocity_control=velocity_control,
                model1300=model1300,
                robot_sn=robot_sn,
                attach_to=attach_to,
                attach_xyz=attach_xyz,
                attach_rpy=attach_rpy,
                kinematics_suffix=kinematics_suffix,
                ros2_control_plugin=ros2_control_plugin,
                ros2_control_params=ros2_control_params,
                add_gripper=add_gripper,
                add_vacuum_gripper=add_vacuum_gripper,
                add_bio_gripper=add_bio_gripper,
                add_realsense_d435i=add_realsense_d435i,
                add_d435i_links=add_d435i_links,
                add_other_geometry=add_other_geometry,
                geometry_type=geometry_type,
                geometry_mass=geometry_mass,
                geometry_height=geometry_height,
                geometry_radius=geometry_radius,
                geometry_length=geometry_length,
                geometry_width=geometry_width,
                geometry_mesh_filename=geometry_mesh_filename,
                geometry_mesh_origin_xyz=geometry_mesh_origin_xyz,
                geometry_mesh_origin_rpy=geometry_mesh_origin_rpy,
                geometry_mesh_tcp_xyz=geometry_mesh_tcp_xyz,
                geometry_mesh_tcp_rpy=geometry_mesh_tcp_rpy,
            )
        }
        robot_description = yaml.dump(robot_description)
    #     moveit_config_dict = robot_description
    # else:
    #     robot_description = {'robot_description': moveit_config_dict['robot_description']}
    #----------------------------New
        moveit_config = MoveItConfigsBuilder(
                context=context,
                controllers_name=controllers_name,
                dof=dof,
                robot_type=robot_type,
                prefix=prefix,
                hw_ns=hw_ns,
                limited=limited,
                effort_control=effort_control,
                velocity_control=velocity_control,
                model1300=model1300,
                robot_sn=robot_sn,
                attach_to=attach_to,
                attach_xyz=attach_xyz,
                attach_rpy=attach_rpy,
                mesh_suffix='stl',
                kinematics_suffix=kinematics_suffix,
                ros2_control_plugin=ros2_control_plugin,
                ros2_control_params=ros2_control_params,
                add_gripper=add_gripper,
                add_vacuum_gripper=add_vacuum_gripper,
                add_bio_gripper=add_bio_gripper,
                add_realsense_d435i=add_realsense_d435i,
                add_d435i_links=add_d435i_links,
                add_other_geometry=add_other_geometry,
                geometry_type=geometry_type,
                geometry_mass=geometry_mass,
                geometry_height=geometry_height,
                geometry_radius=geometry_radius,
                geometry_length=geometry_length,
                geometry_width=geometry_width,
                geometry_mesh_filename=geometry_mesh_filename,
                geometry_mesh_origin_xyz=geometry_mesh_origin_xyz,
                geometry_mesh_origin_rpy=geometry_mesh_origin_rpy,
                geometry_mesh_tcp_xyz=geometry_mesh_tcp_xyz,
                geometry_mesh_tcp_rpy=geometry_mesh_tcp_rpy,
            ).to_moveit_configs()
            
        # kinematics.yaml check
        moveit_config_dict = moveit_config.to_dict()
    #     if not moveit_config_dict.get('robot_description_kinematics'):
    #         kin_path = os.path.join(
    #             get_package_share_directory(moveit_config_package_name), 'config', 'kinematics.yaml')
    #         if os.path.exists(kin_path):
    #             with open(kin_path, 'r') as kin_file:
    #                 moveit_config_dict['robot_description_kinematics'] = yaml.load(kin_file, Loader=yaml.FullLoader)
        robot_description = {'robot_description': moveit_config_dict['robot_description']}
    else:
        robot_description = {'robot_description': moveit_config_dict['robot_description']}
    
        # robot state publisher node
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'use_sim_time': True}, robot_description]
    )
    # ---------------------#
    # Gazebo launch related
    #----------------------#
    # Bridge between ROS and Ignition
    ign_bridge = Node(
        package='ros_ign_bridge',
        executable='parameter_bridge',
        arguments=[
            # Clock (IGN -> ROS2)
            '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock]'
            # # # Joint states (IGN -> ROS2)
            # '/world/default/model/SLEEK/joint_state@sensor_msgs/msg/JointState[ignition.msgs.Model',
            '/xarm/joint_states@sensor_msgs/msg/JointState[ignition.msgs.Model]',
            # # # Joint commands (ROS2 -> IGN)
            # '/lite6_traj_controller/joint_trajectory@trajectory_msgs/msg/JointTrajectory]ignition.msgs.JointTrajectory',
            # #'/lite6_traj_controller/joint_trajectory@trajectory_msgs/msg/JointTrajectory[ignition.msgs.Model',
            # # # Gripper commands (ROS2 -> IGN)

        ],
        remappings=[
             ('/xarm/joint_states', 'joint_states'),
            # ('/world/default/model/SLEEK/joint_state', 'joint_states'),
        ],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )
    #---------------ros_ign_gazebo/launch/ign_gazebo.launch.py---------------#
    xarm_gazebo_world = PathJoinSubstitution(
        [FindPackageShare('my_simulation'), 'worlds', 'custom_empty.sdf']
    )
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([FindPackageShare('ros_ign_gazebo'), 'launch', 'ign_gazebo.launch.py'])),
        launch_arguments={
            'ign_args': ' -r -v 3 {}'.format(xarm_gazebo_world.perform(context)),
            #'gz_args': ' -r -v 3 {}'.format(xarm_gazebo_world.perform(context)),
        'on_exit_shutdown': 'true'
    }.items(),
    )
    #----------------Gazebo spawn entity node----------------#
    gazebo_spawn_robot_node = Node(
        package="ros_ign_gazebo",
        executable="create",
        output='screen',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'SLEEK',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.0',
            '-Y', '0.0'
        ],
        parameters=[{'use_sim_time': True}],
    )

    #----------------Aluminium profile----------------------#
    object_20x20_profile_urdf_path = os.path.join(
        get_package_share_directory('my_simulation'), 'objects','urdf', 'F20_20_B.urdf')
    
    with open(object_20x20_profile_urdf_path, 'r') as urdf_file:
        object_20x20_profile_urdf_content = urdf_file.read()

    #object_20x20_profile_description = {
    # 'object_20x20_profile_description': object_20x20_profile_urdf_content}

    object_20x20_profile_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='object_20x20_profile_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'robot_description': object_20x20_profile_urdf_content,
            'publish_frequency': 10.0
        }],
        remappings=[
            ('robot_description', 'object_20x20_profile_description')
        ]
    )

    gazebo_spawn_object_20x20_profile_node = Node(
        package="ros_ign_gazebo",
        executable="create",
        output='screen',
        arguments=[
            '-topic', 'object_20x20_profile_description',
            '-name', '20x20_Profile',
            '-x', '1.52',
            '-y', '0.05',
            '-z', '1.0',
            '-Y', '0.0',
            '-R', '1.5708'
        ],
        parameters=[{'use_sim_time': True}]
    )
    # Testing to see if timing mismatch is the issue with the object not spawning
    # gazebo_spawn_object_20x20_profile_node_delayed = TimerAction(
    #     period=3.0,
    #     actions=[gazebo_spawn_object_20x20_profile_node]
    # )

    #--------------Rviz with moveit configuration---------------------------
    rviz_config_file = PathJoinSubstitution([FindPackageShare(moveit_config_package_name), 'rviz', 'planner.rviz' if no_gui_ctrl.perform(context) == 'true' else 'moveit.rviz'])
    rviz2_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file],
        parameters=[
            {
                **moveit_config_dict,
                # 'robot_description': moveit_config_dict.get('robot_description', ''),
                # 'robot_description_semantic': moveit_config_dict.get('robot_description_semantic', ''),
                # 'robot_description_kinematics': moveit_config_dict.get('robot_description_kinematics', {}),
                # 'robot_description_planning': moveit_config_dict.get('robot_description_planning', {}),
                # 'planning_pipelines': moveit_config_dict.get('planning_pipelines', {}),
                'use_sim_time': True
            }
        ]
    )

    #-----------------------Loading controllers----------------------#
    controllers = [
        'joint_state_broadcaster',
        '{}{}_traj_controller'.format(prefix.perform(context), xarm_type),
    ]
    if add_gripper.perform(context) in ('True', 'true'):
        controllers.append(
            '{}{}_gripper_traj_controller'.format(prefix.perform(context), xarm_type))#robot_type.perform(context)
    elif robot_type.perform(context) != 'lite' and add_bio_gripper.perform(context) in ('True', 'true'):
        controllers.append(
            '{}bio_gripper_traj_controller'.format(prefix.perform(context)))
    
    move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        name='move_group_arm',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            moveit_config_dict,# add ** if it does not work initially
            {'robot_description': moveit_config_dict.get('robot_description', '')},
            {'robot_description_semantic': moveit_config_dict.get('robot_description_semantic', '')},
            {'robot_description_kinematics': moveit_config_dict.get('robot_description_kinematics', {})},
            {'robot_description_planning': moveit_config_dict.get('robot_description_planning', {})},
            {'planning_pipelines': moveit_config_dict.get('planning_pipelines', {})},
        ]
    )
    #------------Testing for gripper control----#
    xarm_driver_node = Node(
        package='xarm_api',
        executable='xarm_driver_node',
        name='xarm_driver',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'add_gripper': True}
        ]
    )
    xarm_driver_node_delayed = TimerAction(
        period=3.0,
        actions=[xarm_driver_node]
    )
    #-----------------------------------------#
    # controller_manager_node = Node(
    #     package='controller_manager',
    #     executable='ros2_control_node',
    #     parameters=[
    #         {'use_sim_time': True},
    #         ros2_control_params,
    #     ],
    #     output='screen'
    # )

    controller_nodes = []
    if load_controller.perform(context) in ('True', 'true'):
        for controller in controllers:
            controller_nodes.append(Node(
                package='controller_manager',
                executable='spawner',
                output='screen',
                arguments=[
                    controller,
                    #'--controller-manager', 'controller_manager' if not ros_namespace else f'{ros_namespace}/controller_manager'
                    '--controller-manager', '/controller_manager' if not ros_namespace else f'{ros_namespace}/controller_manager'
                ],
                parameters=[{'use_sim_time': True}],
            ))

    # joint_state_publisher_node = Node(
    # package='joint_state_publisher',
    # executable='joint_state_publisher',
    # name='joint_state_publisher',
    # output='screen',
    # parameters=[{
    #     'robot_description': robot_description['robot_description'],
    #     'use_sim_time': True,
    #     'publish_frequency': 50.0  # 50 Hz
    # }]
    # )

    tp_control_node = Node(
    package='tp_control_cpp',
    executable='tp_control_node',
    output='screen',
    parameters=[{
        **moveit_config_dict,
        'planning_group': 'lite6',
        'ee_link': 'link_tcp', 
        'base_frame': 'link_base',
        'timing_method': 'totg',
        'z_hover': 0.20,
        'z_grasp': 0.05,
        't_hover': 3.0,
        't_grasp': 4.5,
        'vel_scale': 1.0,
        'acc_scale': 1.0,
        'use_sim_time': True
    }]
    )

    # Event Handlers
    # -------------------------------------------------------------------------------------------------------------#
    # IMPORTANT for avoiding timing mismatch
    # Order goes as follows:
    # 
    # robot_state_publisher_node -> ign_bridge -> gazebo_spawn_robot_node -> controller_nodes
    #                            -> gazebo_launch                         -> rviz2_node -> controller_manager_node
    #                            -> move_group_node
    #-------------------------------------------------------------------------------------------------------------#
    if len(controller_nodes) > 0:
        return [
            RegisterEventHandler(
                event_handler=OnProcessStart(
                    target_action=robot_state_publisher_node,
                    on_start=ign_bridge,
                )
            ),
            RegisterEventHandler(
                event_handler=OnProcessStart(
                    target_action=robot_state_publisher_node,
                    on_start=gazebo_launch,
                )
            ),
            RegisterEventHandler(
                event_handler=OnProcessStart(
                    target_action=ign_bridge,
                    on_start=gazebo_spawn_robot_node,
                )
            ),
            RegisterEventHandler(
                event_handler=OnProcessStart(
                    target_action=robot_state_publisher_node,
                    on_start=move_group_node,
                )
            ),
            RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=gazebo_spawn_robot_node,
                    on_exit=controller_nodes,
                )
            ),
            RegisterEventHandler(
                condition=IfCondition(show_rviz),
                event_handler=OnProcessExit(
                    target_action=gazebo_spawn_robot_node,
                    on_exit=rviz2_node,
                )
            ),
            RegisterEventHandler(
                event_handler=OnProcessStart(
                    target_action=gazebo_spawn_robot_node,
                    on_start=object_20x20_profile_state_publisher,
                )
            ),
            RegisterEventHandler(
                event_handler=OnProcessStart(
                    target_action=object_20x20_profile_state_publisher,
                    on_start=gazebo_spawn_object_20x20_profile_node,
                )
            ),
            # RegisterEventHandler(
            #     event_handler=OnProcessExit(
            #         target_action=rviz2_node,
            #         on_exit= controller_manager_node,
            #     )
            # ),
            # RegisterEventHandler(
            #     event_handler=OnProcessExit(
            #         target_action=gazebo_spawn_object_20x20_profile_node,
            #         on_exit= TimerAction(period=10.0, actions=[tp_control_node]),
            #     )
            # ),
            # RegisterEventHandler(
            #     event_handler=OnProcessStart(
            #         target_action=robot_state_publisher_node,
            #         on_start=joint_state_publisher_node,
            #     )
            # ),
            robot_state_publisher_node
        ]
    else:
        return [
            RegisterEventHandler(
                event_handler=OnProcessStart(
                    target_action=robot_state_publisher_node,
                    on_start=gazebo_launch,
                )
            ),
            RegisterEventHandler(
                event_handler=OnProcessStart(
                    target_action=robot_state_publisher_node,
                    on_start=ign_bridge,
                )
            ),
            RegisterEventHandler(
                event_handler=OnProcessStart(
                    target_action=robot_state_publisher_node,
                    on_start=gazebo_spawn_robot_node,
                )
            ),
            RegisterEventHandler(
                condition=IfCondition(show_rviz),
                event_handler=OnProcessExit(
                    target_action=robot_state_publisher_node,
                    on_exit=rviz2_node,
                )
            ),
            RegisterEventHandler(
                event_handler=OnProcessStart(
                    target_action=gazebo_spawn_robot_node,
                    on_start=object_20x20_profile_state_publisher,
                )
            ),
            RegisterEventHandler(
                event_handler=OnProcessStart(
                    target_action=object_20x20_profile_state_publisher,
                    on_start=gazebo_spawn_object_20x20_profile_node,
                )
            ),
            # RegisterEventHandler(
            #     event_handler=OnProcessExit(
            #         target_action=rviz2_node,
            #         on_exit= controller_manager_node,
            #     )
            # ),
            # RegisterEventHandler(
            #     event_handler=OnProcessStart(
            #         target_action=robot_state_publisher_node,
            #         on_start=joint_state_publisher_node,
            #     )
            # ),
            # RegisterEventHandler(
            #     event_handler=OnProcessExit(
            #         target_action=gazebo_spawn_object_20x20_profile_node,
            #         on_exit= TimerAction(period=10.0, actions=[tp_control_node]),
            #     )
            # ),
            robot_state_publisher_node
        ]

def generate_launch_description():
    return LaunchDescription([
        #----------Added for troubleshooting---------------#
        DeclareLaunchArgument('prefix', default_value='', description= 'Robot prefix'),
        DeclareLaunchArgument('hw_ns', default_value='ufactory', description='Hardware namespace'),
        DeclareLaunchArgument('dof', default_value='6', description='Degrees of freedom'),
        DeclareLaunchArgument('robot_type', default_value='lite', description='Type of robot'),
        DeclareLaunchArgument('add_gripper', default_value='true', description='Add gripper'),
        DeclareLaunchArgument('show_rviz', default_value='true', description='Show RViz'),
        DeclareLaunchArgument('load_controller', default_value='true', description='Load controllers'),
        DeclareLaunchArgument('no_gui_ctrl', default_value='false', description='No GUI control'),
        #--------------------------------------------------#
        SetEnvironmentVariable('IGN_GAZEBO_RESOURCE_PATH', '/usr/share/ignition/ignition-gazebo6'),
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', '/usr/share/gazebo-11/models:/usr/share/ignition/ignition-gazebo6/worlds'),
        SetEnvironmentVariable('GZ_SIM_MODEL_PATH', '/usr/share/gazebo-11/models:/usr/share/ignition/ignition-gazebo6/models'+
                               get_package_share_directory('xarm_description')),# + ':'
                               #+ get_package_share_directory('my_simulation')),
        #SetEnvironmentVariable('ROS_DOMAIN_ID', '0'), <- this is not useful, maybe
        OpaqueFunction(function=launch_setup)
    ])