import os

from arena_bringup.future import IfElseSubstitution, PythonExpression
from arena_bringup.substitutions import (LaunchArgument, YAMLFileSubstitution,
                                         YAMLMergeSubstitution,
                                         YAMLReplaceSubstitution,
                                         YAMLRetrieveSubstitution)
from launch import LaunchDescription
from launch.actions import GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node, SetRemap
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    workspace_dir = os.environ.get('WORKSPACE_DIR', os.path.expanduser('~/arena5_ws'))
    source_ss_root = os.path.join(workspace_dir, 'src', 'Arena', 'arena_simulation_setup')
    source_robots_root = os.path.join(workspace_dir, 'src', 'Arena', 'arena_robots', 'arena_robots')

    ss_root = source_ss_root if os.path.exists(source_ss_root) else FindPackageShare('arena_simulation_setup')
    robots_root = source_robots_root if os.path.exists(source_robots_root) else FindPackageShare('arena_robots')
    pkg_nav2_bringup = FindPackageShare('nav2_bringup')

    ld_items = []
    LaunchArgument.auto_append(ld_items)

    robot = LaunchArgument('robot')
    task_generator_node = LaunchArgument('task_generator_node')
    namespace = LaunchArgument('namespace')
    frame = LaunchArgument('frame')
    use_sim_time = LaunchArgument('use_sim_time')
    global_planner = LaunchArgument('global_planner')
    local_planner = LaunchArgument('local_planner')
    inter_planner = LaunchArgument('inter_planner')

    amcl = LaunchArgument('amcl')
    agent_name = LaunchArgument('agent_name', default_value='')
    train_mode = LaunchArgument('train_mode', default_value='false')

    ai_dwb_agent = PythonExpression([
        '("', agent_name.substitution, '".strip().startswith("SocialNav") or ',
        '"', agent_name.substitution, '".strip().startswith("UrbanNav") or ',
        '"', agent_name.substitution, '".strip().startswith("CityWalker") or ',
        '"', agent_name.substitution, '".strip().startswith("LeLan") or ',
        '"', agent_name.substitution, '".strip().startswith("LeLaN"))'
    ])
    ai_hybrid_dwb = PythonExpression([
        '("', local_planner.substitution, '".strip().lower() == "dwb" and ',
        '"', train_mode.substitution, '".strip().lower() in ("false", "0", "no", "") and ',
        ai_dwb_agent,
        ')'
    ])
    suppress_nav_cmd_vel = PythonExpression([
        '("', train_mode.substitution, '".strip().lower() in ("true", "1", "yes") or ',
        ai_hybrid_dwb,
        ')'
    ])

    substitutions = YAMLMergeSubstitution(
        YAMLFileSubstitution(
            PathJoinSubstitution([
                ss_root,
                'configs',
                'nav2',
                'model_params.yaml'
            ])
        ),
        YAMLFileSubstitution(
            PathJoinSubstitution([
                robots_root,
                'robots',
                robot.substitution,
                'model_params.yaml'
            ])
        ),
        # Load controller-specific configuration based on local_planner argument
        YAMLFileSubstitution(
            PathJoinSubstitution([
                ss_root,
                'configs',
                'nav2',
                'controllers',
                local_planner.substitution,
                'controller_config.yaml'
            ])
        ),
        # Load robot-specific controller overrides (optional, falls back to empty dict)
        YAMLFileSubstitution(
            PathJoinSubstitution([
                robots_root,
                'robots',
                robot.substitution,
                'configs',
                'controller_config.yaml'
            ]),
            default={},
        ),
        # Load controller-specific configuration based on global_planner argument
        YAMLFileSubstitution(
            PathJoinSubstitution([
                ss_root,
                'configs',
                'nav2',
                'planners',
                global_planner.substitution,
                'planner_config.yaml'
            ])
        ),
        YAMLFileSubstitution(
            PathJoinSubstitution([
                ss_root,
                'configs',
                'nav2',
                'interplanners',
                inter_planner.substitution,
                'interplanner_config.yaml'
            ])
        ),
        YAMLFileSubstitution.from_dict(
            {
                'frame': frame.substitution,
                **task_generator_node.dict,
                'namespace': namespace.substitution,
                # In train_mode and AI-hybrid DWB modes, avoid overwriting
                # controller-side velocity commands on the robot cmd_vel topic.
                'cmd_vel_out_topic': IfElseSubstitution(
                    suppress_nav_cmd_vel,
                    'cmd_vel_sink',
                    'cmd_vel',
                ),
                'default_nav_to_pose_bt_xml': YAMLRetrieveSubstitution(
                    YAMLFileSubstitution(
                        PathJoinSubstitution([
                            ss_root,
                            'configs',
                            'nav2',
                            'interplanners',
                            inter_planner.substitution,
                            'interplanner_config.yaml'
                        ])
                    ),
                    'bt_navigator/ros__parameters/default_nav_to_pose_bt_xml'
                ),
                'default_nav_through_poses_bt_xml': YAMLRetrieveSubstitution(
                    YAMLFileSubstitution(
                        PathJoinSubstitution([
                            ss_root,
                            'configs',
                            'nav2',
                            'interplanners',
                            inter_planner.substitution,
                            'interplanner_config.yaml'
                        ])
                    ),
                    'bt_navigator/ros__parameters/default_nav_through_poses_bt_xml'
                ),
                'plugin_lib_names': YAMLRetrieveSubstitution(
                    YAMLFileSubstitution(
                        PathJoinSubstitution([
                            ss_root,
                            'configs',
                            'nav2',
                            'interplanners',
                            inter_planner.substitution,
                            'interplanner_config.yaml'
                        ])
                    ),
                    'bt_navigator/ros__parameters/plugin_lib_names'
                ),
            },
            substitute=True
        ),
    )

    substituted_parameters = YAMLReplaceSubstitution(
        obj=YAMLFileSubstitution(
            PathJoinSubstitution([
                ss_root,
                'configs',
                'nav2',
                'nav2.yaml'
            ])
        ),
        substitutions=YAMLFileSubstitution(substitutions)
    )

    robot_base_frame = YAMLRetrieveSubstitution(
        YAMLFileSubstitution(substitutions),
        os.path.join('robot_base_frame'),
    )

    robot_odom_frame = YAMLRetrieveSubstitution(
        YAMLFileSubstitution(substitutions),
        os.path.join('robot_odom_frame'),
    )

    first_observation_source_topic = YAMLRetrieveSubstitution(
        YAMLFileSubstitution(
            YAMLReplaceSubstitution(
                obj=YAMLFileSubstitution(substitutions),
                substitutions=YAMLFileSubstitution(substitutions)
            )
        ),
        PathJoinSubstitution([
            'observation_sources_dict',
            YAMLRetrieveSubstitution(
                YAMLFileSubstitution(substitutions),
                os.path.join('observation_sources', '0'),
            ),
            'topic',
        ]),
    )

    controller_cmd_topic = IfElseSubstitution(
        ai_hybrid_dwb,
        'cmd_vel_nav_raw',
        'cmd_vel_nav',
    )

    remappings = [
        ('map_server', '/map_server'),
        ('/tf', '/tf'),
        ('/tf_static', '/tf_static'),
        ('map', PathJoinSubstitution([task_generator_node.substitution, 'map'])),
    ]

    bringup_cmd_group = GroupAction([
        *(SetRemap(src=r[0], dst=r[1]) for r in remappings),
        Node(
            package='topic_tools',
            executable='relay',
            name='goal_pose_relay',
            arguments=['/goal_pose', 'goal_pose'],
        ),
        # Node(
        #     package='tf2_ros',
        #     executable='static_transform_publisher',
        #     name='map_broadcaster',
        #     arguments=['0', '0', '0', '0', '0', '0', 'world', 'map'],
        #     parameters=[{'use_sim_time': use_sim_time.substitution}]
        # ),
        # TF publishers
        # Node(
        #     package="tf2_ros",
        #     executable="static_transform_publisher",
        #     name="odomframe_to_baseframe_publisher",
        #     namespace=namespace.substitution,
        #     arguments=["0", "0", "0", "0", "0", "0", [frame.substitution, robot_odom_frame], [frame.substitution, robot_base_frame]],
        #     parameters=[use_sim_time.dict],
        # ),
        # Node(
        #     package='robot_localization',
        #     executable='ekf_node',
        #     name='ekf_node',
        #     output='screen',
        #     parameters=[{
        #         'use_sim_time': True,
        #         'odom0': PathJoinSubstitution([task_generator_node.substitution, frame.substitution, robot_odom_frame]),
        #         'odom0_config': [False, False, False, False, False, False, True, True, False, False, False, True, False, False, False],
        #         'imu0': PathJoinSubstitution([task_generator_node.substitution, frame.substitution, 'imu/data']),
        #         'imu0_config': [False, False, False, False, False, False, True, True, True, False, False, True, False, False, False],
        #         'odom_frame': PathJoinSubstitution([frame.substitution, robot_odom_frame]),
        #         'base_link_frame': PathJoinSubstitution([frame.substitution, robot_base_frame]),
        #         'world_frame': PathJoinSubstitution([frame.substitution, robot_odom_frame]),
        #         'map_frame': 'map',
        #         'publish_tf': True,
        #         "two_d_mode": True,
        #         "frequency": 30.0,
        #         "transform_time_offset": 0.0,
        #         "transform_timeout": 0.1,
        #     }],
        # ),
        # AMCL Node
        # GroupAction([
        #     Node(
        #         package='nav2_amcl',
        #         executable='amcl',
        #         name='amcl',
        #         namespace=namespace.substitution,
        #         output='screen',
        #         parameters=[{
        #             'use_sim_time': True,
        #             'global_frame_id': 'map',
        #             'map_topic': 'map',
        #             'odom_frame_id': PathJoinSubstitution([frame.substitution, robot_odom_frame]),
        #             'base_frame_id': PathJoinSubstitution([frame.substitution, robot_base_frame]),
        #             'scan_topic': first_observation_source_topic,
        #             'set_initial_pose': True,
        #             'always_reset_on_initial_pose': True,
        #             'initial_pose_x': 0.0,
        #             'initial_pose_y': 0.0,
        #             'initial_pose_a': 0.0,
        #             'odom_model_type': 'diff',
        #             'alpha1': 0.2,
        #             'alpha2': 0.2,
        #             'alpha3': 0.2,
        #             'alpha4': 0.2,
        #             'min_particles': 200,
        #             'max_particles': 2000,
        #             'pf_err': 0.05,
        #             'pf_z': 0.99,
        #             'update_min_d': 0.1,
        #             'update_min_a': 0.2,
        #             'laser_max_beams': 640,
        #             'laser_z_hit': 0.95,
        #             'laser_z_rand': 0.05,
        #             'laser_sigma_hit': 0.2,
        #             'laser_likelihood_max_dist': 2.0,
        #             'recovery_alpha_slow': 0.001,
        #             'recovery_alpha_fast': 0.1,
        #             'transform_tolerance': 0.2,
        #             'tf_broadcast': True,
        #         }],
        #         remappings=[
        #             ('/map', PathJoinSubstitution([task_generator_node.substitution, 'map'])),
        #             ('initialpose', PathJoinSubstitution([task_generator_node.substitution, frame.substitution, 'initialpose'])),
        #         ],
        #     ),

        #     # Lifecycle Manager for AMCL
        #     Node(
        #         package='nav2_lifecycle_manager',
        #         executable='lifecycle_manager',
        #         name='lifecycle_manager_navigation',
        #         output='screen',
        #         parameters=[
        #             {'use_sim_time': True},
        #             {'autostart': True},
        #             {'node_names': ["amcl"]},
        #             {'bond_timeout': 0.0},
        #             {'attempt_respawn_reconnection': True}
        #         ]
        #     )],
        #     condition=IfCondition(PythonExpression(["'", amcl.substitution, "' == 'true'"]))
        # ),
        # # Navigation Stack
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution(
                    [
                        pkg_nav2_bringup,
                        'launch',
                        'navigation_launch.py'
                    ]
                )
            ),
            launch_arguments={
                **use_sim_time.dict,
                'local_planner': local_planner.substitution,
                'inter_planner': inter_planner.substitution,
                'autostart': 'True',
                'params_file': substituted_parameters,
                'use_composition': 'False',
                'controller_cmd_topic': controller_cmd_topic,
            }.items()
        ),
    ])

    # Create the launch description and populate
    ld = LaunchDescription([
        *ld_items,
        bringup_cmd_group,
    ])

    return ld
