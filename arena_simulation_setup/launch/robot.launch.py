import os

import launch_ros
from arena_bringup.future import PythonExpression
from arena_bringup.substitutions import LaunchArgument
from launch.conditions import IfCondition
from launch_ros.actions import PushRosNamespace
from launch_ros.substitutions import FindPackageShare

import launch
import launch.actions
import launch.launch_description_sources
import launch.substitutions


def generate_launch_description():

    ss_path = FindPackageShare('arena_simulation_setup')

    ld_items = []
    LaunchArgument.auto_append(ld_items)

    use_sim_time = LaunchArgument("use_sim_time")

    task_generator_node = LaunchArgument('task_generator_node')
    namespace = LaunchArgument("namespace")
    robot = LaunchArgument("robot")
    frame = LaunchArgument("frame")
    agent_name = LaunchArgument('agent_name', default_value='')

    global_planner = LaunchArgument("global_planner")
    local_planner = LaunchArgument("local_planner")
    inter_planner = LaunchArgument("inter_planner", default_value="navigate_to_pose")
    map_file = LaunchArgument("map_file", default_value="")
    scenario_file = LaunchArgument("scenario_file", default_value="")

    record_data_dir = LaunchArgument('record_data_dir', default_value='')
    amcl = LaunchArgument('amcl', default_value='false')
    train_mode = LaunchArgument('train_mode', default_value='false')
    agents_dir = LaunchArgument(
        'agents_dir',
        default_value=launch.substitutions.EnvironmentVariable('ROSNAV_AGENTS_DIR', default_value=''),
        description=(
            'Base directory for agent artifacts. '
            'Forwarded as ROSNAV_AGENTS_DIR to the action server. '
            'Defaults to the ROSNAV_AGENTS_DIR env var.'
        ),
    )

    # Include the Nav2 launch file
    nav2_launch = launch.actions.IncludeLaunchDescription(
        launch.launch_description_sources.PythonLaunchDescriptionSource(
            launch.substitutions.PathJoinSubstitution(
                [
                    ss_path,
                    "launch",
                    "nav2.launch.py",
                ]
            )),
        launch_arguments={
            **use_sim_time.dict,
            **robot.dict,
            **task_generator_node.dict,
            **namespace.dict,
            **global_planner.dict,
            **local_planner.dict,
            **inter_planner.dict,
            **frame.dict,
            **amcl.dict,
            **agent_name.dict,
            **train_mode.dict,
        }.items(),
    )

    workspace_dir = os.environ.get('WORKSPACE_DIR', os.path.expanduser('~/arena5_ws'))

    # ---- SocialNav / UrbanNav paths ----
    social_nav_root = next(
        (
            candidate
            for candidate in (
                os.path.join(workspace_dir, 'src', 'arena-social-nav'),
                os.path.join(workspace_dir, 'src', 'social-nav'),
            )
            if os.path.exists(candidate)
        ),
        os.path.join(workspace_dir, 'src', 'arena-social-nav'),
    )
    socialnav_bridge_script = os.path.join(
        social_nav_root,
        'ros2_nodes',
        'socialnav',
        'human_states_bridge.py',
    )
    socialnav_controller_script = os.path.join(
        social_nav_root,
        'ros2_nodes',
        'socialnav',
        'socialnav_dwb_node.py',
    )
    urbannav_controller_script = os.path.join(
        social_nav_root,
        'ros2_nodes',
        'urbannav',
        'urbannav_dwb_node.py',
    )
    socialnav_config_path = os.path.join(social_nav_root, 'configs', 'socialnav_film.yaml')
    urbannav_config_path = os.path.join(social_nav_root, 'configs', 'urbannav_film.yaml')

    socialnav_ckpt_path = os.path.join(social_nav_root, 'ckpt', 'SocialNav_margin.pth')
    urbannav_ckpt_path = os.path.join(social_nav_root, 'ckpt', 'UrbanNav_FiLM.pth')

    # ---- CityWalker paths ----
    # Thêm block này khi tích hợp arena-citywalker
    citywalker_root = os.path.join(workspace_dir, 'src', 'arena-citywalker')
    citywalker_controller_script = os.path.join(
        citywalker_root,
        'ros2_nodes',
        'citywalker',
        'citywalker_dwb_node.py',
    )
    citywalker_config_path = os.path.join(citywalker_root, 'config', 'citywalker_one.yaml')

    data_recorder = launch_ros.actions.Node(
        package='arena_evaluation',
        executable='record',
        namespace=namespace.substitution, 
        name=PythonExpression(['"data_recorder" + "', namespace.substitution, '".replace("/","_")']),
        arguments=[
            ['--dir', ' ', record_data_dir.substitution],
        ],
        parameters=[{
            'model': robot.substitution,
            'local_planner': local_planner.substitution,
            'inter_planner': inter_planner.substitution,
            'agent_name': agent_name.substitution,
            'map_file': map_file.substitution,
            'scenario_file': scenario_file.substitution,
        }],
        condition=launch.conditions.IfCondition(PythonExpression(['bool("', record_data_dir.substitution, '")'])),
    )

    # Launch the rosnav_rl action server when using DRL local planner
    rosnav_rl_action_server = launch.actions.IncludeLaunchDescription(
        launch.launch_description_sources.PythonLaunchDescriptionSource(
            launch.substitutions.PathJoinSubstitution([
                FindPackageShare('rosnav_rl'),
                'launch',
                'action_server.launch.py',
            ])
        ),
        launch_arguments={
            'agent_name': agent_name.substitution,
            'namespace': namespace.substitution,
            'agents_dir': agents_dir.substitution,
        }.items(),
        condition=IfCondition(
            PythonExpression([
                "('", local_planner.substitution, "' == 'rosnav_rl' or '",
                local_planner.substitution, "' == 'rosnav') and '",
                train_mode.substitution, "' == 'false'"
            ])
        ),
    )

    # ---- SocialNav bridge (human detection) ----
    socialnav_bridge = launch.actions.ExecuteProcess(
        cmd=[
            'python3',
            socialnav_bridge_script,
            '--ros-args',
            '-r',
            PythonExpression([
                '"__node:=human_states_bridge_" + "',
                namespace.substitution,
                '".strip("/").replace("/", "_")'
            ]),
        ],
        output='screen',
        condition=IfCondition(
            PythonExpression([
                '"', agent_name.substitution, '".startswith("SocialNav") and "',
                local_planner.substitution, '" == "dwb" and "',
                train_mode.substitution, '" == "false"'
            ])
        ),
    )

    # ---- SocialNav controller ----
    socialnav_controller = launch.actions.ExecuteProcess(
        cmd=[
            'python3', socialnav_controller_script,
            '--ros-args',
            '-r', PythonExpression(['"__node:=socialnav_dwb_controller_" + "', namespace.substitution, '".strip("/").replace("/", "_")']),
            '-p', f'model_config_path:={socialnav_config_path}',
            '-p', PythonExpression(['"agent_name:=', agent_name.substitution, '"']),
            '-p', 'history_length:=8',
            '-p', 'control_frequency:=10.0',
            '-p', 'look_ahead_distance:=0.5',
            '-p', 'max_linear_velocity:=1.0',
            '-p', 'max_angular_velocity:=1.5',
            '-p', 'arrival_threshold:=0.25',
            '-p', 'use_arrival_completion:=false',
            '-p', 'goal_completion_radius:=0.25',
            '-p', 'enable_human_tracking:=true',
            '-p', 'max_humans:=10',
            '-p', PythonExpression(['"robot_namespace:=', namespace.substitution, '"']),
            '-p', 'instruction_topic:=/nav_instruction',
        ],
        output='screen',
        condition=IfCondition(
            PythonExpression([
                '"', agent_name.substitution, '".startswith("SocialNav") and "',
                local_planner.substitution, '" == "dwb" and "',
                train_mode.substitution, '" == "false"'
            ])
        ),
    )

    # ---- UrbanNav controller ----
    urbannav_controller = launch.actions.ExecuteProcess(
        cmd=[
            'python3',
            urbannav_controller_script,
            '--ros-args',
            '-r',
            PythonExpression([
                '"__node:=urbannav_dwb_controller_" + "',
                namespace.substitution,
                '".strip("/").replace("/", "_")'
            ]),
            '-p',
            f'model_config_path:={urbannav_config_path}',
            '-p',
            PythonExpression(['"agent_name:=', agent_name.substitution, '"']),
            '-p', 'history_length:=8',
            '-p', 'control_frequency:=10.0',
            '-p', 'look_ahead_distance:=0.5',
            '-p', 'max_linear_velocity:=1.0',
            '-p', 'max_angular_velocity:=1.5',
            '-p', 'arrival_threshold:=0.7',
            '-p', 'use_arrival_completion:=false',
            '-p', 'goal_completion_radius:=0.25',
            '-p',
            PythonExpression(['"robot_namespace:=', namespace.substitution, '"']),
            '-p', 'instruction_topic:=/nav_instruction',
        ],
        output='screen',
        condition=IfCondition(
            PythonExpression([
                '"', agent_name.substitution, '".startswith("UrbanNav") and "',
                local_planner.substitution, '" == "dwb" and "',
                train_mode.substitution, '" == "false"'
            ])
        ),
    )

    # ---- CityWalker controller ----
    # Thêm block này khi tích hợp arena-citywalker.
    # Điều kiện: agent_name phải bắt đầu bằng "CityWalker"
    citywalker_controller = launch.actions.ExecuteProcess(
        cmd=[
            'python3',
            citywalker_controller_script,
            '--ros-args',
            '-r',
            PythonExpression([
                '"__node:=citywalker_controller_" + "',
                namespace.substitution,
                '".strip("/").replace("/", "_")'
            ]),
            '-p', f'model_config_path:={citywalker_config_path}',
            '-p',
            PythonExpression(['"agent_name:=', agent_name.substitution, '"']),
            '-p', 'control_frequency:=10.0',
            '-p', 'look_ahead_distance:=0.5',
            '-p', 'max_linear_velocity:=1.0',
            '-p', 'max_angular_velocity:=1.5',
            '-p', 'use_arrival_completion:=false',
            '-p', 'goal_completion_radius:=0.25',
            '-p',
            PythonExpression(['"robot_namespace:=', namespace.substitution, '"']),
            '-p', 'instruction_topic:=/nav_instruction',
        ],
        output='screen',
        condition=IfCondition(
            PythonExpression([
                '"', agent_name.substitution, '".startswith("CityWalker") and "',
                local_planner.substitution, '" == "dwb" and "',
                train_mode.substitution, '" == "false"'
            ])
        ),
    )

    # ---- Pure DWB Baseline Controller (khi không dùng AI) ----
    # pure_dwb_controller = launch_ros.actions.Node(
    #     package='nav2_controller',
    #     executable='controller_server',
    #     name='controller_server',
    #     namespace=namespace.substitution,
    #     output='screen',
    #     parameters=[{
    #         'use_sim_time': True,
    #         'controller_server.ros__parameters': {
    #             'controller_plugins': ['FollowPath'],
    #             'FollowPath': {
    #                 'plugin': 'dwb_core::DWBLocalPlanner',
    #                 'max_vel_x': 1.0,
    #                 'min_vel_x': -0.5,
    #                 'max_vel_theta': 2.0,
    #             }
    #         }
    #     }],
    #     condition=IfCondition(
    #         PythonExpression(['"', agent_name.substitution, '" == ""'])
    #     ),
    # )

    # ---- Thêm block cấu hình và khởi chạy Hunav Evaluator ----
    metrics_config_path = os.path.join(workspace_dir, 'results', 'metrics.yaml')

    hunav_evaluator_node = launch_ros.actions.Node(
        package='hunav_evaluator',
        executable='hunav_evaluator_node',
        name='hunav_evaluator_node',
        # Ép node này chạy ở root namespace để khớp với service mapping của benchmark.py
        namespace='/', 
        parameters=[metrics_config_path],
        output='screen',
        # Tùy chọn: Bạn có thể thêm condition nếu chỉ muốn bật evaluator khi đang lưu data
        # condition=launch.conditions.IfCondition(PythonExpression(['bool("', record_data_dir.substitution, '")']))
    )
    # ---------------------------------------------------------

    lidar_relay = launch_ros.actions.Node(
        package='topic_tools',
        executable='relay',
        name='lidar_to_scan_relay',
        arguments=['lidar', 'scan'], 
        output='screen',
    )

    ld = launch.LaunchDescription([
        *ld_items,
        launch.actions.DeclareLaunchArgument(
            name='complexity',
            default_value='1'
        ),
        PushRosNamespace(namespace=namespace.substitution),
        lidar_relay,
        nav2_launch,
        rosnav_rl_action_server,
        socialnav_bridge,
        socialnav_controller,
        urbannav_controller,
        citywalker_controller,   
        # pure_dwb_controller,
        data_recorder,
        hunav_evaluator_node, 
    ])
    return ld


if __name__ == '__main__':
    print("[INFO] Robot is launching ... ")
    generate_launch_description()
