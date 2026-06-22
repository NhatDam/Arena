import glob
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


def _build_ai_library_path(ai_python: str) -> str:
    paths = []

    configured_path = os.environ.get('SOCIALNAV_AI_RUNTIME_LIBRARY_PATH', '')
    if configured_path:
        paths.extend(configured_path.split(':'))

    if os.path.isabs(ai_python):
        ai_prefix = os.path.abspath(os.path.join(os.path.dirname(ai_python), os.pardir))
        paths.append(os.path.join(ai_prefix, 'lib'))
        for site_packages in glob.glob(os.path.join(ai_prefix, 'lib', 'python*', 'site-packages')):
            paths.extend(sorted(glob.glob(os.path.join(site_packages, 'nvidia', '*', 'lib'))))

    seen = set()
    valid = []
    for path in paths:
        real_path = os.path.realpath(path) if path else ''
        if real_path and os.path.isdir(real_path) and real_path not in seen:
            seen.add(real_path)
            valid.append(real_path)

    return ':'.join(valid)


def generate_launch_description():

    workspace_dir = os.environ.get('WORKSPACE_DIR', os.path.expanduser('~/arena5_ws'))
    source_sim_setup_root = os.path.join(workspace_dir, 'src', 'Arena', 'arena_simulation_setup')
    ss_path = (
        source_sim_setup_root
        if os.path.exists(source_sim_setup_root)
        else FindPackageShare('arena_simulation_setup')
    )

    ld_items = []
    LaunchArgument.auto_append(ld_items)

    use_sim_time = LaunchArgument("use_sim_time")

    task_generator_node = LaunchArgument('task_generator_node')
    namespace = LaunchArgument("namespace")
    robot = LaunchArgument("robot")
    frame = LaunchArgument("frame")
    agent_name = LaunchArgument('agent_name', default_value='')
    goal_tolerance_radius = LaunchArgument('goal_tolerance_radius', default_value='0.25')

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

    ai_python = os.environ.get('SOCIALNAV_AI_PYTHON', 'python3')
    ai_process_env = {
        'PYTHONUNBUFFERED': '1',
        'RCUTILS_LOGGING_BUFFERED_STREAM': '0',
    }
    ai_cuda_visible_devices = os.environ.get('AI_CUDA_VISIBLE_DEVICES', '').strip()
    if ai_cuda_visible_devices:
        ai_process_env['CUDA_VISIBLE_DEVICES'] = ai_cuda_visible_devices
    ai_python_no_user_site = os.environ.get('SOCIALNAV_AI_PYTHONNOUSERSITE', '1')
    if ai_python_no_user_site:
        ai_process_env['PYTHONNOUSERSITE'] = ai_python_no_user_site
    ai_library_path = _build_ai_library_path(ai_python)
    if ai_library_path:
        current_library_path = os.environ.get('LD_LIBRARY_PATH', '')
        ai_process_env['LD_LIBRARY_PATH'] = (
            f'{ai_library_path}:{current_library_path}'
            if current_library_path
            else ai_library_path
        )

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

    # ---- Social AI bridge (human detection) ----
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
                '("', agent_name.substitution, '".startswith("SocialNav") or "',
                agent_name.substitution, '".startswith("UrbanNav") or "',
                agent_name.substitution, '".startswith("LeLan") or "',
                agent_name.substitution, '".startswith("LeLaN")) and "',
                local_planner.substitution, '" == "dwb" and "',
                train_mode.substitution, '" == "false"'
            ])
        ),
    )

    # ---- Unified AI + DWB FollowPath adapter ----
    ai_controller = launch.actions.ExecuteProcess(
        cmd=[
            ai_python,
            '-m',
            'arena_ai_integration.nodes.ai_controller_node',
            '--ros-args',
            '-r',
            PythonExpression([
                '"__node:=ai_controller_" + "',
                namespace.substitution,
                '".strip("/").replace("/", "_")'
            ]),
            '-p', PythonExpression(['"use_sim_time:=', use_sim_time.substitution, '"']),
            '-p',
            PythonExpression([
                '"agent_type:=socialnav" if "',
                agent_name.substitution,
                '".startswith("SocialNav") else "agent_type:=urbannav" if "',
                agent_name.substitution,
                '".startswith("UrbanNav") else "agent_type:=lelan"'
            ]),
            '-p', PythonExpression(['"agent_name:=', agent_name.substitution, '"']),
            '-p', 'history_length:=8',
            '-p', 'control_frequency:=10.0',
            '-p', 'look_ahead_distance:=0.5',
            '-p',
            PythonExpression([
                '"max_linear_velocity:=0.5" if ("',
                agent_name.substitution,
                '".startswith("LeLan") or "',
                agent_name.substitution,
                '".startswith("LeLaN")) else "max_linear_velocity:=1.0"'
            ]),
            '-p',
            PythonExpression([
                '"max_angular_velocity:=1.0" if ("',
                agent_name.substitution,
                '".startswith("LeLan") or "',
                agent_name.substitution,
                '".startswith("LeLaN")) else "max_angular_velocity:=1.5"'
            ]),
            '-p', 'fallback_to_dwb:=true',
            '-p', 'reset_on_eval_dropout:=false',
            '-p', 'initial_eval_wait_sec:=5.0',
            '-p', 'max_eval_staleness_sec:=5.0',
            '-p', 'startup_data_timeout_sec:=30.0',
            '-p', 'enable_bev_visualization:=true',
            '-p', 'bev_visualization_period_sec:=1.0',
            '-p', 'dwb_cmd_staleness_sec:=1.0',
            '-p', PythonExpression(['"arrival_threshold:=', goal_tolerance_radius.substitution, '"']),
            '-p', 'use_arrival_completion:=false',
            '-p', PythonExpression(['"goal_completion_radius:=', goal_tolerance_radius.substitution, '"']),
            '-p',
            PythonExpression([
                '"enable_human_tracking:=true" if "',
                agent_name.substitution,
                '".startswith("SocialNav") else "enable_human_tracking:=false"'
            ]),
            '-p', 'max_humans:=10',
            '-p', PythonExpression(['"robot_namespace:=', namespace.substitution, '"']),
            '-p', PythonExpression(['"image_topic:=', namespace.substitution, '/rgbd_camera/image"']),
            '-p', PythonExpression(['"dwb_cmd_topic:=', namespace.substitution, '/cmd_vel_nav_raw"']),
            '-p', 'instruction_topic:=/nav_instruction',
            '-p', 'coordinate_mode:=xz_to_ros',
        ],
        output='screen',
        condition=IfCondition(
            PythonExpression([
                '("', agent_name.substitution, '".startswith("SocialNav") or "',
                agent_name.substitution, '".startswith("UrbanNav") or "',
                agent_name.substitution, '".startswith("LeLan") or "',
                agent_name.substitution, '".startswith("LeLaN")) and "',
                local_planner.substitution, '" == "dwb" and "',
                train_mode.substitution, '" == "false"'
            ])
        ),
        additional_env=ai_process_env,
    )

    # ---- CityWalker controller ----
    # Thêm block này khi tích hợp arena-citywalker.
    # Điều kiện: agent_name phải bắt đầu bằng "CityWalker"
    citywalker_controller = launch.actions.ExecuteProcess(
        cmd=[
            ai_python,
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
            '-p', 'allow_pure_pursuit_fallback:=true',
            '-p', 'use_arrival_completion:=false',
            '-p', PythonExpression(['"goal_completion_radius:=', goal_tolerance_radius.substitution, '"']),
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
        additional_env=ai_process_env,
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
        ai_controller,
        citywalker_controller,
        # pure_dwb_controller,
        data_recorder,
    ])
    return ld


if __name__ == '__main__':
    print("[INFO] Robot is launching ... ")
    generate_launch_description()
