from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from arena_bringup.substitutions import LaunchArgument


def generate_launch_description():
    # Launch Arguments
    use_sim_time = LaunchArgument('use_sim_time', default_value='true')
    namespace = LaunchArgument('namespace')

    # Metrics configuration for hunav evaluator
    metrics_file = PathJoinSubstitution([
        FindPackageShare('hunav_evaluator'),
        'config',
        'metrics.yaml',
    ])

    return LaunchDescription([
        use_sim_time,
        namespace,

        # Agent Manager Node
        Node(
            package='hunav_agent_manager',
            executable='arena_hunav_agent_manager',
            namespace=namespace.substitution,
            name='hunav_agent_manager',
            output='screen',
            parameters=[
                use_sim_time.param(bool)
            ]
        ),

        # Hunav Evaluator Node – records metrics per episode via service calls
        Node(
            package='hunav_evaluator',
            executable='hunav_evaluator_node',
            name='hunav_evaluator_node',
            output='screen',
            parameters=[metrics_file],
            remappings=[
                ('human_states', [namespace.substitution, '/human_states']),
                ('robot_states', [namespace.substitution, '/robot_states']),
            ],
        ),
    ])
