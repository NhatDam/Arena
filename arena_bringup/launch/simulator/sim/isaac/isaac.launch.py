from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

from arena_bringup.substitutions import LaunchArgument
from launch import LaunchDescription


def generate_launch_description():
    ld = []
    LaunchArgument.auto_append(ld)
    log_level = LaunchArgument(
        name='log_level',
        default_value='debug',
        description='Logging level',
    )
    headless = LaunchArgument(
        name='headless',
        default_value='False',
        description='Run Isaac Sim in headless mode (no GUI)',
    )
    return LaunchDescription([
        *ld,
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('arena_isaac'),
                    'launch',
                    'run_isaacsim.launch.py'
                ]),
            ),
            launch_arguments={
                **log_level.dict,
                **headless.dict,
            }.items(),
        )
    ])
