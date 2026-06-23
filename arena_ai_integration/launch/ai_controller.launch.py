#!/usr/bin/env python3
"""Launch unified AI controller with agent_type parameter."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg_share = get_package_share_directory('arena_ai_integration')

    agent_type_arg = DeclareLaunchArgument(
        'agent_type',
        default_value='socialnav',
        description='AI agent: socialnav, urbannav, lelan',
    )

    agent_type = LaunchConfiguration('agent_type')

    ai_python = os.environ.get('ARENA_AI_PYTHON', 'python3')

    controller_node = ExecuteProcess(
        cmd=[
            ai_python,
            '-m',
            'arena_ai_integration.nodes.ai_controller_node',
            '--ros-args',
            '-r',
            '__node:=ai_controller',
            '-p',
            ['agent_type:=', agent_type],
            '--params-file',
            os.path.join(pkg_share, 'config', 'base_params.yaml'),
            '--params-file',
            LaunchConfiguration('params_file'),
        ],
        output='screen',
    )

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(pkg_share, 'config', 'socialnav_params.yaml'),
        description='Agent-specific parameter YAML: socialnav_params, urbannav_params, or lelan_params',
    )

    return LaunchDescription([
        agent_type_arg,
        params_file_arg,
        controller_node,
    ])
