#!/usr/bin/env python3
"""Launch unified AI controller with agent_type parameter."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('arena_ai_integration')

    agent_type_arg = DeclareLaunchArgument(
        'agent_type',
        default_value='socialnav',
        description='AI agent: socialnav, urbannav, lelan',
    )

    agent_type = LaunchConfiguration('agent_type')

    config_map = {
        'socialnav': os.path.join(pkg_share, 'config', 'socialnav_params.yaml'),
        'urbannav': os.path.join(pkg_share, 'config', 'urbannav_params.yaml'),
    }

    # Default config file resolved at launch time via agent_type env substitution
    default_params = [
        os.path.join(pkg_share, 'config', 'base_params.yaml'),
        os.path.join(pkg_share, 'config', 'socialnav_params.yaml'),
    ]

    controller_node = Node(
        package='arena_ai_integration',
        executable='ai_controller',
        name='ai_controller',
        output='screen',
        parameters=[
            os.path.join(pkg_share, 'config', 'base_params.yaml'),
            LaunchConfiguration('params_file'),
            {'agent_type': agent_type},
        ],
    )

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(pkg_share, 'config', 'socialnav_params.yaml'),
        description='Agent-specific parameter YAML',
    )

    return LaunchDescription([
        agent_type_arg,
        params_file_arg,
        controller_node,
    ])
