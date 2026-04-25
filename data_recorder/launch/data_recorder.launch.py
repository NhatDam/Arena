from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Launch the data recorder node."""

    data_recorder_node = Node(
        package="data_recorder",
        executable="data_recorder_node",
        output="screen",
        parameters=[
            # Optional: override topic names if needed
            # {"image_topic": "/custom/image_topic"},
            # {"depth_topic": "/custom/depth_topic"},
            # {"detections_topic": "/custom/detections_topic"},
            # {"robot_state_topic": "/custom/robot_state_topic"},
        ],
    )

    ld = LaunchDescription()
    ld.add_action(data_recorder_node)

    return ld
