#!/usr/bin/env python3
"""
Bridge: /task_generator_node/human_states → /detections/humans (JSON)
Tọa độ giữ nguyên map frame để semantic_laser_filter dùng với tf2 transform.
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json

# Arena message type - thử import, fallback nếu không có
try:
    from pedsim_msgs.msg import AgentStates
    MSG_TYPE = AgentStates
    USE_PEDSIM = True
except ImportError:
    try:
        from arena_msgs.msg import AgentStates
        MSG_TYPE = AgentStates
        USE_PEDSIM = True
    except ImportError:
        USE_PEDSIM = False


class HumanStatesBridge(Node):
    def __init__(self):
        super().__init__('human_states_bridge')

        self.declare_parameter('task_namespace', '/task_generator_node')
        self.declare_parameter('output_topic', '/detections/humans')

        task_namespace = self.get_parameter('task_namespace').value.strip('/')
        output_topic = self.get_parameter('output_topic').value
        human_states_topic = f'/{task_namespace}/human_states'
        pedestrian_markers_topic = f'/{task_namespace}/pedestrian_markers'

        self.pub = self.create_publisher(String, output_topic, 10)

        if USE_PEDSIM:
            self.create_subscription(
                MSG_TYPE,
                human_states_topic,
                self.cb, 10)
            self.get_logger().info(
                f"Bridge started (AgentStates mode): {human_states_topic} -> {output_topic}"
            )
        else:
            # Fallback: dùng topic /task_generator_node/pedestrian_markers (MarkerArray)
            from visualization_msgs.msg import MarkerArray
            self.create_subscription(
                MarkerArray,
                pedestrian_markers_topic,
                self.marker_cb, 10)
            self.get_logger().info(
                f"Bridge started (MarkerArray fallback mode): {pedestrian_markers_topic} -> {output_topic}"
            )

    def cb(self, msg):
        """Parse AgentStates, lấy position của từng agent type=1 (người)"""
        positions = []
        for agent in msg.agents:
            if agent.type == 1:  # type 1 = pedestrian
                x = agent.position.position.x
                y = agent.position.position.y
                positions.append([x, y])

        self._publish(positions)

    def marker_cb(self, msg):
        """
        Fallback: parse MarkerArray, lấy markers thuộc ns='pedestrian_meshes'
        vì chúng có pose.position = vị trí thực của người.
        """
        positions = []
        for marker in msg.markers:
            if marker.ns == 'pedestrian_meshes' and marker.action == 0:
                x = marker.pose.position.x
                y = marker.pose.position.y
                positions.append([x, y])

        self._publish(positions)

    def _publish(self, positions):
        if not positions:
            return
        msg = String()
        msg.data = json.dumps(positions)
        self.pub.publish(msg)
        self.get_logger().info(
            f"[Bridge] Published {len(positions)} humans",
            throttle_duration_sec=2.0
        )


def main(args=None):
    rclpy.init(args=args)
    node = HumanStatesBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
