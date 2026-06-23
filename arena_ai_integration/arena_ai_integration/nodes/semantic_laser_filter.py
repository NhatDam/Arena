#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import json
import math
import numpy as np

import tf2_ros
from geometry_msgs.msg import PointStamped
import tf2_geometry_msgs  # cần import để register transform


class SemanticLaserFilter(Node):
    def __init__(self):
        super().__init__('semantic_laser_filter')
        
        self.human_positions_map = []  # tọa độ global (map frame)
        self.clear_radius = 0.5

        # TF2 buffer để transform map → base_link
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(String, '/detections/humans', self.humans_cb, 10)
        self.create_subscription(
            LaserScan,
            '/task_generator_node/turtlebot/lidar',
            self.laser_cb,
            qos_profile_sensor_data)

        self.laser_pub = self.create_publisher(
            LaserScan,
            '/task_generator_node/turtlebot/lidar_static',
            10)

        self.get_logger().info("✅ Semantic Laser Filter Started (use_sim_time=True)!")

    def humans_cb(self, msg):
        try:
            self.human_positions_map = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"Lỗi parse JSON: {e}")

    def _transform_humans_to_robot_frame(self, stamp):
        """
        Chuyển tọa độ người từ map frame → base_link frame.
        Trả về list [[lx, ly], ...] trong robot frame.
        """
        local_positions = []
        try:
            # Lấy transform map --> base_link tại thời điểm scan
            t = self.tf_buffer.lookup_transform(
                'turtlebot/base_link',  
                'map',         
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05)
            )
        except Exception as e:
            self.get_logger().warn(
                f"TF lookup failed: {e}",
                throttle_duration_sec=2.0)
            return local_positions

        for h_pos in self.human_positions_map:
            pt = PointStamped()
            pt.header.frame_id = 'map'
            pt.header.stamp    = stamp
            pt.point.x = float(h_pos[0])
            pt.point.y = float(h_pos[1])
            pt.point.z = 0.0
            try:
                pt_local = tf2_geometry_msgs.do_transform_point(pt, t)
                local_positions.append([pt_local.point.x, pt_local.point.y])
            except Exception:
                pass

        return local_positions

    def laser_cb(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)

        if self.human_positions_map:
            # Transform sang robot frame
            local_humans = self._transform_humans_to_robot_frame(msg.header.stamp)

            if local_humans:
                angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
                valid  = ~(np.isinf(ranges) | np.isnan(ranges))
                ray_x  = np.where(valid, ranges * np.cos(angles), np.nan)
                ray_y  = np.where(valid, ranges * np.sin(angles), np.nan)

                for hx, hy in local_humans:
                    dist_sq = (ray_x - hx)**2 + (ray_y - hy)**2
                    ranges[dist_sq < self.clear_radius**2] = np.inf

                self.get_logger().info(
                    f"[Filter] Cleared laser around {len(local_humans)} humans",
                    throttle_duration_sec=2.0)

        filtered_scan = LaserScan()
        filtered_scan.header          = msg.header
        filtered_scan.angle_min       = msg.angle_min
        filtered_scan.angle_max       = msg.angle_max
        filtered_scan.angle_increment = msg.angle_increment
        filtered_scan.time_increment  = msg.time_increment
        filtered_scan.scan_time       = msg.scan_time
        filtered_scan.range_min       = msg.range_min
        filtered_scan.range_max       = msg.range_max
        filtered_scan.ranges          = ranges.tolist()
        filtered_scan.intensities     = msg.intensities

        self.laser_pub.publish(filtered_scan)


def main(args=None):
    rclpy.init(args=args)
    node = SemanticLaserFilter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()