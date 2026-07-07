#!/usr/bin/env python3
from typing import List, Tuple, Optional

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import qos_profile_sensor_data

# Use LaserScan instead of PointCloud2
from sensor_msgs.msg import CameraInfo, LaserScan
from arena_people_msgs.msg import Pedestrians
from visualization_msgs.msg import Marker, MarkerArray

from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose

import tf2_ros
from geometry_msgs.msg import TransformStamped


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def quat_to_rot_matrix_xyzw(x: float, y: float, z: float, w: float):
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return [
        [1 - 2 * (yy + zz),     2 * (xy - wz),         2 * (xz + wy)],
        [2 * (xy + wz),         1 - 2 * (xx + zz),     2 * (yz - wx)],
        [2 * (xz - wy),         2 * (yz + wx),         1 - 2 * (xx + yy)],
    ]


def apply_tf(T: TransformStamped, p: Tuple[float, float, float]) -> Tuple[float, float, float]:
    tx = T.transform.translation.x
    ty = T.transform.translation.y
    tz = T.transform.translation.z
    q = T.transform.rotation
    R = quat_to_rot_matrix_xyzw(q.x, q.y, q.z, q.w)

    x, y, z = p
    X = R[0][0] * x + R[0][1] * y + R[0][2] * z + tx
    Y = R[1][0] * x + R[1][1] * y + R[1][2] * z + ty
    Z = R[2][0] * x + R[2][1] * y + R[2][2] * z + tz
    return (X, Y, Z)


def rotz(yaw: float, x: float, y: float) -> Tuple[float, float]:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return (c * x - s * y, s * x + c * y)


class GTHumanBboxProjector(Node):
    def __init__(self):
        super().__init__("gt_human_bbox_projector")

        # Topics / frames
        self.declare_parameter("peds_topic", "/task_generator_node/arena_peds")
        self.declare_parameter("camera_info_topic", "/task_generator_node/turtlebot/rgbd_camera/camera_info")
        # Change to your exact LaserScan topic
        self.declare_parameter("scan_topic", "/task_generator_node/turtlebot/lidar") 
        self.declare_parameter("camera_frame", "turtlebot/oakd_rgb_camera_optical_frame")
        self.declare_parameter("peds_frame_override", "map") 

        # Human box dimensions
        self.declare_parameter("human_height", 1.70)
        self.declare_parameter("human_width", 0.50)
        self.declare_parameter("human_depth", 0.30)

        self.declare_parameter("ped_pose_is_center", False)
        self.declare_parameter("min_depth_m", 0.20)
        self.declare_parameter("publish_topic", "/task_generator_node/turtlebot/gt_human_bboxes_2d")
        
        # Occlusion Tolerance (0.4m buffer for walls)
        self.declare_parameter("occlusion_tolerance_m", 0.40)

        self.peds_topic = self.get_parameter("peds_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.scan_topic = self.get_parameter("scan_topic").value
        self.camera_frame = self.get_parameter("camera_frame").value
        self.src_override = self.get_parameter("peds_frame_override").value

        self.H = float(self.get_parameter("human_height").value)
        self.W = float(self.get_parameter("human_width").value)
        self.D = float(self.get_parameter("human_depth").value)
        self.pose_is_center = bool(self.get_parameter("ped_pose_is_center").value)
        self.min_depth = float(self.get_parameter("min_depth_m").value)
        self.occlusion_tolerance = float(self.get_parameter("occlusion_tolerance_m").value)
        self.pub_topic = self.get_parameter("publish_topic").value

        # State variables
        self.fx = self.fy = self.cx = self.cy = None
        self.img_w = self.img_h = None
        self.have_cam = False
        
        self.last_scan: Optional[LaserScan] = None

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=60.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ROS I/O — use sensor_data QoS everywhere (Best Effort) to match Isaac Sim
        self.create_subscription(CameraInfo, self.camera_info_topic, self.on_cam_info, qos_profile_sensor_data)
        self.create_subscription(LaserScan, self.scan_topic, self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Pedestrians, self.peds_topic, self.on_peds, qos_profile_sensor_data)
        
        self.pub = self.create_publisher(Detection2DArray, self.pub_topic, qos_profile_sensor_data)
        self.marker_pub = self.create_publisher(MarkerArray, self.pub_topic + "_3d_markers", qos_profile_sensor_data)

        self.get_logger().info(f"✅ Projector started!")
        self.get_logger().info(f"  camera_info: {self.camera_info_topic}")
        self.get_logger().info(f"  peds:        {self.peds_topic}")
        self.get_logger().info(f"  scan:        {self.scan_topic}")
        self.get_logger().info(f"  camera_frame:{self.camera_frame}")
        self.get_logger().info(f"  publish:     {self.pub_topic}")

    def on_cam_info(self, msg: CameraInfo):
        if not self.have_cam:
            self.get_logger().info(f"📷 Got camera_info: {msg.width}x{msg.height}, fx={msg.k[0]:.1f}")
        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])
        self.img_w = int(msg.width)
        self.img_h = int(msg.height)
        self.have_cam = True
        
    def on_scan(self, msg: LaserScan):
        self.last_scan = msg

    def project_point(self, X: float, Y: float, Z: float) -> Optional[Tuple[float, float]]:
        if Z <= self.min_depth:
            return None
        u = self.fx * (X / Z) + self.cx
        v = self.fy * (Y / Z) + self.cy
        return (u, v)

    def human_box_corners_world(self, px: float, py: float, pz: float, yaw: float) -> List[Tuple[float, float, float]]:
        half_w = self.W / 2.0
        half_d = self.D / 2.0

        if self.pose_is_center:
            z0 = pz - self.H / 2.0
            z1 = pz + self.H / 2.0
        else:
            z0 = pz
            z1 = pz + self.H

        base_xy = [(-half_d, -half_w), (-half_d, half_w), (half_d, -half_w), (half_d, half_w)]

        corners = []
        for (dx, dy) in base_xy:
            rx, ry = rotz(yaw, dx, dy)
            corners.append((px + rx, py + ry, z0))
            corners.append((px + rx, py + ry, z1))
        return corners

    def on_peds(self, msg: Pedestrians):
        if not self.have_cam:
            self.get_logger().warn("Waiting for camera_info...", throttle_duration_sec=2.0)
            return

        src_frame = msg.header.frame_id if msg.header.frame_id else self.src_override

        try:
            query_time = Time.from_msg(msg.header.stamp) if (msg.header.stamp.sec != 0 or msg.header.stamp.nanosec != 0) else Time()
        except Exception:
            query_time = Time()

        try:
            T_cam = self.tf_buffer.lookup_transform(
                self.camera_frame, src_frame, query_time, timeout=rclpy.duration.Duration(seconds=0.1)
            )
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed ({src_frame} -> {self.camera_frame}): {e}", throttle_duration_sec=2.0)
            return

        out = Detection2DArray()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.camera_frame
        marker_array = MarkerArray()

        peds_list = getattr(msg, "pedestrians", None)
        if peds_list is None:
            self.pub.publish(out)
            return

        for ped in peds_list:
            pid = int(getattr(ped, "id", 0))
            if not hasattr(ped, "pose"): continue

            px, py, pz = float(ped.pose.position.x), float(ped.pose.position.y), float(ped.pose.position.z)
            q = ped.pose.orientation
            yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))

            # ==========================================
            # 1. SIMPLE 2D LASERSCAN OCCLUSION CHECK
            # ==========================================
            is_occluded = False
            scan = self.last_scan
            
            if scan is not None:
                try:
                    # Look up transform from Map -> LiDAR
                    T_lidar = self.tf_buffer.lookup_transform(
                        scan.header.frame_id, src_frame, query_time, timeout=rclpy.duration.Duration(seconds=0.1)
                    )
                    
                    # Convert Human Pose to LiDAR Frame
                    Hx, Hy, Hz = apply_tf(T_lidar, (px, py, pz))
                    
                    # Calculate true 2D distance and angle to human
                    dist_h = math.hypot(Hx, Hy)
                    angle_h = math.atan2(Hy, Hx)
                    
                    # Check if human is in the front FOV (-45 to +45 degrees)
                    if abs(angle_h) < math.radians(45.0):
                        
                        # Find the index in the ranges array that corresponds to the human's angle
                        # We subtract the min_angle and divide by the increment
                        index_float = (angle_h - scan.angle_min) / scan.angle_increment
                        
                        if 0 <= index_float < len(scan.ranges):
                            center_idx = int(round(index_float))
                            
                            # Look at a small window of beams (+/- 3 degrees) around the human
                            # 3 degrees = ~0.052 rad. Divide by increment to get number of array slots
                            window = int(0.052 / scan.angle_increment)
                            
                            start_idx = max(0, center_idx - window)
                            end_idx = min(len(scan.ranges) - 1, center_idx + window)
                            
                            # Grab those laser beams
                            beams = np.array(scan.ranges[start_idx:end_idx+1])
                            
                            # Filter out infinities, zeros, or anything outside min/max range
                            valid_beams = beams[(beams > scan.range_min) & (beams < scan.range_max)]
                            
                            if len(valid_beams) > 0:
                                # What is the closest object the LiDAR sees in that direction?
                                closest_obstacle_dist = np.min(valid_beams)
                                
                                # If the obstacle is significantly closer than the human, it's a wall!
                                if closest_obstacle_dist < (dist_h - self.occlusion_tolerance):
                                    is_occluded = True
                                    
                except Exception as e:
                    pass

            if is_occluded:
                continue # Skip drawing this pedestrian entirely!

            # ==========================================
            # 2. GENERATE MARKERS & 2D BOUNDING BOX
            # ==========================================
            marker = Marker()
            marker.header.frame_id = src_frame
            marker.header.stamp = out.header.stamp
            marker.ns = "gt_human_boxes"
            marker.id = pid
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x, marker.pose.position.y = px, py
            marker.pose.position.z = pz if self.pose_is_center else pz + (self.H / 2.0)
            marker.pose.orientation = q
            marker.scale.x, marker.scale.y, marker.scale.z = self.D, self.W, self.H
            marker.color.g, marker.color.a = 1.0, 0.5
            marker.lifetime = rclpy.duration.Duration(seconds=0.2).to_msg()
            marker_array.markers.append(marker)

            corners_src = self.human_box_corners_world(px, py, pz, yaw)
            uv = []
            for p in corners_src:
                X_cam, Y_cam, Z_cam = apply_tf(T_cam, p)
                pr = self.project_point(X_cam, Y_cam, Z_cam)
                if pr is not None:
                    uv.append(pr)

            if len(uv) < 2: continue

            us, vs = [p[0] for p in uv], [p[1] for p in uv]
            xmin, xmax = clamp(min(us), 0.0, float(self.img_w - 1)), clamp(max(us), 0.0, float(self.img_w - 1))
            ymin, ymax = clamp(min(vs), 0.0, float(self.img_h - 1)), clamp(max(vs), 0.0, float(self.img_h - 1))

            bbox_width, bbox_height = xmax - xmin, ymax - ymin
            if bbox_width < 2.0 or bbox_height < 2.0: continue

            det = Detection2D()
            det.header = out.header
            det.id = str(pid)

            bbox = BoundingBox2D()
            bbox.center.position.x, bbox.center.position.y = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
            bbox.size_x, bbox.size_y = bbox_width, bbox_height
            det.bbox = bbox

            # TRUE DEPTH LOGIC
            X_true, Y_true, Z_true = apply_tf(T_cam, (px, py, pz))
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = "person"
            hyp.hypothesis.score = 1.0
            hyp.pose.pose.position.x, hyp.pose.pose.position.y, hyp.pose.pose.position.z = X_true, Y_true, Z_true
            hyp.pose.pose.orientation.w = 1.0 
            
            det.results.append(hyp)
            out.detections.append(det)

        self.marker_pub.publish(marker_array)
        self.pub.publish(out)
        # if len(out.detections) > 0:
        #     self.get_logger().info(f"Published {len(out.detections)} detections from {len(peds_list)} peds", throttle_duration_sec=2.0)
        # elif peds_list and len(peds_list) > 0:
        #     self.get_logger().info(f"0 detections from {len(peds_list)} peds (all occluded/behind camera?)", throttle_duration_sec=2.0)

def main():
    rclpy.init()
    node = GTHumanBboxProjector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
