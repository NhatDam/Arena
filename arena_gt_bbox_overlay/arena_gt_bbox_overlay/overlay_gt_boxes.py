#!/usr/bin/env python3
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

from cv_bridge import CvBridge
import cv2


def clamp_int(v: float, lo: int, hi: int) -> int:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return int(v)


class OverlayGTBoxes(Node):
    def __init__(self):
        super().__init__("overlay_gt_boxes")

        self.declare_parameter("image_topic", "/task_generator_node/turtlebot/rgbd_camera/image")
        self.declare_parameter("bboxes_topic", "/task_generator_node/turtlebot/gt_human_bboxes_2d")
        self.declare_parameter("out_topic", "/task_generator_node/turtlebot/rgbd_camera/image_with_gt_boxes")

        self.image_topic = self.get_parameter("image_topic").value
        self.bboxes_topic = self.get_parameter("bboxes_topic").value
        self.out_topic = self.get_parameter("out_topic").value

        self.bridge = CvBridge()
        self.last_dets: Optional[Detection2DArray] = None

        # Subscribers
        self.create_subscription(
            Detection2DArray, self.bboxes_topic, self.on_dets, qos_profile_sensor_data
        )
        self.create_subscription(
            Image, self.image_topic, self.on_image, qos_profile_sensor_data
        )

        # Publisher
        self.pub = self.create_publisher(Image, self.out_topic, 10)

        self.get_logger().info(f"Sub image : {self.image_topic}")
        self.get_logger().info(f"Sub bboxes: {self.bboxes_topic}")
        self.get_logger().info(f"Pub out   : {self.out_topic}")

    def on_dets(self, msg: Detection2DArray):
        self.last_dets = msg

    def on_image(self, msg: Image):
        # Convert image to OpenCV
        try:
            # Most Isaac RGB streams are "rgb8" or "bgra8"; let cv_bridge handle
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge convert failed: {e}")
            return

        h, w = cv_img.shape[:2]

        dets = self.last_dets
        if dets is not None:
            for det in dets.detections:
                # bbox center + size in pixels
                cx = det.bbox.center.position.x
                cy = det.bbox.center.position.y
                sx = det.bbox.size_x
                sy = det.bbox.size_y

                xmin = clamp_int(cx - sx / 2.0, 0, w - 1)
                xmax = clamp_int(cx + sx / 2.0, 0, w - 1)
                ymin = clamp_int(cy - sy / 2.0, 0, h - 1)
                ymax = clamp_int(cy + sy / 2.0, 0, h - 1)

                if xmax <= xmin or ymax <= ymin:
                    continue

                # Draw rectangle (default OpenCV green)
                cv2.rectangle(cv_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                # Label: id + class if available
                label = det.id if det.id else "person"
                if det.results:
                    cls = det.results[0].hypothesis.class_id
                    if cls:
                        label = f"{label}:{cls}"
                cv2.putText(
                    cv_img,
                    label,
                    (xmin, max(0, ymin - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        # Convert back to ROS Image
        out = self.bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
        out.header = msg.header  # keep same stamp/frame
        self.pub.publish(out)


def main():
    rclpy.init()
    node = OverlayGTBoxes()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
