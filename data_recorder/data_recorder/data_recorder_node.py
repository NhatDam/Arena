#!/usr/bin/env python3
"""
ROS2 node to record RGB frames, depth frames, detection bboxes, and robot trajectory.

Subscribes to:
- Image frames (RGB)
- Image frames (Depth)
- Detection2DArray (bounding boxes)
- Agent (robot trajectory)

Services:
- /data_recorder_start_recording (StartEvaluation): Start recording
- /data_recorder_stop_recording (Empty): Stop recording

Output structure:
- ~/arena5_ws/output/<run_id>/
  - 0000.jpg, 0001.jpg, ... (RGB frames)
  - depth_vis/
    - 0000.jpg, 0001.jpg, ... (depth visualization JPGs)
  - 0000.depth.npy, 0001.depth.npy, ... (raw depth arrays)
  - 0000.pred_label.json, 0001.pred_label.json, ... (detections with root-level RGB paths)
  - traj_data.txt (robot trajectory)

Design: Write-through (streaming) I/O — every frame, detection JSON, and
trajectory point is flushed to disk as it arrives.  This means a Ctrl-C
at any moment leaves a consistent, readable dataset on disk.

Depth sync / sampling design:
  1. Timestamp-based pairing — depth frames are stored in a small ring buffer
     (DEPTH_BUFFER_SIZE entries). When an RGB frame arrives, the node picks the
     depth frame whose header stamp is closest to the RGB stamp. If the best
     match is older than DEPTH_MAX_AGE_SEC the depth fields are recorded as null.
  2. Camera intrinsics — the node subscribes to the camera_info topic and
     latches the first message, saving fx/fy/cx/cy from the K matrix. These are
     written to camera_info.json once per run at recording start.
  3. Single-point back-projection — for each detected human the node reads the
     raw depth at the bbox centre pixel (u, v) and back-projects it to a 3D
     point (X, Y, Z) in the camera frame using the pinhole model:
       X = (u - cx) * d / fx
       Y = (v - cy) * d / fy
       Z = d
     All four values are stored in "depth_point" in the JSON. d is recorded
     raw with no range filter; null only when the sensor returned 0/inf/NaN.
     If camera_info has not arrived yet, X/Y/Z are null but d is still stored.
"""

import collections
import os
import sys
import json
import threading
import time
from typing import Optional, List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

try:
    import cv2
except ImportError:
    print("WARNING: cv2 not found, installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
    import cv2

try:
    import numpy as np
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    import numpy as np

try:
    from cv_bridge import CvBridge
except ImportError:
    print("ERROR: cv_bridge not available. This is a ROS2 dependency issue.")
    raise

from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray
from hunav_msgs.msg import Agent, Agents
from hunav_msgs.srv import StartEvaluation
from std_srvs.srv import Empty


# ── Depth sync tuning constants ───────────────────────────────────────────────
# Number of depth frames kept in the ring buffer for timestamp matching.
DEPTH_BUFFER_SIZE = 10

# Maximum age (seconds) between an RGB frame stamp and its best-matching depth
# frame before depth is recorded as null for that frame.
DEPTH_MAX_AGE_SEC = 0.15
# ─────────────────────────────────────────────────────────────────────────────


class DataRecorderNode(Node):
    """ROS2 node for recording robot run data with write-through I/O."""

    def __init__(self):
        super().__init__("data_recorder_node")

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter("image_topic", "/task_generator_node/turtlebot/rgbd_camera/image")
        self.declare_parameter("depth_topic", "/task_generator_node/turtlebot/rgbd_camera/depth")
        self.declare_parameter("camera_info_topic", "/task_generator_node/turtlebot/rgbd_camera/camera_info")
        self.declare_parameter("detections_topic", "/task_generator_node/turtlebot/gt_human_bboxes_2d")
        self.declare_parameter("robot_state_topic", "/task_generator_node/robot_states")
        self.declare_parameter("human_states_topic", "/task_generator_node/human_states")
        self.declare_parameter("startup_delay", 5.0)
        self.declare_parameter("topic_timeout", 120.0)
        self.declare_parameter("min_frames_before_record", 5)
        self.declare_parameter("image_fps", 1.0)
        self.declare_parameter("trajectory_fps", 5.0)

        self.image_topic = self.get_parameter("image_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.detections_topic = self.get_parameter("detections_topic").value
        self.robot_state_topic = self.get_parameter("robot_state_topic").value
        self.human_states_topic = self.get_parameter("human_states_topic").value
        self.startup_delay = self.get_parameter("startup_delay").value
        self.topic_timeout = self.get_parameter("topic_timeout").value
        self.min_frames_before_record = self.get_parameter("min_frames_before_record").value
        self.image_interval = 1.0 / self.get_parameter("image_fps").value
        self.traj_interval = 1.0 / self.get_parameter("trajectory_fps").value
        self.last_image_recorded_time = 0.0
        self.last_traj_recorded_time = 0.0

        # ── Output directory ─────────────────────────────────────────────────
        home_dir = os.path.expanduser("~")
        self.output_base_dir = os.path.join(home_dir, "arena5_ws", "output")
        os.makedirs(self.output_base_dir, exist_ok=True)

        # ── Recording state ──────────────────────────────────────────────────
        self.recording = False
        self.run_id: Optional[int] = None
        self.output_dir: Optional[str] = None
        self.frame_count = 0          # frames written to disk so far
        self.traj_count = 0           # trajectory points written to disk
        self.start_time: Optional[float] = None  # for relative timestamps

        # ── Latest detections (kept in memory; written per-frame) ─────────────
        self.latest_detections: Optional[Detection2DArray] = None
        self.latest_human_states: Optional[List] = None  # list of Agent

        # ── Depth ring buffer ─────────────────────────────────────────────────
        # Each entry is a (stamp_sec: float, array: np.ndarray, encoding: str)
        # tuple.  stamp_sec is the message header stamp converted to seconds.
        # Protected by _frame_lock (same lock as frame_count / disk writes).
        self._depth_buffer: collections.deque = collections.deque(
            maxlen=DEPTH_BUFFER_SIZE
        )

        # ── Camera intrinsics (latched from camera_info topic) ────────────────
        # Populated on the first CameraInfo message; used for back-projection.
        # fx, fy  — focal lengths in pixels
        # cx, cy  — principal point (optical centre) in pixels
        self._cam_fx: Optional[float] = None
        self._cam_fy: Optional[float] = None
        self._cam_cx: Optional[float] = None
        self._cam_cy: Optional[float] = None
        self._camera_info_raw: Optional[dict] = None   # full K/D/P for JSON

        # ── Open file handle for streaming trajectory writes ─────────────────
        # Opened when recording starts, closed when recording stops / node shuts down.
        self._traj_file = None
        self._human_file = None
        self._traj_lock = threading.Lock()   # protect concurrent writes
        self._frame_lock = threading.Lock()  # protect frame_count / disk writes

        # ── Topic readiness ──────────────────────────────────────────────────
        self.image_received = False
        self.trajectory_received = False
        self.first_image_time: Optional[float] = None
        self.first_trajectory_time: Optional[float] = None

        # ── Startup thread (waits for topics before enabling recording) ───────
        self.startup_thread: Optional[threading.Thread] = None
        self.startup_thread_stop = threading.Event()

        # ── CV Bridge ────────────────────────────────────────────────────────
        self.bridge = CvBridge()

        # ── Services ─────────────────────────────────────────────────────────
        self.recording_service_start = self.create_service(
            StartEvaluation,
            "data_recorder_start_recording",
            self.start_recording_callback,
        )
        self.recording_service_stop = self.create_service(
            Empty,
            "data_recorder_stop_recording",
            self.stop_recording_callback,
        )

        # ── Subscribers ───────────────────────────────────────────────────────
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )
        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            qos_profile_sensor_data,
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            qos_profile_sensor_data,
        )
        self.detections_sub = self.create_subscription(
            Detection2DArray,
            self.detections_topic,
            self.detections_callback,
            qos_profile_sensor_data,
        )
        self.robot_state_sub = self.create_subscription(
            Agent,
            self.robot_state_topic,
            self.robot_state_callback,
            1,
        )
        self.human_states_sub = self.create_subscription(
            Agents,
            self.human_states_topic,
            self.human_states_callback,
            1,
        )

        self.get_logger().info(f"  Image topic:        {self.image_topic}")
        self.get_logger().info(f"  Depth topic:        {self.depth_topic}")
        self.get_logger().info(f"  Camera info topic:  {self.camera_info_topic}")
        self.get_logger().info(f"  Detections topic:   {self.detections_topic}")
        self.get_logger().info(f"  Robot state topic:  {self.robot_state_topic}")
        self.get_logger().info(f"  Human states topic: {self.human_states_topic}")
        self.get_logger().info(f"  Output base dir:    {self.output_base_dir}")
        self.get_logger().info(
            f"  Depth sync: buffer={DEPTH_BUFFER_SIZE} frames, "
            f"max_age={DEPTH_MAX_AGE_SEC*1000:.0f} ms, "
            f"sampling=bbox-center back-projection (X,Y,Z)"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _open_trajectory_file(self) -> None:
        """Open (or re-open) the trajectory file for streaming appends."""
        traj_path = os.path.join(self.output_dir, "traj_data.txt")
        # 'w' truncates so each new run starts clean
        self._traj_file = open(traj_path, "w", buffering=1)  # line-buffered
        self._traj_file.write("timestamp x y z qx qy qz qw\n")
        self._traj_file.flush()

    def _open_human_file(self) -> None:
        """Open the human states file for streaming appends."""
        human_path = os.path.join(self.output_dir, "human_.txt")
        self._human_file = open(human_path, "w", buffering=1)
        self._human_file.write("timestamp id x y z\n")
        self._human_file.flush()

    def _close_trajectory_file(self) -> None:
        """Flush and close the trajectory file handle safely."""
        with self._traj_lock:
            if self._traj_file is not None:
                try:
                    self._traj_file.flush()
                    os.fsync(self._traj_file.fileno())
                    self._traj_file.close()
                except Exception as e:
                    print(f"  Error closing trajectory file: {e}", flush=True)
                finally:
                    self._traj_file = None

    def _close_human_file(self) -> None:
        """Flush and close the human file handle safely."""
        with self._traj_lock:
            if self._human_file is not None:
                try:
                    self._human_file.flush()
                    os.fsync(self._human_file.fileno())
                    self._human_file.close()
                except Exception as e:
                    print(f"  Error closing human file: {e}", flush=True)
                finally:
                    self._human_file = None

    def _startup_thread_worker(self) -> None:
        """Wait for both topics to publish, then enable recording."""
        try:
            self.get_logger().info(f" Waiting {self.startup_delay}s for simulation to stabilise...")
            time.sleep(self.startup_delay)

            self.get_logger().info(" Waiting for topics to publish...")
            start_time = time.time()
            deadline = start_time + self.topic_timeout
            last_log_time = start_time

            while time.time() < deadline:
                if self.startup_thread_stop.is_set():
                    self.get_logger().info("Startup thread cancelled")
                    return

                if self.image_received and self.trajectory_received:
                    elapsed = time.time() - start_time
                    self.get_logger().info(
                        f"✓ Topics ready! (startup: {self.startup_delay}s, discovery: {elapsed:.2f}s)"
                    )
                    # Open trajectory file *before* flipping the recording flag
                    self._open_trajectory_file()
                    self._open_human_file()
                    self.frame_count = 0
                    self.traj_count = 0
                    self.recording = True
                    self.start_time = time.time()  # Record start time for relative timestamps
                    # Write camera_info.json if intrinsics already arrived
                    if self._camera_info_raw is not None:
                        self._save_camera_info_json()
                    else:
                        self.get_logger().warn(
                            "camera_info not yet received — camera_info.json will be "
                            "written when the first CameraInfo message arrives."
                        )
                    self.get_logger().info(f"✓ Recording started — writing directly to {self.output_dir}")
                    return

                current_time = time.time()
                if current_time - last_log_time >= 2.0:
                    elapsed = current_time - start_time
                    self.get_logger().info(
                        f"  Waiting {elapsed:.1f}s…  "
                        f"Image: {self.image_received}, Trajectory: {self.trajectory_received}"
                    )
                    last_log_time = current_time

                time.sleep(0.1)

            self.get_logger().error(
                f"  Timeout after {self.topic_timeout}s. "
                f"Image received: {self.image_received}, Trajectory received: {self.trajectory_received}"
            )
        except Exception as e:
            self.get_logger().error(f"Error in startup thread: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def _wait_for_topics_ready(self) -> None:
        """Launch the startup background thread."""
        self.startup_thread_stop.clear()
        self.startup_thread = threading.Thread(
            target=self._startup_thread_worker, daemon=False
        )
        self.startup_thread.start()

    # ─────────────────────────────────────────────────────────────────────────
    # Depth sync helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _best_depth_for_stamp(self, rgb_stamp_sec: float) -> Tuple[Optional[np.ndarray], Optional[str], float]:
        """
        Return the depth frame from the ring buffer whose timestamp is closest
        to *rgb_stamp_sec*.

        Returns (array, encoding, age_sec).  Returns (None, None, inf) when the
        buffer is empty or the closest match exceeds DEPTH_MAX_AGE_SEC.

        Must be called with _frame_lock already held.
        """
        if not self._depth_buffer:
            return None, None, float("inf")

        best_entry = min(self._depth_buffer, key=lambda e: abs(e[0] - rgb_stamp_sec))
        age = abs(best_entry[0] - rgb_stamp_sec)

        if age > DEPTH_MAX_AGE_SEC:
            return None, None, age

        return best_entry[1], best_entry[2], age

    def _save_camera_info_json(self) -> None:
        """Write camera_info.json to the current run directory (call once per run)."""
        if self._camera_info_raw is None or self.output_dir is None:
            return
        path = os.path.join(self.output_dir, "camera_info.json")
        try:
            with open(path, "w") as f:
                json.dump(self._camera_info_raw, f, indent=2)
            self.get_logger().info(f"✓ camera_info.json written: {path}")
        except Exception as e:
            self.get_logger().warn(f"Failed to write camera_info.json: {e}")

    def _backproject_center(
        self,
        depth_array: np.ndarray,
        cx_px: float,
        cy_px: float,
    ) -> dict:
        """
        Read raw depth at pixel (cx_px, cy_px) and back-project to 3D camera frame.

        Returns a dict:
          u, v  — pixel coordinates (float, as received from bbox)
          d     — raw depth in metres; null if sensor returned 0 / inf / NaN
          X     — (u - cx) * d / fx  (right in camera frame)
          Y     — (v - cy) * d / fy  (down in camera frame)
          Z     — d                   (forward / optical axis)
          intrinsics_available — bool, False when camera_info not yet received

        X/Y/Z are null when d is null OR when intrinsics are not yet available.
        No range filtering is applied to d — the raw sensor value is preserved.
        """
        h, w = depth_array.shape[:2]

        # Clamp to image bounds
        px = int(round(max(0.0, min(cx_px, w - 1))))
        py = int(round(max(0.0, min(cy_px, h - 1))))

        raw = float(depth_array[py, px])

        # 16UC1 stores millimetres; convert to metres
        if depth_array.dtype == np.uint16 and raw > 1000.0:
            raw = raw / 1000.0

        d: Optional[float] = None if (not np.isfinite(raw) or raw == 0.0) else raw

        intrinsics_available = (
            self._cam_fx is not None
            and self._cam_fy is not None
            and self._cam_cx is not None
            and self._cam_cy is not None
        )

        X: Optional[float] = None
        Y: Optional[float] = None
        Z: Optional[float] = None
        if d is not None and intrinsics_available:
            X = (cx_px - self._cam_cx) * d / self._cam_fx
            Y = (cy_px - self._cam_cy) * d / self._cam_fy
            Z = d

        return {
            "u": cx_px,
            "v": cy_px,
            "d": d,
            "X": X,
            "Y": Y,
            "Z": Z,
            "intrinsics_available": intrinsics_available,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Service callbacks
    # ─────────────────────────────────────────────────────────────────────────

    def start_recording_callback(
        self, request: StartEvaluation.Request, response: StartEvaluation.Response
    ) -> StartEvaluation.Response:
        """Start recording data."""
        if self.recording:
            self.get_logger().warn("Already recording!")
            response.success = False
            return response

        if self.startup_thread is not None and self.startup_thread.is_alive():
            self.get_logger().warn("Startup already in progress!")
            response.success = False
            return response

        self.run_id = request.run_id
        self.output_dir = os.path.join(self.output_base_dir, str(self.run_id))
        os.makedirs(self.output_dir, exist_ok=True)

        # Create depth visualization folder only; RGB and annotations remain at run root
        self.depth_vis_dir = os.path.join(self.output_dir, "depth_vis")
        os.makedirs(self.depth_vis_dir, exist_ok=True)

        # Reset topic-readiness flags and depth buffer
        self.image_received = False
        self.trajectory_received = False
        self.first_image_time = None
        self.first_trajectory_time = None
        with self._frame_lock:
            self._depth_buffer.clear()

        self.get_logger().info(f"Preparing to record for run_id: {self.run_id}")
        self.get_logger().info(f"Output directory: {self.output_dir}")

        self._wait_for_topics_ready()

        response.success = True
        return response

    def stop_recording_callback(self, request, response) -> Empty.Response:
        """Stop recording — all data is already on disk."""
        if not self.recording:
            self.get_logger().warn("Not currently recording!")
            return response

        self.recording = False  # Stop accepting new data first

        # Flush & close the trajectory file
        self._close_trajectory_file()
        self._close_human_file()

        self.get_logger().info(
            f"  Recording stopped. "
            f"Frames saved: {self.frame_count}, "
            f"Trajectory points saved: {self.traj_count}"
        )
        self.get_logger().info(f" Data is in: {self.output_dir}")

        return response

    # ─────────────────────────────────────────────────────────────────────────
    # Topic callbacks — write directly to disk
    # ─────────────────────────────────────────────────────────────────────────

    def image_callback(self, msg: Image) -> None:
        """Convert and immediately write each frame + detection JSON to disk."""
        current_time = self.get_clock().now().nanoseconds / 1e9

        if not self.image_received:
            self.image_received = True
            self.first_image_time = current_time
            self.get_logger().info(f"✓ First image received at t={self.first_image_time:.2f}")

        if not self.recording:
            return

        # Throttle to configured FPS
        if (current_time - self.last_image_recorded_time) < self.image_interval:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"Failed to convert image: {e}")
            return

        # Derive RGB header stamp in seconds for depth matching
        rgb_stamp_sec = (
            msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        )

        with self._frame_lock:
            self.last_image_recorded_time = current_time
            frame_idx = self.frame_count
            frame_num_str = f"{frame_idx:04d}"

            # ── Find the best-matching depth frame ────────────────────────
            depth_array, depth_encoding, depth_age = self._best_depth_for_stamp(rgb_stamp_sec)

            if depth_array is None:
                self.get_logger().debug(
                    f"Frame {frame_num_str}: no depth match "
                    f"(buffer size={len(self._depth_buffer)}, age={depth_age*1000:.1f} ms)"
                )
            else:
                self.get_logger().debug(
                    f"Frame {frame_num_str}: depth matched, age={depth_age*1000:.1f} ms"
                )

            # ── Write JPEG ────────────────────────────────────────────────
            jpg_filename = f"{frame_num_str}.jpg"
            jpg_path = os.path.join(self.output_dir, jpg_filename)
            if not cv2.imwrite(jpg_path, cv_image):
                self.get_logger().warn(f"cv2.imwrite failed for {jpg_path}")
                return

            # ── Write depth visualization and raw depth array ─────────────
            if depth_array is not None:
                depth_vis_filename = f"{frame_num_str}.jpg"
                depth_vis_path = os.path.join(self.depth_vis_dir, depth_vis_filename)
                depth_npy_path = os.path.join(self.output_dir, f"{frame_num_str}.depth.npy")

                if depth_array.dtype == np.float32:
                    depth_normalized = np.clip(depth_array, 0, 10) / 10.0 * 255.0
                    depth_8bit = depth_normalized.astype(np.uint8)
                elif depth_array.dtype == np.uint16:
                    depth_8bit = (depth_array / 65535.0 * 255.0).astype(np.uint8)
                else:
                    depth_8bit = depth_array.astype(np.uint8)

                cv2.imwrite(depth_vis_path, depth_8bit)

                try:
                    np.save(depth_npy_path, depth_array)
                except Exception as e:
                    self.get_logger().warn(f"Failed to save depth array {depth_npy_path}: {e}")

            # ── Write detection JSON ───────────────────────────────────────
            json_filename = f"{frame_num_str}.pred_label.json"
            json_path = os.path.join(self.output_dir, json_filename)
            json_data = self._build_detection_json(jpg_filename, cv_image, depth_array)
            try:
                with open(json_path, "w") as f:
                    json.dump(json_data, f, indent=2)
            except Exception as e:
                self.get_logger().warn(f"Failed to write JSON {json_path}: {e}")
                return

            # ── Write human positions to human_.txt ─────────────────────────
            if self.latest_human_states is not None:
                relative_time = current_time - self.start_time if self.start_time else current_time
                with self._traj_lock:
                    if self._human_file is not None:
                        try:
                            for human in self.latest_human_states:
                                hid = human.id
                                hx = human.position.position.x
                                hy = human.position.position.y
                                hz = human.position.position.z
                                self._human_file.write(
                                    f"{relative_time:.6f} {hid} {hx:.6f} {hy:.6f} {hz:.6f}\n"
                                )
                        except Exception as e:
                            self.get_logger().warn(f"Failed to write human positions: {e}")

            self.frame_count += 1

        # Periodic progress log
        if self.frame_count % 30 == 0:
            try:
                self.get_logger().info(f" {self.frame_count} frames written to disk")
            except Exception:
                pass

    def depth_callback(self, msg: Image) -> None:
        """
        Convert each incoming depth frame and push it into the ring buffer.

        The buffer stores (stamp_sec, array, encoding) tuples so that
        image_callback can find the depth frame whose stamp is closest to
        the RGB frame being processed, rather than relying on arrival order.
        """
        # Accept depth frames even before recording starts so the buffer is
        # pre-filled by the time the first RGB frame arrives.
        try:
            if msg.encoding == "32FC1":
                cv_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
                encoding = "32FC1"
            elif msg.encoding == "16UC1":
                cv_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
                encoding = "16UC1"
            else:
                cv_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                encoding = msg.encoding
        except Exception as e:
            try:
                self.get_logger().warn(f"Failed to convert depth image: {e}")
            except Exception:
                pass
            return

        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        with self._frame_lock:
            self._depth_buffer.append((stamp_sec, cv_depth, encoding))

    def camera_info_callback(self, msg: CameraInfo) -> None:
        """
        Latch camera intrinsics from the first CameraInfo message received.

        The K matrix is row-major 3×3:
          K = [fx,  0, cx,
                0, fy, cy,
                0,  0,  1]
        so K[0]=fx, K[2]=cx, K[4]=fy, K[5]=cy.

        Once latched, the subscriber is destroyed to avoid unnecessary CPU use.
        """
        if self._cam_fx is not None:
            return  # already latched

        K = msg.k  # flat 9-element array
        if len(K) < 6:
            self.get_logger().warn("CameraInfo K matrix too short, ignoring.")
            return

        self._cam_fx = float(K[0])
        self._cam_fy = float(K[4])
        self._cam_cx = float(K[2])
        self._cam_cy = float(K[5])

        self._camera_info_raw = {
            "width":            msg.width,
            "height":           msg.height,
            "distortion_model": msg.distortion_model,
            "K":                list(msg.k),
            "D":                list(msg.d),
            "R":                list(msg.r),
            "P":                list(msg.p),
            "fx":               self._cam_fx,
            "fy":               self._cam_fy,
            "cx":               self._cam_cx,
            "cy":               self._cam_cy,
        }

        self.get_logger().info(
            f"✓ Camera intrinsics latched — "
            f"fx={self._cam_fx:.2f}, fy={self._cam_fy:.2f}, "
            f"cx={self._cam_cx:.2f}, cy={self._cam_cy:.2f}"
        )

        # Save to disk immediately if recording has already started
        if self.recording and self.output_dir is not None:
            self._save_camera_info_json()

        # No longer need this subscriber
        self.destroy_subscription(self.camera_info_sub)

    def detections_callback(self, msg: Detection2DArray) -> None:
        """Keep the latest detections so image_callback can pair them."""
        if not self.recording:
            return
        self.latest_detections = msg

    def human_states_callback(self, msg: Agents) -> None:
        """Keep the latest human states for global positions."""
        if not self.recording:
            return
        self.latest_human_states = msg.agents

    def robot_state_callback(self, msg: Agent) -> None:
        """Immediately append each trajectory point to traj_data.txt."""
        current_time = self.get_clock().now().nanoseconds / 1e9

        if not self.trajectory_received:
            self.trajectory_received = True
            self.first_trajectory_time = current_time
            self.get_logger().info(f"✓ First trajectory received at t={self.first_trajectory_time:.2f}")

        if not self.recording:
            return

        # Throttle to configured FPS
        if (current_time - self.last_traj_recorded_time) < self.traj_interval:
            return

        x = msg.position.position.x
        y = msg.position.position.y
        z = msg.position.position.z
        qx = msg.position.orientation.x
        qy = msg.position.orientation.y
        qz = msg.position.orientation.z
        qw = msg.position.orientation.w

        with self._traj_lock:
            if self._traj_file is not None:
                self.last_traj_recorded_time = current_time
                relative_time = current_time - self.start_time if self.start_time else current_time
                try:
                    self._traj_file.write(
                        f"{relative_time:.6f} {x:.6f} {y:.6f} {z:.6f} "
                        f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
                    )
                    self.traj_count += 1
                except Exception as e:
                    self.get_logger().warn(f"Failed to write trajectory point: {e}")

        if self.traj_count % 10 == 0:
            try:
                self.get_logger().info(f"   {self.traj_count} trajectory points written")
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────────────
    # Detection helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_detection_json(
        self,
        image_filename: str,
        cv_image: np.ndarray,
        depth_array: Optional[np.ndarray],
    ) -> dict:
        """
        Build JSON annotation dict for the current frame.

        *depth_array* is the timestamp-matched depth frame (may be None when
        no suitable depth frame is available within DEPTH_MAX_AGE_SEC).

        Each detection entry contains:
          bbox_xyxy            — [xmin, ymin, xmax, ymax] in pixels
          gt_depth             — ground-truth camera-to-human distance (metres)
                                 from the simulation pose
          depth_frame_available— False when no depth frame was matched within
                                 the timestamp tolerance
          depth_point          — back-projected 3D point in camera frame:
                                   u, v  pixel coords of bbox centre
                                   d     raw depth in metres (null = no reading)
                                   X, Y, Z  3D position in camera frame (metres)
                                            null when d is null or intrinsics
                                            not yet received
                                   intrinsics_available  bool
        """
        img_h, img_w = cv_image.shape[:2]

        detection_data = {
            "video_id": str(self.run_id),
            "file_name": image_filename,
            "detections": [],
        }

        if self.latest_detections is None:
            return detection_data

        for idx, det in enumerate(self.latest_detections.detections, start=1):
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            sx = det.bbox.size_x
            sy = det.bbox.size_y

            xmin = max(0.0, cx - sx / 2.0)
            ymin = max(0.0, cy - sy / 2.0)
            xmax = min(float(img_w), cx + sx / 2.0)
            ymax = min(float(img_h), cy + sy / 2.0)

            gt_depth = None
            if hasattr(det, "results") and det.results:
                first = det.results[0]
                if (
                    hasattr(first, "pose")
                    and hasattr(first.pose, "pose")
                    and hasattr(first.pose.pose, "position")
                ):
                    gt_depth = float(first.pose.pose.position.z)

            # ── GT back-projection (camera frame, using gt_depth) ─────────
            # Same pinhole formula as depth_point but driven by the simulation
            # ground-truth distance, so (gt_X, gt_Y, gt_depth) is where the
            # human *should* be in the camera frame according to the sim.
            intrinsics_ok = (
                self._cam_fx is not None
                and self._cam_fy is not None
                and self._cam_cx is not None
                and self._cam_cy is not None
            )
            gt_X: Optional[float] = None
            gt_Y: Optional[float] = None
            if gt_depth is not None and intrinsics_ok:
                gt_X = (cx - self._cam_cx) * gt_depth / self._cam_fx
                gt_Y = (cy - self._cam_cy) * gt_depth / self._cam_fy

            # ── Back-project bbox centre to 3D camera frame ───────────────
            depth_frame_available = depth_array is not None
            if depth_frame_available:
                depth_point = self._backproject_center(depth_array, cx, cy)
            else:
                depth_point = {
                    "u": cx,
                    "v": cy,
                    "d": None,
                    "X": None,
                    "Y": None,
                    "Z": None,
                    "intrinsics_available": self._cam_fx is not None,
                }

            detection_data["detections"].append(
                {
                    "id": self._extract_detection_id(det, idx),
                    "category": "human",
                    "bbox_xyxy": [xmin, ymin, xmax, ymax],
                    "conf": 1.0,
                    "gt_depth": gt_depth,
                    "gt_X": gt_X,
                    "gt_Y": gt_Y,
                    "depth_frame_available": depth_frame_available,
                    "depth_point": depth_point,
                }
            )

        return detection_data

    def _extract_detection_id(self, det, default_index: int) -> str:
        """Use the pedestrian instance ID if available, otherwise fall back to a sequential ID."""
        if hasattr(det, "id") and det.id:
            return str(det.id)
        if hasattr(det, "results") and getattr(det, "results"):
            first = det.results[0]
            if hasattr(first, "hypothesis") and hasattr(first.hypothesis, "class_id"):
                return str(first.hypothesis.class_id)
        return f"human_{default_index}"


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    node = DataRecorderNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n  Ctrl+C received — data already on disk, nothing to flush.", flush=True)
    except Exception as e:
        print(f"\n {type(e).__name__} during spin: {e}", flush=True)
    finally:
        # ── Make sure the trajectory file is properly closed ──────────────
        if node._traj_file is not None:
            print(" Flushing trajectory file...", flush=True)
            node._close_trajectory_file()
            print("✓ Trajectory file closed.", flush=True)

        if node._human_file is not None:
            print(" Flushing human file...", flush=True)
            node._close_human_file()
            print("✓ Human file closed.", flush=True)

        # ── Cancel any in-progress startup thread ─────────────────────────
        if node.startup_thread is not None and node.startup_thread.is_alive():
            node.startup_thread_stop.set()
            node.startup_thread.join(timeout=5.0)

        if node.recording:
            print(
                f"  Interrupted while recording. "
                f"Frames on disk: {node.frame_count}, "
                f"Trajectory points on disk: {node.traj_count}",
                flush=True,
            )

        print(f" Data directory: {node.output_dir}", flush=True)

        try:
            node.destroy_node()
        except Exception as e:
            print(f"  Error destroying node: {e}", flush=True)

        try:
            rclpy.shutdown()
        except Exception as e:
            if "already called" not in str(e):
                print(f"  ROS2 shutdown error: {e}", flush=True)

        print("✓ Shutdown complete.", flush=True)


if __name__ == "__main__":
    main()