# Data Recorder

A ROS2 node to record robot RGB frames, detection bboxes, and trajectory data during simulation runs.

## Features

- Records RGB frames as JPG images
- Saves per-frame detection bounding boxes as JSON
- Records robot trajectory (x, y, yaw) with timestamps
- Uses same start/stop service pattern as hunav_evaluator
- Synchronized frame and detection capture

## Output Structure

```
~/arena5_ws/output/
├── <run_id>/
│   ├── 0000.jpg
│   ├── 0001.jpg
│   ├── ...
│   ├── 0000.pred_label.json
│   ├── 0001.pred_label.json
│   ├── ...
│   └── traj_data.txt
```

## JSON Format

Each frame gets a corresponding JSON annotation file:

```json
{
  "video_id": "run_id_123",
  "file_name": "0000.jpg",
  "detections": [
    {
      "bbox_xyxy": [xmin, ymin, xmax, ymax],
      "conf": 1.0
    },
    ...
  ]
}
```

## Trajectory Format

The `traj_data.txt` file contains robot trajectory with timestamps:

```
timestamp x y yaw
0.000000 x0 y0 yaw0
0.033333 x1 y1 yaw1
...
```

## Installation

```bash
cd ~/arena5_ws
colcon build --packages-select data_recorder
source install/setup.bash
```

## Usage

### Start the node

```bash
ros2 launch data_recorder data_recorder.launch.py
```

Or directly:

```bash
ros2 run data_recorder data_recorder_node
```

### Control recording

In another terminal:

```bash
# Start recording
ros2 service call /data_recorder_start_recording hunav_msgs/srv/StartEvaluation "{run_id: 123, experiment_tag: 'my_exp'}"

# Stop recording (saves all data)
ros2 service call /data_recorder_stop_recording std_srvs/srv/Empty
```

### Check output

```bash
ls -la ~/arena5_ws/output/123/
```

## Parameters

The node accepts the following parameters (can be overridden in launch file):

- `image_topic`: RGB image topic (default: `/task_generator_node/turtlebot/rgbd_camera/image`)
- `detections_topic`: Detection2DArray topic (default: `/task_generator_node/turtlebot/gt_human_bboxes_2d`)
- `robot_state_topic`: Robot state topic (default: `/task_generator_node/robot_states`)

## Services

### `/data_recorder_start_recording` (hunav_msgs/srv/StartEvaluation)

Start recording data. Request fields:
- `run_id` (int): Unique identifier for this run
- `experiment_tag` (str): Tag for the experiment
- `robot_goal` (optional): Goal pose

Response:
- `success` (bool): Whether the request succeeded

### `/data_recorder_stop_recording` (std_srvs/srv/Empty)

Stop recording and save all collected data.

## Notes

- Frames and detection JSON are synchronized at capture time
- Trajectory data is collected independently and saved with timestamps
- Ground truth detections always have confidence = 1.0
- Bounding boxes are in xyxy format (xmin, ymin, xmax, ymax)
- Output directory is created automatically if it doesn't exist

## Dependencies

- rclpy
- sensor_msgs
- vision_msgs
- hunav_msgs
- cv_bridge
- opencv-python
- numpy
