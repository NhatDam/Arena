#!/bin/bash
set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-$HOME/arena5_ws}"
RUN_ID="${RUN_ID:-$(date +%s)}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-data_recorder_${RUN_ID}}"
AUTO_START="${AUTO_START:-1}"
AUTO_STOP_ON_EXIT="${AUTO_STOP_ON_EXIT:-1}"
SERVICE_TIMEOUT_SEC="${SERVICE_TIMEOUT_SEC:-60}"
USE_SIM_TIME="${USE_SIM_TIME:-true}"

IMAGE_TOPIC="${IMAGE_TOPIC:-/task_generator_node/turtlebot/rgbd_camera/image}"
DEPTH_TOPIC="${DEPTH_TOPIC:-/task_generator_node/turtlebot/rgbd_camera/depth}"
CAMERA_INFO_TOPIC="${CAMERA_INFO_TOPIC:-/task_generator_node/turtlebot/rgbd_camera/camera_info}"
DETECTIONS_TOPIC="${DETECTIONS_TOPIC:-/task_generator_node/turtlebot/gt_human_bboxes_2d}"
ROBOT_STATE_TOPIC="${ROBOT_STATE_TOPIC:-/task_generator_node/robot_states}"
HUMAN_STATES_TOPIC="${HUMAN_STATES_TOPIC:-/task_generator_node/human_states}"

IMAGE_FPS="${IMAGE_FPS:-1.0}"
TRAJECTORY_FPS="${TRAJECTORY_FPS:-5.0}"
STARTUP_DELAY="${STARTUP_DELAY:-5.0}"
TOPIC_TIMEOUT="${TOPIC_TIMEOUT:-120.0}"
MIN_FRAMES_BEFORE_RECORD="${MIN_FRAMES_BEFORE_RECORD:-5}"

RECORDER_PID=""
RECORDING_STARTED=0
IN_CLEANUP=0

usage() {
    cat <<'EOF'
Usage:
  start_data_recorder.sh
  start_data_recorder.sh start <run_id> [experiment_tag]
  start_data_recorder.sh stop
  start_data_recorder.sh status

Default mode launches data_recorder_node, waits for its service, starts
recording, and stops recording on Ctrl+C.

Common environment overrides:
  RUN_ID=1001
  EXPERIMENT_TAG=hospital_1_default_05_ep001
  AUTO_START=0
  AUTO_STOP_ON_EXIT=1
  IMAGE_FPS=1.0
  TRAJECTORY_FPS=5.0
  IMAGE_TOPIC=/task_generator_node/turtlebot/rgbd_camera/image
  DEPTH_TOPIC=/task_generator_node/turtlebot/rgbd_camera/depth
  DETECTIONS_TOPIC=/task_generator_node/turtlebot/gt_human_bboxes_2d
  ROBOT_STATE_TOPIC=/task_generator_node/robot_states
  HUMAN_STATES_TOPIC=/task_generator_node/human_states
EOF
}

source_workspace() {
    if [ ! -f "$WORKSPACE_DIR/install/setup.bash" ]; then
        echo "[ERROR] Workspace setup not found: $WORKSPACE_DIR/install/setup.bash"
        echo "[ERROR] Build first, for example: colcon build --packages-select data_recorder"
        exit 1
    fi

    set +u
    source /opt/ros/humble/setup.bash
    source "$WORKSPACE_DIR/install/setup.bash"
    set -u
}

wait_for_service() {
    local service_name="$1"
    local elapsed=0

    echo "[INFO] Waiting for service: $service_name"
    until ros2 service list 2>/dev/null | grep -qx "$service_name"; do
        if [ "$elapsed" -ge "$SERVICE_TIMEOUT_SEC" ]; then
            echo "[ERROR] Service not available after ${SERVICE_TIMEOUT_SEC}s: $service_name"
            return 1
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
}

call_start_recording() {
    local run_id="$1"
    local experiment_tag="${2:-data_recorder_${run_id}}"

    wait_for_service "/data_recorder_start_recording"
    echo "[INFO] Starting data recording: run_id=$run_id experiment_tag=$experiment_tag"
    ros2 service call /data_recorder_start_recording hunav_msgs/srv/StartEvaluation \
        "{run_id: ${run_id}, experiment_tag: '${experiment_tag}'}"
}

call_stop_recording() {
    if ! ros2 service list 2>/dev/null | grep -qx "/data_recorder_stop_recording"; then
        echo "[WARN] /data_recorder_stop_recording is not available; nothing to stop."
        return 0
    fi

    echo "[INFO] Stopping data recording..."
    ros2 service call /data_recorder_stop_recording std_srvs/srv/Empty "{}" || true
}

show_status() {
    echo "[INFO] Recorder services:"
    ros2 service list 2>/dev/null | grep -E "/data_recorder_(start|stop)_recording" || true
    echo
    echo "[INFO] Recorder node:"
    ros2 node list 2>/dev/null | grep -E "data_recorder" || true
    echo
    echo "[INFO] Output root: $WORKSPACE_DIR/output"
}

cleanup() {
    if [ "$IN_CLEANUP" -eq 1 ]; then
        return 0
    fi
    IN_CLEANUP=1
    trap '' INT TERM

    echo
    echo "[INFO] Shutting down data recorder session..."

    if [ "$AUTO_STOP_ON_EXIT" -eq 1 ] && [ "$RECORDING_STARTED" -eq 1 ]; then
        call_stop_recording
    fi

    if [ -n "$RECORDER_PID" ] && kill -0 "$RECORDER_PID" 2>/dev/null; then
        kill "$RECORDER_PID" 2>/dev/null || true
        wait "$RECORDER_PID" 2>/dev/null || true
    fi
}

main_session() {
    echo "==========================================================================="
    echo "[INFO] Starting Arena data recorder"
    echo "==========================================================================="
    echo "[INFO] WORKSPACE_DIR=$WORKSPACE_DIR"
    echo "[INFO] RUN_ID=$RUN_ID"
    echo "[INFO] EXPERIMENT_TAG=$EXPERIMENT_TAG"
    echo "[INFO] Output root: $WORKSPACE_DIR/output"
    echo

    trap cleanup INT TERM EXIT

    echo "[INFO] Launching data_recorder_node..."
    ros2 run data_recorder data_recorder_node --ros-args \
        -p use_sim_time:="$USE_SIM_TIME" \
        -p image_topic:="$IMAGE_TOPIC" \
        -p depth_topic:="$DEPTH_TOPIC" \
        -p camera_info_topic:="$CAMERA_INFO_TOPIC" \
        -p detections_topic:="$DETECTIONS_TOPIC" \
        -p robot_state_topic:="$ROBOT_STATE_TOPIC" \
        -p human_states_topic:="$HUMAN_STATES_TOPIC" \
        -p image_fps:="$IMAGE_FPS" \
        -p trajectory_fps:="$TRAJECTORY_FPS" \
        -p startup_delay:="$STARTUP_DELAY" \
        -p topic_timeout:="$TOPIC_TIMEOUT" \
        -p min_frames_before_record:="$MIN_FRAMES_BEFORE_RECORD" &
    RECORDER_PID=$!

    wait_for_service "/data_recorder_start_recording"

    if [ "$AUTO_START" -eq 1 ]; then
        call_start_recording "$RUN_ID" "$EXPERIMENT_TAG"
        RECORDING_STARTED=1
        echo "[INFO] Recording active. Press Ctrl+C to stop and flush."
    else
        echo "[INFO] Recorder is running. AUTO_START=0, so no recording has been started."
        echo "[INFO] Start manually with:"
        echo "       ros2 service call /data_recorder_start_recording hunav_msgs/srv/StartEvaluation \"{run_id: ${RUN_ID}, experiment_tag: '${EXPERIMENT_TAG}'}\""
    fi

    wait "$RECORDER_PID"
}

source_workspace

case "${1:-run}" in
    run)
        main_session
        ;;
    start)
        call_start_recording "${2:-$RUN_ID}" "${3:-$EXPERIMENT_TAG}"
        ;;
    stop)
        call_stop_recording
        ;;
    status)
        show_status
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "[ERROR] Unknown command: $1"
        usage
        exit 1
        ;;
esac
