#!/bin/bash
set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-$HOME/arena5_ws}"

# Arena/data-collection run configuration.
SIM="${SIM:-isaac}"
WORLD="${WORLD:-hospital_1}"
ROBOT="${ROBOT:-turtlebot}"
LOCAL_PLANNER="${LOCAL_PLANNER:-dwb}"
INTER_PLANNER="${INTER_PLANNER:-navigate_w_replanning_time}"
GLOBAL_PLANNER="${GLOBAL_PLANNER:-navfn}"
AGENT_NAME="${AGENT_NAME:-}"
TM_OBSTACLES="${TM_OBSTACLES:-scenario}"
TM_ROBOTS="${TM_ROBOTS:-scenario}"
TM_MODULES="${TM_MODULES:-rviz_ui}"
HUMAN="${HUMAN:-hunav}"
HEADLESS="${HEADLESS:-1}"
USE_SIM_TIME="${USE_SIM_TIME:-true}"
ENV_N="${ENV_N:-1}"

# Scenario/episode configuration.
SCENARIOS="${SCENARIOS:-default_05}"
EPISODES="${EPISODES:-1}"
RUN_DURATION="${RUN_DURATION:-180}"
SETTLE_TIME="${SETTLE_TIME:-20}"
BETWEEN_EPISODE_WAIT="${BETWEEN_EPISODE_WAIT:-8}"
RUN_ID_START="${RUN_ID_START:-${RUN_ID:-$(date +%s)}}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-arena_data}"

# Scenario preparation. The scenario task mode expects the active scenario in
# worlds/<world>/scenarios/default, so this mirrors run_scenario_recording.sh.
PREPARE_SCENARIO="${PREPARE_SCENARIO:-1}"
BUILD_BEFORE_SCENARIO="${BUILD_BEFORE_SCENARIO:-1}"
BUILD_TARGET="${BUILD_TARGET:-arena_simulation_setup}"
SCENARIO_ROOT="${SCENARIO_ROOT:-$WORKSPACE_DIR/src/Arena/arena_simulation_setup/worlds/$WORLD/scenarios}"
DEFAULT_SCENARIO_DIR="${DEFAULT_SCENARIO_DIR:-$SCENARIO_ROOT/default}"

# Recorder topics. Override these if ros2 topic list shows different names.
IMAGE_TOPIC="${IMAGE_TOPIC:-/task_generator_node/turtlebot/rgbd_camera/image}"
DEPTH_TOPIC="${DEPTH_TOPIC:-/task_generator_node/turtlebot/rgbd_camera/depth}"
CAMERA_INFO_TOPIC="${CAMERA_INFO_TOPIC:-/task_generator_node/turtlebot/rgbd_camera/camera_info}"
DETECTIONS_TOPIC="${DETECTIONS_TOPIC:-/task_generator_node/turtlebot/gt_human_bboxes_2d}"
ROBOT_STATE_TOPIC="${ROBOT_STATE_TOPIC:-/task_generator_node/robot_states}"
HUMAN_STATES_TOPIC="${HUMAN_STATES_TOPIC:-/task_generator_node/human_states}"

IMAGE_FPS="${IMAGE_FPS:-1.0}"
TRAJECTORY_FPS="${TRAJECTORY_FPS:-5.0}"
RECORDER_STARTUP_DELAY="${RECORDER_STARTUP_DELAY:-2.0}"
RECORDER_TOPIC_TIMEOUT="${RECORDER_TOPIC_TIMEOUT:-120.0}"
MIN_FRAMES_BEFORE_RECORD="${MIN_FRAMES_BEFORE_RECORD:-5}"

SERVICE_TIMEOUT_SEC="${SERVICE_TIMEOUT_SEC:-60}"
TOPIC_WAIT_SEC="${TOPIC_WAIT_SEC:-180}"
TOPIC_SAMPLE_TIMEOUT_SEC="${TOPIC_SAMPLE_TIMEOUT_SEC:-5}"
CLEAN_STALE_PROCESSES="${CLEAN_STALE_PROCESSES:-1}"
ROS_DAEMON_RESTART_BETWEEN_EPISODES="${ROS_DAEMON_RESTART_BETWEEN_EPISODES:-1}"

ARENA_PID=""
RECORDER_PID=""
RECORDING_STARTED=0
IN_CLEANUP=0

usage() {
    cat <<'EOF'
Usage:
  start_data_recorder.sh [run]
  start_data_recorder.sh status
  start_data_recorder.sh stop

This script is a full data-collection orchestrator:
  1. prepare scenario
  2. launch Arena with a baseline planner
  3. wait for image/trajectory topics
  4. launch data_recorder
  5. start recording
  6. stop recording after the episode duration
  7. cleanup and repeat

Common environment overrides:
  SCENARIOS=default_05,default_04
  EPISODES=2
  RUN_DURATION=180
  LOCAL_PLANNER=dwb
  INTER_PLANNER=navigate_w_replanning_time
  AGENT_NAME=
  RUN_ID_START=1000
  EXPERIMENT_PREFIX=hospital_baseline
  HEADLESS=1
  SIM=isaac
  WORLD=hospital_1
  ROBOT=turtlebot

Recorder topic overrides:
  IMAGE_TOPIC=/task_generator_node/turtlebot/rgbd_camera/image
  ROBOT_STATE_TOPIC=/task_generator_node/robot_states
  DETECTIONS_TOPIC=/task_generator_node/turtlebot/gt_human_bboxes_2d

Advanced:
  PREPARE_SCENARIO=0             # do not copy scenario into default
  BUILD_BEFORE_SCENARIO=0        # skip arena build after scenario copy
  CLEAN_STALE_PROCESSES=0        # do not pkill old Arena/recorder processes
EOF
}

source_workspace() {
    if [ ! -f "$WORKSPACE_DIR/install/setup.bash" ]; then
        echo "[ERROR] Workspace setup not found: $WORKSPACE_DIR/install/setup.bash"
        echo "[ERROR] Build first, for example: colcon build --packages-select data_recorder arena_ai_integration"
        exit 1
    fi

    set +u
    if [ -f "$WORKSPACE_DIR/arena" ]; then
        pushd "$WORKSPACE_DIR" >/dev/null
        source "$WORKSPACE_DIR/arena"
        popd >/dev/null
    else
        source /opt/ros/humble/setup.bash
        source "$WORKSPACE_DIR/install/setup.bash"
    fi
    set -u
}

normalize_integer() {
    local label="$1"
    local value="$2"

    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] $label must be a non-negative integer, got: $value" >&2
        exit 1
    fi

    echo $((10#$value))
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

wait_for_topic_message() {
    local topic_name="$1"
    local elapsed=0

    echo "[INFO] Waiting for topic data: $topic_name"
    while [ "$elapsed" -lt "$TOPIC_WAIT_SEC" ]; do
        if ros2 topic list 2>/dev/null | grep -qx "$topic_name"; then
            if timeout "$TOPIC_SAMPLE_TIMEOUT_SEC" ros2 topic echo --once "$topic_name" >/dev/null 2>&1; then
                echo "[INFO] Topic is publishing: $topic_name"
                return 0
            fi
        fi

        sleep 2
        elapsed=$((elapsed + 2))
    done

    echo "[ERROR] No messages received on $topic_name after ${TOPIC_WAIT_SEC}s"
    echo "[ERROR] Check topic names with: ros2 topic list | grep -E 'image|rgbd|robot_states|bbox|human_states'"
    return 1
}

prepare_scenario() {
    local scenario="$1"
    local source_dir="$SCENARIO_ROOT/$scenario"

    if [ "$PREPARE_SCENARIO" -ne 1 ]; then
        echo "[INFO] PREPARE_SCENARIO=0, using current active scenario."
        return 0
    fi

    if [ ! -d "$source_dir" ]; then
        echo "[ERROR] Scenario directory not found: $source_dir"
        exit 1
    fi

    echo "[INFO] Preparing scenario '$scenario'"
    mkdir -p "$DEFAULT_SCENARIO_DIR"
    rm -rf "${DEFAULT_SCENARIO_DIR:?}"/*
    cp -a "$source_dir"/. "$DEFAULT_SCENARIO_DIR/"

    if [ ! -f "$DEFAULT_SCENARIO_DIR/bt.xml" ] && [ -f "$DEFAULT_SCENARIO_DIR/bt1.xml" ]; then
        ln -sf "$DEFAULT_SCENARIO_DIR/bt1.xml" "$DEFAULT_SCENARIO_DIR/bt.xml"
    fi

    if [ "$BUILD_BEFORE_SCENARIO" -eq 1 ]; then
        echo "[INFO] Building scenario assets: arena build $BUILD_TARGET"
        set +u
        arena build "$BUILD_TARGET" || {
            echo "[WARN] arena build reported an issue; continuing anyway."
        }
        set -u
        source_workspace
    fi
}

clean_stale_processes() {
    if [ "$CLEAN_STALE_PROCESSES" -ne 1 ]; then
        return 0
    fi

    echo "[INFO] Cleaning stale Arena/recorder processes..."
    pkill -f "data_recorder_node" 2>/dev/null || true
    pkill -f "arena.launch.py" 2>/dev/null || true
    pkill -f "task_generator" 2>/dev/null || true
    pkill -f "hunav_evaluator_node" 2>/dev/null || true
    pkill -f "nav2_container" 2>/dev/null || true
    pkill -f "rviz2" 2>/dev/null || true
}

launch_arena() {
    local launch_args=(
        sim:="$SIM"
        world:="$WORLD"
        robot:="$ROBOT"
        local_planner:="$LOCAL_PLANNER"
        inter_planner:="$INTER_PLANNER"
        global_planner:="$GLOBAL_PLANNER"
        tm_obstacles:="$TM_OBSTACLES"
        tm_robots:="$TM_ROBOTS"
        tm_modules:="$TM_MODULES"
        human:="$HUMAN"
        use_sim_time:="$USE_SIM_TIME"
        env_n:="$ENV_N"
        headless:="$HEADLESS"
    )

    if [ -n "$AGENT_NAME" ]; then
        launch_args+=(agent_name:="$AGENT_NAME")
    fi

    echo "[INFO] Launching Arena baseline:"
    echo "[INFO]   sim=$SIM world=$WORLD robot=$ROBOT local_planner=$LOCAL_PLANNER inter_planner=$INTER_PLANNER agent_name='${AGENT_NAME}'"

    setsid ros2 launch arena_bringup arena.launch.py "${launch_args[@]}" &

    ARENA_PID=$!
}

launch_recorder() {
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
        -p startup_delay:="$RECORDER_STARTUP_DELAY" \
        -p topic_timeout:="$RECORDER_TOPIC_TIMEOUT" \
        -p min_frames_before_record:="$MIN_FRAMES_BEFORE_RECORD" &

    RECORDER_PID=$!
    wait_for_service "/data_recorder_start_recording"
}

call_start_recording() {
    local run_id="$1"
    local experiment_tag="$2"
    local output

    echo "[INFO] Starting data recording: run_id=$run_id experiment_tag=$experiment_tag"
    output="$(ros2 service call /data_recorder_start_recording hunav_msgs/srv/StartEvaluation \
        "{run_id: ${run_id}, experiment_tag: '${experiment_tag}'}" 2>&1)"
    echo "$output"

    if ! grep -q "success=True" <<<"$output"; then
        echo "[ERROR] data_recorder rejected start request."
        echo "[ERROR] Common causes: stale recorder node, duplicate service, or previous startup still in progress."
        return 1
    fi

    RECORDING_STARTED=1
}

call_stop_recording() {
    if ! ros2 service list 2>/dev/null | grep -qx "/data_recorder_stop_recording"; then
        echo "[WARN] /data_recorder_stop_recording is not available; nothing to stop."
        return 0
    fi

    echo "[INFO] Stopping data recording..."
    ros2 service call /data_recorder_stop_recording std_srvs/srv/Empty "{}" || true
    RECORDING_STARTED=0
}

shutdown_current_episode() {
    if [ "$IN_CLEANUP" -eq 1 ]; then
        return 0
    fi
    IN_CLEANUP=1

    if [ "$RECORDING_STARTED" -eq 1 ]; then
        call_stop_recording
    fi

    if [ -n "$RECORDER_PID" ] && kill -0 "$RECORDER_PID" 2>/dev/null; then
        echo "[INFO] Stopping data_recorder_node..."
        kill "$RECORDER_PID" 2>/dev/null || true
        wait "$RECORDER_PID" 2>/dev/null || true
    fi
    RECORDER_PID=""

    if [ -n "$ARENA_PID" ] && kill -0 "$ARENA_PID" 2>/dev/null; then
        echo "[INFO] Stopping Arena..."
        kill -- -"$ARENA_PID" 2>/dev/null || true
        sleep 2
        kill -9 -- -"$ARENA_PID" 2>/dev/null || true
        wait "$ARENA_PID" 2>/dev/null || true
    fi
    ARENA_PID=""

    pkill -f "data_recorder_node" 2>/dev/null || true

    if [ "$ROS_DAEMON_RESTART_BETWEEN_EPISODES" -eq 1 ]; then
        ros2 daemon stop 2>/dev/null || true
        sleep 1
        ros2 daemon start 2>/dev/null || true
    fi

    sleep "$BETWEEN_EPISODE_WAIT"
    IN_CLEANUP=0
}

handle_signal() {
    echo
    echo "[INFO] Shutdown requested."
    trap '' INT TERM
    shutdown_current_episode
    exit 130
}

handle_exit() {
    local status=$?

    if [ "$RECORDING_STARTED" -eq 1 ] || [ -n "$RECORDER_PID" ] || [ -n "$ARENA_PID" ]; then
        echo
        echo "[INFO] Exiting; cleaning up active episode."
        shutdown_current_episode
    fi

    return "$status"
}

show_status() {
    echo "[INFO] Recorder services:"
    ros2 service list 2>/dev/null | grep -E "/data_recorder_(start|stop)_recording" || true
    echo
    echo "[INFO] Recorder nodes:"
    ros2 node list 2>/dev/null | grep -E "data_recorder" || true
    echo
    echo "[INFO] Arena/task topics:"
    ros2 topic list 2>/dev/null | grep -E "image|rgbd|robot_states|human_states|bbox" || true
    echo
    echo "[INFO] Output root: $WORKSPACE_DIR/output"
}

run_episode_timer() {
    local duration="$1"
    local elapsed=0

    echo "[INFO] Recording for ${duration}s..."
    while [ "$elapsed" -lt "$duration" ]; do
        if [ -n "$ARENA_PID" ] && ! kill -0 "$ARENA_PID" 2>/dev/null; then
            echo "[ERROR] Arena exited before episode finished."
            return 1
        fi
        if [ -n "$RECORDER_PID" ] && ! kill -0 "$RECORDER_PID" 2>/dev/null; then
            echo "[ERROR] data_recorder_node exited before episode finished."
            return 1
        fi

        sleep 5
        elapsed=$((elapsed + 5))
        if [ $((elapsed % 30)) -eq 0 ]; then
            echo "[INFO]   elapsed ${elapsed}/${duration}s"
        fi
    done
}

run_collection() {
    local run_id_start
    local episodes
    local run_offset=0

    run_id_start="$(normalize_integer RUN_ID_START "$RUN_ID_START")"
    episodes="$(normalize_integer EPISODES "$EPISODES")"

    if [ "$episodes" -lt 1 ]; then
        echo "[ERROR] EPISODES must be at least 1."
        exit 1
    fi

    IFS=',' read -ra scenario_list <<<"$SCENARIOS"

    echo "==========================================================================="
    echo "[INFO] Arena data collection runner"
    echo "==========================================================================="
    echo "[INFO] workspace=$WORKSPACE_DIR"
    echo "[INFO] scenarios=$SCENARIOS"
    echo "[INFO] episodes=$episodes duration=${RUN_DURATION}s"
    echo "[INFO] baseline local_planner=$LOCAL_PLANNER agent_name='${AGENT_NAME}'"
    echo "[INFO] output root=$WORKSPACE_DIR/output"

    clean_stale_processes

    for scenario in "${scenario_list[@]}"; do
        scenario="${scenario//[[:space:]]/}"
        if [ -z "$scenario" ]; then
            continue
        fi

        prepare_scenario "$scenario"

        for episode in $(seq 1 "$episodes"); do
            local run_id
            local tag

            run_id=$((run_id_start + run_offset))
            tag="${EXPERIMENT_PREFIX}_${WORLD}_${scenario}_${LOCAL_PLANNER}_ep$(printf "%03d" "$episode")"
            run_offset=$((run_offset + 1))

            echo
            echo "==========================================================================="
            echo "[INFO] Scenario=$scenario episode=$episode/$episodes run_id=$run_id"
            echo "==========================================================================="

            launch_arena
            echo "[INFO] Waiting ${SETTLE_TIME}s for Arena startup..."
            sleep "$SETTLE_TIME"
            ros2 param set /task_generator_node auto_reset false 2>/dev/null || true

            wait_for_topic_message "$IMAGE_TOPIC"
            wait_for_topic_message "$ROBOT_STATE_TOPIC"

            launch_recorder
            call_start_recording "$run_id" "$tag"

            run_episode_timer "$RUN_DURATION" || {
                shutdown_current_episode
                return 1
            }

            shutdown_current_episode
            echo "[INFO] Episode complete. Data directory should be: $WORKSPACE_DIR/output/$run_id"
        done
    done

    echo
    echo "[INFO] Data collection complete."
}

source_workspace

case "${1:-run}" in
    run)
        trap handle_signal INT TERM
        trap handle_exit EXIT
        run_collection
        ;;
    status)
        show_status
        ;;
    stop)
        call_stop_recording
        clean_stale_processes
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
