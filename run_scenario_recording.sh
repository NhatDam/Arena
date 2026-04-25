#!/usr/bin/env bash
###############################################################################
# run_scenario_recording.sh
#
# Launches the Arena in scenario mode on hospital_1 (Isaac sim).
# Manually swaps scenario files into 'default' and runs 'arena build' once.
###############################################################################
set -euo pipefail

# ──────────────── Configuration ────────────────
NUM_RUNS=1
SETTLE_TIME=1           # seconds to wait for nodes to come up
KILL_WAIT=8               # seconds to wait after killing before next run
SIM="isaac"
WORLD="hospital_1"
ROBOT="turtlebot"
LOCAL_PLANNER="mppi"
TM_OBSTACLES="scenario"
TM_ROBOTS="scenario"
HUMAN="hunav"
HEADLESS="1"              # 2 = fully headless (no rviz, no gui)
SCENARIO="${1:-corridor_head_on}"

# ── Per-scenario Configuration ───────────────────────────────────────────
declare -A GOAL_X GOAL_Y GOAL_Z EXP_TAG SCENARIO_DURATION

# Goal Coordinates
GOAL_X[corridor_head_on]=9.5;    GOAL_Y[corridor_head_on]=30.0;  GOAL_Z[corridor_head_on]=0.0
GOAL_X[crossing_intersection]=11.0; GOAL_Y[crossing_intersection]=2.0; GOAL_Z[crossing_intersection]=0.0
GOAL_X[doorway_bottleneck]=13.5; GOAL_Y[doorway_bottleneck]=32.25; GOAL_Z[doorway_bottleneck]=0.0
GOAL_X[group_blocking]=9.5;     GOAL_Y[group_blocking]=30.0;     GOAL_Z[group_blocking]=0.0
GOAL_X[overtaking]=9.5;         GOAL_Y[overtaking]=30.0;         GOAL_Z[overtaking]=0.0

# Dynamic Run Durations
SCENARIO_DURATION[corridor_head_on]=300
SCENARIO_DURATION[crossing_intersection]=180
SCENARIO_DURATION[doorway_bottleneck]=180
SCENARIO_DURATION[group_blocking]=300
SCENARIO_DURATION[overtaking]=400

# Experiment Tags
EXP_TAG[corridor_head_on]="corridor_head_on"
EXP_TAG[crossing_intersection]="crossing_intersection"
EXP_TAG[doorway_bottleneck]="doorway_bottleneck"
EXP_TAG[group_blocking]="group_blocking"
EXP_TAG[overtaking]="overtaking"

# Validate scenario name
if [[ -z "${GOAL_X[$SCENARIO]+set}" ]]; then
    echo "ERROR: Unknown scenario '$SCENARIO'"
    echo "Available: ${!GOAL_X[*]}"
    exit 1
fi

RUN_DURATION=${SCENARIO_DURATION[$SCENARIO]}
RESULTS_DIR="$HOME/arena_hunav_results/${WORLD}_${SCENARIO}/$(date +%Y%m%d_%H%M%S)"

# ── Scenario File Setup (Run Once) ───────────────────────────────────────
WORLD_BASE_DIR="$HOME/arena5_ws/src/Arena/arena_simulation_setup/worlds/hospital_1/scenarios"
DEFAULT_DIR="$WORLD_BASE_DIR/default"
SOURCE_DIR="$WORLD_BASE_DIR/$SCENARIO"

echo "[setup] Preparing scenario: $SCENARIO (Duration: ${RUN_DURATION}s)"

# 1. Clear default directory
if [ -d "$DEFAULT_DIR" ]; then
    echo "[setup] Cleaning default scenario directory..."
    rm -rf "${DEFAULT_DIR:?}"/*
else
    mkdir -p "$DEFAULT_DIR"
fi

# 2. Copy selected scenario files
if [ -d "$SOURCE_DIR" ]; then
    echo "[setup] Copying files from $SOURCE_DIR to $DEFAULT_DIR..."
    cp -r "$SOURCE_DIR"/. "$DEFAULT_DIR/"
else
    echo "ERROR: Source scenario directory $SOURCE_DIR does not exist."
    exit 1
fi

# 3. Rebuild Arena environment
echo "[setup] Running 'arena build'..."

if [ ! -f "$DEFAULT_DIR/bt.xml" ] && [ -f "$DEFAULT_DIR/bt1.xml" ]; then
        echo "[setup] Creating bt.xml symlink for build compatibility..."
        ln -sf "$DEFAULT_DIR/bt1.xml" "$DEFAULT_DIR/bt.xml"
fi
    
# Temporarily disable strict variable checking to prevent the ZSH_VERSION error
set +u
# Run build, but allow it to continue even if there is stderr output
arena build || {
    echo "WARNING: 'arena build' reported an issue, but attempting to continue..."
}
set -u

echo "[setup] Scenario environment ready. Proceeding to runs..."

# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "$RESULTS_DIR"
echo "Results directory: $RESULTS_DIR"
echo "Starting $NUM_RUNS runs of ${RUN_DURATION}s each..."

cleanup() {
    echo "[cleanup] Killing arena processes …"
    if [[ -n "${LAUNCH_PID:-}" ]] && kill -0 "$LAUNCH_PID" 2>/dev/null; then
        kill -- -"$LAUNCH_PID" 2>/dev/null || true
        sleep 2
        kill -9 -- -"$LAUNCH_PID" 2>/dev/null || true
    fi

    local PATTERNS=("ros2 launch" "arena.launch.py" "task_generator" "hunav_evaluator_node" "rviz2" "nav2_container" "isaac" "gazebo")
    for pat in "${PATTERNS[@]}"; do
        pkill -9 -f "$pat" 2>/dev/null || true
    done
    pkill -9 -f "gt_bbox_projector.py" || true
    pkill -9 -f "overlay_gt_boxes.py" || true
    pkill -9 -f "socialwalker_state.py" || true
    pkill -9 -f "socialwalker_controller.py" || true
    
    ros2 daemon stop 2>/dev/null || true
    sleep "$KILL_WAIT"
    ros2 daemon start 2>/dev/null || true
    echo "[cleanup] Done."
}

trap cleanup EXIT

for RUN in $(seq 1 "$NUM_RUNS"); do
    echo "=============================================="
    echo " RUN $RUN / $NUM_RUNS | Scenario: $SCENARIO"
    echo "=============================================="

    setsid ros2 launch arena_bringup arena.launch.py \
        sim:="$SIM" \
        world:="$WORLD" \
        robot:="$ROBOT" \
        local_planner:="$LOCAL_PLANNER" \
        tm_obstacles:="$TM_OBSTACLES" \
        tm_robots:="$TM_ROBOTS" \
        human:="$HUMAN" \
        use_sim_time:=true \
        headless:="$HEADLESS" &
    LAUNCH_PID=$!
    
    sleep "$SETTLE_TIME"
    ros2 param set /task_generator_node auto_reset false 2>/dev/null || true
    python3 ~/arena5_ws/src/SocialWalker-lam_task1/arena_gt_bbox_projector/arena_gt_bbox_projector/gt_bbox_projector.py &
    python3 ~/arena5_ws/src/SocialWalker-lam_task1/arena_gt_bbox_overlay/arena_gt_bbox_overlay/overlay_gt_boxes.py &
    python3 ~/arena5_ws/src/SocialWalker-lam_task1/lam_task/socialwalker_state.py &
    python3 ~/arena5_ws/src/SocialWalker-lam_task1/lam_task/socialwalker_controller.py &

    echo "All SocialWalker nodes started. Press Ctrl+C to stop all."
done

echo "ALL $NUM_RUNS RUNS COMPLETE. Results: $RESULTS_DIR"
