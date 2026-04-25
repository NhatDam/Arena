#!/usr/bin/env bash
###############################################################################
# run_all_scenarios.sh
#
# Iterates through all scenarios, performs setup, and runs NUM_RUNS for each.
###############################################################################
set -euo pipefail

# ── Configuration ────────────────
NUM_RUNS=2
SETTLE_TIME=30
KILL_WAIT=8
SIM="isaac"
WORLD="hospital_1"
ROBOT="turtlebot"
LOCAL_PLANNER="sicnav"
TM_OBSTACLES="scenario"
TM_ROBOTS="scenario"
HUMAN="hunav"
HEADLESS="1"

# ── Global State for Safe Exit ──
INTERRUPTED=false

# ── Define Scenario Order (doorway_bottleneck first) ─────────────────────
SCENARIOS=(
"group_blocking"
    "overtaking"
)

# ── Per-scenario Configuration ───────────────────────────────────────────
declare -A GOAL_X GOAL_Y GOAL_Z EXP_TAG SCENARIO_DURATION

GOAL_X[corridor_head_on]=9.5;    GOAL_Y[corridor_head_on]=30.0;  GOAL_Z[corridor_head_on]=0.0
GOAL_X[crossing_intersection]=11.0; GOAL_Y[crossing_intersection]=2.0; GOAL_Z[crossing_intersection]=0.0
GOAL_X[doorway_bottleneck]=13.5; GOAL_Y[doorway_bottleneck]=32.25; GOAL_Z[doorway_bottleneck]=0.0
GOAL_X[group_blocking]=9.5;     GOAL_Y[group_blocking]=30.0;     GOAL_Z[group_blocking]=0.0
GOAL_X[overtaking]=9.5;         GOAL_Y[overtaking]=30.0;         GOAL_Z[overtaking]=0.0

SCENARIO_DURATION[corridor_head_on]=300
SCENARIO_DURATION[crossing_intersection]=240
SCENARIO_DURATION[doorway_bottleneck]=80
SCENARIO_DURATION[group_blocking]=300
SCENARIO_DURATION[overtaking]=180

for s in "${SCENARIOS[@]}"; do EXP_TAG[$s]="$s"; done

# ── Paths ────────────────────────────────────────────────────────────────
WORLD_BASE_DIR="$HOME/arena5_ws/src/Arena/arena_simulation_setup/worlds/hospital_1/scenarios"
DEFAULT_DIR="$WORLD_BASE_DIR/default"

cleanup() {
    echo -e "\n[cleanup] Shutting down Arena and ROS 2 processes..."
    
    # 1. Kill the main launch process group if it exists
    if [[ -n "${LAUNCH_PID:-}" ]]; then
        # Sending SIGTERM to the process group (using negative PID)
        kill -TERM -"$LAUNCH_PID" 2>/dev/null || true
        sleep 2
        kill -9 -"$LAUNCH_PID" 2>/dev/null || true
    fi

    # 2. Kill any background jobs started by this shell script
    kill $(jobs -p) 2>/dev/null || true

    # 3. Force-kill stubborn patterns
    local PATTERNS=("ros2 launch" "arena.launch.py" "task_generator" "hunav_evaluator_node" "isaac" "rviz2")
    for pat in "${PATTERNS[@]}"; do 
        pkill -9 -f "$pat" 2>/dev/null || true
    done
    
    # Specific cleanup for your SocialWalker/LAM tasks
    pkill -9 -f "gt_bbox_projector.py" || true
    pkill -9 -f "overlay_gt_boxes.py" || true
    pkill -9 -f "socialwalker_state.py" || true
    pkill -9 -f "socialwalker_controller.py" || true

    # 4. Reset ROS 2 Daemon to clear middleware hang-ups
    ros2 daemon stop 2>/dev/null || true
    sleep 2
    ros2 daemon start 2>/dev/null || true
    echo "[cleanup] Done."
}

# ── Signal Handlers ──
term_handler() {
    echo -e "\n\n[!!!] CTRL+C DETECTED. Terminating entire sequence..."
    INTERRUPTED=true
    cleanup
    exit 1
}

trap term_handler SIGINT SIGTERM
trap cleanup EXIT

# ── Main Outer Loop (Scenario Switcher) ──────────────────────────────────
for SCENARIO in "${SCENARIOS[@]}"; do
    echo "################################################"
    echo " STARTING MASTER SEQUENCE: $SCENARIO"
    echo "################################################"

    RUN_DURATION=${SCENARIO_DURATION[$SCENARIO]}
    SOURCE_DIR="$WORLD_BASE_DIR/$SCENARIO"
    RESULTS_DIR="$HOME/arena_hunav_results/${WORLD}_${SCENARIO}/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$RESULTS_DIR"

    # 1. Setup Scenario Files
    echo "[setup] Swapping files for $SCENARIO..."
    rm -rf "${DEFAULT_DIR:?}"/*
    cp -r "$SOURCE_DIR"/. "$DEFAULT_DIR/"
    
    # 2. Build Once per Scenario
    echo "[setup] Running 'arena build'..."
if [ "$SCENARIO" == "group_blocking" ]; then
        echo "[setup] Detected group_blocking: Performing CLEAN build..."
        rm -rf "$HOME/arena5_ws/build/arena_simulation_setup"
        rm -rf "$HOME/arena5_ws/install/arena_simulation_setup"
        
        set +u
        arena build arena_simulation_setup --cmake-clean-cache || {
            echo "WARNING: Clean build failed for group_blocking, attempting to continue..."
        }
        set -u
    else
        echo "[setup] Performing INCREMENTAL build for $SCENARIO..."
        set +u
        arena build arena_simulation_setup || {
            echo "WARNING: Incremental build reported an issue, attempting to continue..."
        }
        set -u
    fi

echo "[setup] Scenario environment ready. Proceeding to runs..."

    # 3. Run inner loop
    for RUN in $(seq 1 "$NUM_RUNS"); do
        echo "--- Run $RUN / $NUM_RUNS for $SCENARIO ---"
        
        setsid ros2 launch arena_bringup arena.launch.py \
            sim:="$SIM" world:="$WORLD" robot:="$ROBOT" \
            local_planner:="$LOCAL_PLANNER" tm_obstacles:="$TM_OBSTACLES" \
            tm_robots:="$TM_ROBOTS" human:="$HUMAN" \
            use_sim_time:=true headless:="$HEADLESS" &
        LAUNCH_PID=$!
        
        sleep "$SETTLE_TIME"

        ros2 param set /task_generator_node auto_reset false 2>/dev/null || true
        #python3 ~/arena5_ws/src/SocialWalker-lam_task1/arena_gt_bbox_projector/arena_gt_bbox_projector/gt_bbox_projector.py &
        #python3 ~/arena5_ws/src/SocialWalker-lam_task1/arena_gt_bbox_overlay/arena_gt_bbox_overlay/overlay_gt_boxes.py &
        #python3 ~/arena5_ws/src/SocialWalker-lam_task1/lam_task/socialwalker_state.py &
        #python3 ~/arena5_ws/src/SocialWalker-lam_task1/lam_task/socialwalker_controller.py &
        

        EXPERIMENT_TAG="${EXP_TAG[$SCENARIO]}_run_${RUN}"
        GX="${GOAL_X[$SCENARIO]}"; GY="${GOAL_Y[$SCENARIO]}"; GZ="${GOAL_Z[$SCENARIO]}"
        
        ros2 service call /hunav_start_recording hunav_msgs/srv/StartEvaluation \
            "{robot_goal: {header: {frame_id: 'map'}, pose: {position: {x: $GX, y: $GY, z: $GZ}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}, experiment_tag: '$EXPERIMENT_TAG', run_id: $RUN}" || true

        sleep "$RUN_DURATION"
        ros2 service call /hunav_stop_recording std_srvs/srv/Empty "{}" || true
        sleep 3

        find . -maxdepth 1 -name "metrics*.csv" -newer /tmp/.arena_run_marker 2>/dev/null \
            | xargs -I{} cp -v {} "$RESULTS_DIR/" 2>/dev/null || true

        cleanup
        touch /tmp/.arena_run_marker
    done
done

echo "COMPLETED ALL SCENARIOS."
