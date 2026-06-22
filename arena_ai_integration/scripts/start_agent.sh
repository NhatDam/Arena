#!/bin/bash
set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-$HOME/arena5_ws}"
AGENT_TYPE="${AGENT_TYPE:-socialnav}"
ENABLE_HUMAN_TRACKING="${ENABLE_HUMAN_TRACKING:-true}"
ENABLE_VISUALIZATION="${ENABLE_VISUALIZATION:-true}"
TRAIN_MODE="${TRAIN_MODE:-false}"

AI_INTEGRATION_DIR="$WORKSPACE_DIR/src/Arena/arena_ai_integration"
SOCIAL_NAV_DIR="$WORKSPACE_DIR/src/arena-social-nav"

case "$AGENT_TYPE" in
  socialnav)
    MODEL_CONFIG="${MODEL_CONFIG:-$AI_INTEGRATION_DIR/config/models/socialnav_film.yaml}"
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$AI_INTEGRATION_DIR/checkpoints/SocialNav_1_path.pth}"
    PARAMS_FILE="socialnav_params.yaml"
    ;;
  urbannav)
    MODEL_CONFIG="${MODEL_CONFIG:-$AI_INTEGRATION_DIR/config/models/urbannav_film.yaml}"
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$AI_INTEGRATION_DIR/checkpoints/UrbanNav_FiLM.pth}"
    PARAMS_FILE="urbannav_params.yaml"
    ENABLE_HUMAN_TRACKING="${ENABLE_HUMAN_TRACKING:-false}"
    ;;
  lelan)
    MODEL_CONFIG="${MODEL_CONFIG:-$AI_INTEGRATION_DIR/config/models/lelan.yaml}"
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$AI_INTEGRATION_DIR/checkpoints/LeLan_latest.pth}"
    PARAMS_FILE="lelan_params.yaml"
    ENABLE_HUMAN_TRACKING="${ENABLE_HUMAN_TRACKING:-false}"
    ;;
  *)
    echo "[ERROR] Unsupported AGENT_TYPE=$AGENT_TYPE (use socialnav, urbannav, or lelan)"
    exit 1
    ;;
esac

cleanup() {
    echo
    echo "[INFO] Stopping AI controller session (agent=$AGENT_TYPE)..."
    pkill -P $$ 2>/dev/null || true
    sleep 1
    pkill -f "ai_controller|human_states_bridge.py|semantic_laser_filter.py" 2>/dev/null || true
    pkill -f "nav2_lifecycle_manager|task_generator_node|bt_navigator|planner_server" 2>/dev/null || true
}
trap cleanup INT TERM

echo "═══════════════════════════════════════════════════════════════════════════"
echo "[INFO] Starting Arena AI Integration (agent=$AGENT_TYPE)"
echo "═══════════════════════════════════════════════════════════════════════════"

if [ ! -f "$WORKSPACE_DIR/install/setup.bash" ]; then
    echo "[ERROR] Workspace not built: $WORKSPACE_DIR/install/setup.bash missing"
    exit 1
fi

set +u
source "$WORKSPACE_DIR/install/setup.bash"
set -u

if [ ! -f "$MODEL_CHECKPOINT" ]; then
    echo "[ERROR] Checkpoint not found: $MODEL_CHECKPOINT"
    exit 1
fi

if [ "$AGENT_TYPE" = "socialnav" ]; then
    echo "[INFO] Starting Human States Bridge..."
    python3 "$SOCIAL_NAV_DIR/ros2_nodes/socialnav/human_states_bridge.py" &
    sleep 1
fi

echo "[INFO] Launching Arena simulator..."
arena launch \
    sim:=isaac \
    world:=hospital_1 \
    robot:=turtlebot \
    tm_robots:=scenario \
    tm_obstacles:=scenario \
    use_sim_time:=true \
    headless:=1 \
    train_mode:="$TRAIN_MODE" &
ARENA_PID=$!
sleep 12

if [ "$AGENT_TYPE" = "socialnav" ]; then
    echo "[INFO] Starting Semantic Laser Filter..."
    python3 "$SOCIAL_NAV_DIR/ros2_nodes/socialnav/semantic_laser_filter.py" --ros-args -p use_sim_time:=true &
    sleep 1
fi

PKG_SHARE="$(ros2 pkg prefix arena_ai_integration)/share/arena_ai_integration"

echo "[INFO] Starting unified AI controller..."
ros2 launch arena_ai_integration ai_controller.launch.py \
    agent_type:="$AGENT_TYPE" \
    params_file:="$PKG_SHARE/config/$PARAMS_FILE" \
    --ros-args \
    -p model_config_path:="$MODEL_CONFIG" \
    -p model_checkpoint_path:="$MODEL_CHECKPOINT" \
    -p enable_human_tracking:="$ENABLE_HUMAN_TRACKING" \
    -p enable_bev_visualization:="$ENABLE_VISUALIZATION" &

sleep 3

if [ -f "$SOCIAL_NAV_DIR/ros2_nodes/urbannav/arena_urbannav_bridge.py" ]; then
    python3 "$SOCIAL_NAV_DIR/ros2_nodes/urbannav/arena_urbannav_bridge.py" &
fi

if [ "$ENABLE_VISUALIZATION" = "true" ]; then
    RVIZ_CONFIG="$WORKSPACE_DIR/src/Arena/arena_bringup/config/default.rviz"
    if [ -f "$RVIZ_CONFIG" ]; then
        ros2 run rviz2 rviz2 -d "$RVIZ_CONFIG" &
    fi
fi

wait
