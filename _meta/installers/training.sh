#!/bin/bash -i

install(){
    set -e
    
    cd "$ARENA_DIR"

    echo "Initializing arena_training submodule..."
    git submodule update --init --checkout --remote --depth 1 arena_training

    echo "Initializing rosnav_rl submodule..."
    pushd "$ARENA_DIR/arena_training" > /dev/null
    git submodule update --init --depth 1 deps/rosnav_rl
    popd > /dev/null

    echo "Installing training Python dependencies into Arena venv..."
    pushd "$ARENA_DIR/arena_training" > /dev/null
    uv pip install --python "$ARENA_DIR/.venv/bin/python" \
        -e "." \
        -e "./deps/rosnav_rl/rosnav_rl"
    popd > /dev/null

    echo "Building rosnav_rl and arena_training packages..."
    cd "$ARENA_WS_DIR"
    arena build
    
    echo ""
    echo "=== Training feature installed successfully ==="
    echo ""
    echo "Usage:"
    echo "  Train:   [without simulation] ros2 run arena_training train_agent.py --config sb_training_config.yaml"
    echo "  Train:   arena launch sim:=gazebo local_planner:=rosnav_rl env_n:=2 train_config:=<path to config.yaml>"
    echo "  Deploy:  arena launch local_planner:=rosnav_rl agent_name:=<your_agent>"
    echo "  Deploy:  arena launch local_planner:=rosnav_rl agent_name:=<your_agent> agents_dir:=<path/to/agents>"
    echo "  Tune:    python3 scripts/tune_agent.py --config tuning_config.yaml"
    echo ""
}

uninstall(){
    echo "To uninstall training support, remove the arena_training package"
    echo "and rebuild the workspace."
    echo "  rm -rf $ARENA_DIR/arena_training"
    echo "  arena build"
}

# === MAIN SCRIPT ===
help(){
    echo "Usage: training.sh <install|uninstall>"
    echo ""
    echo "Installs arena_training and rosnav_rl for DRL-based navigation."
    echo "This enables:"
    echo "  - Training RL agents with Stable Baselines 3 and DreamerV3"
    echo "  - Deploying trained agents as nav2 local planners"
    echo "  - Action server for real-time model inference"
}
if [ $# -ne 1 ]; then
    help
    exit 1
fi
case "$1" in
    install)
        install
    ;;
    uninstall)
        uninstall
    ;;
    source)
    ;;
    *)
        help
        exit 1
    ;;
esac
