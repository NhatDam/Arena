#!/bin/bash

echo "=== CHECKING SYSTEM RESOURCES ==="

# Get current parameters
WATCHES=$(cat /proc/sys/fs/inotify/max_user_watches)
INSTANCES=$(cat /proc/sys/fs/inotify/max_user_instances)
# Get free space of /home (remove 'G' to get only the number)
FREE_SPACE_GB=$(df -BG /home | awk 'NR==2 {print $4}' | tr -d 'G')

# Status flag (0 is Pass, 1 is Fail)
FAIL_FLAG=0

# 1. Check max_user_watches (Requires >= 524288)
if [ "$WATCHES" -lt 524288 ]; then
    echo -e "\e[31m[FAIL]\e[0m fs.inotify.max_user_watches is too low: $WATCHES (Requires >= 524288)"
    FAIL_FLAG=1
else
    echo -e "\e[32m[PASS]\e[0m inotify watches: $WATCHES"
fi

# 2. Check max_user_instances (Requires >= 1024)
if [ "$INSTANCES" -lt 1024 ]; then
    echo -e "\e[31m[FAIL]\e[0m fs.inotify.max_user_instances is too low: $INSTANCES (Requires >= 1024)"
    FAIL_FLAG=1
else
    echo -e "\e[32m[PASS]\e[0m inotify instances: $INSTANCES"
fi

# 3. Check /home free space (Requires >= 10GB)
if [ "$FREE_SPACE_GB" -lt 4 ]; then
    echo -e "\e[31m[FAIL]\e[0m /home free space is too low: ${FREE_SPACE_GB}GB (Recommended >= 10GB)"
    FAIL_FLAG=1
else
    echo -e "\e[32m[PASS]\e[0m /home free space: ${FREE_SPACE_GB}GB"
fi

# Decide whether to stop or continue
if [ "$FAIL_FLAG" -eq 1 ]; then
    echo -e "\e[31m>>> INSUFFICIENT RESOURCES. ABORTING BENCHMARK TO PREVENT CRASH! <<<\e[0m"
    exit 1
fi

echo -e "\e[32m=== ALL CHECKS PASSED. STARTING BENCHMARK ===\e[0m"
echo ""

set -e
SCRIPT_START_TS=$(date +%s)

export WORKSPACE_DIR="$HOME/arena5_ws"
ARENA_EVAL_DIR="$WORKSPACE_DIR/src/Arena/arena_evaluation/arena_evaluation/arena_evaluation"
BENCHMARK_CONFIG_ROOT="$WORKSPACE_DIR/src/Arena/arena_bringup/configs/benchmark"
BENCHMARK_CONFIG="$BENCHMARK_CONFIG_ROOT/config.yaml"
HOST_PATH_BEFORE_CONDA="${PATH:-}"
HOST_LD_LIBRARY_PATH_BEFORE_CONDA="${LD_LIBRARY_PATH:-}"
HOST_PYTHONPATH_BEFORE_CONDA="${PYTHONPATH:-}"
CONTEST_NAME=$(awk '
    $1 == "contest:" { in_contest = 1; next }
    in_contest && $1 == "config:" {
        gsub(/\.yaml$/, "", $2)
        print $2
        exit
    }
' "$BENCHMARK_CONFIG")
BENCHMARK_LOG_DIR="$WORKSPACE_DIR/src/Arena/arena_bringup/configs/benchmark/logs"
BENCHMARK_RESULTS_ROOT="$WORKSPACE_DIR/results/${CONTEST_NAME:-social_contest}"
HUNAV_METRICS_FILE="$WORKSPACE_DIR/results/metrics.csv"
PLOTS_OUTPUT_DIR="$WORKSPACE_DIR/results/postprocess"

SOURCE_PYTHONPATH="$WORKSPACE_DIR/src/Arena/task_generator:$WORKSPACE_DIR/src/Arena/arena_simulation_setup:$WORKSPACE_DIR/src/Arena/arena_bringup:$WORKSPACE_DIR/src/Arena/arena_evaluation"
export SOURCE_PYTHONPATH
ARENA_ASSETS_DIR="${ARENA_ASSETS_DIR:-$WORKSPACE_DIR/src/Arena/_assets}"
ARENA_ASSETS_DIR_LOCAL="${ARENA_ASSETS_DIR_LOCAL:-$ARENA_ASSETS_DIR/_local}"
ASSET_BUCKETS="${ASSET_BUCKETS:-default}"
export ARENA_ASSETS_DIR ARENA_ASSETS_DIR_LOCAL ASSET_BUCKETS
BENCHMARK_CONDA_ENV="${BENCHMARK_CONDA_ENV:-socialnav}"
ARENA_AI_PYTHONNOUSERSITE="${ARENA_AI_PYTHONNOUSERSITE:-1}"
export ARENA_AI_PYTHONNOUSERSITE
SYSTEM_PYTHON="${SYSTEM_PYTHON:-/usr/bin/python3}"
ISAACSIM_PYTHON_SH="${ISAACSIM_PYTHON_SH:-$HOME/isaacsim-4.2.0/python.sh}"
FASTDDS_TRANSPORT_MODE="${FASTDDS_TRANSPORT_MODE:-udp_only}"
FASTDDS_BENCHMARK_PROFILE="${FASTDDS_BENCHMARK_PROFILE:-$WORKSPACE_DIR/.arena_fastdds_udp_only.xml}"
RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export RMW_IMPLEMENTATION
if [ -n "${PYTHONNOUSERSITE:-}" ]; then
    echo "[INFO] Clearing global PYTHONNOUSERSITE for Arena/Hunav; AI controllers set it locally."
    unset PYTHONNOUSERSITE
fi

ARENA_PID=""
STOP_REQUESTED=0
IN_CLEANUP=0
LAST_FORCE_MODE=""
PRESTART_GRACE_SEC=10
MAP_SERVER_WAIT_SEC=90
TASK_GENERATOR_WAIT_SEC="${TASK_GENERATOR_WAIT_SEC:-300}"
HUNAV_NODE_WAIT_SEC=90
RECOMMENDED_INOTIFY_WATCHES=524288
RECOMMENDED_INOTIFY_INSTANCES=1024
DISABLE_AIOMONITOR_FOR_BENCHMARK="${DISABLE_AIOMONITOR_FOR_BENCHMARK:-1}"
export DISABLE_AIOMONITOR_FOR_BENCHMARK
ARENA_BENCHMARK_MIN_TIMEOUT_SEC="${ARENA_BENCHMARK_MIN_TIMEOUT_SEC:-180}"
export ARENA_BENCHMARK_MIN_TIMEOUT_SEC
ROS_DISABLE_DAEMON="${ROS_DISABLE_DAEMON:-1}"
export ROS_DISABLE_DAEMON
RESTART_STACK_EACH_EPISODE="${RESTART_STACK_EACH_EPISODE:-1}"
ARENA_HEADLESS="${ARENA_HEADLESS:-1}"
BATCH_CONTINUE_ON_FAILURE="${BATCH_CONTINUE_ON_FAILURE:-1}"
BENCHMARK_BATCH_ROOT="${BENCHMARK_BATCH_ROOT:-$WORKSPACE_DIR/results/benchmark_batch_configs}"
BENCHMARK_RUN_ID="${BENCHMARK_RUN_ID:-$(date +%s)}"
export BENCHMARK_RUN_ID
ISAAC_CUDA_VISIBLE_DEVICES="${ISAAC_CUDA_VISIBLE_DEVICES:-}"
AI_CUDA_VISIBLE_DEVICES="${AI_CUDA_VISIBLE_DEVICES:-}"
ARENA_AI_DWB_INTEGRATION="${ARENA_AI_DWB_INTEGRATION:-path_adapter}"
ARENA_AI_DWB_HARD_GATE="${ARENA_AI_DWB_HARD_GATE:-false}"
ARENA_AI_COORDINATE_MODE="${ARENA_AI_COORDINATE_MODE:-xz_to_ros}"
export ARENA_AI_DWB_INTEGRATION ARENA_AI_DWB_HARD_GATE ARENA_AI_COORDINATE_MODE

ros2_cli() {
    local -a clean_env=(
        env
        -u CONDA_PREFIX
        -u CONDA_DEFAULT_ENV
        -u CONDA_PROMPT_MODIFIER
        -u CONDA_SHLVL
        -u CONDA_EXE
        -u CONDA_PYTHON_EXE
        -u _CE_CONDA
        -u _CE_M
        -u PYTHONHOME
        -u PYTHONNOUSERSITE
    )

    if [ -z "$HOST_LD_LIBRARY_PATH_BEFORE_CONDA" ]; then
        clean_env+=(-u LD_LIBRARY_PATH)
    fi

    if [ -z "$HOST_PYTHONPATH_BEFORE_CONDA" ]; then
        clean_env+=(-u PYTHONPATH)
    fi

    clean_env+=(
        PATH="$HOST_PATH_BEFORE_CONDA"
        RMW_IMPLEMENTATION="$RMW_IMPLEMENTATION"
        ROS_DISABLE_DAEMON=1
    )

    if [ -n "${FASTRTPS_DEFAULT_PROFILES_FILE:-}" ]; then
        clean_env+=(FASTRTPS_DEFAULT_PROFILES_FILE="$FASTRTPS_DEFAULT_PROFILES_FILE")
    fi

    if [ -n "${FASTDDS_DEFAULT_PROFILES_FILE:-}" ]; then
        clean_env+=(FASTDDS_DEFAULT_PROFILES_FILE="$FASTDDS_DEFAULT_PROFILES_FILE")
    fi

    if [ -n "$HOST_LD_LIBRARY_PATH_BEFORE_CONDA" ]; then
        clean_env+=(LD_LIBRARY_PATH="$HOST_LD_LIBRARY_PATH_BEFORE_CONDA")
    fi

    if [ -n "$HOST_PYTHONPATH_BEFORE_CONDA" ]; then
        clean_env+=(PYTHONPATH="$HOST_PYTHONPATH_BEFORE_CONDA")
    fi

    "${clean_env[@]}" bash -c '
        source /opt/ros/humble/setup.bash
        source "$WORKSPACE_DIR/install/setup.bash"
        ros2 "$@"
    ' ros2_cli "$@"
}

build_ai_runtime_library_path() {
    local python_bin="$1"

    PYTHONNOUSERSITE="$ARENA_AI_PYTHONNOUSERSITE" "$python_bin" - <<'PY' 2>/dev/null || true
from pathlib import Path
import sys

prefix = Path(sys.executable).resolve().parents[1]
paths = [prefix / "lib"]
for site_packages in (prefix / "lib").glob("python*/site-packages"):
    paths.extend(sorted((site_packages / "nvidia").glob("*/lib")))

seen = set()
valid = []
for path in paths:
    if path.is_dir():
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            valid.append(resolved)

print(":".join(valid))
PY
}

activate_benchmark_conda_env() {
    local conda_sh=""
    local conda_base=""

    if [ "${CONDA_DEFAULT_ENV:-}" = "$BENCHMARK_CONDA_ENV" ]; then
        echo "[INFO] Conda environment already active: $BENCHMARK_CONDA_ENV"
    else
        if [ -n "${CONDA_EXE:-}" ]; then
            conda_base="$(dirname "$(dirname "$CONDA_EXE")")"
            conda_sh="$conda_base/etc/profile.d/conda.sh"
        fi

        if [ ! -f "$conda_sh" ] && [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
            conda_sh="$HOME/miniconda3/etc/profile.d/conda.sh"
        fi

        if [ ! -f "$conda_sh" ] && [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
            conda_sh="$HOME/anaconda3/etc/profile.d/conda.sh"
        fi

        if [ ! -f "$conda_sh" ] && command -v conda >/dev/null 2>&1; then
            conda_base="$(conda info --base 2>/dev/null || true)"
            if [ -n "$conda_base" ]; then
                conda_sh="$conda_base/etc/profile.d/conda.sh"
            fi
        fi

        if [ ! -f "$conda_sh" ]; then
            echo "[ERROR] Could not find conda initialization script."
            echo "[ERROR] Expected one of:"
            echo "[ERROR]   $HOME/miniconda3/etc/profile.d/conda.sh"
            echo "[ERROR]   $HOME/anaconda3/etc/profile.d/conda.sh"
            echo "[ERROR] Or run with CONDA_EXE set by an initialized conda shell."
            exit 1
        fi

        # shellcheck disable=SC1090
        source "$conda_sh"
        echo "[INFO] Activating conda environment: $BENCHMARK_CONDA_ENV"
        conda activate "$BENCHMARK_CONDA_ENV"
    fi

    local python_bin
    python_bin="$(python3 -c 'import sys; print(sys.executable)')"
    ARENA_AI_PYTHON="${ARENA_AI_PYTHON:-$python_bin}"
    export ARENA_AI_PYTHON

    ARENA_AI_RUNTIME_LIBRARY_PATH="$(
        build_ai_runtime_library_path "$ARENA_AI_PYTHON"
    )"
    export ARENA_AI_RUNTIME_LIBRARY_PATH

    ARENA_AI_LD_LIBRARY_PATH="$ARENA_AI_RUNTIME_LIBRARY_PATH"
    if [ -n "${LD_LIBRARY_PATH:-}" ]; then
        ARENA_AI_LD_LIBRARY_PATH="${ARENA_AI_LD_LIBRARY_PATH:+$ARENA_AI_LD_LIBRARY_PATH:}$LD_LIBRARY_PATH"
    fi
    export ARENA_AI_LD_LIBRARY_PATH

    echo "[INFO] Benchmark Python: $python_bin"
    echo "[INFO] Global PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-<unset>}"
    echo "[INFO] Arena AI Python: $ARENA_AI_PYTHON"
    echo "[INFO] Arena AI PYTHONNOUSERSITE=$ARENA_AI_PYTHONNOUSERSITE"
    echo "[INFO] Arena AI runtime library path: ${ARENA_AI_RUNTIME_LIBRARY_PATH:-<empty>}"

    local clip_path
    clip_path="$(LD_LIBRARY_PATH="$ARENA_AI_LD_LIBRARY_PATH" PYTHONNOUSERSITE="$ARENA_AI_PYTHONNOUSERSITE" "$ARENA_AI_PYTHON" -c 'import clip; print(clip.__file__)' 2>/dev/null || true)"
    if [ -n "$clip_path" ]; then
        echo "[INFO] CLIP module: $clip_path"
    else
        echo "[WARN] CLIP module is not importable from $ARENA_AI_PYTHON with AI isolation"
    fi

    local torch_info
    torch_info="$(LD_LIBRARY_PATH="$ARENA_AI_LD_LIBRARY_PATH" PYTHONNOUSERSITE="$ARENA_AI_PYTHONNOUSERSITE" "$ARENA_AI_PYTHON" -c 'import torch; print(f"{torch.__version__} {torch.__file__} cuda_available={torch.cuda.is_available()}")' 2>/dev/null || true)"
    if [ -n "$torch_info" ]; then
        echo "[INFO] Torch: $torch_info"
    else
        echo "[WARN] Torch is not importable from $ARENA_AI_PYTHON with AI isolation"
    fi
}

run_python_preflight_checks() {
    echo "=== CHECKING PYTHON RUNTIMES ==="
    local fail=0

    if [ ! -x "$SYSTEM_PYTHON" ]; then
        echo "[ERROR] System Python is not executable: $SYSTEM_PYTHON"
        fail=1
    elif "$SYSTEM_PYTHON" -c 'import cattrs' >/dev/null 2>&1; then
        echo "[PASS] System Python can import cattrs: $SYSTEM_PYTHON"
    else
        echo "[ERROR] System Python cannot import cattrs: $SYSTEM_PYTHON"
        echo "[ERROR] Install with: $SYSTEM_PYTHON -m pip install --user cattrs attrs shapely"
        fail=1
    fi

    if [ ! -x "$ISAACSIM_PYTHON_SH" ]; then
        echo "[ERROR] IsaacSim python.sh is not executable: $ISAACSIM_PYTHON_SH"
        fail=1
    elif "$ISAACSIM_PYTHON_SH" -c 'import cattrs' >/dev/null 2>&1; then
        echo "[PASS] IsaacSim Python can import cattrs: $ISAACSIM_PYTHON_SH"
    else
        echo "[ERROR] IsaacSim Python cannot import cattrs: $ISAACSIM_PYTHON_SH"
        echo "[ERROR] Install with: $ISAACSIM_PYTHON_SH -m pip install cattrs attrs shapely"
        fail=1
    fi

    local ai_check
    if ! ai_check="$(LD_LIBRARY_PATH="$ARENA_AI_LD_LIBRARY_PATH" \
        PYTHONNOUSERSITE="$ARENA_AI_PYTHONNOUSERSITE" \
        "$ARENA_AI_PYTHON" - <<'PY' 2>&1
import sys
import torch

torch_path = torch.__file__
cuda_ok = torch.cuda.is_available()
device_count = torch.cuda.device_count()

print(f"torch={torch.__version__}")
print(f"torch_path={torch_path}")
print(f"cuda_available={cuda_ok}")
print(f"device_count={device_count}")

if ".local/lib" in torch_path:
    raise SystemExit("Torch resolved from ~/.local; Arena AI must use the conda env torch")
if not cuda_ok:
    raise SystemExit("Torch CUDA is unavailable")
if device_count < 1:
    raise SystemExit("No CUDA devices visible to Torch")
PY
    )"; then
        echo "[ERROR] Arena AI Python failed CUDA/Torch preflight:"
        printf '%s\n' "$ai_check"
        fail=1
    else
        echo "[PASS] Arena AI Torch/CUDA preflight:"
        printf '%s\n' "$ai_check" | sed 's/^/[INFO]   /'
    fi

    if [ "$fail" -ne 0 ]; then
        echo "[ERROR] Python runtime preflight failed. Aborting before launching Arena."
        exit 1
    fi

    echo -e "\e[32m=== PYTHON RUNTIME CHECKS PASSED ===\e[0m"
    echo ""
}

metrics_line_count() {
    if [ -f "$HUNAV_METRICS_FILE" ]; then
        wc -l < "$HUNAV_METRICS_FILE"
    else
        echo 0
    fi
}

wait_for_metrics_flush() {
    local baseline_lines="$1"
    local waited=0

    while [ "$waited" -lt 30 ]; do
        local current_lines
        current_lines=$(metrics_line_count)

        if [ "$current_lines" -gt "$baseline_lines" ]; then
            echo "[INFO] Hunav metrics flushed to $HUNAV_METRICS_FILE"
            return 0
        fi

        sleep 1
        waited=$((waited + 1))
    done

    echo "[WARN] Timed out waiting for $HUNAV_METRICS_FILE to update"
    return 1
}

read_string_param() {
    local node_name="$1"
    local param_name="$2"
    local output

    output=$(ros2_cli param get "$node_name" "$param_name" 2>/dev/null || true)
    if [[ "$output" != String\ value\ is:* ]]; then
        return 1
    fi

    printf '%s' "${output#String value is: }"
    return 0
}

normalize_fastdds_profile() {
    local profile_file="${FASTRTPS_DEFAULT_PROFILES_FILE:-${FASTDDS_DEFAULT_PROFILES_FILE:-}}"
    if [ -z "$profile_file" ]; then
        return 0
    fi

    case "$profile_file" in
        "~/"*) profile_file="$HOME/${profile_file#~/}" ;;
    esac

    if [ ! -f "$profile_file" ]; then
        echo "[WARN] FastDDS profile points to missing file: $profile_file; unsetting it."
        unset FASTRTPS_DEFAULT_PROFILES_FILE
        unset FASTDDS_DEFAULT_PROFILES_FILE
        return 0
    fi

    export FASTRTPS_DEFAULT_PROFILES_FILE="$profile_file"
    export FASTDDS_DEFAULT_PROFILES_FILE="$profile_file"
}

configure_fastdds_transport() {
    case "$FASTDDS_TRANSPORT_MODE" in
        udp_only)
            mkdir -p "$(dirname "$FASTDDS_BENCHMARK_PROFILE")"
            cat > "$FASTDDS_BENCHMARK_PROFILE" <<'XML'
<?xml version="1.0" encoding="UTF-8" ?>
<profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
  <transport_descriptors>
    <transport_descriptor>
      <transport_id>arena_udp_transport</transport_id>
      <type>UDPv4</type>
    </transport_descriptor>
  </transport_descriptors>
  <participant profile_name="arena_udp_only_participant" is_default_profile="true">
    <rtps>
      <useBuiltinTransports>false</useBuiltinTransports>
      <userTransports>
        <transport_id>arena_udp_transport</transport_id>
      </userTransports>
    </rtps>
  </participant>
</profiles>
XML
            export FASTRTPS_DEFAULT_PROFILES_FILE="$FASTDDS_BENCHMARK_PROFILE"
            export FASTDDS_DEFAULT_PROFILES_FILE="$FASTDDS_BENCHMARK_PROFILE"
            echo "[INFO] FastDDS transport: UDP-only profile at $FASTDDS_BENCHMARK_PROFILE"
            ;;
        inherit)
            normalize_fastdds_profile
            echo "[INFO] FastDDS transport: inherited profile ${FASTRTPS_DEFAULT_PROFILES_FILE:-<none>}"
            ;;
        *)
            echo "[ERROR] Unsupported FASTDDS_TRANSPORT_MODE=$FASTDDS_TRANSPORT_MODE"
            echo "[ERROR] Use udp_only or inherit."
            exit 1
            ;;
    esac
}

warn_inotify_limits() {
    local watches instances
    watches=$(cat /proc/sys/fs/inotify/max_user_watches 2>/dev/null || echo 0)
    instances=$(cat /proc/sys/fs/inotify/max_user_instances 2>/dev/null || echo 0)

    if [ "$watches" -lt "$RECOMMENDED_INOTIFY_WATCHES" ] || [ "$instances" -lt "$RECOMMENDED_INOTIFY_INSTANCES" ]; then
        echo "[WARN] Inotify limits are low for Isaac + ROS2 workloads:"
        echo "[WARN]   fs.inotify.max_user_watches=$watches (recommended >= $RECOMMENDED_INOTIFY_WATCHES)"
        echo "[WARN]   fs.inotify.max_user_instances=$instances (recommended >= $RECOMMENDED_INOTIFY_INSTANCES)"
    fi
}

stop_core_benchmark_processes() {
    echo "[INFO] Stopping stale Arena benchmark core processes..."
    pkill -TERM -f "ros2 launch arena_bringup arena.launch.py" 2>/dev/null || true
    pkill -TERM -f "$ISAACSIM_PYTHON_SH" 2>/dev/null || true
    pkill -TERM -f "$HOME/isaacsim-4.2.0/kit/kit" 2>/dev/null || true
    pkill -TERM -f "run_isaacsim" 2>/dev/null || true
    pkill -TERM -f "/install/task_generator/lib/task_generator/task_generator_node" 2>/dev/null || true
    pkill -TERM -f "/install/nav2_map_server/lib/nav2_map_server/map_server" 2>/dev/null || true
    pkill -TERM -f "/install/nav2_lifecycle_manager/lib/nav2_lifecycle_manager/lifecycle_manager" 2>/dev/null || true
    pkill -TERM -f "/install/arena_simulation_setup/lib/arena_simulation_setup/world_generator" 2>/dev/null || true
    pkill -TERM -f "$WORKSPACE_DIR/install/hunav_agent_manager/lib/hunav_agent_manager/arena_hunav_agent_manager" 2>/dev/null || true
    pkill -TERM -f "$WORKSPACE_DIR/install/hunav_evaluator/lib/hunav_evaluator/hunav_evaluator_node" 2>/dev/null || true
    pkill -TERM -f "$WORKSPACE_DIR/install/rviz_utils/lib/rviz_utils/pedestrian_marker_publisher" 2>/dev/null || true
    pkill -TERM -f "$WORKSPACE_DIR/install/rviz_utils/lib/rviz_utils/rviz_config" 2>/dev/null || true
    pkill -TERM -f "$WORKSPACE_DIR/install/nav2_[^ ]+/lib/nav2_[^ ]+/" 2>/dev/null || true
    pkill -TERM -f "$WORKSPACE_DIR/install/ros2cli/bin/ros2 action send_goal /task_generator_node" 2>/dev/null || true
    pkill -TERM -f "/opt/ros/humble/lib/topic_tools/relay .*__ns:=/task_generator_node" 2>/dev/null || true

    sleep 2

    pkill -KILL -f "ros2 launch arena_bringup arena.launch.py" 2>/dev/null || true
    pkill -KILL -f "$ISAACSIM_PYTHON_SH" 2>/dev/null || true
    pkill -KILL -f "$HOME/isaacsim-4.2.0/kit/kit" 2>/dev/null || true
    pkill -KILL -f "run_isaacsim" 2>/dev/null || true
    pkill -KILL -f "/install/task_generator/lib/task_generator/task_generator_node" 2>/dev/null || true
    pkill -KILL -f "/install/nav2_map_server/lib/nav2_map_server/map_server" 2>/dev/null || true
    pkill -KILL -f "/install/nav2_lifecycle_manager/lib/nav2_lifecycle_manager/lifecycle_manager" 2>/dev/null || true
    pkill -KILL -f "/install/arena_simulation_setup/lib/arena_simulation_setup/world_generator" 2>/dev/null || true
    pkill -KILL -f "$WORKSPACE_DIR/install/hunav_agent_manager/lib/hunav_agent_manager/arena_hunav_agent_manager" 2>/dev/null || true
    pkill -KILL -f "$WORKSPACE_DIR/install/hunav_evaluator/lib/hunav_evaluator/hunav_evaluator_node" 2>/dev/null || true
    pkill -KILL -f "$WORKSPACE_DIR/install/rviz_utils/lib/rviz_utils/pedestrian_marker_publisher" 2>/dev/null || true
    pkill -KILL -f "$WORKSPACE_DIR/install/rviz_utils/lib/rviz_utils/rviz_config" 2>/dev/null || true
    pkill -KILL -f "$WORKSPACE_DIR/install/nav2_[^ ]+/lib/nav2_[^ ]+/" 2>/dev/null || true
    pkill -KILL -f "$WORKSPACE_DIR/install/ros2cli/bin/ros2 action send_goal /task_generator_node" 2>/dev/null || true
    pkill -KILL -f "/opt/ros/humble/lib/topic_tools/relay .*__ns:=/task_generator_node" 2>/dev/null || true
}

cleanup_fastdds_shm() {
    if command -v fastdds >/dev/null 2>&1; then
        echo "[INFO] Cleaning stale FastDDS SHM artifacts via 'fastdds shm clean'..."
        fastdds shm clean >/dev/null 2>&1 || true
        return 0
    fi

    echo "[INFO] Cleaning stale FastDDS SHM artifacts under /dev/shm..."
    find /dev/shm -maxdepth 1 -type f \( -name 'fastrtps_*' -o -name 'fastdds*' -o -name 'sem.fastrtps_*' \) -delete 2>/dev/null || true
}

restart_ros_daemon() {
    echo "[INFO] Restarting ROS 2 daemon..."
    ros2 daemon stop >/dev/null 2>&1 || true
    if [ "${ROS_DISABLE_DAEMON:-0}" = "1" ]; then
        echo "[INFO] ROS_DISABLE_DAEMON=1; skipping daemon start for direct discovery."
    else
        ros2 daemon start >/dev/null 2>&1 || true
    fi
}

prelaunch_cleanup() {
    stop_sidecars
    stop_core_benchmark_processes
    cleanup_fastdds_shm
    restart_ros_daemon
}

wait_for_map_server_active() {
    local waited=0
    while [ "$waited" -lt "$MAP_SERVER_WAIT_SEC" ]; do
        if ! kill -0 "$ARENA_PID" 2>/dev/null; then
            echo "[ERROR] Arena process exited before map_server reached ACTIVE."
            return 1
        fi

        local state_output state_name
        state_output=$(ros2_cli lifecycle get /task_generator_node/map_server 2>&1 || true)
        state_name=$(printf '%s\n' "$state_output" | awk 'NF { print tolower($1); exit }')

        if [ "$state_name" = "active" ]; then
            echo "[INFO] map_server lifecycle state is ACTIVE."
            return 0
        fi

        sleep 1
        waited=$((waited + 1))
    done

    echo "[ERROR] map_server did not reach ACTIVE within ${MAP_SERVER_WAIT_SEC}s."
    echo "[ERROR] Last lifecycle output: ${state_output:-unavailable}"
    echo "[ERROR] Visible ROS nodes:"
    ros2_cli node list 2>&1 | sed 's/^/[ERROR]   /' || true
    return 1
}

wait_for_task_generator_initialized() {
    local waited=0
    local init_output=""

    while [ "$waited" -lt "$TASK_GENERATOR_WAIT_SEC" ]; do
        if ! kill -0 "$ARENA_PID" 2>/dev/null; then
            echo "[ERROR] Arena process exited before task_generator_node initialized."
            return 1
        fi

        init_output=$(ros2_cli param get /task_generator_node initialized 2>&1 || true)
        if printf '%s\n' "$init_output" | grep -Eq 'Boolean value is: (True|true)'; then
            echo "[INFO] task_generator_node initialized."
            return 0
        fi

        sleep 1
        waited=$((waited + 1))
    done

    echo "[ERROR] task_generator_node did not initialize within ${TASK_GENERATOR_WAIT_SEC}s."
    echo "[ERROR] Last initialized param output: ${init_output:-unavailable}"
    echo "[ERROR] Visible ROS nodes:"
    ros2_cli node list 2>&1 | sed 's/^/[ERROR]   /' || true
    echo "[ERROR] Visible Isaac services:"
    ros2_cli service list 2>&1 | grep '^/isaac/' | sed 's/^/[ERROR]   /' || true
    return 1
}

wait_for_hunav_evaluator_node() {
    local waited=0
    while [ "$waited" -lt "$HUNAV_NODE_WAIT_SEC" ]; do
        if ! kill -0 "$ARENA_PID" 2>/dev/null; then
            echo "[ERROR] Arena process exited before hunav evaluator node was ready."
            return 1
        fi

        if ros2_cli node list 2>/dev/null | grep -qx '/hunav_evaluator_node'; then
            echo "[INFO] hunav_evaluator_node is available."
            return 0
        fi

        sleep 1
        waited=$((waited + 1))
    done

    echo "[ERROR] hunav_evaluator_node not detected within ${HUNAV_NODE_WAIT_SEC}s."
    return 1
}

configure_hunav_metrics_output() {
    mkdir -p "$(dirname "$HUNAV_METRICS_FILE")"

    if ros2_cli param set /hunav_evaluator_node result_file "$HUNAV_METRICS_FILE" >/dev/null 2>&1; then
        echo "[INFO] hunav evaluator result_file set to $HUNAV_METRICS_FILE"
        return 0
    fi

    echo "[ERROR] Failed to set /hunav_evaluator_node result_file to $HUNAV_METRICS_FILE"
    local current_value
    current_value=$(read_string_param /hunav_evaluator_node result_file || echo unavailable)
    echo "[ERROR] Current /hunav_evaluator_node result_file: $current_value"
    return 1
}

current_contestant_from_logs() {
    local latest_log
    local contestant_line

    latest_log=$(ls -t "$BENCHMARK_LOG_DIR"/*.log 2>/dev/null | head -n 1 || true)
    [ -n "$latest_log" ] || return 1

    contestant_line=$(grep -E "C \\[" "$latest_log" | tail -n 1 || true)
    [ -n "$contestant_line" ] || return 1

    printf '%s' "${contestant_line##*] }"
    return 0
}

stop_hunav_recording() {
    if [ "$STOP_REQUESTED" -eq 1 ]; then
        return 0
    fi

    STOP_REQUESTED=1
    local baseline_lines
    baseline_lines=$(metrics_line_count)
    local waited=0
    local max_wait="${HUNAV_STOP_SERVICE_WAIT_SEC:-10}"

    while [ "$waited" -lt "$max_wait" ]; do
        if ros2_cli service list 2>/dev/null | grep -qx '/hunav_stop_recording'; then
            break
        fi
        sleep 1
        waited=$((waited + 1))
    done

    if ! ros2_cli service list 2>/dev/null | grep -qx '/hunav_stop_recording'; then
        echo "[WARN] /hunav_stop_recording is not available after ${max_wait}s; skipping metrics flush request"
        return 1
    fi

    local attempt=1
    local max_attempts="${HUNAV_STOP_ATTEMPTS:-3}"
    while [ "$attempt" -le "$max_attempts" ]; do
        echo "[INFO] Requesting hunav evaluator stop (attempt $attempt/$max_attempts)..."
        ros2_cli service call /hunav_stop_recording std_srvs/srv/Empty "{}" >/dev/null 2>&1 || true
        if wait_for_metrics_flush "$baseline_lines"; then
            echo "[INFO] Hunav evaluator stop completed; metrics flushed"
            return 0
        fi
        attempt=$((attempt + 1))
    done

    echo "[WARN] Hunav evaluator stop was requested, but $HUNAV_METRICS_FILE did not gain a row"
    return 1
}

stop_sidecars() {
    echo "[INFO] Force stopping benchmark sidecar processes..."
    
    # Kill sidecar/controller processes from the Arena AI integration path.
    pkill -9 -f "citywalker_dwb_node.py" 2>/dev/null || true
    pkill -9 -f "arena_ai_integration.nodes.ai_controller_node" 2>/dev/null || true
    pkill -9 -f "/arena_ai_integration/ai_controller" 2>/dev/null || true
    pkill -9 -f "arena_ai_integration.nodes.human_states_bridge" 2>/dev/null || true
    pkill -9 -f "arena_ai_integration.nodes.semantic_laser_filter" 2>/dev/null || true
    pkill -9 -f "/arena_ai_integration/human_states_bridge" 2>/dev/null || true
    pkill -9 -f "/arena_ai_integration/semantic_laser_filter" 2>/dev/null || true
    
    # Kill Python controller processes launched by benchmark wrappers.
    pkill -9 -f "citywalker_dwb_controller" 2>/dev/null || true
    pkill -9 -f "ai_controller_" 2>/dev/null || true
    
    sleep 6
    
    echo "[INFO] Sidecars stopped."
}

create_restart_episode_plan() {
    local plan_file="$1"
    mkdir -p "$(dirname "$plan_file")"

    python3 - "$BENCHMARK_CONFIG_ROOT" "$plan_file" "$BENCHMARK_BATCH_ROOT" "${BENCHMARK_RUN_ID#t}" <<'PY'
import copy
import os
import re
import sys
from pathlib import Path

import yaml

config_root = Path(sys.argv[1]).expanduser().resolve()
plan_file = Path(sys.argv[2]).expanduser().resolve()
batch_root = Path(sys.argv[3]).expanduser().resolve()
run_id = str(sys.argv[4])


def split_filter(value: str) -> set[str]:
    return {
        item.strip()
        for item in re.split(r"[,\s]+", value or "")
        if item.strip()
    }


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    return cleaned or "unknown"


def scenario_file(stage: dict) -> str:
    return str(stage.get("config", {}).get("SCENARIO", {}).get("file", ""))


def contestant_label(contestant: dict) -> str:
    return str(
        contestant.get("name")
        or contestant.get("agent_name")
        or contestant.get("local_planner")
        or "contestant"
    )


def stage_label(stage: dict) -> str:
    return str(stage.get("name") or scenario_file(stage) or "stage")


model_filter = split_filter(os.environ.get("BENCHMARK_MODELS", ""))
scenario_filter = split_filter(os.environ.get("BENCHMARK_SCENARIOS", ""))
episodes_override = os.environ.get("BENCHMARK_EPISODES", "").strip()

base_config_path = config_root / "config.yaml"
base_config = yaml.safe_load(base_config_path.read_text())
contest_config_name = base_config["contest"]["config"]
suite_config_name = base_config["suite"]["config"]
scale_episodes = float(base_config.get("suite", {}).get("scale_episodes", 1) or 1)

contest_path = config_root / "contests" / contest_config_name
suite_path = config_root / "suites" / suite_config_name
contest_data = yaml.safe_load(contest_path.read_text())
suite_data = yaml.safe_load(suite_path.read_text())

contestants = []
for contestant in contest_data.get("contestants", []):
    keys = {
        str(contestant.get("name", "")),
        str(contestant.get("agent_name", "")),
        str(contestant.get("local_planner", "")),
    }
    if model_filter and keys.isdisjoint(model_filter):
        continue
    contestants.append(contestant)

stages = []
for stage in suite_data.get("stages", []):
    keys = {stage_label(stage), scenario_file(stage)}
    if scenario_filter and keys.isdisjoint(scenario_filter):
        continue
    stages.append(stage)

if not contestants:
    raise SystemExit("No benchmark contestants matched BENCHMARK_MODELS")
if not stages:
    raise SystemExit("No benchmark stages matched BENCHMARK_SCENARIOS")

batch_dir = batch_root / f"run_{safe_name(run_id)}"
batch_dir.mkdir(parents=True, exist_ok=True)
rows = []
index = 0

for contestant in contestants:
    contestant_name = contestant_label(contestant)
    for stage in stages:
        stage_name = stage_label(stage)
        if episodes_override:
            episode_count = int(float(episodes_override))
        else:
            episode_count = int(float(stage.get("episodes", 1) or 1) * scale_episodes)
        episode_count = max(1, episode_count)

        for episode_index in range(episode_count):
            config_dir = (
                batch_dir
                / safe_name(contestant_name)
                / safe_name(stage_name)
                / f"ep_{episode_index:03d}"
            )
            (config_dir / "contests").mkdir(parents=True, exist_ok=True)
            (config_dir / "suites").mkdir(parents=True, exist_ok=True)

            temp_config = copy.deepcopy(base_config)
            temp_config["contest"]["config"] = contest_config_name
            temp_config["suite"]["config"] = suite_config_name
            temp_config["suite"]["scale_episodes"] = 1

            temp_contest = {"contestants": [copy.deepcopy(contestant)]}
            temp_stage = copy.deepcopy(stage)
            temp_stage["episodes"] = 1
            temp_suite = {"stages": [temp_stage]}

            (config_dir / "config.yaml").write_text(
                yaml.safe_dump(temp_config, sort_keys=False),
                encoding="utf-8",
            )
            (config_dir / "contests" / contest_config_name).write_text(
                yaml.safe_dump(temp_contest, sort_keys=False),
                encoding="utf-8",
            )
            (config_dir / "suites" / suite_config_name).write_text(
                yaml.safe_dump(temp_suite, sort_keys=False),
                encoding="utf-8",
            )

            rows.append(
                (
                    index,
                    str(config_dir),
                    contestant_name,
                    stage_name,
                    episode_index,
                    episode_count,
                )
            )
            index += 1

plan_file.parent.mkdir(parents=True, exist_ok=True)
with plan_file.open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write("\t".join(map(str, row)) + "\n")

print(f"[INFO] Restart-per-episode batch plan: {len(rows)} run(s)")
print(f"[INFO]   models={len(contestants)} scenarios={len(stages)}")
print(f"[INFO]   plan={plan_file}")
PY
}

start_arena_process() {
    local -a clean_env=(
        env
        -u CONDA_PREFIX
        -u CONDA_DEFAULT_ENV
        -u CONDA_PROMPT_MODIFIER
        -u CONDA_SHLVL
        -u CONDA_EXE
        -u CONDA_PYTHON_EXE
        -u _CE_CONDA
        -u _CE_M
        -u PYTHONHOME
        -u PYTHONNOUSERSITE
    )

    if [ -z "$HOST_LD_LIBRARY_PATH_BEFORE_CONDA" ]; then
        clean_env+=(-u LD_LIBRARY_PATH)
    fi

    if [ -z "$HOST_PYTHONPATH_BEFORE_CONDA" ]; then
        clean_env+=(-u PYTHONPATH)
    fi

    clean_env+=(
        PATH="$HOST_PATH_BEFORE_CONDA"
        RMW_IMPLEMENTATION="$RMW_IMPLEMENTATION"
        ROS_DISABLE_DAEMON="$ROS_DISABLE_DAEMON"
    )

    if [ -n "${FASTRTPS_DEFAULT_PROFILES_FILE:-}" ]; then
        clean_env+=(FASTRTPS_DEFAULT_PROFILES_FILE="$FASTRTPS_DEFAULT_PROFILES_FILE")
    fi

    if [ -n "${FASTDDS_DEFAULT_PROFILES_FILE:-}" ]; then
        clean_env+=(FASTDDS_DEFAULT_PROFILES_FILE="$FASTDDS_DEFAULT_PROFILES_FILE")
    fi

    if [ -n "$HOST_LD_LIBRARY_PATH_BEFORE_CONDA" ]; then
        clean_env+=(LD_LIBRARY_PATH="$HOST_LD_LIBRARY_PATH_BEFORE_CONDA")
    fi

    if [ -n "$HOST_PYTHONPATH_BEFORE_CONDA" ]; then
        clean_env+=(PYTHONPATH="$HOST_PYTHONPATH_BEFORE_CONDA")
    fi

    if [ -n "${ISAAC_CUDA_VISIBLE_DEVICES:-}" ]; then
        clean_env+=(
            CUDA_VISIBLE_DEVICES="$ISAAC_CUDA_VISIBLE_DEVICES"
            ISAAC_CUDA_VISIBLE_DEVICES="$ISAAC_CUDA_VISIBLE_DEVICES"
        )
    fi

    if [ -n "${AI_CUDA_VISIBLE_DEVICES:-}" ]; then
        clean_env+=(AI_CUDA_VISIBLE_DEVICES="$AI_CUDA_VISIBLE_DEVICES")
    fi

    clean_env+=(
        ARENA_AI_DWB_INTEGRATION="$ARENA_AI_DWB_INTEGRATION"
        ARENA_AI_DWB_HARD_GATE="$ARENA_AI_DWB_HARD_GATE"
        ARENA_AI_COORDINATE_MODE="$ARENA_AI_COORDINATE_MODE"
    )

    if [ -n "${ARENA_BENCHMARK_CONFIG_DIR:-}" ]; then
        clean_env+=(ARENA_BENCHMARK_CONFIG_DIR="$ARENA_BENCHMARK_CONFIG_DIR")
    fi

    if [ -n "${ARENA_BENCHMARK_RUN_ID:-}" ]; then
        clean_env+=(ARENA_BENCHMARK_RUN_ID="$ARENA_BENCHMARK_RUN_ID")
    fi

    if [ -n "${ARENA_BENCHMARK_EPISODE_OFFSET:-}" ]; then
        clean_env+=(ARENA_BENCHMARK_EPISODE_OFFSET="$ARENA_BENCHMARK_EPISODE_OFFSET")
    fi

    if [ -n "${ARENA_BENCHMARK_EPISODE_TOTAL:-}" ]; then
        clean_env+=(ARENA_BENCHMARK_EPISODE_TOTAL="$ARENA_BENCHMARK_EPISODE_TOTAL")
    fi

    clean_env+=(
        ARENA_ASSETS_DIR="$ARENA_ASSETS_DIR"
        ARENA_ASSETS_DIR_LOCAL="$ARENA_ASSETS_DIR_LOCAL"
        ASSET_BUCKETS="$ASSET_BUCKETS"
    )

    if [ -n "${ARENA_MODELS_FORMATS:-}" ]; then
        clean_env+=(ARENA_MODELS_FORMATS="$ARENA_MODELS_FORMATS")
    fi

    echo "[INFO] Launching Arena/Isaac with clean non-conda runtime env."
    if [ -n "${ISAAC_CUDA_VISIBLE_DEVICES:-}" ] || [ -n "${AI_CUDA_VISIBLE_DEVICES:-}" ]; then
        echo "[INFO] GPU routing: Isaac CUDA_VISIBLE_DEVICES=${ISAAC_CUDA_VISIBLE_DEVICES:-<inherited>} AI CUDA_VISIBLE_DEVICES=${AI_CUDA_VISIBLE_DEVICES:-<inherited>}"
    else
        echo "[INFO] GPU routing: inherited CUDA_VISIBLE_DEVICES for both Isaac and AI."
    fi
    echo "[INFO] Arena AI DWB integration: $ARENA_AI_DWB_INTEGRATION"
    echo "[INFO] Arena AI DWB hard gate: $ARENA_AI_DWB_HARD_GATE"
    echo "[INFO] Arena AI coordinate mode: $ARENA_AI_COORDINATE_MODE"
    setsid "${clean_env[@]}" bash -c '
        source /opt/ros/humble/setup.bash
        source "$WORKSPACE_DIR/install/setup.bash"
        export PYTHONPATH="$SOURCE_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"
        export ARENA_DISABLE_AIOMONITOR="$DISABLE_AIOMONITOR_FOR_BENCHMARK"
        ros2 launch arena_bringup arena.launch.py \
            sim:=isaac \
            world:=hospital_1 \
            robot:=turtlebot \
            tm_robots:=scenario \
            tm_obstacles:=scenario \
            tm_modules:=benchmark \
            use_sim_time:=true \
            local_planner:=dwb \
            headless:="$ARENA_HEADLESS"
    ' &
    ARENA_PID=$!
}

shutdown_current_stack() {
    echo -e "\n[INFO] Shutting down current Arena/Isaac stack..."
    stop_hunav_recording
    stop_sidecars

    if [ -n "$ARENA_PID" ] && kill -0 "$ARENA_PID" 2>/dev/null; then
        kill -INT -- "-$ARENA_PID" 2>/dev/null || true
        wait "$ARENA_PID" 2>/dev/null || true
    fi

    stop_core_benchmark_processes
    cleanup_fastdds_shm
    ARENA_PID=""
}

run_arena_once() {
    STOP_REQUESTED=0
    LAST_FORCE_MODE=""
    local completed=0

    echo "[INFO] Running pre-launch cleanup to avoid stale DDS/SHM and ROS processes..."
    prelaunch_cleanup

    echo "[INFO] Starting Arena Benchmark..."
    if declare -F arena >/dev/null 2>&1; then
        export -f arena
    fi
    start_arena_process

    echo "[INFO] Arena started (PID: $ARENA_PID). Monitoring system..."

    sleep "$PRESTART_GRACE_SEC"

    if ! wait_for_task_generator_initialized; then
        shutdown_current_stack
        return 1
    fi

    if ! wait_for_map_server_active; then
        echo "[WARN] Continuing after map_server lifecycle wait failure; task_generator_node is initialized."
    fi

    if ! wait_for_hunav_evaluator_node; then
        shutdown_current_stack
        return 1
    fi

    if ! configure_hunav_metrics_output; then
        shutdown_current_stack
        return 1
    fi

    while kill -0 "$ARENA_PID" 2>/dev/null; do
        LATEST_LAUNCH_LOG=$(ls -t ~/.ros/log/*/launch.log 2>/dev/null | head -n 1 || true)
        if [ -f "$LATEST_LAUNCH_LOG" ]; then
            PROGRESS=$(grep -E "C \[|S \[|E \[" "$LATEST_LAUNCH_LOG" | tail -n 1 || true)
            if [ -n "$PROGRESS" ]; then
                echo -ne "\r[PROGRESS] $PROGRESS"
            fi
        fi

        CURRENT_AGENT=""
        CURRENT_CONTESTANT=""
        if AGENT_VALUE=$(read_string_param /task_generator_node agent_name); then
            CURRENT_AGENT="$AGENT_VALUE"
        fi
        if CONTESTANT_VALUE=$(current_contestant_from_logs); then
            CURRENT_CONTESTANT="$CONTESTANT_VALUE"
        fi

        SHOULD_FORCE_BASELINE=0
        if [ "$CURRENT_CONTESTANT" = "DWB-Baseline" ]; then
            SHOULD_FORCE_BASELINE=1
        fi

        if [ "$SHOULD_FORCE_BASELINE" -eq 1 ]; then
            if [ "$LAST_FORCE_MODE" != "baseline" ]; then
                echo "[INFO] DWB Baseline active; leaving costmaps to Nav2/reset handling."
                LAST_FORCE_MODE="baseline"
            fi
        elif [ "$LAST_FORCE_MODE" = "baseline" ]; then
            echo "[INFO] Active contestant is AI-driven: ${CURRENT_CONTESTANT:-$CURRENT_AGENT}"
            LAST_FORCE_MODE="ai"
        fi

        LATEST_BENCHMARK_LOG=$(ls -t "$BENCHMARK_LOG_DIR"/*.log 2>/dev/null | head -n 1 || true)
        if [ "$STOP_REQUESTED" -eq 0 ] && [ -f "$LATEST_BENCHMARK_LOG" ]; then
            if grep -q "Benchmark completed" "$LATEST_BENCHMARK_LOG"; then
                echo -e "\n[INFO] Benchmark completion detected."
                completed=1
                stop_hunav_recording
                break
            fi
        fi

        sleep 5
    done

    echo -e "\n[INFO] Benchmark run finished."
    if [ -n "$ARENA_PID" ] && ! kill -0 "$ARENA_PID" 2>/dev/null; then
        wait "$ARENA_PID" 2>/dev/null || true
    fi
    shutdown_current_stack

    if [ "$completed" -eq 1 ]; then
        return 0
    fi

    echo "[WARN] Arena process ended before benchmark completion was detected."
    return 1
}

cleanup() {
    if [ "$IN_CLEANUP" -eq 1 ]; then
        return 0
    fi
    IN_CLEANUP=1
    trap '' INT TERM

    echo -e "\n[INFO] Shutdown signal received."
    shutdown_current_stack
    exit 0
}

postprocess_results() {
    mapfile -t RECENT_RUN_DIRS < <(
        find "$BENCHMARK_RESULTS_ROOT" -type f -name "odom.csv" -printf '%T@ %h\n' 2>/dev/null \
            | awk -v start="$SCRIPT_START_TS" '$1 >= start { $1=""; sub(/^ /, ""); print }' \
            | sort -u
    )

    if [ "${#RECENT_RUN_DIRS[@]}" -eq 0 ]; then
        LATEST_RUN_DIR=$(find "$BENCHMARK_RESULTS_ROOT" -type f -name "odom.csv" -printf '%T@ %h\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
        if [ -n "$LATEST_RUN_DIR" ]; then
            RECENT_RUN_DIRS=("$LATEST_RUN_DIR")
        fi
    fi

    if [ "${#RECENT_RUN_DIRS[@]}" -gt 0 ]; then
        echo "[INFO] Computing arena_evaluation metrics for ${#RECENT_RUN_DIRS[@]} raw run(s)"
        for run_dir in "${RECENT_RUN_DIRS[@]}"; do
            echo "[INFO]   get_metrics.py --dir $run_dir"
            python3 "$ARENA_EVAL_DIR/get_metrics.py" --dir "$run_dir"
        done
    else
        echo "[WARN] No raw run directory found under $BENCHMARK_RESULTS_ROOT"
    fi

    if [ -d "$BENCHMARK_RESULTS_ROOT" ]; then
        mkdir -p "$PLOTS_OUTPUT_DIR"
        echo "[INFO] Aggregating run data from $BENCHMARK_RESULTS_ROOT"
        python3 "$ARENA_EVAL_DIR/process_data.py" "$BENCHMARK_RESULTS_ROOT" --output "$PLOTS_OUTPUT_DIR"
        if [ -f "$HUNAV_METRICS_FILE" ]; then
            CURRENT_RUN_ID="${BENCHMARK_RUN_ID#t}"
            echo "[INFO] Writing scenario averages to $PLOTS_OUTPUT_DIR/metrics_agv.csv"
            python3 -m arena_ai_integration.tools.aggregate_benchmark_metrics \
                --input "$HUNAV_METRICS_FILE" \
                --output "$PLOTS_OUTPUT_DIR/metrics_agv.csv" \
                --run-id "$CURRENT_RUN_ID"
        fi
        echo "[INFO] Aggregated plots saved to $PLOTS_OUTPUT_DIR"
    else
        echo "[WARN] Benchmark results root not found: $BENCHMARK_RESULTS_ROOT"
    fi
}

run_restart_episode_batch() {
    local plan_file="$BENCHMARK_BATCH_ROOT/run_${BENCHMARK_RUN_ID#t}/plan.tsv"
    local failures=0
    local total_runs=0

    export BENCHMARK_CONFIG_ROOT
    create_restart_episode_plan "$plan_file"
    total_runs=$(wc -l < "$plan_file" | tr -d ' ')

    while IFS=$'\t' read -r batch_index config_dir model_name stage_name episode_index episode_total; do
        echo ""
        echo "[BATCH] [$((batch_index + 1))/$total_runs] model=$model_name scenario=$stage_name episode=$((episode_index + 1))/$episode_total"

        export ARENA_BENCHMARK_CONFIG_DIR="$config_dir"
        export ARENA_BENCHMARK_RUN_ID="$BENCHMARK_RUN_ID"
        export ARENA_BENCHMARK_EPISODE_OFFSET="$episode_index"
        export ARENA_BENCHMARK_EPISODE_TOTAL="$episode_total"
        BENCHMARK_LOG_DIR="$config_dir/logs"

        if ! run_arena_once; then
            failures=$((failures + 1))
            echo "[ERROR] Batch run failed: model=$model_name scenario=$stage_name episode=$((episode_index + 1))"
            if [ "$BATCH_CONTINUE_ON_FAILURE" -ne 1 ]; then
                break
            fi
        fi
    done < "$plan_file"

    unset ARENA_BENCHMARK_CONFIG_DIR
    unset ARENA_BENCHMARK_EPISODE_OFFSET
    unset ARENA_BENCHMARK_EPISODE_TOTAL
    BENCHMARK_LOG_DIR="$BENCHMARK_CONFIG_ROOT/logs"

    if [ "$failures" -gt 0 ]; then
        echo "[WARN] Restart-per-episode batch completed with $failures failed run(s)."
        return 1
    fi

    echo "[INFO] Restart-per-episode batch completed successfully."
    return 0
}

trap cleanup INT TERM

activate_benchmark_conda_env
source "$WORKSPACE_DIR/install/setup.bash"
configure_fastdds_transport
echo "[INFO] ROS middleware: RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"
export PYTHONPATH="$SOURCE_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"
run_python_preflight_checks
mkdir -p "$WORKSPACE_DIR/results"

warn_inotify_limits

RUN_STATUS=0
export ARENA_BENCHMARK_RUN_ID="$BENCHMARK_RUN_ID"
if [ "$RESTART_STACK_EACH_EPISODE" -eq 1 ]; then
    echo "[INFO] Restart-per-episode mode enabled."
    echo "[INFO] Optional filters: BENCHMARK_MODELS='$BENCHMARK_MODELS' BENCHMARK_SCENARIOS='$BENCHMARK_SCENARIOS' BENCHMARK_EPISODES='${BENCHMARK_EPISODES:-}'"
    run_restart_episode_batch || RUN_STATUS=$?
else
    echo "[INFO] Single Arena-session mode enabled."
    unset ARENA_BENCHMARK_CONFIG_DIR
    unset ARENA_BENCHMARK_EPISODE_OFFSET
    unset ARENA_BENCHMARK_EPISODE_TOTAL
    BENCHMARK_LOG_DIR="$BENCHMARK_CONFIG_ROOT/logs"
    run_arena_once || RUN_STATUS=$?
fi

postprocess_results
exit "$RUN_STATUS"
