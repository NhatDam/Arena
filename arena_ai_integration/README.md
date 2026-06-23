# Arena AI Integration

Unified ROS 2 integration for AI navigation agents in Arena.

## Architecture

`arena_ai_integration` owns the AI-DWB runtime for SocialNav, UrbanNav, and
LeLaN benchmark runs. Model inference code, ROS sidecars, and benchmark helpers
are packaged inside this ROS 2 package.

Runtime flow:

```text
RGB / odom / goal / instruction
  -> arena_ai_integration.nodes.ai_controller_node
  -> SocialNavAgent | UrbanNavAgent | LeLanAgent
  -> local waypoints + arrival score
  -> BaseAINode FollowPath path-adapter
  -> Nav2 DWB FollowPath
  -> cmd_vel_nav_raw relay to cmd_vel
```

UrbanNav and LeLaN use the same FollowPath path-adapter as SocialNav. The old
hard-gate DWB candidate selection and waypoint-regeneration logic is intentionally
not used by the unified controller.

## Checkpoints

Place model weights here:

```text
arena_ai_integration/checkpoints/
  SocialNav_1_path.pth
  UrbanNav_FiLM.pth
  LeLan_latest.pth
```

The model YAML files are installed from:

```text
arena_ai_integration/config/models/
  socialnav_film.yaml
  urbannav_film.yaml
  lelan.yaml
```

## Agents

| agent_type | Default checkpoint | Runtime module |
| --- | --- | --- |
| `socialnav` | `SocialNav_1_path.pth` | `arena_ai_integration.models.socialnav.runtime.SocialNavModel` |
| `urbannav` | `UrbanNav_FiLM.pth` | `arena_ai_integration.models.socialnav.runtime.UrbanNavModel` |
| `lelan` | `LeLan_latest.pth` | `arena_ai_integration.models.lelan.runtime.LeLaNInferenceModel` |

`agent_name` still selects the checkpoint name when a matching
`checkpoints/<agent_name>.pth` exists. `agent_type` selects the wrapper class.

## Build

```bash
cd ~/arena5_ws
colcon build --packages-select arena_ai_integration arena_simulation_setup
source install/setup.bash
```

## Run

Standalone controller:

```bash
AGENT_TYPE=socialnav ./src/Arena/arena_ai_integration/scripts/start_agent.sh
AGENT_TYPE=urbannav ./src/Arena/arena_ai_integration/scripts/start_agent.sh
AGENT_TYPE=lelan ./src/Arena/arena_ai_integration/scripts/start_agent.sh
```

Benchmark runs use `arena_simulation_setup/launch/robot.launch.py` to map:

```text
SocialNav*      -> agent_type=socialnav
UrbanNav*       -> agent_type=urbannav
LeLan* / LeLaN* -> agent_type=lelan
```

Pure DWB baselines should use an empty/non-AI `agent_name`, so no AI controller is
launched.

Runtime modules:

```text
python3 -m arena_ai_integration.nodes.ai_controller_node
python3 -m arena_ai_integration.nodes.human_states_bridge
python3 -m arena_ai_integration.nodes.semantic_laser_filter
python3 -m arena_ai_integration.tools.aggregate_benchmark_metrics
```

## Conflict Rule

Only one AI controller should run for a robot namespace. Do not launch
additional external controllers together with `arena_ai_integration` for the same
robot namespace.
