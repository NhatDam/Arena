# Arena 5 Debug Notes

Tai lieu nay tong hop cac loi da gap khi setup va chay benchmark SocialNav/UrbanNav trong `arena5_ws`.
Muc tieu la giup debug nhanh theo tang: Conda/Python, ROS build, Isaac/assets, RViz robot, va AI/BEV.

Workspace mac dinh:

```bash
cd ~/arena5_ws
```

Lenh chay benchmark thuong dung:

```bash
source /opt/ros/humble/setup.bash
source ~/arena5_ws/install/setup.bash
conda activate socialnav
export ISAAC_PATH=$HOME/isaacsim-4.2.0

ARENA_AI_PYTHONNOUSERSITE=1 ARENA_HEADLESS=1 RESTART_STACK_EACH_EPISODE=0 FORCE_COLOR=1 \
bash ./src/Arena/arena_ai_integration/scripts/start_benchmark.sh
```

## 1. Conda va Python Environment

### Loi: khong tim thay conda init script

Log:

```text
[ERROR] Could not find conda initialization script.
Expected one of:
  /home/.../miniconda3/etc/profile.d/conda.sh
  /home/.../anaconda3/etc/profile.d/conda.sh
```

Nguyen nhan: script can activate conda nhung shell chua duoc init, hoac conda khong o duong dan mac dinh.

Kiem tra:

```bash
which conda
echo "$CONDA_EXE"
ls ~/miniconda3/etc/profile.d/conda.sh
```

Fix:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
```

Neu conda nam o vi tri khac:

```bash
export CONDA_EXE=/path/to/conda
```

### Loi: `EnvironmentNameNotFound: socialnav`

Log:

```text
EnvironmentNameNotFound: Could not find conda environment: socialnav
```

Nguyen nhan: env `socialnav` chua duoc tao.

Kiem tra:

```bash
conda info --envs
```

Fix:

```bash
conda create -n socialnav python=3.10 -y
conda activate socialnav
```

### Loi: Conda Terms of Service chua accept

Log:

```text
CondaToSNonInteractiveError: Terms of Service have not been accepted
```

Fix:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

Sau do tao lai env:

```bash
conda create -n socialnav python=3.10 -y
```

### Loi: pip build CLIP thieu `pkg_resources`

Log:

```text
ModuleNotFoundError: No module named 'pkg_resources'
ERROR: Failed to build 'clip'
```

Nguyen nhan: `setuptools` moi co the khong expose `pkg_resources` trong build env.

Fix:

```bash
conda activate socialnav
python -m pip install --upgrade "pip<25" "setuptools<70" wheel
python -m pip install -r src/arena-social-nav/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu121
```

### Loi: Torch import fail vi thieu `typing_extensions`

Log:

```text
ModuleNotFoundError: No module named 'typing_extensions'
```

Fix:

```bash
conda activate socialnav
python -m pip install typing_extensions
```

## 2. System Python, ROS Python va NumPy

Arena dung nhieu Python runtime:

- System Python: `/usr/bin/python3`
- Isaac Python: `$ISAAC_PATH/python.sh`
- AI Python: `~/miniconda3/envs/socialnav/bin/python3`

Script benchmark se preflight ca 3 runtime.

### Loi: system/Isaac Python thieu `cattrs`

Log:

```text
[ERROR] System Python cannot import cattrs
[ERROR] IsaacSim Python cannot import cattrs
```

Fix:

```bash
/usr/bin/python3 -m pip install --user cattrs attrs shapely
$HOME/isaacsim-4.2.0/python.sh -m pip install cattrs attrs shapely
```

### Loi: `tf_transformations` fail voi NumPy 2.x

Log:

```text
AttributeError: np.maximum_sctype was removed in the NumPy 2.0 release
```

Nguyen nhan: ROS Humble package `transforms3d` trong system Python khong tuong thich NumPy 2.x trong `~/.local`.

Fix system Python:

```bash
/usr/bin/python3 -m pip install --user "numpy<2"
```

Hoac tranh user-site khi chay ROS:

```bash
export PYTHONNOUSERSITE=1
```

Luu y: AI env `socialnav` co the can NumPy 2.0 de tuong thich PyTorch 2.5.0. Khong tron system Python va AI Python.

## 3. APT va ROS Dependencies

### Loi: apt unmet dependencies / broken packages

Log:

```text
E: Unmet dependencies. Try 'apt --fix-broken install'
```

Kiem tra truoc khi sua:

```bash
sudo apt -s --fix-broken install
```

Neu output hop ly thi chay that:

```bash
sudo apt --fix-broken install
sudo apt update
```

### Loi: GraphicsMagick C++ missing

Log:

```text
GRAPHICSMAGICKCPP_LIBRARIES (missing: GRAPHICSMAGICKCPP_INCLUDE_DIRS)
```

Fix:

```bash
sudo apt install -y graphicsmagick libgraphicsmagick++1-dev
```

### Loi: linker khong tim thay yaml-cpp

Log:

```text
/usr/bin/ld: cannot find -lyaml-cpp
```

Fix:

```bash
sudo apt install -y libyaml-cpp-dev
```

### Loi: gtest include lay tu conda

Log:

```text
/home/.../miniconda3/include/gtest/gtest.h
error: redundant redeclaration ... [-Werror=deprecated]
```

Nguyen nhan: build ROS khi conda dang active, compiler lay header tu conda.

Fix:

```bash
conda deactivate
unset CMAKE_PREFIX_PATH CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH
source /opt/ros/humble/setup.bash
colcon build --symlink-install --cmake-args -DBUILD_TESTING=OFF
```

## 4. Colcon Build va Workspace Overlay

### Loi: build package rieng le thieu dependency da build

Log:

```text
Failed to find:
install/arena_robots/share/arena_robots/package.sh
Check that package has been built: arena_robots
```

Nguyen nhan: dang build package con, nhung package phu thuoc chua co trong `install/`.

Fix:

```bash
conda deactivate
source /opt/ros/humble/setup.bash
colcon build --symlink-install --cmake-args -DBUILD_TESTING=OFF
source ~/arena5_ws/install/setup.bash
```

Hoac build kem dependency:

```bash
colcon build --packages-up-to task_generator --symlink-install --cmake-args -DBUILD_TESTING=OFF
```

### Loi: `em.BUFFERED_OPT` trong rosidl/slam_toolbox

Log:

```text
AttributeError: module 'em' has no attribute 'BUFFERED_OPT'
```

Nguyen nhan: user-site dang co `empy` 4.x, ROS Humble can API cua empy 3.x.

Kiem tra:

```bash
/usr/bin/python3 - <<'PY'
import em
print(em.__file__)
print(getattr(em, "__version__", "no version"))
print(hasattr(em, "BUFFERED_OPT"))
PY
```

Fix:

```bash
/usr/bin/python3 -m pip uninstall -y empy
/usr/bin/python3 -m pip install --user "empy==3.3.4"
```

Neu pip bao khong co `--user` cho uninstall, dung:

```bash
/usr/bin/python3 -m pip uninstall -y empy
```

## 5. Isaac Sim, Headless va Assets

### Loi: thieu `ISAAC_PATH`

Log:

```text
environment variable 'ISAAC_PATH' does not exist
```

Fix:

```bash
export ISAAC_PATH=$HOME/isaacsim-4.2.0
```

Them vao `~/.bashrc` neu can:

```bash
echo 'export ISAAC_PATH=$HOME/isaacsim-4.2.0' >> ~/.bashrc
```

### Headless modes

Trong script benchmark:

- `ARENA_HEADLESS=0`: mo Isaac GUI va RViz.
- `ARENA_HEADLESS=1`: tat GUI Isaac, van co RViz.
- `ARENA_HEADLESS=2`: tat ca Isaac GUI va RViz.

Lenh hay dung:

```bash
ARENA_HEADLESS=1 RESTART_STACK_EACH_EPISODE=0 FORCE_COLOR=1 \
bash ./src/Arena/arena_ai_integration/scripts/start_benchmark.sh
```

### Loi: malformed launch argument `headless:=`

Log:

```text
malformed launch argument 'headless:=', expected format '<name>:=<value>'
```

Nguyen nhan: bien `ARENA_HEADLESS` rong.

Fix:

```bash
export ARENA_HEADLESS=1
```

### Loi: thieu assets Hospital

Log:

```text
ObjectIdentifier(name='SM_SupplyCart_03a', domain='Hospital') not found
DynamicPathResolver<Object>(path=.../worlds/hospital_1/assets)
DynamicPathResolver<Object>(path=.../_assets/_local)
NetResolver<Object>(path=.../_assets/default)
```

Nguyen nhan: assets Hospital chua duoc tai dung cho Arena.

Kiem tra bien:

```bash
export ARENA_ASSETS_DIR="$PWD/src/Arena/_assets"
export ARENA_ASSETS_DIR_LOCAL="$PWD/src/Arena/_assets/_local"
export ARENA_MODELS_FORMATS="yaml,usdz,usd,usda,usdc"
```

Neu `ros2 run arena_models ...` bao:

```text
Package 'arena_models' not found
```

thi package `arena_models` chua duoc build/source hoac khong co trong workspace. Can build/source lai workspace hoac cai dung tool assets theo huong dan Arena.

## 6. Robot Khong Hien Trong RViz

### Kiem tra description package

```bash
sudo apt install ros-humble-irobot-create-description ros-humble-turtlebot4-description
ros2 pkg prefix irobot_create_description
ros2 pkg prefix turtlebot4_description
```

Neu tra ve `/opt/ros/humble` la da cai thanh cong.

### Kiem tra robot_description va TF

```bash
ros2 topic list | grep robot_description
ros2 topic list | grep tf_static
ros2 node list | grep robot_state_publisher
```

Topic dung da tung thay:

```text
/task_generator_node/turtlebot/robot_description
/tf_static
```

Neu robot khong hien:

- RViz Fixed Frame nen la `map` hoac frame co TF.
- RobotModel display phai tro toi topic `/task_generator_node/turtlebot/robot_description`.
- Can co TF tu `map -> odom -> turtlebot/base_link`.

Code lien quan:

- `src/Arena/arena_simulation_setup/launch/robot.launch.py`
- `src/Arena/arena_simulation_setup/launch/state_publisher.launch.py`
- `src/Arena/arena_robots/arena_robots/robots/turtlebot/...`

## 7. AI Controller va BEV/Path

### Topic BEV/path dung

Voi `UrbanNav_FiLM`, topic prefix la `urbannav`:

```text
/urbannav/bev_viz
/urbannav/path
/urbannav/arrival_score
```

Voi SocialNav, topic prefix la `socialnav`.

Kiem tra:

```bash
ROS_DISABLE_DAEMON=1 ros2 node list | grep ai_controller
ROS_DISABLE_DAEMON=1 ros2 topic list | grep -E 'urbannav|socialnav|bev|arrival|path'
```

Neu khong co gi, AI controller khong song hoac chua duoc launch.

### Dieu kien launch AI

AI controller chi duoc launch trong `robot.launch.py` khi:

- `agent_name` bat dau bang `SocialNav`, `UrbanNav`, `LeLan`, hoac `LeLaN`.
- `local_planner == dwb`.
- `train_mode == false`.

Code:

```text
src/Arena/arena_simulation_setup/launch/robot.launch.py
```

Config contest hien tai nen co dang:

```yaml
contestants:
  - name: UrbanNav_FiLM
    local_planner: dwb
    inter_planner: navigate_w_replanning_time
    agent_name: UrbanNav_FiLM
```

### AI load model OK nhung khong inference

Log tot ban dau:

```text
UrbanNav model loaded ...
AI DWB Path Adapter initialized ... model_ready=True
```

Neu log dung o:

```text
New instruction received ... Waiting for goal_pose
```

thi AI chua nhan benchmark goal. AI chi inference sau khi `goal_callback()` dat `episode_active=True`.

Code lien quan:

```text
src/Arena/arena_ai_integration/arena_ai_integration/core/base_ai_node.py
```

Can thay log:

```text
Benchmark goal received; starting SocialNav DWB path-adapter episode.
Raw model waypoint
[AI] dist_to_goal=... inference active
AI path adapter: phase=...
```

### Loi OpenCV khong nhan NumPy array

Log:

```text
cv2.error: src is not a numpy array, neither a scalar
```

Kiem tra toi thieu:

```bash
conda activate socialnav
PYTHONNOUSERSITE=1 env -u PYTHONPATH -u LD_LIBRARY_PATH python - <<'PY'
import numpy as np, cv2
x = np.zeros((480, 640, 3), dtype=np.uint8)
print(np.__version__, cv2.__version__)
print(cv2.resize(x, (224, 224)).shape)
PY
```

Neu test nay fail thi khong phai loi Arena code, ma la OpenCV/NumPy ABI trong env.

Fix da dung:

```bash
conda activate socialnav
python -m pip install --no-cache-dir --force-reinstall \
  "numpy==2.0.0" \
  "opencv-python-headless==4.10.0.84"
```

### Loi PyTorch khong nhan NumPy array

Log:

```text
TypeError: expected np.ndarray (got numpy.ndarray)
```

Kiem tra:

```bash
PYTHONNOUSERSITE=1 env -u PYTHONPATH -u LD_LIBRARY_PATH \
~/miniconda3/envs/socialnav/bin/python - <<'PY'
import numpy as np, torch
x = np.zeros((4, 4, 3), dtype=np.uint8)
print(np.__version__, torch.__version__)
print(torch.from_numpy(x).shape)
PY
```

Fix da dung:

```bash
conda activate socialnav
python -m pip install --no-cache-dir --force-reinstall \
  "numpy==2.0.0" \
  "opencv-python-headless==4.10.0.84"
```

Expected:

```text
torch.Size([4, 4, 3])
```

### AI controller bien mat khoi graph, khong co BEV topic

Trieu chung:

```bash
ROS_DISABLE_DAEMON=1 ros2 node list | grep ai_controller
ROS_DISABLE_DAEMON=1 ros2 topic list | grep -E 'urbannav|socialnav|bev|arrival|path'
```

Khong in ra gi.

Kiem tra process:

```bash
ps -ef | grep arena_ai_integration.nodes.ai_controller_node
```

Kiem tra kernel segfault:

```bash
grep -E "segfault|cv_bridge_boost|python3" /var/log/kern.log /var/log/syslog | tail -80
```

Loi da gap:

```text
python3[PID]: segfault ... in cv_bridge_boost.so
```

Nguyen nhan: AI controller crash native trong `cv_bridge_boost.so` khi convert `sensor_msgs/Image`.
Vi la segfault nen Python log khong co traceback ro rang.

Fix da ap dung:

- Bo `cv_bridge` trong `base_ai_node.py`.
- Chuyen `sensor_msgs/Image` sang RGB `np.ndarray` bang NumPy thuan.
- Publish BEV Image bang `sensor_msgs.msg.Image` thu cong.

File lien quan:

```text
src/Arena/arena_ai_integration/arena_ai_integration/core/base_ai_node.py
```

Build lai:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select arena_ai_integration --symlink-install
source ~/arena5_ws/install/setup.bash
```

Sau do restart benchmark hoan toan.

## 8. Nav2/Goal va AI khong nhan goal

RobotManager publish goal topic va send Nav2 action.

Code:

```text
src/Arena/task_generator/task_generator/manager/robot_manager/robot_manager.py
```

Topic goal:

```text
/task_generator_node/turtlebot/goal_pose
```

AI subscribe topic nay:

```text
src/Arena/arena_ai_integration/arena_ai_integration/core/base_ai_node.py
```

Kiem tra:

```bash
ros2 topic echo --once /task_generator_node/turtlebot/goal_pose
ros2 topic info /task_generator_node/turtlebot/goal_pose -v
```

Neu AI start muon va bo lo goal, can xem QoS:

- RobotManager publish goal bang `TRANSIENT_LOCAL`.
- AI subscriber goal hien co the dung `VOLATILE`.

Trieu chung:

```text
New instruction received ... Waiting for goal_pose
```

ma khong bao gio co:

```text
Benchmark goal received
```

Huong fix: de AI subscriber goal dung QoS `TRANSIENT_LOCAL`, hoac publish lai goal sau khi AI node da song.

## 9. Clean Restart Checklist

Khi ROS graph bi stale, Nav2 timeout, hoac AI topic bien mat:

```bash
# Dung benchmark bang Ctrl+C truoc.

pkill -f "arena_ai_integration.nodes.ai_controller_node" || true
pkill -f "arena_ai_integration.nodes.human_states_bridge" || true
pkill -f "ros2 launch arena_bringup arena.launch.py" || true
pkill -f "/install/task_generator/lib/task_generator/task_generator_node" || true
pkill -f "/opt/ros/humble/lib/topic_tools/relay .*__ns:=/task_generator_node" || true
```

Sau do:

```bash
source /opt/ros/humble/setup.bash
source ~/arena5_ws/install/setup.bash
conda activate socialnav
export ISAAC_PATH=$HOME/isaacsim-4.2.0

ARENA_AI_PYTHONNOUSERSITE=1 ARENA_HEADLESS=1 RESTART_STACK_EACH_EPISODE=0 FORCE_COLOR=1 \
bash ./src/Arena/arena_ai_integration/scripts/start_benchmark.sh
```

Kiem tra nhanh:

```bash
ROS_DISABLE_DAEMON=1 ros2 node list | grep ai_controller
ROS_DISABLE_DAEMON=1 ros2 topic list | grep -E 'urbannav|socialnav|bev|arrival|path'
```

Expected:

```text
/ai_controller_task_generator_node_turtlebot
/urbannav/bev_viz
/urbannav/path
/urbannav/arrival_score
```

## 10. Log Files Hay Can Xem

Log ROS moi nhat:

```bash
ls -td ~/.ros/log/* | head
```

AI controller log thuong co ten:

```text
~/.ros/log/python3_<pid>_<timestamp>.log
```

Tim loi AI:

```bash
grep -R "ai_controller\\|AI inference\\|UrbanNav\\|SocialNav\\|Traceback\\|segfault" ~/.ros/log -n | tail -100
```

Tim crash native:

```bash
grep -E "segfault|cv_bridge_boost|python3" /var/log/kern.log /var/log/syslog | tail -80
```

## 11. Thu Tu Debug Nen Theo

1. Conda env co ton tai khong:

   ```bash
   conda info --envs
   ```

2. AI Python co import torch/cv2/numpy dung khong:

   ```bash
   conda activate socialnav
   PYTHONNOUSERSITE=1 env -u PYTHONPATH -u LD_LIBRARY_PATH python - <<'PY'
   import numpy as np, cv2, torch
   x = np.zeros((10, 10, 3), dtype=np.uint8)
   print(np.__version__, cv2.__version__, torch.__version__)
   print(cv2.resize(x, (5, 5)).shape)
   print(torch.from_numpy(x).shape)
   PY
   ```

3. ROS workspace da build/source chua:

   ```bash
   source /opt/ros/humble/setup.bash
   source ~/arena5_ws/install/setup.bash
   ros2 pkg prefix arena_ai_integration
   ```

4. AI node co song khong:

   ```bash
   ROS_DISABLE_DAEMON=1 ros2 node list | grep ai_controller
   ```

5. BEV/path topic co khong:

   ```bash
   ROS_DISABLE_DAEMON=1 ros2 topic list | grep -E 'urbannav|socialnav|bev|arrival|path'
   ```

6. Neu AI node bien mat, kiem tra segfault:

   ```bash
   grep -E "segfault|cv_bridge_boost|python3" /var/log/kern.log /var/log/syslog | tail -80
   ```

7. Neu AI node song nhung khong inference, xem log co dung o `Waiting for goal_pose` hay `waiting for RGB history` khong.

8. Neu co `Raw model waypoint` va `[AI] dist_to_goal`, inference da chay. Luc do debug RViz display/topic QoS, khong debug model nua.
