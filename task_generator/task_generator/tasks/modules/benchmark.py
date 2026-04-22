import asyncio
import hashlib
import json
import logging
import os
import pathlib
import time
import typing
from logging import FileHandler, Formatter, StreamHandler

import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from hunav_msgs.srv import StartEvaluation
from rclpy.parameter import Parameter
from std_srvs.srv import Empty

from task_generator.constants import Constants
from task_generator.tasks.modules import TM_Module


class _Config(typing.NamedTuple):
    @classmethod
    def parse(cls, obj: typing.Dict):
        return cls(
            suite=cls.Suite(**obj["suite"]),
            contest=cls.Contest(**obj["contest"]),
            general=cls.General(**obj["general"]),
        )

    class Suite(typing.NamedTuple):
        config: str
        scale_episodes: float = 1

    class Contest(typing.NamedTuple):
        config: str

    class General(typing.NamedTuple):
        simulator: str

    suite: Suite
    contest: Contest
    general: General


class Suite(typing.NamedTuple):
    @classmethod
    def parse(cls, name: str, obj: typing.Dict, config_class=None):
        return cls(
            name=name,
            stages=[cls.Stage.parse(stage, config_class) for stage in obj["stages"]]
        )

    class Index(int):
        pass

    class Stage(typing.NamedTuple):
        name: str
        episodes: int
        robot: str
        map: str
        tm_robots: Constants.TaskMode.TM_Robots
        tm_obstacles: Constants.TaskMode.TM_Obstacles
        config: typing.Dict
        seed: int
        timeout: str

        @classmethod
        def _make_serializable(cls, item):
            if isinstance(item, dict):
                return {k: cls._make_serializable(v) for k, v in item.items()}
            elif isinstance(item, (list, tuple)):
                return [cls._make_serializable(i) for i in item]
            elif isinstance(item, (Constants.TaskMode.TM_Robots, Constants.TaskMode.TM_Obstacles)):
                return item.value
            elif hasattr(item, 'value'):
                return cls._make_serializable(item.value)
            elif hasattr(item, 'get_value'):
                return cls._make_serializable(item.get_value())
            elif hasattr(item, '__str__'):
                logging.getLogger("benchmark").debug(f"Converting unknown type {type(item)} to str: {str(item)}")
                return str(item)
            return item

        @classmethod
        def hash(cls, obj: typing.Dict) -> int:
            logger = logging.getLogger("benchmark")
            hashable_obj = {k: v for k, v in obj.items() if k != "config"}
            hashable_obj = cls._make_serializable(hashable_obj)
            try:
                return 0x7fffffff & int.from_bytes(
                    hashlib.sha1(json.dumps(hashable_obj).encode()).digest()[-4:], byteorder="big"
                )
            except Exception as e:
                logger.error(f"Hash failed: {e}, using fallback seed")
                return 0

        @classmethod
        def parse(cls, obj: typing.Dict, config_class=None) -> "Suite.Stage":
            if config_class is None:
                raise ValueError("Configuration class must be provided")
            logger = logging.getLogger("benchmark")
            if "tm_robots" in obj:
                if isinstance(obj["tm_robots"], str):
                    try:
                        obj["tm_robots"] = Constants.TaskMode.TM_Robots[obj["tm_robots"].upper()]
                    except KeyError:
                        logger.error(f"Invalid tm_robots value: {obj['tm_robots']}")
                        raise
                elif hasattr(obj["tm_robots"], 'value'):
                    obj["tm_robots"] = Constants.TaskMode.TM_Robots[obj["tm_robots"].value.upper()]
                elif hasattr(obj["tm_robots"], 'get_value'):
                    obj["tm_robots"] = Constants.TaskMode.TM_Obstacles[obj["tm_robots"].get_value().upper()]
                else:
                    logger.error(f"Invalid tm_robots type: {type(obj['tm_robots'])}")
                    raise ValueError(f"Invalid tm_robots type: {type(obj['tm_robots'])}")
            if "tm_obstacles" in obj:
                if isinstance(obj["tm_obstacles"], str):
                    try:
                        obj["tm_obstacles"] = Constants.TaskMode.TM_Obstacles[obj["tm_obstacles"].upper()]
                    except KeyError:
                        logger.error(f"Invalid tm_obstacles value: {obj['tm_obstacles']}")
                        raise
                elif hasattr(obj["tm_obstacles"], 'value'):
                    obj["tm_obstacles"] = Constants.TaskMode.TM_Obstacles[obj["tm_obstacles"].value.upper()]
                elif hasattr(obj["tm_obstacles"], 'get_value'):
                    obj["tm_obstacles"] = Constants.TaskMode.TM_Obstacles[obj["tm_obstacles"].get_value().upper()]
                else:
                    logger.error(f"Invalid tm_obstacles type: {type(obj['tm_obstacles'])}")
                    raise ValueError(f"Invalid tm_obstacles type: {type(obj['tm_obstacles'])}")
            obj.setdefault("timeout", str(config_class.Robot.TIMEOUT))
            obj.setdefault("seed", cls.hash(obj))
            return cls(**obj)

    name: str
    stages: typing.List[Stage]

    @property
    def min_index(self):
        return self.Index()

    @property
    def max_index(self) -> Index:
        return self.Index(len(self.stages) - 1)

    def config(self, index: Index) -> Stage:
        return self.stages[index]


class Contest(typing.NamedTuple):
    @classmethod
    def parse(cls, name: str, obj: dict):
        return cls(
            name=name,
            contestants=[cls.Contestant.parse(contestant) for contestant in obj["contestants"]]
        )

    class Index(int):
        pass

    class Contestant(typing.NamedTuple):
        name: str
        local_planner: str
        inter_planner: str
        agent_name: str = ""

        @classmethod
        def parse(cls, obj: typing.Dict) -> "Contest.Contestant":
            obj.setdefault("inter_planner", "navigate_w_replanning_time")
            return cls(**obj)

    name: str
    contestants: typing.List[Contestant]

    @property
    def min_index(self):
        return self.Index()

    @property
    def max_index(self) -> Index:
        return self.Index(len(self.contestants) - 1)

    def config(self, index: int) -> Contestant:
        return self.contestants[index]


class Mod_Benchmark(TM_Module):
    DIR = pathlib.Path(os.path.join(get_package_share_directory("arena_bringup"), "configs", "benchmark"))
    LOCK_FILE = "resume.lock"
    LOG_DIR = DIR / "logs"


    _config: _Config
    _suite: Suite
    _contest: Contest
    _episode_index: int
    _runid: str
    _contest_index: Contest.Index
    _suite_index: Suite.Index
    _headless: int
    _config_class: typing.Any
    _primary_node: str
    _logger_object: logging.Logger = None  # type: ignore

    @classmethod
    def _load_config(cls) -> _Config:
        with open(cls.DIR / "config.yaml") as f:
            config = yaml.safe_load(f)
            assert isinstance(config, dict), "expected a dict in config.yaml"
            return _Config.parse(config)

    @classmethod
    def _load_contest(cls, contest: str) -> Contest:
        with open(cls.DIR / "contests" / contest) as f:
            config = yaml.safe_load(f)
            assert isinstance(config, dict), "expected a dict in contest.yaml"
            return Contest.parse(pathlib.Path(contest).name.strip(".yaml"), config)

    @classmethod
    def _load_suite(cls, suite: str, config_class):
        with open(cls.DIR / "suites" / suite) as f:
            config = yaml.safe_load(f)
            assert isinstance(config, dict), "expected a dict in suite.yaml"
            return Suite.parse(pathlib.Path(suite).name.strip(".yaml"), config, config_class)

    @classmethod
    def _resume(cls):
        with open(cls.DIR / cls.LOCK_FILE) as f:
            runid, contest, suite, headless = f.read().split(" ")
            return runid, Contest.Index(contest), Suite.Index(suite), int(headless)

    def _normalize_namespace(self, namespace: str) -> str:
        """Normalize namespace by removing extra slashes and ensuring proper format."""
        namespace = os.path.normpath(namespace)
        return os.path.join("/", namespace) if namespace else self.node.service_namespace()


    def _set_node_parameters(self, suite_config):
        """
        Apply map/world, task modes, and scenario file for the current stage.
        Only touches parameters that actually need to change.
        Returns True on success, False on error.
        """
        logger = self._logger
        updated = False

        # Task-mode enums (robots / obstacles)
        new_tm_r = Constants.TaskMode.TM_Robots(suite_config.tm_robots).value
        new_tm_o = Constants.TaskMode.TM_Obstacles(suite_config.tm_obstacles).value

        if self.node.conf.TaskMode.TM_ROBOTS.value != new_tm_r:
            self.node.conf.TaskMode.TM_ROBOTS.value = new_tm_r
            logger.info(f"[Benchmark] TM_ROBOTS → {new_tm_r}")
            updated = True

        if self.node.conf.TaskMode.TM_OBSTACLES.value != new_tm_o:
            self.node.conf.TaskMode.TM_OBSTACLES.value = new_tm_o
            logger.info(f"[Benchmark] TM_OBSTACLES → {new_tm_o}")
            updated = True

        # Map / world
        if suite_config.map and self.node.conf.Arena.WORLD.value != suite_config.map:
            self.node.conf.Arena.WORLD.value = suite_config.map
            logger.info(f"[Benchmark] World  → {suite_config.map}")
            updated = True

        # Scenario JSON
        scenario_file = suite_config.config.get("SCENARIO", {}).get("file")
        if scenario_file:
            # Declare once (ignored if already declared)
            if not self.node.has_parameter("task.scenario.file"):
                self.node.declare_parameter("task.scenario.file", scenario_file)

            current = self.node.get_parameter("task.scenario.file").value
            if current != scenario_file:
                try:
                    self.node.set_parameters([
                        Parameter("task.scenario.file",
                                  Parameter.Type.STRING,
                                  scenario_file)
                    ])
                    logger.info(f"[Benchmark] Scenario file → {scenario_file}")
                    updated = True
                except Exception as e:
                    logger.error(f"[Benchmark] Could not set scenario file: {e}")
                    return False

        # Apply per-stage timeout
        try:
            stage_timeout = float(suite_config.timeout)
            if self.node.conf.Robot.TIMEOUT.value != stage_timeout:
                self.node.conf.Robot.TIMEOUT.value = stage_timeout
                logger.info(f"[Benchmark] Timeout → {stage_timeout}s")
                updated = True
        except Exception as e:
            logger.warning(f"[Benchmark] Could not set stage timeout: {e}")

        if not updated:
            logger.debug("[Benchmark] No parameter changes for this stage.")
        return True

    def __init__(self, task, **kwargs):
        super().__init__(task=task, **kwargs)

        self.needs_reincarnation: bool = True
        self._hunav_recording: bool = False

        self._runid = f"t{int(time.time())}"
        # Log detected task_generator_nodes
        self._primary_node = self.node.service_namespace()

        # self._config_class = Configuration(ROSParamServer(f'benchmark_param_server_{int(time.time())}'))
        self._config = self._load_config()
        self._suite = self._load_suite(suite=self._config.suite.config, config_class=self.node.conf)
        self._contest = self._load_contest(self._config.contest.config)
        self._episode_index = -1
        self._contest_index = self._contest.min_index
        self._suite_index = Suite.Index(self._suite.min_index - 1)
        self._headless = 1

        # Hunav evaluator service clients (absolute names – evaluator runs in root namespace)
        self._hunav_start_client = self.node.create_client(
            StartEvaluation, '/hunav_start_recording'
        )
        self._hunav_stop_client = self.node.create_client(
            Empty, '/hunav_stop_recording'
        )

        os.makedirs(self.LOG_DIR, exist_ok=True)
        with open(self.LOG_DIR / f"{self._runid}.log", "w") as f:
            f.write(f"run {self._runid}\n")
            f.write(f"contest {self._contest.name}\n")
            f.write(f"suite {self._suite.name}\n")

        self._log_contest()
        # suite_index starts at -1; first before_reset() will advance to 0
        # self._reincarnate()

    # ---- hunav evaluator helpers ----

    def _hunav_start_recording(self):
        """Call hunav_start_recording service (non-blocking fire-and-forget)."""
        logger = self._logger

        if not self._hunav_start_client.service_is_ready():
            logger.warning("[Benchmark] hunav_start_recording service not available – metrics will NOT be recorded")
            return

        contest_cfg = self._contest.config(self._contest_index)
        suite_cfg = self._suite.config(self._suite_index)

        # Build a meaningful experiment tag:  <planner>_<stage>_ep<N>
        experiment_tag = f"{contest_cfg.name}_{suite_cfg.name}_ep{self._episode_index}"

        # Try to get the robot goal from the first robot manager
        robot_goal = PoseStamped()
        try:
            first_robot = next(iter(self._TASK.robots.values()), None)
            if first_robot is not None:
                goal_pose = first_robot.goal_pos
                if goal_pose is not None:
                    robot_goal.header.frame_id = "map"
                    robot_goal.header.stamp = self.node.get_clock().now().to_msg()
                    robot_goal.pose = goal_pose.to_msg()
        except Exception as e:
            logger.debug(f"[Benchmark] Could not retrieve robot goal for evaluator: {e}")

        request = StartEvaluation.Request()
        request.experiment_tag = experiment_tag
        request.robot_goal = robot_goal
        request.run_id = int(self._runid.lstrip('t')) % (2**31)

        # Fire-and-forget: never block the node's executor
        self._hunav_recording = True
        logger.info(f"[Benchmark] Requesting hunav start recording: {experiment_tag}")
        future = self._hunav_start_client.call_async(request)
        future.add_done_callback(
            lambda f: self._on_hunav_start_result(f, experiment_tag)
        )

    def _on_hunav_start_result(self, future, experiment_tag):
        """Async callback – logs the evaluator's response."""
        try:
            result = future.result()
            if result and result.success:
                self._logger.info(f"[Benchmark] Hunav evaluator confirmed recording: {experiment_tag}")
            else:
                self._logger.warning(f"[Benchmark] Hunav evaluator rejected recording (tag={experiment_tag})")
                self._hunav_recording = False
        except Exception as e:
            self._logger.error(f"[Benchmark] hunav_start_recording callback error: {e}")
            self._hunav_recording = False

    def _hunav_stop_recording(self):
        """Call hunav_stop_recording service (non-blocking fire-and-forget)."""
        if not self._hunav_recording:
            return

        logger = self._logger
        self._hunav_recording = False

        if not self._hunav_stop_client.service_is_ready():
            logger.warning("[Benchmark] hunav_stop_recording service not available")
            return

        logger.info("[Benchmark] Requesting hunav stop recording")
        future = self._hunav_stop_client.call_async(Empty.Request())
        future.add_done_callback(self._on_hunav_stop_result)

    def _on_hunav_stop_result(self, future):
        """Async callback - logs stop recording result."""
        try:
            future.result()
            self._logger.info("[Benchmark] Hunav evaluator stopped recording – metrics computed")
        except Exception as e:
            self._logger.error(f"[Benchmark] hunav_stop_recording callback error: {e}")

    def before_reset(self):
        self._logger.debug("Before task reset")
        # Stop hunav recording for the previous episode (if active)
        self._hunav_stop_recording()
        if self.needs_reincarnation:
            self.needs_reincarnation = False
            self._episode_index = -1

            # Remember the TM types BEFORE reincarnation updates conf
            old_tm_robots = self.node.conf.TaskMode.TM_ROBOTS.value
            old_tm_obstacles = self.node.conf.TaskMode.TM_OBSTACLES.value

            self.suite_index += 1
            self._reincarnate()

            # Only force TM re-instantiation when the TM *type* has actually
            # changed (e.g. RANDOM → SCENARIO).  _reset_task() already checked
            # the TM type BEFORE calling before_reset(), so if the type changed
            # during _reincarnate() we must apply it now.
            #
            # When the TM type stays the same (e.g. SCENARIO → SCENARIO with a
            # different scenario file), do NOT re-instantiate: the existing
            # TM_Scenario._config ROSParam callback was already triggered by
            # _set_node_parameters() and the instance already holds the new
            # scenario data.  Re-instantiating would discard that state and
            # register duplicate ROSParam callbacks.
            new_tm_robots = self.node.conf.TaskMode.TM_ROBOTS.value
            new_tm_obstacles = self.node.conf.TaskMode.TM_OBSTACLES.value

            if new_tm_robots != old_tm_robots:
                self._TASK.set_tm_robots(new_tm_robots)
                self._logger.info(
                    f"[Benchmark] TM type changed: robots {old_tm_robots} → {new_tm_robots}"
                )

            if new_tm_obstacles != old_tm_obstacles:
                self._TASK.set_tm_obstacles(new_tm_obstacles)
                self._logger.info(
                    f"[Benchmark] TM type changed: obstacles {old_tm_obstacles} → {new_tm_obstacles}"
                )

            if new_tm_robots == old_tm_robots and new_tm_obstacles == old_tm_obstacles:
                self._logger.info(
                    f"[Benchmark] TM types unchanged ({new_tm_robots}, {new_tm_obstacles})"
                    " — scenario file updated via ROSParam callback"
                )

    def after_reset(self):
        self._episode_index += 1
        self._hunav_start_recording()
        self._log_episode()
        episode_limit = int(self._suite.config(self._suite_index).episodes * self._config.suite.scale_episodes)
        if self._episode_index >= episode_limit:
            self.needs_reincarnation = True

    def _reset_task(self):
        result = self._TASK.reset()
        if asyncio.iscoroutine(result):
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(result)
            except RuntimeError:
                asyncio.run(result)

    @property
    def _logger(self) -> logging.Logger:  # type: ignore
        if self._logger_object is None:
            handler = FileHandler(self.LOG_DIR / f"{self._runid}.log")
            handler.setFormatter(Formatter("%(asctime)s: %(levelname)s: %(message)s"))
            console_handler = StreamHandler()
            console_handler.setFormatter(Formatter("%(asctime)s: %(levelname)s: %(message)s"))
            logger = logging.getLogger("benchmark")
            logger.setLevel(logging.DEBUG)
            logger.addHandler(handler)
            logger.addHandler(console_handler)
            self._logger_object = logger
        return self._logger_object

    def _log_contest(self):
        self._logger.info(f"C [{1+self._contest_index}/{1+self._contest.max_index}] {self._contest.config(self._contest_index).name}")

    def _log_suite(self):
        self._logger.info(f"S [{1+self._suite_index}/{1+self._suite.max_index}] {self._suite.config(self._suite_index).name}")

    def _log_episode(self):
        if self._episode_index < 0:
            return
        episode_limit = int(self._suite.config(self._suite_index).episodes * self._config.suite.scale_episodes)
        self._logger.info(f"E [{1+self._episode_index}/{episode_limit}]")

    @property
    def contest_index(self) -> Contest.Index:
        return self._contest_index

    @contest_index.setter
    def contest_index(self, index: int):
        self._contest_index = Contest.Index(index)
        if self._contest_index > self._contest.max_index:
            self._logger.info("Benchmark completed")
        else:
            self._log_contest()
            # self._reincarnate()

    @property
    def suite_index(self) -> Suite.Index:
        return self._suite_index

    @suite_index.setter
    def suite_index(self, index: int):
        self._suite_index = Suite.Index(index)
        if self._suite_index > self._suite.max_index:
            self._suite_index = self._suite.min_index
            self.contest_index += 1
        else:
            self._log_suite()
            # self._reincarnate()

    @property
    def _episode(self) -> int:
        return self._episode_index

    @_episode.setter
    def _episode(self, episode: int):
        episode_limit = int(self._suite.config(self._suite_index).episodes * self._config.suite.scale_episodes)
        if episode >= episode_limit:
            self._episode_index = -1
            self.suite_index += 1
        else:
            self._episode_index = episode
            self._log_episode()

    def _reincarnate(self):
        logger = self._logger
        logger.debug("Starting reincarnation process")
        suite_config = self._suite.config(self._suite_index)
        logger.info(f"Transitioning to stage: {suite_config.name} (tm_robots={suite_config.tm_robots.value}, tm_obstacles={suite_config.tm_obstacles.value})")
        success = False
        selected_node = self._primary_node
        if self._set_node_parameters(suite_config):
            logger.info(f"Stage setup complete for {suite_config.name} on {selected_node}")
            success = True
        else:
            logger.error(f"Failed to set parameters for {suite_config.name} on {selected_node}")

        if not success:
            logger.error(f"Failed to set parameters for {suite_config.name} on any task_generator_node. Ensure tm_robots and tm_obstacles are declared in task_generator_node.py.")
        # self._episode = 0
        # self._reset_task()
        # NOTE: Do NOT call _reset_task() or set _episode here.
        # The framework reset cycle that called before_reset() will continue
        # with __tm_robots.reset() / __tm_obstacles.reset() using the params
        # we just set above.  after_reset() handles the episode counter.