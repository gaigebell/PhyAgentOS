"""
FrankaSimulationDriver — PhyAgentOS HAL driver for Franka arm simulation.

Bridges ACTION.md to FrankaManipulationAPI (internutopia sim stack).
Supports pick/place presets with Cartesian motion and gripper control.
"""

from __future__ import annotations

import importlib
import json
import threading
import time
from pathlib import Path
from typing import Any

from hal.base_driver import BaseDriver

_PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"


class FrankaSimulationDriver(BaseDriver):
    """Bridge ACTION.md calls to FrankaManipulationAPI methods.

    This driver wraps the FrankaManipulationAPI from internutopia to provide
    Franka arm manipulation capabilities in Isaac Sim simulation.

    Unlike PiperGo2 (mobile robot), Franka is a fixed-base 7-DOF arm, so
    navigation actions are not applicable.
    """

    def __init__(self, gui: bool = False, **kwargs: Any) -> None:
        self._gui = gui
        self._api = None
        self._env = None
        self._env_lock = threading.RLock()
        self._last_obs: Any = None
        self._last_scene: dict[str, dict] = {}

        self._scene_asset_path = str(kwargs.get("scene_asset_path", "")).strip()
        self._robot_position = tuple(kwargs.get("robot_position", (0.0, 0.0, 0.0)))
        self._robot_cfg = None

        self._grasp_targets: dict[str, dict] = kwargs.get("grasp_targets", {})
        self._place_targets: dict[str, dict] = kwargs.get("place_targets", {})

        self._output_dir = Path(kwargs.get("output_dir", "/tmp/paos_franka_sim_logs"))
        self._grasp_dump_name = str(kwargs.get("grasp_dump", "grasp.json"))
        self._place_dump_name = str(kwargs.get("place_dump", "place.json"))

        self._pause_steps = int(kwargs.get("pause_steps", 45))
        self._gripper_settle_steps = int(kwargs.get("gripper_settle_steps", 30))
        self._arm_waypoint_count = int(kwargs.get("arm_waypoint_count", 8))

        self._pythonpath_entries = self._normalize_pythonpath(kwargs.get("pythonpath", []))
        self._ensure_pythonpath()

        self._scene_narration_cn = ""
        self._last_action_summary = ""

        self._idle_step_enabled = bool(kwargs.get("idle_step_enabled", True))
        self._idle_steps_per_cycle = int(kwargs.get("idle_steps_per_cycle", 1))
        self._idle_step_interval_s = float(kwargs.get("idle_step_interval_s", 1.0 / 30.0))
        self._last_idle_step_ts = 0.0

        self._objects_spec = kwargs.get("objects", [])
        self._api_kwargs = dict(kwargs.get("api_kwargs", {}))

    @staticmethod
    def _normalize_pythonpath(raw: Any) -> list[str]:
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            return []
        out: list[str] = []
        for item in raw:
            p = str(item).strip()
            if not p:
                continue
            out.append(str(Path(p).expanduser().resolve()))
        return out

    def _ensure_pythonpath(self) -> None:
        import sys
        for entry in reversed(self._pythonpath_entries):
            if entry in sys.path:
                continue
            sys.path.insert(0, entry)

    def get_profile_path(self) -> Path:
        return _PROFILES_DIR / "franka_simulation.md"

    def load_scene(self, scene: dict[str, dict]) -> None:
        pass

    def execute_action(self, action_type: str, params: dict) -> str:
        handlers = {
            "start": self._start_from_action,
            "enter_simulation": self._start_from_action,
            "close": self._close_api,
            "step": self._step_env,
            "api_call": self._api_call,
            "grasp": self._grasp,
            "pick": self._grasp,
            "place": self._place,
            "release": self._place,
            "register_grasp_target": self._register_grasp_target,
            "register_place_target": self._register_place_target,
            "get_robot_state": self._get_robot_state,
        }
        handler = handlers.get(action_type)
        if handler is None:
            return f"Unknown action: {action_type}"
        try:
            return handler(params or {})
        except Exception as exc:
            return f"Error: {type(exc).__name__}: {exc}"

    def get_scene(self) -> dict[str, Any]:
        return {
            "manipulation_runtime": {
                "location": "sim",
                "state": "running" if self._api is not None else "idle",
                "robot_type": "franka",
                "scene_asset_path": self._scene_asset_path,
                "robot_position": self._robot_position,
                "grasp_targets": list(self._grasp_targets.keys()),
                "place_targets": list(self._place_targets.keys()),
                "last_action_summary": self._last_action_summary,
            }
        }

    def get_runtime_state(self) -> dict[str, Any]:
        if self._api is None or self._env is None:
            return {}
        try:
            obs = self._env.get_observations()
            if isinstance(obs, dict):
                eef_pos = obs.get("eef_position", [0.0, 0.0, 0.0])
                return {
                    "eef_position": list(eef_pos) if hasattr(eef_pos, "__iter__") else [float(eef_pos)],
                }
        except Exception:
            pass
        return {}

    def health_check(self) -> bool:
        if self._idle_step_enabled:
            self._idle_step_if_due()
        return self._api is not None

    def close(self) -> None:
        self._close_api({})

    def _start_from_action(self, params: dict[str, Any]) -> str:
        if self._api is not None:
            return "Franka Manipulation API already started."

        scene_asset_path = str(params.get("scene_asset_path", self._scene_asset_path)).strip()
        if not scene_asset_path:
            return "Error: missing scene_asset_path in driver config or action params."
        if not Path(scene_asset_path).exists():
            return f"Error: scene file not found: {scene_asset_path}"

        robot_position = params.get("robot_position", list(self._robot_position))
        if isinstance(robot_position, list) and len(robot_position) >= 3:
            self._robot_position = tuple(float(x) for x in robot_position[:3])

        api_kwargs = dict(self._api_kwargs)
        api_kwargs.update(params.get("api_kwargs", {}))

        self._api = self._build_api(
            scene_asset_path=scene_asset_path,
            robot_position=self._robot_position,
            api_kwargs=api_kwargs,
        )

        self._last_obs = self._api.start()
        self._env = getattr(self._api, "_env", None)

        self._last_action_summary = "started"
        return "Franka Manipulation API started."

    def _build_api(self, *, scene_asset_path: str, robot_position: tuple[float, ...], api_kwargs: dict[str, Any]):
        import sys
        from pathlib import Path

        pythonpath = self._pythonpath_entries
        for entry in reversed(pythonpath):
            ep = str(Path(entry).expanduser().resolve())
            if ep not in sys.path:
                sys.path.insert(0, ep)

        pythonpath_from_kwargs = api_kwargs.pop("pythonpath", None)
        if pythonpath_from_kwargs:
            if isinstance(pythonpath_from_kwargs, str):
                pythonpath_from_kwargs = [pythonpath_from_kwargs]
            for entry in reversed(pythonpath_from_kwargs):
                ep = str(Path(entry).expanduser().resolve())
                if ep not in sys.path:
                    sys.path.insert(0, ep)

        bridge = importlib.import_module("internutopia.bridge")
        FrankaManipulationAPI = bridge.FrankaManipulationAPI
        create_franka_robot_cfg = bridge.create_franka_robot_cfg

        robot_cfg = create_franka_robot_cfg(position=robot_position)
        robot_cfg.name = "franka"
        robot_cfg.prim_path = "/franka"

        pause_steps = int(api_kwargs.pop("pause_steps", self._pause_steps))
        gripper_settle_steps = int(api_kwargs.pop("gripper_settle_steps", self._gripper_settle_steps))
        arm_waypoint_count = int(api_kwargs.pop("arm_waypoint_count", self._arm_waypoint_count))

        headless = api_kwargs.pop("headless", None)
        if headless is None:
            headless = not self._gui

        api = FrankaManipulationAPI(
            scene_asset_path=scene_asset_path,
            robot_cfg=robot_cfg,
            headless=headless,
            pause_steps=pause_steps,
            gripper_settle_steps=gripper_settle_steps,
            arm_waypoint_count=arm_waypoint_count,
            **api_kwargs,
        )

        for name, target in self._grasp_targets.items():
            api.register_grasp_target(name, target)

        for name, target in self._place_targets.items():
            api.register_place_target(name, target)

        api.start()
        return api

    def _close_api(self, _params: dict[str, Any]) -> str:
        if self._api is None:
            return "Franka Manipulation API already closed."
        try:
            self._api.close()
        finally:
            self._api = None
            self._env = None
        return "Franka Manipulation API closed."

    def _step_env(self, params: dict[str, Any]) -> str:
        if self._env is None:
            return "Error: API not started. Dispatch action_type='start' first."
        action = params.get("action", {})
        with self._env_lock:
            self._last_obs, _, _, _, _ = self._env.step(action=action)
        return "Environment stepped."

    def _api_call(self, params: dict[str, Any]) -> str:
        if self._api is None:
            return "Error: API not started. Dispatch action_type='start' first."
        method_name = str(params.get("method", "")).strip()
        if not method_name:
            return "Error: parameters.method is required for api_call."
        method = getattr(self._api, method_name, None)
        if method is None or not callable(method):
            return f"Error: method not found on API: {method_name}"
        args = params.get("args", [])
        kwargs = params.get("kwargs", {})
        if not isinstance(args, list):
            return "Error: parameters.args must be a list."
        if not isinstance(kwargs, dict):
            return "Error: parameters.kwargs must be a dict."
        result = method(*args, **kwargs)
        return f"api_call ok: {method_name} => {self._safe_json(result)}"

    def _grasp(self, params: dict[str, Any]) -> str:
        if self._api is None:
            return "Error: API not started. Dispatch action_type='start' first."

        target = params.get("target")
        if not target:
            return "Error: target is required for grasp action."

        target_dict = self._resolve_target(target, self._grasp_targets)

        out_dir = Path(params.get("output_dir", self._output_dir))
        out_dir.mkdir(parents=True, exist_ok=True)
        dump_path = out_dir / params.get("dump_name", self._grasp_dump_name)

        result = self._api.grasp(target_dict, dump_path=dump_path)
        self._last_action_summary = (
            f"grasp success={result.success} steps={result.steps}"
        )
        return self._last_action_summary

    def _place(self, params: dict[str, Any]) -> str:
        if self._api is None:
            return "Error: API not started. Dispatch action_type='start' first."

        target = params.get("target")
        if not target:
            return "Error: target is required for place action."

        target_dict = self._resolve_target(target, self._place_targets)

        out_dir = Path(params.get("output_dir", self._output_dir))
        out_dir.mkdir(parents=True, exist_ok=True)
        dump_path = out_dir / params.get("dump_name", self._place_dump_name)

        result = self._api.place(target_dict, dump_path=dump_path)
        self._last_action_summary = (
            f"place success={result.success} steps={result.steps}"
        )
        return self._last_action_summary

    def _resolve_target(
        self,
        target: str | dict[str, Any],
        registry: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        if isinstance(target, dict):
            return target
        if isinstance(target, str):
            if target in registry:
                return registry[target]
            raise KeyError(f"Unknown target: {target}. Available: {list(registry.keys())}")
        raise TypeError(f"Unsupported target type: {type(target)}")

    def _register_grasp_target(self, params: dict[str, Any]) -> str:
        name = str(params.get("name", "")).strip()
        if not name:
            return "Error: name is required for register_grasp_target."
        target = params.get("target", {})
        if not isinstance(target, dict):
            return "Error: target must be a dict."
        self._grasp_targets[name] = target
        if self._api is not None:
            self._api.register_grasp_target(name, target)
        return f"grasp_target registered: {name}"

    def _register_place_target(self, params: dict[str, Any]) -> str:
        name = str(params.get("name", "")).strip()
        if not name:
            return "Error: name is required for register_place_target."
        target = params.get("target", {})
        if not isinstance(target, dict):
            return "Error: target must be a dict."
        self._place_targets[name] = target
        if self._api is not None:
            self._api.register_place_target(name, target)
        return f"place_target registered: {name}"

    def _get_robot_state(self, params: dict[str, Any]) -> str:
        if self._api is None:
            return "Error: API not started. Dispatch action_type='start' first."
        try:
            obs = self._env.get_observations() if self._env else None
            if isinstance(obs, dict):
                eef_pos = obs.get("eef_position", [0.0, 0.0, 0.0])
                if hasattr(eef_pos, "__iter__"):
                    eef_pos = list(eef_pos)
                else:
                    eef_pos = [float(eef_pos)]
                state = {
                    "eef_position": eef_pos,
                    "last_action": self._last_action_summary,
                    "grasp_targets": list(self._grasp_targets.keys()),
                    "place_targets": list(self._place_targets.keys()),
                }
                return f"Robot State: {self._safe_json(state)}"
            return "Robot State: observation format unknown"
        except Exception as exc:
            return f"Error getting robot state: {exc}"

    def _idle_step_if_due(self) -> None:
        if self._api is None or self._env is None:
            return
        now = time.monotonic()
        if now - self._last_idle_step_ts < self._idle_step_interval_s:
            return
        self._last_idle_step_ts = now
        steps = max(1, self._idle_steps_per_cycle)
        for _ in range(steps):
            with self._env_lock:
                self._last_obs, _, _, _, _ = self._env.step(action={})

    @staticmethod
    def _safe_json(value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except TypeError:
            return str(value)
