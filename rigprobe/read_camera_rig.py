"""Read the real camera rig out of RoboDojo's simulator. Read-only, no episode runs.

Why this exists
---------------
Everything I know about RoboDojo's camera geometry so far came from a third party
(PyRUA-Lean hard-coded constants "read off the simulator once") and from config files. Config
can disagree with what the simulator actually builds, and I already found one apparent
conflict: ``Assets/Robots/x5/robot_config.yml`` gives the wrist camera ``ori: [0, -80, -90]``,
while that third-party code models the wrist mount as a 30-degree downward tilt. Those cannot
both be right, and every baseline/co-visibility number I computed depends on the wrist pose.

So this script asks the simulator, not the YAML:

* the camera names the manager actually created,
* their intrinsics, as the manager computes them,
* their world poses, including the wrist cameras at a known joint configuration,
* the robot's end-effector pose in the same frame, so the two can be cross-checked.

What it does NOT do
-------------------
No episode loop, no policy, no randomness, no scoring, no writes into the repo. It builds the
env, reads, and exits. Nothing here touches ground-truth depth or object poses; camera extrinsics
are simulator state, not privileged evaluation data, and they are read offline.

Output
------
``cam_rig_<tag>.npz`` plus a JSON summary. Kept outside the RoboDojo tree.

Usage
-----
    python read_camera_rig.py --task stack_blocks --out /path/to/out --tag baseline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    """RoboDojo checkout this script is run against (set by --repo or ROBODOJO_DIR)."""
    root = os.environ.get("ROBODOJO_DIR")
    if not root:
        raise SystemExit("set ROBODOJO_DIR to the RoboDojo checkout, or pass --repo")
    return Path(root).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=None, help="RoboDojo checkout (else $ROBODOJO_DIR)")
    parser.add_argument("--task", default="stack_blocks", help="task whose env config to load")
    parser.add_argument("--env-cfg", default="arx_x5", help="env_cfg type, e.g. arx_x5")
    parser.add_argument("--device-id", type=int, default=None, help="CUDA device index")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--tag", default="baseline", help="label for the output files")
    parser.add_argument("--warmup-steps", type=int, default=8, help="sim steps before reading")
    args = parser.parse_args()

    repo = Path(args.repo).resolve() if args.repo else _repo_root()
    os.environ.setdefault("ROBODOJO_DIR", str(repo))
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "XPolicyLab"))

    if args.device_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

    # --- Kit / AppLauncher must be configured before anything imports omni ------------------
    # AppLauncher wants an argparse.Namespace (it pops keys off a dict copy), so build one with
    # exactly the flags Isaac Lab registers; anything missing raises inside its config resolution.
    import argparse as _argparse

    from isaaclab.app import AppLauncher

    app_parser = _argparse.ArgumentParser(add_help=False)
    # Registered by Isaac Lab's AppLauncher itself - let it add its own arguments.
    from isaaclab.app import AppLauncher as _AppLauncher

    _AppLauncher.add_app_launcher_args(app_parser)
    app_args = app_parser.parse_args(
        [
            "--headless",
            "--enable_cameras",
            "--device",
            "cuda:0",
        ]
    )
    # Belt and braces: some Isaac Lab revisions read these without defaults.
    for _name, _value in (
        ("livestream", 0),
        ("kit_args", ""),
        ("experience", ""),
        ("rendering_mode", None),
        ("enable_cameras", True),
        ("headless", True),
        ("device", "cuda:0"),
    ):
        if not hasattr(app_args, _name):
            setattr(app_args, _name, _value)

    # Same camera/physics sync the eval client forces; without it frames lag physics and the
    # extrinsics we read would not correspond to the pixels.
    from env.camera_manager.capture.render_sync import add_zero_delay_kit_args

    add_zero_delay_kit_args(app_args)

    print("[rig] launching Isaac Sim (headless)...", flush=True)
    launcher = AppLauncher(app_args)
    sim_app = launcher.app
    print("[rig] app up", flush=True)

    import importlib

    from env.global_configs import BENCHMARK, ENV_CONFIG_PATH
    from omegaconf import OmegaConf
    from src.eval_client.eval_env import create_eval_env
    from utils.load_file import load_yaml
    from utils.pipeline_utils import process_config, process_randomization

    task_registry = importlib.import_module(f"task.{BENCHMARK}.task_registry")
    _task_name, _task_class = task_registry.load_task_class(args.task)
    task_yml = task_registry.task_config_path(
        os.path.join(repo, "task", BENCHMARK, "config"), args.task
    )

    eval_cfg = load_yaml(os.path.join(ENV_CONFIG_PATH, args.env_cfg + ".yml"))
    eval_cfg["task_name"] = args.task
    eval_cfg["num_envs"] = 1
    eval_cfg["device_id"] = args.device_id if args.device_id is not None else 0
    eval_cfg["eval_batch"] = False
    eval_cfg["policy_name"] = "_rigprobe"
    eval_cfg["additional_info"] = args.tag
    eval_cfg["seed"] = 0
    eval_cfg["physx_monitor_enabled"] = False
    eval_cfg["eval_num"] = 1

    env_cfg = OmegaConf.create(
        {
            "sim": load_yaml(
                os.path.join(ENV_CONFIG_PATH, "sim", eval_cfg["config"]["sim"] + ".yml")
            ),
            "scene": load_yaml(
                os.path.join(ENV_CONFIG_PATH, "scene", eval_cfg["config"]["scene"] + ".yml")
            ),
            "camera": load_yaml(
                os.path.join(ENV_CONFIG_PATH, "camera", eval_cfg["config"]["camera"] + ".yml")
            ),
            "robot": load_yaml(
                os.path.join(ENV_CONFIG_PATH, "robot", eval_cfg["config"]["robot"] + ".yml")
            ),
            "task_env": load_yaml(task_yml),
            "eval_cfg": eval_cfg,
            "deploy_cfg": {"policy_name": "_rigprobe", "port": 0, "host": "127.0.0.1"},
        }
    )
    OmegaConf.update(env_cfg, "sim.scene.num_envs", 1, force_add=True)
    OmegaConf.update(env_cfg, "eval_cfg.num_envs", 1, force_add=True)
    env_cfg = process_randomization(env_cfg)
    env_cfg, _eval_num = process_config(env_cfg, task_name=args.task)
    env_cfg.sim.seed = [0]

    # CameraManager reads ``camera.default_frequency`` and uses it unconditionally at
    # camera-creation time (camera_manager.py:257), but only assigns it if the key is present
    # (line 183). main.py writes it here, which is why the eval path never trips over it; a
    # caller that builds the env directly does. Set it to the same value main.py uses.
    OmegaConf.update(
        env_cfg,
        "camera.default_frequency",
        env_cfg.eval_cfg.get("observation", {}).get("collect_freq", 0),
        force_add=True,
    )

    os.environ.setdefault("ROBODOJO_RUN_ID", "rigprobe-" + args.tag)

    # create_eval_env builds a WsModelClient in __init__ and then waits for the policy server to
    # answer. This probe never runs a policy: it only needs the scene, the robots and the cameras.
    # Replace the client with a stub that satisfies attribute access and refuses to be called, so
    # a silent policy round-trip can never sneak into a geometry measurement.
    import src.eval_client.eval_env as _eval_env_mod

    class _NoPolicy:
        """Stand-in for WsModelClient: answers lifecycle calls, refuses anything else.

        ``env.reset()`` calls ``model_client.call("reset")`` to tell the policy server a new
        episode started. Raising there aborts the reset — and the reset is exactly what writes
        the camera poses, so the probe has to tolerate it. Answer the calls the environment
        makes and raise on everything else, so a real policy round-trip still cannot slip in
        unnoticed.
        """

        #: Lifecycle verbs the environment sends that carry no geometry meaning.
        _ALLOWED = {"reset": None, "close": None, "handshake": {}, "ping": "pong"}

        def __init__(self, *args, **kwargs) -> None:
            self.url = kwargs.get("url", "")
            self.calls: list[str] = []

        def call(self, func_name=None, *args, **kwargs):
            self.calls.append(str(func_name))
            if func_name in self._ALLOWED:
                return self._ALLOWED[func_name]
            raise RuntimeError(
                f"rig probe must not call the policy server (attempted .call({func_name!r}))"
            )

        def __getattr__(self, item):
            if item in {"close", "disconnect", "shutdown"}:
                return lambda *a, **k: None

            def _refuse(*a, **k):
                raise RuntimeError(
                    f"rig probe must not use the policy client (attempted .{item}())"
                )

            return _refuse

        def close(self) -> None:
            pass

    _eval_env_mod.WsModelClient = _NoPolicy
    print("[rig] building env (policy client stubbed out)...", flush=True)
    env = create_eval_env(env_cfg, sim_app)

    cm = env.camera_manager

    # Two separate things have to happen here, and confusing them cost a debugging round:
    #
    #   * ``launch_sim`` builds the world and sets ``env.sim``.
    #   * ``reset`` is what runs ``CameraManager.reset()`` -> ``init_camera_pose()``, the only
    #     place the configured ``pos``/``ori`` are written onto each camera prim.
    #
    # The first version called reset only when ``env.sim`` was still None, so after a successful
    # launch_sim it never ran — and the head camera kept the identity transform USD gives a fresh
    # prim, reporting [0,0,0] instead of its configured [0,-0.41,1.308]. The wrist cameras looked
    # correct anyway because they inherit their pose from the robot link they are mounted on,
    # which masked the bug. Always reset.
    if getattr(env, "sim", None) is None:
        print("[rig] launching sim...", flush=True)
        env.launch_sim(env.sim_config)
    if getattr(env, "sim", None) is None:
        raise SystemExit("sim not available after launch_sim")

    print("[rig] resetting env (writes camera poses)...", flush=True)
    # EvalEnv.reset does ``seed = list(seed)``, so it needs a sequence, not a bare int.
    env.reset(seed=[0] * int(env.num_envs))

    for _ in range(max(1, args.warmup_steps)):
        env.sim_step(render=True)

    names = list(cm.camera_names[0])
    print(f"[rig] cameras: {names}", flush=True)

    payload = {}
    arrays = {}
    for cam_id, name in enumerate(names):
        K = np.asarray(cm.get_camera_intrinsics(cam_id, 0), dtype=np.float64)
        T = np.asarray(cm.get_camera_extrinsics(cam_id, 0), dtype=np.float64)
        arrays[f"K_{name}"] = K
        arrays[f"T_{name}"] = T
        payload[name] = {
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "position": [float(v) for v in T[:3, 3]],
            "optical_axis": [float(v) for v in -T[:3, 2]],
            "T_world_cam": T.tolist(),
        }

        # The pose above comes from get_world_pose(). Read the prim path and the configured
        # pose too: a mismatch is the difference between "the camera really is at the origin"
        # and "the local pose was never applied to the prim we asked about".
        prim_path = None
        try:
            prim_path = str(cm.cameras_xform_path[0][cam_id])
        except Exception:
            pass
        cfg_pos, cfg_ori = None, None
        try:
            cam_cfg = cm.camera_config[name]
            cam_node = cam_cfg["camera"] if "camera" in cam_cfg else None
            if cam_node is not None:
                if "pos" in cam_node:
                    cfg_pos = [float(v) for v in cam_node["pos"]]
                if "ori" in cam_node:
                    cfg_ori = [float(v) for v in cam_node["ori"]]
        except Exception as exc:
            payload[name]["config_probe_error"] = repr(exc)
        local_pose = None
        try:
            lp = cm.cameras_xform[0][cam_id].get_local_pose()
            local_pose = [float(v) for v in np.asarray(lp[0], dtype=np.float64).ravel()]
        except Exception:
            pass
        payload[name].update(
            {
                "prim_path": prim_path,
                "config_pos": cfg_pos,
                "config_ori": cfg_ori,
                "local_position": local_pose,
            }
        )
        print(
            f"[rig] {name:16s} fx={K[0, 0]:8.3f} pos={np.round(T[:3, 3], 4)}",
            flush=True,
        )

    # Robot state in the same frame, for cross-checking the wrist mount.
    try:
        robot_manager = env.robot_manager
        for idx, robot in enumerate(getattr(robot_manager, "robot_list", [])):
            key = getattr(robot, "robot_key", None)
            payload[f"robot_{idx}_name"] = str(getattr(robot, "robot_name", "?"))
            payload[f"robot_{idx}_key"] = str(key)
            pose = None
            try:
                pose = robot_manager.get_link_pose(robot, robot.base_link)
            except Exception:
                pass
            if pose is not None:
                arr = np.asarray(pose, dtype=np.float64).ravel()
                arrays[f"robot_{idx}_base"] = arr
                payload[f"robot_{idx}_base_link_pose"] = [float(v) for v in arr]
    except Exception as exc:  # pragma: no cover - diagnostics only
        payload["robot_state_error"] = repr(exc)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez(outdir / f"cam_rig_{args.tag}.npz", **arrays)
    (outdir / f"cam_rig_{args.tag}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"[rig] wrote {outdir}/cam_rig_{args.tag}.{{npz,json}}", flush=True)

    try:
        env.close()
    except Exception:
        pass
    sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
