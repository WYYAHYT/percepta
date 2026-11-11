"""Sweep the arms through joint configurations and record where the wrist cameras go.

The reset pose answers one question: what happens when both arms hang at their default. It is
the *worst* case for the wrist pair, because the two cameras sit 0.60 m apart pointing the same
way and their frustums only intersect in a narrow strip near x=0.

The arms can move, and that is the whole point of mounting a camera on each one. Bring both
hands over the same object and the wrist cameras converge on it -- the geometry that matters for
grasping, and the one a static pose cannot show.

This script drives each arm to a set of joint configurations, steps the simulator until the arm
settles, and records every camera's pose at each stop. It is a geometry measurement: no policy,
no episode, no scoring, no object interaction. The arms move through free space above the table.

Sampling strategy
-----------------
A 6-DoF arm has an enormous configuration space, so this does not try to cover it. It sweeps the
joints that actually change where a wrist camera points and how far apart the two are:

* shoulder pan (joint 0) -- swings the arm left/right, moving the camera's azimuth;
* shoulder lift (joint 1) -- raises/lowers the whole arm;
* elbow (joint 2) -- changes reach, hence the left-right baseline.

The wrist's own joints (3-5) are left at their reset values: they rotate the camera about its
own axis and barely move its position, so they add samples without adding information.

Read-only with respect to the repo; writes NPZ/JSON outside it.

Usage
-----
    python sweep_arm_poses.py --repo <RoboDojo> --out <dir> [--max-configs 120]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=os.environ.get("ROBODOJO_DIR"))
    parser.add_argument("--task", default="stack_blocks")
    parser.add_argument("--env-cfg", default="arx_x5")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tag", default="sweep")
    parser.add_argument("--settle-steps", type=int, default=24)
    parser.add_argument("--max-configs", type=int, default=120)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    os.environ.setdefault("ROBODOJO_DIR", str(repo))
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "XPolicyLab"))

    import argparse as _ap

    from isaaclab.app import AppLauncher as _AL

    ap = _ap.ArgumentParser(add_help=False)
    _AL.add_app_launcher_args(ap)
    app_args = ap.parse_args(["--headless", "--enable_cameras", "--device", "cuda:0"])
    for k, v in (
        ("livestream", 0),
        ("kit_args", ""),
        ("experience", ""),
        ("rendering_mode", None),
        ("enable_cameras", True),
        ("headless", True),
        ("device", "cuda:0"),
    ):
        if not hasattr(app_args, k):
            setattr(app_args, k, v)

    from env.camera_manager.capture.render_sync import add_zero_delay_kit_args

    add_zero_delay_kit_args(app_args)

    print("[sweep] launching Isaac Sim...", flush=True)
    sim_app = _AL(app_args).app
    print("[sweep] app up", flush=True)

    import importlib

    from env.global_configs import BENCHMARK, ENV_CONFIG_PATH
    from omegaconf import OmegaConf
    from src.eval_client.eval_env import create_eval_env
    from utils.load_file import load_yaml
    from utils.pipeline_utils import process_config, process_randomization

    task_registry = importlib.import_module(f"task.{BENCHMARK}.task_registry")
    task_registry.load_task_class(args.task)
    task_yml = task_registry.task_config_path(
        os.path.join(repo, "task", BENCHMARK, "config"), args.task
    )

    eval_cfg = load_yaml(os.path.join(ENV_CONFIG_PATH, args.env_cfg + ".yml"))
    eval_cfg.update(
        {
            "task_name": args.task,
            "num_envs": 1,
            "device_id": 0,
            "eval_batch": False,
            "policy_name": "_sweep",
            "additional_info": args.tag,
            "seed": 0,
            "physx_monitor_enabled": False,
            "eval_num": 1,
        }
    )
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
            "deploy_cfg": {"policy_name": "_sweep", "port": 0, "host": "127.0.0.1"},
        }
    )
    OmegaConf.update(env_cfg, "sim.scene.num_envs", 1, force_add=True)
    OmegaConf.update(env_cfg, "eval_cfg.num_envs", 1, force_add=True)
    env_cfg = process_randomization(env_cfg)
    env_cfg, _ = process_config(env_cfg, task_name=args.task)
    env_cfg.sim.seed = [0]
    OmegaConf.update(
        env_cfg,
        "camera.default_frequency",
        env_cfg.eval_cfg.get("observation", {}).get("collect_freq", 0),
        force_add=True,
    )

    import src.eval_client.eval_env as _mod

    class _NoPolicy:
        """Answers the environment's lifecycle calls; refuses everything else."""

        _ALLOWED = {"reset": None, "close": None, "handshake": {}, "ping": "pong"}

        def __init__(self, *a, **k) -> None:
            pass

        def call(self, func_name=None, *a, **k):
            if func_name in self._ALLOWED:
                return self._ALLOWED[func_name]
            raise RuntimeError(f"sweep must not talk to a policy (.call({func_name!r}))")

        def __getattr__(self, item):
            if item in {"close", "disconnect", "shutdown"}:
                return lambda *a, **k: None

            def _refuse(*a, **k):
                raise RuntimeError(f"sweep must not use the policy client (.{item})")

            return _refuse

        def close(self) -> None:
            pass

    _mod.WsModelClient = _NoPolicy
    os.environ.setdefault("ROBODOJO_RUN_ID", "armsweep")

    print("[sweep] building env...", flush=True)
    env = create_eval_env(env_cfg, sim_app)
    if getattr(env, "sim", None) is None:
        env.launch_sim(env.sim_config)
    if getattr(env, "sim", None) is None:
        raise SystemExit("sim unavailable")
    print("[sweep] resetting (writes camera poses)...", flush=True)
    env.reset(seed=[0] * int(env.num_envs))

    import torch

    rm = env.robot_manager
    cm = env.camera_manager

    arms = [r for r in rm.robot_list if getattr(r, "robot_type", None) == "arm"]
    print(f"[sweep] arms: {[a.arm_name for a in arms]}", flush=True)
    for a in arms:
        print(
            f"[sweep]   {a.arm_name}: joints={len(a.arm_joint_indices)} "
            f"names={getattr(a, 'arm_joints_name', None)}",
            flush=True,
        )

    base_q = {}
    keys = {}
    for idx, a in enumerate(arms):
        robot_key = rm.robot_key[rm.robot_list.index(a)]
        keys[idx] = robot_key
        q = robot_key.data.joint_pos[0, a.arm_joint_indices].clone()
        base_q[idx] = q
        print(f"[sweep]   {a.arm_name} base q = {np.round(q.cpu().numpy(), 3).tolist()}",
              flush=True)

    cam_names = list(cm.camera_names[0])
    print(f"[sweep] cameras: {cam_names}", flush=True)

    def read_rig() -> dict:
        out = {}
        for cam_id, name in enumerate(cam_names):
            K = np.asarray(cm.get_camera_intrinsics(cam_id, 0), dtype=np.float64)
            T = np.asarray(cm.get_camera_extrinsics(cam_id, 0), dtype=np.float64)
            out[name] = {"K": K, "T": T}
        return out

    def apply_q(offsets: dict[int, np.ndarray]) -> None:
        for idx, a in enumerate(arms):
            target = base_q[idx].clone()
            off = offsets.get(idx)
            if off is not None:
                target = target + torch.as_tensor(off, dtype=target.dtype, device=target.device)
            keys[idx].set_joint_position_target(
                target.unsqueeze(0), joint_ids=a.arm_joint_indices
            )
        for _ in range(max(1, args.settle_steps)):
            env.sim_step(render=False)

    pans = [-0.6, -0.3, 0.0, 0.3, 0.6]
    lifts = [-0.3, 0.0, 0.3]
    elbows = [-0.3, 0.0, 0.3]

    configs: list[dict[int, np.ndarray]] = []
    # Both arms move: a positive pan on the left mirrors a negative pan on the right, so the
    # hands converge and diverge together. That is the motion that changes the LR baseline.
    for pan, lift, elbow in itertools.product(pans, lifts, elbows):
        if len(configs) >= args.max_configs:
            break
        left = np.array([pan, lift, elbow, 0.0, 0.0, 0.0])
        right = np.array([-pan, lift, elbow, 0.0, 0.0, 0.0])
        configs.append({0: left, 1: right})
    # One arm alone, to separate "wrist pair moves together" from "one camera moves".
    for pan, lift in itertools.product([-0.6, 0.0, 0.6], [-0.3, 0.0, 0.3]):
        if len(configs) >= args.max_configs:
            break
        off = np.array([pan, lift, 0.0, 0.0, 0.0, 0.0])
        configs.append({0: off.copy(), 1: np.zeros(6)})
        configs.append({0: np.zeros(6), 1: off.copy()})

    configs = configs[: args.max_configs]
    print(f"[sweep] {len(configs)} configurations", flush=True)

    records = []
    for i, offsets in enumerate(configs):
        apply_q(offsets)
        rig = read_rig()
        rec = {
            "index": i,
            "offsets": {str(k): np.asarray(v).tolist() for k, v in offsets.items()},
        }
        for name, entry in rig.items():
            rec[f"{name}_pos"] = entry["T"][:3, 3].tolist()
            rec[f"{name}_axis"] = (-entry["T"][:3, 2]).tolist()
            rec[f"{name}_T"] = entry["T"].tolist()
        records.append(rec)
        if i % 10 == 0:
            lp = rig.get("cam_left_wrist")
            rp = rig.get("cam_right_wrist")
            line = f"[sweep] {i + 1}/{len(configs)}"
            if lp is not None and rp is not None:
                d = float(np.linalg.norm(lp["T"][:3, 3] - rp["T"][:3, 3]))
                ang = float(
                    np.degrees(
                        np.arccos(
                            np.clip(
                                float(
                                    (-lp["T"][:3, 2]) @ (-rp["T"][:3, 2])
                                ),
                                -1.0,
                                1.0,
                            )
                        )
                    )
                )
                line += f"  LR baseline {d:.3f} m  axis angle {ang:.1f} deg"
            print(line, flush=True)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"arm_sweep_{args.tag}.json").write_text(
        json.dumps(records, indent=1), encoding="utf-8"
    )
    print(f"[sweep] wrote {outdir}/arm_sweep_{args.tag}.json ({len(records)} poses)", flush=True)

    try:
        env.close()
    except Exception:
        pass
    sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
