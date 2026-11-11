"""Why does cam_head read as position [0,0,0] when its config says [0,-0.41,1.308]?

The rig probe found the head camera's xform prim at /World/envs/env_0/Camera_0 with a *local*
pose of zero, while its config carries a real position. The wrist cameras read sensible world
poses, but they hang off robot links that already carry position, so their own near-zero local
pose is expected and proves nothing about whether set_local_pose works.

This script settles it by reading the USD stage directly instead of trusting any wrapper:

  1. every xformOp on the camera xform prim, and on its ancestors
  2. the world transform computed by USD itself (UsdGeom.Xformable.ComputeLocalToWorldTransform)
  3. the same for the actual Camera *sensor* prim inside it, when there is one
  4. whether the two agree

If the wrapper's get_world_pose() disagrees with USD's own computation, the bug is in the
wrapper. If USD also reports zero, the pose was never written. Those lead to opposite fixes, so
guessing is not an option.

Read-only. Writes one JSON report.

Usage
-----
    python probe_cam_prim.py --repo <RoboDojo> --out <dir>
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--tag", default="prim")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    os.environ.setdefault("ROBODOJO_DIR", str(repo))
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "XPolicyLab"))

    import argparse as _ap

    from isaaclab.app import AppLauncher as _AL

    p = _ap.ArgumentParser(add_help=False)
    _AL.add_app_launcher_args(p)
    app_args = p.parse_args(["--headless", "--enable_cameras", "--device", "cuda:0"])
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

    launcher = _AL(app_args)
    sim_app = launcher.app
    print("[prim] app up", flush=True)

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
            "policy_name": "_primprobe",
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
            "deploy_cfg": {"policy_name": "_primprobe", "port": 0, "host": "127.0.0.1"},
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
        def __init__(self, *a, **k) -> None:
            pass

        def __getattr__(self, item):
            def _refuse(*a, **k):
                raise RuntimeError(f"no policy in a prim probe (.{item})")

            return _refuse

        def close(self) -> None:
            pass

    _mod.WsModelClient = _NoPolicy
    os.environ.setdefault("ROBODOJO_RUN_ID", "primprobe")

    env = create_eval_env(env_cfg, sim_app)
    if getattr(env, "sim", None) is None:
        env.launch_sim(env.sim_config)
    if getattr(env, "sim", None) is None:
        env.reset(seed=0)
    for _ in range(4):
        env.sim_step(render=True)

    # ---- read USD directly -----------------------------------------------------------
    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()

    def describe(path: str) -> dict:
        prim = stage.GetPrimAtPath(path)
        out: dict = {"path": path, "exists": bool(prim and prim.IsValid())}
        if not out["exists"]:
            return out
        out["type"] = str(prim.GetTypeName())
        out["children"] = [str(c.GetName()) for c in prim.GetChildren()]

        xf = UsdGeom.Xformable(prim)
        ops = []
        for op in xf.GetOrderedXformOps() if xf else []:
            try:
                val = op.Get()
                ops.append(
                    {
                        "name": str(op.GetOpName()),
                        "type": str(op.GetOpType()),
                        "value": np.asarray(val, dtype=float).ravel().tolist(),
                    }
                )
            except Exception as exc:
                ops.append({"name": str(op.GetOpName()), "error": repr(exc)})
        out["xform_ops"] = ops

        try:
            m = np.asarray(xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=float)
            out["usd_local_to_world"] = m.tolist()
            out["usd_world_translation"] = m[3, :3].tolist()
        except Exception as exc:
            out["usd_local_to_world_error"] = repr(exc)
        return out

    # ancestor chain of the head camera, to see which layer carries the translation
    chain = []
    for path in (
        "/World",
        "/World/envs",
        "/World/envs/env_0",
        "/World/envs/env_0/Camera_0",
    ):
        chain.append(describe(path))

    # the actual camera sensor(s) under the xform, if the mesh is a USD asset
    sensors = []
    head_prim = stage.GetPrimAtPath("/World/envs/env_0/Camera_0")
    if head_prim and head_prim.IsValid():
        for prim in Usd.PrimRange(head_prim):
            if prim.GetTypeName() == "Camera":
                sensors.append(describe(str(prim.GetPath())))

    # what the wrapper says, for the same prim
    cm = env.camera_manager
    wrapper = {}
    for cam_id, name in enumerate(list(cm.camera_names[0])):
        cell = {"prim_path": str(cm.cameras_xform_path[0][cam_id])}
        try:
            wp = cm.cameras_xform[0][cam_id].get_world_pose()
            cell["wrapper_world_pos"] = [
                float(v) for v in np.asarray(wp[0], dtype=float).ravel()
            ]
        except Exception as exc:
            cell["wrapper_world_pos_error"] = repr(exc)
        try:
            lp = cm.cameras_xform[0][cam_id].get_local_pose()
            cell["wrapper_local_pos"] = [
                float(v) for v in np.asarray(lp[0], dtype=float).ravel()
            ]
        except Exception as exc:
            cell["wrapper_local_pos_error"] = repr(exc)
        # USD's own answer for the same prim
        try:
            prim = stage.GetPrimAtPath(cell["prim_path"])
            xf = UsdGeom.Xformable(prim)
            m = np.asarray(xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=float)
            cell["usd_world_pos"] = m[3, :3].tolist()
        except Exception as exc:
            cell["usd_world_pos_error"] = repr(exc)
        wrapper[name] = cell

    report = {
        "ancestor_chain": chain,
        "head_sensor_prims": sensors,
        "per_camera": wrapper,
        "stage_url": str(stage.GetRootLayer().identifier),
    }

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"cam_prim_{args.tag}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"[prim] wrote {outdir}/cam_prim_{args.tag}.json", flush=True)

    try:
        env.close()
    except Exception:
        pass
    sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
