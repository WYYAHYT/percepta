"""Co-visibility and stereo precision from the *measured* rig, not a modelled one.

The earlier analysis (``stereo_feasibility.py``) modelled the arm poses: it assumed a top-down
grasp and a wrist mount taken from a third party. Both turned out to be wrong in ways that
mattered, so this script reads the camera transforms out of a real simulator capture instead.

Source of truth: ``rigprobe/data/cam_rig_fixed.json``, written by ``rigprobe/read_camera_rig.py``
running against the actual RoboDojo build (see that file for how the poses are obtained). The
numbers below are what the simulator reported, after the bug that left the head camera at the
origin was fixed.

What changed against the modelled version, and why it matters:

* both wrist cameras look along +y and 30 degrees down -- they are *parallel*, and their
  baseline runs along x. That is textbook stereo geometry, and it is the opposite of the earlier
  assumption that each wrist looks at its own side of the table.
* the left-right baseline is 0.60 m, not the 0.325 m the modelled poses implied.

Caveat that still applies: this is ONE static pose (the default/reset pose). In a real episode
the arms move and the wrist cameras travel with them. These numbers describe the rig as spawned,
not its distribution during a task.

Run:  python scripts/rig_from_measurement.py [--json <path>]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_RIG = ROOT / "rigprobe" / "data" / "cam_rig_fixed.json"
IMAGE_SHAPE = (480, 640)
TABLE_Z = 0.765
#: Disparity noise of a neural matcher, pixels. An assumption, not a measurement.
SIGMA_D_PX = (0.5, 1.0)


@dataclass
class Cam:
    name: str
    K: np.ndarray
    T: np.ndarray  # world <- camera

    @property
    def pos(self) -> np.ndarray:
        return self.T[:3, 3]

    @property
    def axis(self) -> np.ndarray:
        """Unit vector along the viewing direction (camera looks down -z)."""
        return -self.T[:3, 2] / np.linalg.norm(self.T[:3, 2])

    @property
    def focal(self) -> float:
        return float(self.K[0, 0])

    def project(self, pts_w: np.ndarray):
        local = (pts_w - self.pos) @ self.T[:3, :3]
        depth = -local[:, 2]
        safe = np.where(np.abs(depth) > 1e-9, depth, 1e-9)
        col = self.focal * local[:, 0] / safe + self.K[0, 2]
        row = -self.focal * local[:, 1] / safe + self.K[1, 2]
        return row, col, depth

    def sees(self, pts_w: np.ndarray) -> np.ndarray:
        row, col, depth = self.project(pts_w)
        h, w = IMAGE_SHAPE
        return (depth > 1e-6) & (row >= 0) & (row <= h - 1) & (col >= 0) & (col <= w - 1)

    def pitch_down_deg(self) -> float:
        return math.degrees(math.asin(max(-1.0, min(1.0, -self.axis[2]))))


def load_rig(path: Path) -> dict[str, Cam]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cams: dict[str, Cam] = {}
    for name, entry in data.items():
        if not (isinstance(entry, dict) and "fx" in entry and "T_world_cam" in entry):
            continue
        K = np.asarray(
            [
                [entry["fx"], 0.0, entry["cx"]],
                [0.0, entry["fy"], entry["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        cams[name] = Cam(name=name, K=K, T=np.asarray(entry["T_world_cam"], dtype=np.float64))
    return cams


def sigma_z(z: float, focal: float, baseline: float, sigma_d: float) -> float:
    """``Z^2 / (f B) * sigma_d`` -- idealised depth uncertainty of a stereo pair."""
    denom = focal * baseline
    return float("inf") if denom <= 0 else z * z / denom * sigma_d


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=str(DEFAULT_RIG))
    args = parser.parse_args()

    rig_path = Path(args.json)
    if not rig_path.is_file():
        raise SystemExit(
            f"no measured rig at {rig_path}. Run rigprobe/read_camera_rig.py against RoboDojo "
            "first and copy cam_rig_*.json into rigprobe/data/."
        )
    cams = load_rig(rig_path)
    names = sorted(cams)

    print("=" * 78)
    print("Rig as MEASURED in the simulator (not modelled)")
    print("=" * 78)
    print(f"source: {rig_path}")
    print()
    for name in names:
        c = cams[name]
        print(f"  {name}")
        print(f"    focal      {c.focal:9.4f} px")
        print(f"    position   {np.round(c.pos, 4).tolist()}")
        print(f"    axis       {np.round(c.axis, 3).tolist()}"
              f"   ({c.pitch_down_deg():.1f} deg below horizontal)")
    print()

    # --- pairwise geometry ------------------------------------------------------------
    print("-" * 78)
    print("1. PAIRWISE GEOMETRY")
    print("-" * 78)
    print("   angle = between optical axes. Two cameras looking the same way with a")
    print("   sideways baseline is textbook stereo; two looking along the baseline is not.")
    print()
    header = f"{'pair':>16} {'baseline':>10} {'angle':>8} {'midpoint z':>12}"
    print(header)
    print("-" * len(header))
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ca, cb = cams[a], cams[b]
            base = float(np.linalg.norm(ca.pos - cb.pos))
            ang = math.degrees(
                math.acos(max(-1.0, min(1.0, float(ca.axis @ cb.axis))))
            )
            mid_z = 0.5 * (ca.pos[2] + cb.pos[2])
            pairs.append((a, b, base, ang))
            print(f"{a[4:8] + '-' + b[4:8]:>16} {base:>9.4f}m {ang:>7.1f}d {mid_z:>11.3f}m")
    print()

    # --- the workspace a mask can cover ------------------------------------------------
    # Sample the volume a graspable object occupies on and above the table.
    xs = np.linspace(-0.45, 0.45, 19)
    ys = np.linspace(-0.30, 0.35, 14)
    zs = np.array([TABLE_Z + h for h in (0.0, 0.04, 0.08, 0.12)])
    volume = np.array([[x, y, z] for x in xs for y in ys for z in zs])

    print("-" * 78)
    print("2. CO-VISIBILITY OVER THE TABLE WORKSPACE")
    print("-" * 78)
    print(f"   {len(volume)} sample points: x in [-0.45,0.45], y in [-0.30,0.35],")
    print("   z in [table, table+0.12]. Share of points BOTH cameras project into frame.")
    print()
    print(f"{'pair':>16} {'both see':>10} {'only A':>9} {'only B':>9} {'neither':>9}")
    print("-" * 52)
    for a, b, _base, _ang in pairs:
        sa, sb = cams[a].sees(volume), cams[b].sees(volume)
        both = int((sa & sb).sum())
        oa = int((sa & ~sb).sum())
        ob = int((~sa & sb).sum())
        neither = int((~sa & ~sb).sum())
        n = len(volume)
        print(
            f"{a[4:8] + '-' + b[4:8]:>16} {both / n * 100:>9.1f}% {oa / n * 100:>8.1f}% "
            f"{ob / n * 100:>8.1f}% {neither / n * 100:>8.1f}%"
        )
    print()

    all_three = np.ones(len(volume), dtype=bool)
    for name in names:
        all_three &= cams[name].sees(volume)
    print(f"   All three cameras: {all_three.mean() * 100:.1f}% of the workspace")
    print()

    # --- how well each camera covers the workspace alone -------------------------------
    print("-" * 78)
    print("3. PER-CAMERA COVERAGE AND WHAT IT MEANS FOR DEPTH")
    print("-" * 78)
    print(f"{'camera':>18} {'sees workspace':>16} {'works in stereo with':>24}")
    print("-" * 62)
    for name in names:
        cov = cams[name].sees(volume).mean() * 100
        partners = [b for a, b, _, _ in pairs if a == name]
        partners += [a for a, b, _, _ in pairs if b == name]
        print(f"{name:>18} {cov:>15.1f}% {', '.join(p[4:] for p in partners):>24}")
    print()

    # --- precision ----------------------------------------------------------------
    print("-" * 78)
    print("4. IDEALISED DEPTH PRECISION  sigma_Z = Z^2 / (f B) * sigma_d")
    print("-" * 78)
    print("   Mask-geometry baseline for comparison: 0.3-1.0 cm centroid, 0-2 cm height.")
    print("   A stereo pair only helps where it beats those numbers.")
    print()
    hdr = f"{'pair':>16} {'f(px)':>8} {'B(m)':>7}"
    for z in (0.4, 0.6, 0.8, 1.0):
        hdr += f" {f'Z={z}m':>13}"
    print(hdr)
    print("-" * len(hdr))
    for a, b, base, _ang in pairs:
        focal = 0.5 * (cams[a].focal + cams[b].focal)
        line = f"{a[4:8] + '-' + b[4:8]:>16} {focal:>8.1f} {base:>7.4f}"
        for z in (0.4, 0.6, 0.8, 1.0):
            lo = sigma_z(z, focal, base, SIGMA_D_PX[0]) * 1000
            hi = sigma_z(z, focal, base, SIGMA_D_PX[1]) * 1000
            line += f" {f'{lo:.1f}-{hi:.1f}mm':>13}"
        print(line)
    print()

    # --- disparity ---------------------------------------------------------------
    print("-" * 78)
    print("5. DISPARITY, AGAINST A MATCHER'S SEARCH RANGE")
    print("-" * 78)
    print("   d = f*B/Z. Fast-FoundationStereo defaults to max_disp=192 and calls it")
    print("   tunable; anything above that needs the knob raised (more memory, slower).")
    print()
    hdr2 = f"{'pair':>16} {'B(m)':>7}"
    for z in (0.4, 0.6, 0.8, 1.0):
        hdr2 += f" {f'd@Z={z}m':>11}"
    print(hdr2)
    print("-" * len(hdr2))
    for a, b, base, _ang in pairs:
        focal = 0.5 * (cams[a].focal + cams[b].focal)
        line = f"{a[4:8] + '-' + b[4:8]:>16} {base:>7.4f}"
        for z in (0.4, 0.6, 0.8, 1.0):
            line += f" {focal * base / z:>10.1f}px"
        print(line)
    print()

    print("=" * 78)
    print("WHAT THIS DOES AND DOES NOT SETTLE")
    print("=" * 78)
    print("""
Settled:
  * the camera intrinsics and the reset-pose extrinsics, read from the simulator;
  * whether each pair has a usable baseline and overlapping frustums *at the reset pose*;
  * the disparity each pair needs, hence whether the matcher's default suffices.

Not settled:
  * poses during an episode. The arms move; the wrist cameras ride along, so the
    left-right baseline sweeps through smaller values when the arms converge on one
    object. That convergence is the case that matters for grasping, and it is missing
    here.
  * occlusion. The head camera is blocked by the arms in many tasks; frustum
    intersection cannot see that.
  * the matcher's real error on RoboDojo textures. sigma_d is an assumption.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
