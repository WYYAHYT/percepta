"""What moving the arms buys: co-visibility across a swept set of arm configurations.

The reset pose is one sample of the rig and the worst one for the wrist pair. This reads a sweep
of joint configurations (``rigprobe/sweep_arm_poses.py``) and asks the question that decides
whether three-view stereo is worth building: **can the arms be posed so that the cameras which
see an object see it well?**

For every pose in the sweep it computes, over a table workspace:

* the co-visibility of all three camera pairs,
* the baseline and axis angle of each pair,
* the idealised depth precision at each visible point's own distance,
* a single "how useful is this pose" score: the workspace fraction where a chosen pair both sees
  the point *and* the resulting precision beats the mask-geometry baseline.

The last point is the one worth arguing about. A pair that sees 40% of the table but resolves
depth to 8 mm is not better than the silhouette method's 3-10 mm; a pair that resolves 0.5 mm but
applies to 9% of the table rarely helps. The score counts only points where stereo actually wins.

Run:  python scripts/sweep_analysis.py [--json <path>]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_SWEEP = ROOT / "rigprobe" / "data" / "arm_sweep_sweep1.json"
IMAGE_SHAPE = (480, 640)
TABLE_Z = 0.765
#: Disparity noise of a neural matcher. An assumption, not a measurement.
SIGMA_D_PX = 0.5
#: The mask-geometry baseline this work is trying to beat: 0.3-1.0 cm lateral.
MASK_BASELINE_M = 0.005

NAMES = ["cam_head", "cam_left_wrist", "cam_right_wrist"]
PAIRS = [
    ("cam_head", "cam_left_wrist"),
    ("cam_head", "cam_right_wrist"),
    ("cam_left_wrist", "cam_right_wrist"),
]


def _cam_from_record(rec: dict, name: str):
    T = np.asarray(rec[f"{name}_T"], dtype=np.float64)
    focal = 397.0413 if "wrist" in name else 288.1325
    K = np.array([[focal, 0.0, 320.0], [0.0, focal, 240.0], [0.0, 0.0, 1.0]])
    return K, T


def _sees(K: np.ndarray, T: np.ndarray, pts: np.ndarray, focal: float) -> np.ndarray:
    local = (pts - T[:3, 3]) @ T[:3, :3]
    depth = -local[:, 2]
    safe = np.where(np.abs(depth) > 1e-9, depth, 1e-9)
    col = focal * local[:, 0] / safe + K[0, 2]
    row = -focal * local[:, 1] / safe + K[1, 2]
    h, w = IMAGE_SHAPE
    return (depth > 0.05) & (row >= 0) & (row <= h - 1) & (col >= 0) & (col <= w - 1)


def _pair_key(a: str, b: str) -> str:
    return f"{a[4:]}|{b[4:]}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=str(DEFAULT_SWEEP))
    args = parser.parse_args()

    path = Path(args.json)
    if not path.is_file():
        raise SystemExit(f"no sweep data at {path}")

    records = json.loads(path.read_text(encoding="utf-8"))
    if not records:
        raise SystemExit("sweep file is empty")

    xs = np.linspace(-0.45, 0.45, 19)
    ys = np.linspace(-0.30, 0.35, 14)
    zs = np.array([TABLE_Z + h for h in (0.0, 0.05, 0.10)])
    vol = np.array([[x, y, z] for x in xs for y in ys for z in zs])

    rows = []
    for rec in records:
        cell: dict = {"index": rec["index"], "offsets": rec["offsets"]}
        cams = {n: _cam_from_record(rec, n) for n in NAMES}
        seen = {n: _sees(cams[n][0], cams[n][1], vol, float(cams[n][0][0, 0])) for n in NAMES}
        cell["all_three"] = float(
            (seen[NAMES[0]] & seen[NAMES[1]] & seen[NAMES[2]]).mean()
        )
        for a, b in PAIRS:
            Ka, Ta = cams[a]
            Kb, Tb = cams[b]
            base = float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]))
            ax_a = -Ta[:3, 2] / np.linalg.norm(Ta[:3, 2])
            ax_b = -Tb[:3, 2] / np.linalg.norm(Tb[:3, 2])
            ang = math.degrees(math.acos(max(-1.0, min(1.0, float(ax_a @ ax_b)))))
            both = seen[a] & seen[b]
            focal = 0.5 * (float(Ka[0, 0]) + float(Kb[0, 0]))
            mid = 0.5 * (Ta[:3, 3] + Tb[:3, 3])
            dist = np.linalg.norm(vol - mid, axis=1)
            denom = focal * base
            sigma = (dist * dist) / denom * SIGMA_D_PX if denom > 0 else np.full(
                len(vol), np.inf
            )
            beats = both & (sigma < MASK_BASELINE_M)
            cell[_pair_key(a, b)] = {
                "baseline_m": base,
                "axis_angle_deg": ang,
                "covisibility": float(both.mean()),
                "beats_mask_geometry": float(beats.mean()),
            }
        best = max(PAIRS, key=lambda p: cell[_pair_key(*p)]["beats_mask_geometry"])
        cell["best_pair"] = f"{best[0][4:]}-{best[1][4:]}"
        cell["best_score"] = cell[_pair_key(*best)]["beats_mask_geometry"]
        rows.append(cell)

    ordered = sorted(rows, key=lambda r: -r["best_score"])

    print("=" * 78)
    print(f"Arm sweep: {len(rows)} poses from {path.name}")
    print("=" * 78)
    print()
    print("Score = share of a table workspace where the chosen pair BOTH sees the point")
    print(f"AND predicts depth better than {MASK_BASELINE_M * 1000:.1f} mm. A pair that sees much")
    print("but resolves little scores low; so does a precise pair that rarely applies.")
    print()
    hdr = f"{'rank':>5} {'pose':>6} {'best pair':>14} {'score':>7} {'LR base':>8} {'LR ang':>7}"
    print(hdr)
    print("-" * len(hdr))
    for rank, r in enumerate(ordered[:12], 1):
        lr = r[_pair_key("cam_left_wrist", "cam_right_wrist")]
        print(
            f"{rank:>5} {r['index']:>6} {r['best_pair']:>14} "
            f"{r['best_score'] * 100:>6.1f}% {lr['baseline_m']:>7.3f}m "
            f"{lr['axis_angle_deg']:>6.1f}d"
        )
    print()

    scores = np.array([r["best_score"] for r in rows])
    print(f"  best pose    {scores.max() * 100:5.1f}% of workspace usable")
    print(f"  median pose  {np.median(scores) * 100:5.1f}%")
    print(f"  worst pose   {scores.min() * 100:5.1f}%")
    print()

    print("-" * 78)
    print("WHICH PAIR WINS, AND HOW OFTEN")
    print("-" * 78)
    counts = Counter(r["best_pair"] for r in rows)
    for pair, k in counts.most_common():
        sub = [r for r in rows if r["best_pair"] == pair]
        med = float(np.median([r["best_score"] for r in sub]))
        print(f"  {pair:>16}: best in {k:>3}/{len(rows)} poses, median score {med * 100:5.1f}%")
    print()

    print("-" * 78)
    print("LR BASELINE ACROSS THE SWEEP (what a single pose could not show)")
    print("-" * 78)
    lr = [_pair_key("cam_left_wrist", "cam_right_wrist")]
    lr_base = np.array([r[lr[0]]["baseline_m"] for r in rows])
    lr_cov = np.array([r[lr[0]]["covisibility"] for r in rows])
    lr_ang = np.array([r[lr[0]]["axis_angle_deg"] for r in rows])
    print(
        f"  baseline   : min {lr_base.min():.3f}  median {np.median(lr_base):.3f}  "
        f"max {lr_base.max():.3f} m"
    )
    print(
        f"  axis angle : min {lr_ang.min():.1f}  median {np.median(lr_ang):.1f}  "
        f"max {lr_ang.max():.1f} deg"
    )
    print(
        f"  coverage   : min {lr_cov.min() * 100:.1f}%  median {np.median(lr_cov) * 100:.1f}%  "
        f"max {lr_cov.max() * 100:.1f}%"
    )
    print()
    best_lr = max(rows, key=lambda r: r[lr[0]]["covisibility"])
    print(
        f"  Most co-visible LR pose: index {best_lr['index']}, "
        f"coverage {best_lr[lr[0]]['covisibility'] * 100:.1f}%, "
        f"baseline {best_lr[lr[0]]['baseline_m']:.3f} m, "
        f"angle {best_lr[lr[0]]['axis_angle_deg']:.1f} deg"
    )
    print()

    print("=" * 78)
    print("LIMITS")
    print("=" * 78)
    print("""
* Free-space joint sweeps, not task-driven motion. A real episode reaches poses chosen
  to grasp, not to look; some of the best poses here may never be visited, and the ones
  visited while grasping may be worse.
* No occlusion. The head camera is blocked by the arms in many tasks, which a frustum
  test cannot see - real head-pair co-visibility is lower than reported.
* sigma_d = 0.5 px is an assumption about the matcher; at 1.0 px every score roughly halves.
* The workspace box is a guess at where objects sit; a different box moves the numbers.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
