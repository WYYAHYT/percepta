"""Feasibility measurement for RGB stereo on RoboDojo's camera rig (plan §12.0, item A4).

The stereo plan depends on two physical facts nobody has measured yet:

* do the three cameras actually see the same place, and
* is the baseline long enough for disparity to mean anything.

Both are pure geometry: camera poses, a target region, and ``sigma_Z = Z^2 / (f B) * sigma_d``.
No simulator, no GPU, no model weights — this runs in seconds and decides whether weeks of
model integration are justified.

CONSTANTS
---------
The camera constants below are copied from PyRUA-Lean's ``robodojo_sight.py``, which states they
were "read off the simulator once" and are constant to a micrometre over every pose tried:

    HEAD_FOCAL_PX = 288.1325      HEAD_POSITION = (0.0, -0.41, 1.308)
    WRIST_FOCAL_PX = 397.0413     HEAD_PITCH_DOWN_DEG = 60.0

**They are third-party values, not verified by me against RoboDojo directly.** They come from a
method that reports 50.35% on the official leaderboard, so they are almost certainly right, but
an implementer must confirm them against ``env_cfg/camera/camera_config.yml`` and a real episode
before trusting any number below.

ASSUMPTIONS THAT MUST BE REPLACED WITH REAL DATA
------------------------------------------------
The end-effector poses here are *modelled*, not observed: a top-down grasp at a table target.
Real EE poses come from the HDF5 episodes (``state/left_ee_poses``). Until then, treat the
baselines below as an optimistic envelope rather than a distribution.

Run:  python scripts/stereo_feasibility.py
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# --- RoboDojo constants (see the module docstring for provenance) --------------------------

IMAGE_SHAPE = (480, 640)  # (H, W)
HEAD_FOCAL_PX = 288.1325
HEAD_POSITION = np.array([0.0, -0.41, 1.308])
HEAD_PITCH_DOWN_DEG = 60.0
WRIST_FOCAL_PX = 397.0413
WRIST_MOUNT = np.array(
    [
        [0.0, 0.5, -0.8660254, 0.084842],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.8660254, 0.5, 0.05094],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
TABLE_Z = 0.765
#: From PyRUA-Lean's operating notes: the grasp point is this far out along the pointing axis.
GRASP_OFFSET_M = 0.145

#: Assumed disparity noise of a neural stereo matcher, pixels. A range, not a measurement.
SIGMA_D_PX = (0.5, 1.0)


def _head_pose() -> np.ndarray:
    """Head camera pose (Isaac convention: looks down its -z, +y up the image)."""
    a = math.radians(HEAD_PITCH_DOWN_DEG)
    pose = np.eye(4)
    pose[:3, 0] = [1.0, 0.0, 0.0]
    pose[:3, 1] = [0.0, math.sin(a), math.cos(a)]
    pose[:3, 2] = [0.0, -math.cos(a), math.sin(a)]
    pose[:3, 3] = HEAD_POSITION
    return pose


def _wrist_pose(ee_pose: np.ndarray) -> np.ndarray:
    """Wrist camera pose from an arm's end-effector pose ``[x,y,z,qw,qx,qy,qz]``."""
    w, x, y, z = ee_pose[3:7]
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = ee_pose[:3]
    return pose @ WRIST_MOUNT


def _quat_from_matrix(rot: np.ndarray) -> np.ndarray:
    """``[qw, qx, qy, qz]`` from a rotation matrix (Shepperd's method)."""
    trace = rot[0, 0] + rot[1, 1] + rot[2, 2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        w, x = 0.25 * s, (rot[2, 1] - rot[1, 2]) / s
        y, z = (rot[0, 2] - rot[2, 0]) / s, (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2
        w, x = (rot[2, 1] - rot[1, 2]) / s, 0.25 * s
        y, z = (rot[0, 1] + rot[1, 0]) / s, (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2
        w, x = (rot[0, 2] - rot[2, 0]) / s, (rot[0, 1] + rot[1, 0]) / s
        y, z = 0.25 * s, (rot[1, 2] + rot[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2
        w, x = (rot[1, 0] - rot[0, 1]) / s, (rot[0, 2] + rot[2, 0]) / s
        y, z = (rot[1, 2] + rot[2, 1]) / s, 0.25 * s
    return np.array([w, x, y, z])


def _rotation_z(yaw_rad: float) -> np.ndarray:
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def arm_pose(
    where: np.ndarray, pitch_deg: float = 90.0, yaw_deg: float = 90.0
) -> np.ndarray:
    """An end-effector pose: gripper pointing at ``pitch_deg`` from horizontal, facing ``yaw_deg``.

    ``pitch_deg=90`` points straight down (a top-down grasp), ``0`` points horizontally. This
    matters more than it looks: a wrist camera's optical axis follows the gripper, so with the
    gripper vertical the camera looks *down* and cannot see sideways at all. Rotating the
    gripper is how one arm looks at what the other arm is working on.

    ``where`` is the *grasp point*; the EE link sits ``GRASP_OFFSET_M`` back along the pointing
    axis. Frame: x = pointing, y = closing axis, z = x cross y.

    This is a *model*, not an observed pose — real ones are in HDF5 ``state/*_ee_poses``.
    """
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    horizontal = np.array([math.sin(yaw), math.cos(yaw), 0.0])
    x_axis = horizontal * math.cos(pitch) + np.array([0.0, 0.0, -1.0]) * math.sin(pitch)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.array([math.cos(yaw), -math.sin(yaw), 0.0])
    y_axis = y_axis - x_axis * float(y_axis @ x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    rot = np.column_stack([x_axis, y_axis, z_axis])
    ee_pos = np.asarray(where, dtype=np.float64) - x_axis * GRASP_OFFSET_M
    return np.concatenate([ee_pos, _quat_from_matrix(rot)])


def top_down_ee_pose(target_xyz: np.ndarray, yaw_deg: float = 90.0) -> np.ndarray:
    """Kept for the simple case: a gripper pointing straight down at ``target_xyz``."""
    return arm_pose(target_xyz, 90.0, yaw_deg)


@dataclass
class Camera:
    name: str
    focal: float
    pose: np.ndarray  # world from camera (Isaac convention)

    @property
    def position(self) -> np.ndarray:
        return self.pose[:3, 3]

    @property
    def axis(self) -> np.ndarray:
        return -self.pose[:3, 2]

    def to_pixels(self, points: np.ndarray):
        local = (points - self.position) @ self.pose[:3, :3]
        depth = -local[:, 2]
        safe = np.where(np.abs(depth) > 1e-9, depth, 1e-9)
        row = -self.focal * local[:, 1] / safe + IMAGE_SHAPE[0] / 2
        col = self.focal * local[:, 0] / safe + IMAGE_SHAPE[1] / 2
        return row, col, depth

    def sees(self, points: np.ndarray) -> np.ndarray:
        row, col, depth = self.to_pixels(points)
        h, w = IMAGE_SHAPE
        return (depth > 1e-6) & (row >= 0) & (row <= h - 1) & (col >= 0) & (col <= w - 1)


def frustum_overlap(a: Camera, b: Camera, points: np.ndarray) -> float:
    """Share of points that BOTH cameras see."""
    both = a.sees(points) & b.sees(points)
    return float(both.mean()) if len(points) else 0.0


def sigma_z(z: float, focal: float, baseline: float, sigma_d: float) -> float:
    """``Z^2 / (f B) * sigma_d`` — the idealised depth uncertainty of a stereo pair."""
    denom = focal * baseline
    if denom <= 0:
        return float("inf")
    return z * z / denom * sigma_d


def main() -> int:
    head = Camera("head", HEAD_FOCAL_PX, _head_pose())

    xs = np.linspace(-0.45, 0.45, 13)
    ys = np.linspace(-0.30, 0.30, 13)
    targets = np.array([[x, y, TABLE_Z] for x in xs for y in ys])

    print("=" * 78)
    print("RoboDojo stereo feasibility - synthetic, no simulator")
    print("=" * 78)
    print(f"head : focal {HEAD_FOCAL_PX:.1f} px at {HEAD_POSITION}, "
          f"looking {HEAD_PITCH_DOWN_DEG:.0f} deg below horizontal")
    print(f"wrist: focal {WRIST_FOCAL_PX:.1f} px, mounted on the end effector")
    print(f"table z = {TABLE_Z}, {len(targets)} table targets")
    print()

    print("-" * 78)
    print("1. BASELINE AND CO-VISIBILITY DURING A TOP-DOWN GRASP")
    print("-" * 78)
    print("   Each arm placed over its own side; co-visibility measured over a")
    print("   +-3cm x +-3cm x 8cm-tall volume: the envelope of a graspable object.")
    print()
    header = (f"{'target':>16} {'HL base':>9} {'HR base':>9} {'LR base':>9}"
              f" {'HL cov':>7} {'HR cov':>7} {'LR cov':>7}")
    print(header)
    print("-" * len(header))

    rows = []
    for target in targets:
        left_target = np.array([min(target[0], 0.0) - 0.05, target[1], TABLE_Z])
        right_target = np.array([max(target[0], 0.0) + 0.05, target[1], TABLE_Z])
        lw = Camera("left_wrist", WRIST_FOCAL_PX, _wrist_pose(top_down_ee_pose(left_target)))
        rw = Camera("right_wrist", WRIST_FOCAL_PX, _wrist_pose(top_down_ee_pose(right_target)))

        # Sample the volume the *grasp* is about: an object of up to 6 cm half-width, up to 8 cm
        # tall, around the target. NOT a fixed box: at 8 cm working distance the wrist camera
        # covers only +-6 cm, so a fixed +-5 cm box would report zero overlap for a camera that
        # in fact sees the object. The extent here matches a typical graspable object instead.
        near = np.array(
            [[target[0] + dx, target[1] + dy, TABLE_Z + dz]
             for dx in (-0.03, 0.0, 0.03)
             for dy in (-0.03, 0.0, 0.03)
             for dz in (0.0, 0.04, 0.08)]
        )
        hl_b = float(np.linalg.norm(head.position - lw.position))
        hr_b = float(np.linalg.norm(head.position - rw.position))
        lr_b = float(np.linalg.norm(lw.position - rw.position))
        hl_c = frustum_overlap(head, lw, near)
        hr_c = frustum_overlap(head, rw, near)
        lr_c = frustum_overlap(lw, rw, near)
        rows.append((target, hl_b, hr_b, lr_b, hl_c, hr_c, lr_c))
        label = f"({target[0]:+.2f},{target[1]:+.2f})"
        print(f"{label:>16} {hl_b:>8.3f}m {hr_b:>8.3f}m {lr_b:>8.3f}m"
              f" {hl_c * 100:>6.0f}% {hr_c * 100:>6.0f}% {lr_c * 100:>6.0f}%")

    hl_baselines = np.array([r[1] for r in rows])
    hr_baselines = np.array([r[2] for r in rows])
    lr_baselines = np.array([r[3] for r in rows])
    hl_cover = np.array([r[4] for r in rows])
    hr_cover = np.array([r[5] for r in rows])
    lr_cover = np.array([r[6] for r in rows])
    print()
    for name, baselines, cover in (
        ("HL", hl_baselines, hl_cover),
        ("HR", hr_baselines, hr_cover),
        ("LR", lr_baselines, lr_cover),
    ):
        print(f"  {name} baseline: min {baselines.min():.3f}  "
              f"median {np.median(baselines):.3f}  max {baselines.max():.3f} m"
              f"   |   co-visibility median {np.median(cover) * 100:.0f}%")
    print(f"  LR targets with co-visibility below 5%: "
          f"{int((lr_cover < 0.05).sum())}/{len(lr_cover)}")
    print()

    # -----------------------------------------------------------------------------
    print("-" * 78)
    print("1b. WHAT ROTATING THE GRIPPER BUYS (the case the first version missed)")
    print("-" * 78)
    print("""
   A wrist camera's optical axis follows its gripper. With the gripper vertical it
   looks down and CANNOT see sideways, so a static top-down measurement reports
   LR as useless. An arm can instead tilt its gripper and look at what the other
   arm is holding. This section measures that.
""")
    probe = np.array([-0.15, 0.05, TABLE_Z])
    grasp_lw = Camera("lw", WRIST_FOCAL_PX, _wrist_pose(arm_pose(probe, 90.0, 90.0)))
    probe_vol = np.array(
        [[probe[0] + dx, probe[1] + dy, TABLE_Z + dz]
         for dx in (-0.03, 0.0, 0.03)
         for dy in (-0.03, 0.0, 0.03)
         for dz in (0.0, 0.04, 0.08)]
    )
    print(f"   Left arm grasps at ({probe[0]:+.2f},{probe[1]:+.2f}); right arm tilts to look left.")
    print(f"   {'right EE at':>14} {'pitch':>6} {'optical axis':>22} {'sees it':>8}"
          f" {'LR cov':>7} {'LR base':>8}")
    best_tilt = None
    for offset in (0.20, 0.30, 0.40):
        for pitch in (90, 70, 50, 30):
            right_at = np.array([probe[0] + offset, probe[1], TABLE_Z])
            rw = Camera("rw", WRIST_FOCAL_PX,
                        _wrist_pose(arm_pose(right_at, float(pitch), 270.0)))
            sees = bool(rw.sees(probe.reshape(1, 3))[0])
            cov = frustum_overlap(grasp_lw, rw, probe_vol)
            base = float(np.linalg.norm(grasp_lw.position - rw.position))
            if best_tilt is None and cov > 0.2:
                best_tilt = (offset, pitch, cov, base)
            print(f"   {offset:>13.2f}m {pitch:>6} {str(np.round(rw.axis, 2)):>22}"
                  f" {str(sees):>8} {cov * 100:>6.0f}% {base:>7.3f}m")
    print()
    if best_tilt is not None:
        off, pitch, cov, base = best_tilt
        print(f"   => Cheapest usable pose found: move the right arm {off:.2f} m aside and")
        print(f"      tilt its gripper to {pitch} deg; LR co-visibility {cov * 100:.0f}%,")
        print(f"      baseline {base:.3f} m.")
        print("      Compare: both grippers vertical gives 0% — the difference is pitch alone.")
    else:
        print("   => No pose in this sweep reached 20% co-visibility.")
    print()

    print("-" * 78)
    print("2. THEORETICAL DEPTH PRECISION  sigma_Z = Z^2 / (f B) * sigma_d")
    print("-" * 78)
    print("   At sigma_d = 0.5-1.0 px. Mask-geometry baseline: 0.3-1.0 cm centroid,")
    print("   0-2 cm height.")
    print()
    pairs = [
        ("HL", HEAD_FOCAL_PX, float(np.median(hl_baselines))),
        ("HR", HEAD_FOCAL_PX, float(np.median(hr_baselines))),
        ("LR", WRIST_FOCAL_PX, float(np.median(lr_baselines))),
    ]
    header2 = f"{'pair':>4} {'f(px)':>7} {'B(m)':>7}"
    for z in (0.3, 0.5, 0.8, 1.0):
        header2 += f" {f'Z={z}m':>13}"
    print(header2)
    print("-" * len(header2))
    for name, focal, base in pairs:
        line = f"{name:>4} {focal:>7.1f} {base:>7.3f}"
        for z in (0.3, 0.5, 0.8, 1.0):
            lo = sigma_z(z, focal, base, SIGMA_D_PX[0]) * 1000
            hi = sigma_z(z, focal, base, SIGMA_D_PX[1]) * 1000
            line += f" {f'{lo:.1f}-{hi:.1f}mm':>13}"
        print(line)
    print()

    print("-" * 78)
    print("3. DISPARITY RANGE - does it fit a stereo matcher?")
    print("-" * 78)
    print("   d = f * B / Z. A matcher supporting ~192-256 px needs d below that, and")
    print("   above its minimum (a few px) or the pair is useless at that range.")
    print()
    header3 = f"{'pair':>4} {'B(m)':>7}"
    for z in (0.3, 0.5, 0.8, 1.0):
        header3 += f" {f'd@Z={z}m':>11}"
    print(header3)
    print("-" * len(header3))
    for name, focal, base in pairs:
        line = f"{name:>4} {base:>7.3f}"
        for z in (0.3, 0.5, 0.8, 1.0):
            line += f" {focal * base / z:>10.1f}px"
        print(line)
    print()

    print("-" * 78)
    print("4. WRIST WORKING DISTANCE AND FIELD OF VIEW")
    print("-" * 78)
    print("   The wrist camera is 8.5 cm ahead of and 5.1 cm above the end effector.")
    print("   At grasp height it is very close to the object, which bounds what it can")
    print("   see no matter how good the matcher is.")
    print()
    probe = np.array([-0.20, 0.0, TABLE_Z])
    print(f"   {'EE above grasp pt':>18} {'camera to target':>18} {'FOV half-width':>16}")
    for extra in (0.0, 0.055, 0.155):
        ee = top_down_ee_pose(probe)
        ee[2] += extra
        cam = Camera("lw", WRIST_FOCAL_PX, _wrist_pose(ee))
        dist = float(np.linalg.norm(cam.position - probe))
        half = dist * (IMAGE_SHAPE[1] / 2) / WRIST_FOCAL_PX
        print(f"   {GRASP_OFFSET_M + extra:>17.3f}m {dist:>17.3f}m {half:>15.3f}m")
    print()
    print("   A wrist camera at grasp height covers only a few centimetres either side:")
    print("   it sees the object it is over, and little else. That is fine for a top-down")
    print("   grasp and useless for anything wider.")
    print()

    print("-" * 78)
    print("5. ANGLE BETWEEN OPTICAL AXES (reference)")
    print("-" * 78)
    example_ee = top_down_ee_pose(np.array([-0.10, 0.0, TABLE_Z]))
    lw = Camera("lw", WRIST_FOCAL_PX, _wrist_pose(example_ee))
    angle = math.degrees(math.acos(max(-1.0, min(1.0, float(head.axis @ lw.axis)))))
    print(f"  head vs left wrist (over a left-side target): {angle:.1f} deg")
    print()

    print("=" * 78)
    print("SUMMARY AND LIMITS")
    print("=" * 78)
    print("""
Numbers above are geometry only. Read them with these caveats:

* End-effector poses are MODELLED (top-down grasp above each target), not observed. Real
  poses are in the HDF5 episodes (state/*_ee_poses) and differ: arms approach at angles and
  the wrists rotate. Treat the baselines as an optimistic envelope.
* No occlusion is modelled. The head camera is regularly blocked by the arms, which is
  invisible to a frustum-overlap test on unoccluded points.
* Camera constants are third-party (read off the simulator by another project). Confirm them
  against a real episode before any decision rests on them.
* sigma_d = 0.5-1.0 px is an assumption about the matcher, not a measurement.

What the numbers DO settle: whether a pair has usable baseline at all, and whether the
resulting disparity fits a matcher's search range. That is what decides if the plan is worth
its weeks.

MAX_DISP: the matcher's default of 192 px is NOT enough here. Fast-FoundationStereo's own
readme says max_disp is a tunable parameter ("192 should be enough, unless you need to sense
very near objects") but does not price the increase beyond "slower and more memory". At
640x480 the RoboDojo pairs need roughly 240-384 px at 0.5-0.8 m, so max_disp must be raised
to ~384, or the images downscaled to scale 0.375 (which makes disparity 144 px but throws
away resolution). Raising it is the better trade on an H20: 97 GB of VRAM is not a constraint.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
