"""Draw the explanation figures: how one pixel becomes a 3-D position from silhouettes alone.

No matplotlib, no new dependency: PIL only (already required by percepta). The numbers in the
figures are computed, not drawn by hand — the masks come from the same analytic cube the tests
use, so what the figures show is what the code actually does.

Run:  python scripts/explain_localisation.py [outdir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from percepta import hull  # noqa: E402
from percepta.fusion import _height_curve_grid, _prism_grid, _prism_views, _View  # noqa: E402
from tests.synth_geom import camera_rig, cube_masks  # noqa: E402

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
BG = (18, 20, 26)
FG = (232, 236, 244)
DIM = (130, 140, 158)
ACCENT = (96, 190, 255)
WARN = (255, 150, 90)
GOOD = (110, 220, 150)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = FONT_DIR / name
    if path.is_file():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _text(draw, xy, message, *, size=22, bold=False, fill=FG, anchor="la"):
    draw.text(xy, message, font=_font(size, bold), fill=fill, anchor=anchor)


def _scene(azimuths=(0, 120, 240), top_height=0.10, elevation=20.0):
    """The same analytic cube the tests use: masks are exact silhouettes, not photos."""
    centre = np.array([0.0, 0.0, top_height / 2.0])
    cals = camera_rig(azimuths, elevation_deg=elevation, target=(0, 0, 0.05))
    masks = cube_masks(cals, centre, 0.05)
    views = [
        _View(
            camera=name,
            mask=masks[name],
            K=np.asarray(cals[name].K),
            T=np.asarray(cals[name].T_world_cam),
            score=0.9,
            calibration_id=name,
        )
        for name in sorted(cals)
    ]
    return cals, masks, views


# ---------------------------------------------------------------------------
# Figure 1: one pixel is a ray, not a point
# ---------------------------------------------------------------------------


def figure_ray(path: Path) -> None:
    width, height = 1400, 700
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)

    _text(draw, (40, 26), "1. One pixel fixes a RAY, not a point", size=34, bold=True)
    _text(
        draw,
        (40, 72),
        "The camera can only say: the object is SOMEWHERE along this line. "
        "Depth is the missing number.",
        size=20,
        fill=DIM,
    )

    cam = np.array([130.0, 470.0])
    draw.ellipse([cam[0] - 26, cam[1] - 26, cam[0] + 26, cam[1] + 26], fill=ACCENT)
    _text(draw, (cam[0], cam[1] + 46), "camera", size=18, anchor="ma")

    far = np.array([1230.0, 250.0])
    draw.line([tuple(cam), tuple(far)], fill=ACCENT, width=3)

    for fraction, label, colour in (
        (0.28, "possible\n0.3 m", DIM),
        (0.52, "possible\n0.6 m", WARN),
        (0.80, "possible\n1.2 m", DIM),
    ):
        point = cam + (far - cam) * fraction
        draw.ellipse(
            [point[0] - 13, point[1] - 13, point[0] + 13, point[1] + 13],
            outline=colour,
            width=3,
        )
        _text(draw, (point[0] + 22, point[1] - 10), label, size=17, fill=colour)

    _text(
        draw,
        (40, 590),
        "All three cast the SAME pixel. A single photo cannot tell them apart — "
        "this is the whole problem.",
        size=21,
        fill=FG,
    )
    _text(
        draw,
        (40, 630),
        "So every no-depth method must borrow an extra fact: a known plane, "
        "a second view, or a known size.",
        size=19,
        fill=DIM,
    )
    image.save(path)


# ---------------------------------------------------------------------------
# Figure 2: the table plane supplies the missing fact
# ---------------------------------------------------------------------------


def figure_plane(path: Path) -> None:
    width, height = 1400, 760
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)

    _text(draw, (40, 26), "2. The table plane supplies it (for the x, y)", size=34, bold=True)
    _text(
        draw,
        (40, 72),
        "An object stands ON the table. Its lowest silhouette pixel, cast onto the known "
        "plane, is where it touches. No baseline needed.",
        size=20,
        fill=DIM,
    )

    ground_y = 560
    draw.line([(120, ground_y), (1320, ground_y)], fill=(150, 120, 90), width=6)
    _text(draw, (1200, ground_y + 14), "table  z = 0.765 m", size=18, fill=(190, 160, 120))

    cam = np.array([250.0, 170.0])
    draw.ellipse([cam[0] - 22, cam[1] - 22, cam[0] + 22, cam[1] + 22], fill=ACCENT)

    obj_x0, obj_x1, obj_top = 760.0, 880.0, 380.0
    draw.rectangle([obj_x0, obj_top, obj_x1, ground_y], outline=GOOD, width=4)
    _text(draw, ((obj_x0 + obj_x1) / 2, obj_top - 34), "object", size=19, fill=GOOD, anchor="ma")

    base = np.array([(obj_x0 + obj_x1) / 2, ground_y])
    draw.line([tuple(cam), tuple(base)], fill=WARN, width=3)
    draw.ellipse([base[0] - 11, base[1] - 11, base[0] + 11, base[1] + 11], fill=WARN)
    _text(
        draw,
        (base[0] + 20, ground_y - 66),
        "ray through the lowest\nmask pixel -> contact point",
        size=17,
        fill=WARN,
    )

    top = np.array([(obj_x0 + obj_x1) / 2, obj_top])
    draw.line([tuple(cam), tuple(top)], fill=DIM, width=2)
    _text(draw, (top[0] + 18, obj_top - 6), "top pixel", size=16, fill=DIM)

    _text(
        draw,
        (obj_x1 + 90, ground_y - 190),
        "a SHORT and FAT object\nwould cast the same\nsilhouette",
        size=17,
        fill=(150, 156, 170),
    )

    _text(
        draw,
        (40, 620),
        "x, y: solved exactly (a ray meets a known plane).   z: still unknown — "
        "a short fat thing and a tall thin one look identical.",
        size=20,
        fill=FG,
    )
    _text(draw, (40, 662), "That is why height needs a SECOND fact.", size=19, fill=DIM)
    image.save(path)


# ---------------------------------------------------------------------------
# Figure 3: the real masks, the real footprints
# ---------------------------------------------------------------------------


def _mask_panel(size: tuple[int, int], mask: np.ndarray) -> Image.Image:
    panel = Image.new("RGB", size, BG)
    scale = min(size[0] / mask.shape[1], size[1] / mask.shape[0])
    shown = Image.fromarray((mask * 255).astype(np.uint8)).resize(
        (int(mask.shape[1] * scale), int(mask.shape[0] * scale)), Image.NEAREST
    )
    grey = Image.merge("RGB", ([shown] * 3))
    panel.paste(grey, ((size[0] - shown.width) // 2, (size[1] - shown.height) // 2))
    return panel


def figure_views(path: Path) -> None:
    azimuths = (0, 120, 240)
    top_height = 0.10
    cals, masks, views = _scene(azimuths, top_height)
    prism_views = _prism_views(views)
    grid = _prism_grid(prism_views, 0.0, 0.5)

    width, height = 1500, 900
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)

    _text(
        draw,
        (40, 26),
        "3. Three views: one cone each, their intersection wins",
        size=34,
        bold=True,
    )
    _text(
        draw,
        (40, 72),
        "A 10 cm cube on the table, seen at 20 degrees elevation. The numbers are measured, "
        "not drawn.",
        size=20,
        fill=DIM,
    )

    panel_w, panel_h = 440, 300
    top = 130
    areas = []
    for index, _name in enumerate(sorted(cals)):
        x = 40 + index * (panel_w + 30)
        panel = _mask_panel((panel_w, panel_h), masks[sorted(cals)[index]])
        image.paste(panel, (x, top))
        draw.rectangle([x, top, x + panel_w, top + panel_h], outline=(60, 66, 78), width=2)
        _text(draw, (x + 8, top + 8), f"cam_{index}  ({azimuths[index]} deg)", size=19, bold=True)

        single = hull.footprint_metrics(
            grid, hull.column_inside([prism_views[index]], grid, 0.0, top_height)
        )
        alone = 0.0 if single is None else single.area_m2 * 1e4
        _text(draw, (x + 8, top + panel_h + 12), f"alone: {alone:6.1f} cm2", size=19, fill=WARN)
        areas.append(alone)

    combined = hull.footprint_metrics(
        grid, hull.column_inside(prism_views, grid, 0.0, top_height)
    )
    union_area = 0.0 if combined is None else combined.area_m2 * 1e4

    y = top + panel_h + 70
    _text(draw, (40, y), "Footprint each view alone would allow:", size=22, bold=True)
    _text(
        draw,
        (40, y + 34),
        "  " + "   ".join(f"cam_{i}: {a:6.1f} cm2" for i, a in enumerate(areas)),
        size=20,
        fill=WARN,
    )
    _text(draw, (40, y + 74), "Intersection of all three (what we report):", size=22, bold=True)
    _text(
        draw,
        (40, y + 108),
        f"  {union_area:.1f} cm2      (true cube face: 100.0 cm2)",
        size=20,
        fill=GOOD,
    )
    _text(
        draw,
        (40, y + 160),
        "Each silhouette is a CONE of possible space. One cone is far too generous; "
        "three of them agree only near the object.",
        size=20,
        fill=FG,
    )
    _text(
        draw,
        (40, y + 196),
        "The intersection is still an UPPER BOUND: a silhouette cannot reveal a dent it never "
        "sees. That is a limit of the information, not of the code.",
        size=19,
        fill=DIM,
    )
    image.save(path)
    print(f"  cam areas {[round(a, 1) for a in areas]} cm2 -> combined {union_area:.1f} cm2")


# ---------------------------------------------------------------------------
# Figure 4: the height search, with the real IoU curve
# ---------------------------------------------------------------------------


def figure_height(path: Path) -> None:
    top_height = 0.10
    cals, masks, views = _scene((0, 120, 240), top_height)
    prism_views = _prism_views(views)
    grid = _prism_grid(prism_views, 0.0, 0.5)

    heights = [0.005 * i for i in range(1, 41)]
    curve = _height_curve_grid(grid, prism_views, 0.0, heights)

    width, height = 1400, 780
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)

    _text(
        draw,
        (40, 26),
        "4. Finding the height: try every height, keep the best fit",
        size=34,
        bold=True,
    )
    _text(
        draw,
        (40, 72),
        "For each candidate height we build the prism, project it into the other cameras, "
        "and score the overlap with their masks.",
        size=20,
        fill=DIM,
    )

    left, right = 150, width - 90
    plot_top, plot_bottom = 150, 560
    draw.line([(left, plot_bottom), (right, plot_bottom)], fill=(80, 86, 98), width=2)
    draw.line([(left, plot_top), (left, plot_bottom)], fill=(80, 86, 98), width=2)

    def x_of(h):  # noqa: ANN001
        return left + (right - left) * (h / 0.20)

    def y_of(score):  # noqa: ANN001
        return plot_bottom - (plot_bottom - plot_top) * score

    for score in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = y_of(score)
        draw.line([(left, y), (right, y)], fill=(44, 48, 58), width=1)
        _text(draw, (left - 14, y), f"{score:.2f}", size=16, fill=DIM, anchor="rm")

    y_thr = y_of(0.5)
    draw.line([(left, y_thr), (right, y_thr)], fill=(150, 110, 70), width=2)
    _text(
        draw,
        (right - 6, y_thr - 26),
        "old rule: stop at the first 0.5",
        size=16,
        fill=(200, 160, 110),
        anchor="ra",
    )

    points = [(x_of(h), y_of(s)) for h, s in curve if h <= 0.20]
    draw.line(points, fill=ACCENT, width=3)
    for x, y in points[::4]:
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=ACCENT)

    x_true = x_of(top_height)
    draw.line([(x_true, plot_top), (x_true, plot_bottom)], fill=GOOD, width=2)
    _text(draw, (x_true + 8, plot_top + 6), "true height 0.10 m", size=18, fill=GOOD)

    best_h, best_s = max(curve, key=lambda kv: kv[1])
    px, py = x_of(best_h), y_of(best_s)
    draw.ellipse([px - 9, py - 9, px + 9, py + 9], outline=GOOD, width=3)
    _text(draw, (px + 16, py - 8), f"peak {best_s:.3f} at {best_h:.3f} m", size=18, fill=GOOD)

    crossing = next((h for h, s in curve if s >= 0.5), None)
    if crossing is not None:
        cx, cy = x_of(crossing), y_of(0.5)
        draw.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], outline=WARN, width=3)
        _text(draw, (cx + 14, cy + 14), f"first crossing {crossing:.3f} m", size=17, fill=WARN)

    _text(draw, (left, plot_bottom + 20), "candidate height (m)", size=18, fill=DIM, anchor="ma")
    _text(draw, (32, plot_top - 40), "mask IoU", size=18, fill=DIM)

    _text(
        draw,
        (40, 650),
        "Only ONE height makes the projected prism cover all three masks: too short leaves the "
        "top uncovered, too tall overshoots.",
        size=20,
        fill=FG,
    )
    _text(
        draw,
        (40, 690),
        "Taking the first height that clears 0.5 reports a 4 cm object instead of 10 cm — "
        "that was a real bug, fixed in v0.3.0.",
        size=19,
        fill=WARN,
    )
    image.save(path)
    print(f"  peak at {best_h:.3f} m (true {top_height:.3f}), first crossing at {crossing}")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    outdir = Path(argv[0]) if argv else ROOT / "docs" / "images"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"writing to {outdir}")

    figure_ray(outdir / "explain_1_ray.png")
    figure_plane(outdir / "explain_2_plane.png")
    figure_views(outdir / "explain_3_views.png")
    figure_height(outdir / "explain_4_height.png")
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
