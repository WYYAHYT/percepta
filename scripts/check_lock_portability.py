#!/usr/bin/env python3
"""Verify that every committed ``uv.lock`` is portable.

Why this exists: ``uv`` records the *download URL* of every package in the lockfile. If you
generate a lock while ``UV_DEFAULT_INDEX`` (or ``[tool.uv]``) points at a mirror, those mirror
URLs get baked into all ~600 entries. The lock then only works on machines that can reach that
exact mirror — a fresh clone on CI, a colleague's laptop, or a deployment host will fail with
"not found" on every package, long after the mistake was made.

This has already happened twice in this repository, both times by running ``uv lock`` with a
domestic mirror for speed. Documentation did not prevent it, so it is checked mechanically.

Usage::

    python3 scripts/check_lock_portability.py          # exits 1 on any problem
    python3 scripts/check_lock_portability.py --list   # also print every domain seen

Exit codes: 0 = portable, 1 = problem found, 2 = no lockfiles discovered.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Hosts we accept in a committed lockfile. Deliberately a whitelist rather than a
#: blacklist: a new unknown host is more likely a mistake (or a supply-chain risk) than a
#: deliberate choice, so it should require a human decision.
ALLOWED_HOSTS = frozenset(
    {
        "pypi.org",  # the index itself
        "files.pythonhosted.org",  # PyPI's file CDN
        "download.pytorch.org",  # PyTorch's own wheel index (CUDA builds live here)
        "download-r2.pytorch.org",  # PyTorch's R2 mirror, used in index responses
    }
)

#: Well-known mirrors. Listed separately from "unknown host" purely so the error message can
#: name the likely cause instead of making the reader figure it out.
KNOWN_MIRRORS = {
    "mirrors.aliyun.com": "Aliyun PyPI mirror",
    "pypi.tuna.tsinghua.edu.cn": "Tsinghua PyPI mirror",
    "mirrors.tuna.tsinghua.edu.cn": "Tsinghua mirror",
    "pypi.mirrors.ustc.edu.cn": "USTC PyPI mirror",
    "mirrors.cloud.tencent.com": "Tencent mirror",
    "repo.huaweicloud.com": "Huawei mirror",
    "mirrors.bfsu.edu.cn": "BFSU mirror",
}

_URL_HOST = re.compile(r"https?://([A-Za-z0-9._-]+)")
_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache"}


def find_lockfiles(root: Path = ROOT) -> list[Path]:
    """Every ``uv.lock`` in the tree, excluding generated virtualenvs."""
    found: list[Path] = []
    for path in root.rglob("uv.lock"):
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        found.append(path)
    return sorted(found)


def hosts_in(path: Path) -> Counter[str]:
    return Counter(_URL_HOST.findall(path.read_text(encoding="utf-8", errors="replace")))


def check(path: Path, *, list_domains: bool = False) -> list[str]:
    """Return a list of problems for one lockfile (empty means portable)."""
    counts = hosts_in(path)
    rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path

    if list_domains:
        print(f"  {rel}:")
        for host, count in counts.most_common():
            mark = "ok  " if host in ALLOWED_HOSTS else "BAD "
            print(f"    {mark}{count:5d}  {host}")

    problems: list[str] = []
    for host, count in sorted(counts.items()):
        if host in ALLOWED_HOSTS:
            continue
        reason = KNOWN_MIRRORS.get(host)
        if reason:
            problems.append(
                f"{rel}: {count} URL(s) point at {host} ({reason}).\n"
                f"    Re-generate against the official index:\n"
                f"        cd {rel.parent} && rm uv.lock && uv lock\n"
                f"    If you set UV_DEFAULT_INDEX or a [tool.uv] index for speed, unset it "
                f"for this command — otherwise the mirror URL is written into every entry."
            )
        else:
            problems.append(
                f"{rel}: {count} URL(s) point at unknown host {host!r}.\n"
                f"    If this is intentional, add it to ALLOWED_HOSTS in "
                f"scripts/check_lock_portability.py (and say why in the commit message)."
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="print every domain and its count")
    args = parser.parse_args(argv)

    lockfiles = find_lockfiles()
    if not lockfiles:
        print("no uv.lock found; nothing to check", file=sys.stderr)
        return 2

    print(f"checking {len(lockfiles)} lockfile(s) for portable download URLs")
    problems: list[str] = []
    for path in lockfiles:
        problems.extend(check(path, list_domains=args.list))

    if problems:
        print()
        for problem in problems:
            print(f"FAIL  {problem}")
        print(
            f"\n{len(problems)} problem(s). A non-portable lockfile breaks every machine that "
            "cannot reach the mirror it was built against."
        )
        return 1

    print(f"OK  all {len(lockfiles)} lockfile(s) reference official hosts only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
