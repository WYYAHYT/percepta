"""Portability checker for committed ``uv.lock`` files (``scripts/check_lock_portability.py``).

The checker exists because the same mistake was made twice: running ``uv lock`` with a
domestic PyPI mirror for speed bakes that mirror's URLs into every entry, and the lock then
works only on machines that can reach it. Documentation did not stop it happening, so it is
enforced mechanically — and the enforcement itself needs tests, or a broken checker silently
approves broken lockfiles.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_lock_portability.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_lock_portability", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_lock_portability"] = module
    spec.loader.exec_module(module)
    return module


checker = _load_module()


def _write_lock(tmp_path: Path, urls: list[str], name: str = "uv.lock") -> Path:
    path = tmp_path / name
    body = "\n".join(f'    {{ url = "{url}", hash = "sha256:00" }},' for url in urls)
    path.write_text(f"version = 1\npackages = [\n{body}\n]\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The repository's own lockfiles
# ---------------------------------------------------------------------------


def test_repository_lockfiles_are_portable():
    """The real check: whatever is committed right now must pass."""
    assert checker.main([]) == 0


def test_repository_has_at_least_the_two_expected_lockfiles():
    found = {path.relative_to(checker.ROOT).as_posix() for path in checker.find_lockfiles()}
    assert "uv.lock" in found
    assert "uv_envs/sam3/uv.lock" in found


def test_generated_virtualenvs_are_not_scanned():
    """A venv inside the tree must be skipped, otherwise every check is meaningless."""
    lockfiles = checker.find_lockfiles()
    assert all(".venv" not in path.parts for path in lockfiles)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "host",
    [
        "mirrors.aliyun.com",
        "pypi.tuna.tsinghua.edu.cn",
        "mirrors.tuna.tsinghua.edu.cn",
        "pypi.mirrors.ustc.edu.cn",
        "repo.huaweicloud.com",
    ],
)
def test_known_mirrors_are_rejected(tmp_path, host):
    lock = _write_lock(tmp_path, [f"https://{host}/pypi/packages/foo.whl"])
    problems = checker.check(lock)
    assert problems, f"{host} should be rejected"
    assert host in problems[0]
    # The message must tell the reader how to fix it, not just that it is wrong.
    assert "uv lock" in problems[0]


def test_official_hosts_are_accepted(tmp_path):
    lock = _write_lock(
        tmp_path,
        [
            "https://pypi.org/simple/foo/",
            "https://files.pythonhosted.org/packages/ab/cd/foo.whl",
            "https://download.pytorch.org/whl/cu128/torch-2.10.0%2Bcu128-cp310.whl",
            "https://download-r2.pytorch.org/whl/cu128/bar.whl",
        ],
    )
    assert checker.check(lock) == []


def test_unknown_host_is_rejected_with_an_explicit_message(tmp_path):
    lock = _write_lock(tmp_path, ["https://internal.example.corp/packages/foo.whl"])
    problems = checker.check(lock)
    assert problems
    assert "unknown host" in problems[0]
    assert "internal.example.corp" in problems[0]
    # It must offer the escape hatch: a deliberate host is a decision, not a bug.
    assert "ALLOWED_HOSTS" in problems[0]


def test_mixed_lockfile_reports_only_the_bad_hosts(tmp_path):
    lock = _write_lock(
        tmp_path,
        [
            "https://files.pythonhosted.org/packages/foo.whl",
            "https://mirrors.aliyun.com/pypi/packages/bar.whl",
        ],
    )
    problems = checker.check(lock)
    assert len(problems) == 1
    assert "mirrors.aliyun.com" in problems[0]
    assert "files.pythonhosted.org" not in problems[0]


def test_counts_are_reported_so_scale_is_visible(tmp_path):
    lock = _write_lock(tmp_path, [f"https://mirrors.aliyun.com/p/{i}.whl" for i in range(17)])
    problems = checker.check(lock)
    assert "17 URL(s)" in problems[0]


def test_lockfile_without_urls_is_clean(tmp_path):
    lock = _write_lock(tmp_path, [])
    assert checker.check(lock) == []


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------


def test_main_returns_one_when_a_mirror_is_found(tmp_path, monkeypatch, capsys):
    lock = _write_lock(tmp_path, ["https://mirrors.aliyun.com/p/foo.whl"])
    monkeypatch.setattr(checker, "find_lockfiles", lambda root=None: [lock])
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    assert checker.main([]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_main_returns_zero_when_clean(tmp_path, monkeypatch, capsys):
    lock = _write_lock(tmp_path, ["https://files.pythonhosted.org/p/foo.whl"])
    monkeypatch.setattr(checker, "find_lockfiles", lambda root=None: [lock])
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    assert checker.main([]) == 0
    assert "OK" in capsys.readouterr().out


def test_main_returns_two_when_no_lockfiles(tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "find_lockfiles", lambda root=None: [])
    assert checker.main([]) == 2


def test_list_flag_prints_each_domain(tmp_path, monkeypatch, capsys):
    lock = _write_lock(
        tmp_path,
        [
            "https://files.pythonhosted.org/a.whl",
            "https://mirrors.aliyun.com/b.whl",
        ],
    )
    monkeypatch.setattr(checker, "find_lockfiles", lambda root=None: [lock])
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    checker.main(["--list"])
    out = capsys.readouterr().out
    assert "files.pythonhosted.org" in out
    assert "mirrors.aliyun.com" in out
    assert "BAD" in out  # the offending host is visually marked
