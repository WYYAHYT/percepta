"""S7 — percept cache: hits, TTL, LRU, key canonicalisation, thread safety (spec §11.4)."""

from __future__ import annotations

import hashlib
import threading

import percepta.cache as cache_module
from percepta.cache import NullCache, PerceptCache, build_cache
from percepta.config import CacheConfig
from percepta.schema import PerceiveOptions, Prompt

PROMPT = Prompt(type="text", text="bottle", min_score=0.2, top_k=1)
IMAGE_SHA256 = {"cam_head": "a" * 64, "cam_left": "b" * 64}


def _key(cache: PerceptCache | NullCache, index: int = 0) -> str:
    """A deterministic key, distinct per ``index``."""
    return cache.make_key({"cam_head": f"{index:064x}"}, PROMPT, "fake")


# ---------------------------------------------------------------------------
# Hits, misses, identity
# ---------------------------------------------------------------------------


def test_get_miss_then_hit_returns_the_same_object():
    cache = PerceptCache(max_entries=8, ttl_s=30.0)
    key = cache.make_key(IMAGE_SHA256, PROMPT, "fake")
    value = {"tool": "fake", "observations": [1, 2]}

    assert cache.get(key) is None
    assert cache.stats["misses"] == 1.0
    assert cache.stats["entries"] == 0.0

    cache.put(key, value)
    assert cache.get(key) is value
    assert cache.get(key) is value  # same key twice -> the second get is a hit
    stats = cache.stats
    assert stats["hits"] == 2.0
    assert stats["misses"] == 1.0
    assert stats["entries"] == 1.0


def test_put_replaces_value_and_refreshes_ttl(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(cache_module, "_monotonic", lambda: now[0])
    cache = PerceptCache(max_entries=4, ttl_s=10.0)
    key = _key(cache)

    cache.put(key, "old")
    now[0] += 9.0
    cache.put(key, "new")  # refreshes the TTL from t=9
    now[0] += 9.0  # t=18: the first write would have expired, the second has not
    assert cache.get(key) == "new"
    assert cache.stats["entries"] == 1.0


# ---------------------------------------------------------------------------
# TTL
# ---------------------------------------------------------------------------


def test_ttl_expiry_is_a_miss_and_removes_the_entry(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(cache_module, "_monotonic", lambda: now[0])
    cache = PerceptCache(max_entries=8, ttl_s=10.0)
    key = _key(cache)

    cache.put(key, "payload")
    now[0] += 9.9
    assert cache.get(key) == "payload"  # still inside the TTL

    now[0] += 0.2  # t = 10.1 > ttl_s
    assert cache.get(key) is None
    stats = cache.stats
    assert stats["entries"] == 0.0
    assert stats["expired"] == 1.0
    assert stats["evictions"] == 0.0  # expiry is not capacity pressure
    assert stats["misses"] == 1.0


def test_ttl_zero_is_rejected():
    try:
        PerceptCache(ttl_s=0.0)
    except ValueError:
        pass
    else:  # pragma: no cover - would mean the guard disappeared
        raise AssertionError("PerceptCache(ttl_s=0) should raise ValueError")


# ---------------------------------------------------------------------------
# LRU capacity
# ---------------------------------------------------------------------------


def test_lru_evicts_the_least_recently_used_entry():
    cache = PerceptCache(max_entries=2, ttl_s=60.0)
    first, second, third = (_key(cache, i) for i in range(3))

    cache.put(first, "a")
    cache.put(second, "b")
    assert cache.get(first) == "a"  # refresh `first`, so `second` becomes the LRU end

    cache.put(third, "c")  # evicts `second`
    assert cache.get(second) is None
    assert cache.get(first) == "a"
    assert cache.get(third) == "c"
    stats = cache.stats
    assert stats["entries"] == 2.0
    assert stats["evictions"] == 1.0


# ---------------------------------------------------------------------------
# Key canonicalisation (spec §11.4, §19.6)
# ---------------------------------------------------------------------------


def test_make_key_is_order_independent_and_versioned():
    forward = PerceptCache.make_key(dict(IMAGE_SHA256), PROMPT, "fake")
    backward = PerceptCache.make_key(dict(reversed(list(IMAGE_SHA256.items()))), PROMPT, "fake")
    assert forward == backward
    assert forward.startswith("v1:")
    assert len(forward.split(":", 1)[1]) == 64


def test_make_key_changes_with_prompt_tool_and_image_hash():
    base = PerceptCache.make_key(IMAGE_SHA256, PROMPT, "fake")
    other_prompt = Prompt(type="text", text="cup", min_score=0.2, top_k=1)
    same_phrase_other_threshold = Prompt(type="text", text="bottle", min_score=0.9, top_k=1)

    assert PerceptCache.make_key(IMAGE_SHA256, other_prompt, "fake") != base
    assert PerceptCache.make_key(IMAGE_SHA256, same_phrase_other_threshold, "fake") != base
    assert PerceptCache.make_key(IMAGE_SHA256, PROMPT, "sam3") != base
    changed_image = dict(IMAGE_SHA256, cam_head="c" * 64)
    assert PerceptCache.make_key(changed_image, PROMPT, "fake") != base
    # A camera rename is a different key even when the digests are unchanged.
    renamed = {"front": IMAGE_SHA256["cam_head"], "cam_left": IMAGE_SHA256["cam_left"]}
    assert PerceptCache.make_key(renamed, PROMPT, "fake") != base


def test_make_key_ignores_dict_insertion_order_and_empty_hashes():
    assert PerceptCache.make_key({}, PROMPT, "fake") == PerceptCache.make_key({}, PROMPT, "fake")
    assert PerceptCache.make_key(
        {"cam_head": "a" * 64, "cam_left": "b" * 64}, PROMPT, "fake"
    ) == PerceptCache.make_key(
        {"cam_left": "b" * 64, "cam_head": "a" * 64}, PROMPT, "fake"
    )


# ---------------------------------------------------------------------------
# Options subset in the key (spec §11.4 "options(相关子集)")
# ---------------------------------------------------------------------------


def test_make_key_options_subset_is_part_of_the_key():
    base = PerceptCache.make_key(IMAGE_SHA256, PROMPT, "fake", PerceiveOptions())
    assert PerceptCache.make_key(IMAGE_SHA256, PROMPT, "fake", PerceiveOptions()) == base

    for options in (
        PerceiveOptions(masks="png"),
        PerceiveOptions(min_score=0.5),
        PerceiveOptions(top_k=2),
        PerceiveOptions(masks="png", min_score=0.5, top_k=2),
    ):
        assert PerceptCache.make_key(IMAGE_SHA256, PROMPT, "fake", options) != base


def test_make_key_ignores_non_result_affecting_options():
    base = PerceptCache.make_key(IMAGE_SHA256, PROMPT, "fake", PerceiveOptions())
    irrelevant = PerceiveOptions(fuse=False, timeout_ms=1, cache=False, tools=["fake"])
    assert PerceptCache.make_key(IMAGE_SHA256, PROMPT, "fake", irrelevant) == base


def test_make_key_without_options_is_stable_and_matches_the_legacy_formula():
    legacy = PerceptCache.make_key(IMAGE_SHA256, PROMPT, "fake")
    backward = PerceptCache.make_key(
        dict(reversed(list(IMAGE_SHA256.items()))), PROMPT, "fake", None
    )
    assert legacy == backward

    # Locks backwards compatibility with keys produced before the options argument
    # existed: same three components, no options suffix.
    images = "\x1e".join(
        f"{camera}\x1f{digest}" for camera, digest in sorted(IMAGE_SHA256.items())
    )
    expected = "v1:" + hashlib.sha256(
        f"{images}\x1e{PROMPT.model_dump_json()}\x1efake".encode()
    ).hexdigest()
    assert legacy == expected

    # options=None deliberately stays distinct from the explicit default-options key.
    assert legacy != PerceptCache.make_key(IMAGE_SHA256, PROMPT, "fake", PerceiveOptions())


# ---------------------------------------------------------------------------
# NullCache / build_cache
# ---------------------------------------------------------------------------


def test_null_cache_always_misses_and_never_stores():
    cache = NullCache()
    key = NullCache.make_key(IMAGE_SHA256, PROMPT, "fake")
    cache.put(key, "value")
    assert cache.get(key) is None
    assert cache.get(key) is None  # still a miss: nothing was stored
    stats = cache.stats
    assert stats["hits"] == 0.0
    assert stats["entries"] == 0.0
    assert NullCache.make_key(IMAGE_SHA256, PROMPT, "fake") == PerceptCache.make_key(
        IMAGE_SHA256, PROMPT, "fake"
    )
    assert NullCache.make_key(IMAGE_SHA256, PROMPT, "fake", PerceiveOptions(masks="png")) == (
        PerceptCache.make_key(IMAGE_SHA256, PROMPT, "fake", PerceiveOptions(masks="png"))
    )


def test_build_cache_honours_enabled_and_bounds():
    disabled = build_cache(CacheConfig(enabled=False))
    assert isinstance(disabled, NullCache)
    assert disabled.get("anything") is None

    enabled = build_cache(CacheConfig(enabled=True, max_entries=2, ttl_s=60.0))
    assert isinstance(enabled, PerceptCache)
    for index in range(3):
        enabled.put(_key(enabled, index), index)
    stats = enabled.stats
    assert stats["entries"] == 2.0
    assert stats["evictions"] == 1.0


# ---------------------------------------------------------------------------
# Thread safety (smoke)
# ---------------------------------------------------------------------------


def test_concurrent_get_put_smoke():
    cache = PerceptCache(max_entries=16, ttl_s=60.0)
    workers, iterations = 8, 200
    barrier = threading.Barrier(workers)
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        own_key = _key(cache, index)
        try:
            barrier.wait(timeout=5)
            for step in range(iterations):
                cache.put(own_key, step)
                # `max_entries` exceeds the number of dedicated keys, so this cannot be
                # evicted by a concurrent worker.
                if cache.get(own_key) is None:  # pragma: no cover - race detector
                    raise AssertionError("own key disappeared")
                cache.put(_key(cache, 1000 + index), step)  # churn
                if cache.get(f"v1:missing-{index}-{step}") is not None:  # pragma: no cover
                    raise AssertionError("missing key returned a value")
        except BaseException as exc:  # noqa: BLE001 - report thread failures to the test
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    stats = cache.stats
    # One own-key get and one missing-key get per iteration.
    assert stats["hits"] + stats["misses"] == float(workers * iterations * 2)
    assert stats["hits"] == float(workers * iterations)
    assert 0.0 < stats["entries"] <= 16.0
