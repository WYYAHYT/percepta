"""Percept cache: LRU + TTL (spec §11.4, acceptance S7, milestone M4).

The cache stores whole ``Percept`` objects (or any value) keyed by the *inputs* that
determine a tool result:

    key = "v1:" + sha256( sorted(image_sha256.items()) || prompt.model_dump_json() || tool
                         [ || "opts:<masks>:<min_score>:<top_k>" ] )

The optional options suffix carries the result-affecting subset of ``PerceiveOptions``
(spec §11.4 "options(相关子集)"): ``masks`` changes the mask encoding stored in the cached
observations, ``min_score`` / ``top_k`` change which observations exist at all. When
``options`` is ``None`` the suffix is omitted, so the pre-existing three-component key is
unchanged.

Per spec §19.6 the request ``context`` is deliberately **excluded** from the key: the
pipeline uses the cache only for model-level results, and including free-form context
would make debugging confusing and the hit rate unpredictable.

Conventions
-----------
* LRU: ``OrderedDict``; a ``get`` hit and a ``put`` both move the key to the MRU end,
  capacity pressure evicts from the LRU end.
* TTL: entries expire ``ttl_s`` seconds after they were last written; expiry is checked
  lazily on ``get``. An expired entry is removed and counted as ``expired``, which is
  deliberately *separate* from ``evictions`` (capacity pressure) so operators can tell
  "cache too small" from "cache too old".
* Thread safety: all state is guarded by one ``threading.Lock``. The FastAPI service
  calls the pipeline from several threads (spec §11.2), so ``get``/``put``/``stats`` must
  be safe to call concurrently.
* Clock: ``time.monotonic()`` via the module-level ``_monotonic`` seam, so tests can
  advance time without sleeping and wall-clock jumps cannot resurrect entries.
* ``get`` returns the stored object **unchanged** (identity preserved); the cache does
  not copy or mutate values. Callers that need to annotate a hit (e.g.
  ``provenance.cache_hit=True``) must use their own copy.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from typing import Any

from .config import CacheConfig
from .schema import PerceiveOptions, Prompt

__all__ = ["KEY_VERSION", "NullCache", "PerceptCache", "build_cache"]

KEY_VERSION = "v1"
"""Version tag prefixed to every key; bump when the key material changes."""

_FIELD_SEP = "\x1f"
"""Separator between the parts of one image-hash item (camera name / digest)."""

_ITEM_SEP = "\x1e"
"""Separator between image-hash items and between the key's components."""


def _monotonic() -> float:
    """Clock seam: tests replace this to control TTL without sleeping."""
    return time.monotonic()


class PerceptCache:
    """A bounded, thread-safe LRU cache with a per-entry time-to-live.

    Parameters
    ----------
    max_entries:
        Maximum number of live entries; the least recently used one is evicted when a
        ``put`` would exceed it. Must be >= 1 (disable the cache with :class:`NullCache`
        or ``build_cache()`` instead of setting this to 0).
    ttl_s:
        Seconds an entry stays valid after its last write. Must be > 0.
    """

    def __init__(self, max_entries: int = 256, ttl_s: float = 300.0) -> None:
        if int(max_entries) < 1:
            raise ValueError("PerceptCache.max_entries must be >= 1")
        if float(ttl_s) <= 0.0:
            raise ValueError("PerceptCache.ttl_s must be > 0")
        self._max_entries = int(max_entries)
        self._ttl_s = float(ttl_s)
        # key -> (value, expires_at); iteration order is LRU -> MRU.
        self._entries: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expired = 0

    # -- public API ---------------------------------------------------------

    def get(self, key: str) -> Any | None:
        """Return the cached value for ``key``, or ``None`` on a miss.

        A hit refreshes the key's LRU position. An expired entry is removed and reported
        as a miss. ``None`` values are never stored by ``put``'s callers in practice, so
        ``None`` unambiguously means "miss".
        """
        now = _monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            value, expires_at = entry
            if expires_at <= now:
                del self._entries[key]
                self._expired += 1
                self._misses += 1
                return None
            self._entries.move_to_end(key)
            self._hits += 1
            return value

    def put(self, key: str, value: Any) -> None:
        """Store ``value`` under ``key``, refreshing the TTL and LRU position."""
        now = _monotonic()
        with self._lock:
            self._entries[key] = (value, now + self._ttl_s)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
                self._evictions += 1

    @property
    def stats(self) -> dict[str, float]:
        """Counters plus the current entry count (all floats, safe to expose as metrics).

        ``entries`` counts stored slots; expiry is lazy, so an expired-but-never-read
        entry is still counted until a ``get`` removes it (and increments ``expired``).
        ``evictions`` counts capacity pressure only.
        """
        with self._lock:
            return {
                "hits": float(self._hits),
                "misses": float(self._misses),
                "entries": float(len(self._entries)),
                "evictions": float(self._evictions),
                "expired": float(self._expired),
            }

    @staticmethod
    def make_key(
        image_sha256: dict[str, str],
        prompt: Prompt,
        tool: str,
        options: PerceiveOptions | None = None,
    ) -> str:
        """Build the cache key for one tool call (spec §11.4).

        The key covers exactly the inputs that change a tool's *result*:

        * ``sorted(image_sha256.items())`` — camera name -> sha256, order-independent;
        * ``prompt.model_dump_json()`` — the pydantic JSON of the prompt;
        * ``tool`` — the tool name;
        * ``options`` — when given, only the result-affecting subset ``masks``,
          ``min_score`` and ``top_k`` (spec §11.4's "options(相关子集)"). ``masks``
          changes the stored mask encoding, ``min_score``/``top_k`` change which
          observations are stored. Other options (``tools``/``fuse``/``timeout_ms``/
          ``cache``) do not alter a single tool's output and are not keyed.

        ``context`` is intentionally absent (spec §19.6). ``options=None`` reproduces
        the original three-component key exactly, so callers that do not pass options
        keep stable keys. Components are joined with control-character separators so
        that no field value can fake a boundary, then hashed with sha256 and prefixed
        with the key-format version tag.
        """
        images = _ITEM_SEP.join(
            f"{camera}{_FIELD_SEP}{digest}" for camera, digest in sorted((image_sha256 or {}).items())
        )
        parts = [images, prompt.model_dump_json(), str(tool)]
        if options is not None:
            parts.append(f"opts:{options.masks}:{options.min_score:.6f}:{options.top_k}")
        material = _ITEM_SEP.join(parts)
        return f"{KEY_VERSION}:{hashlib.sha256(material.encode('utf-8')).hexdigest()}"


class NullCache:
    """A cache that never stores anything: same duck-typed interface, always a miss.

    ``build_cache`` returns this when ``cache.enabled`` is false, so callers (the
    pipeline, tools) never need an ``if cache is not None`` branch. ``make_key`` is
    shared with :class:`PerceptCache` so a disabled cache still produces identical keys
    in logs and metrics.
    """

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        """Accept and ignore ``PerceptCache``-style constructor arguments."""

    def get(self, key: str) -> None:
        """Always miss."""
        return None

    def put(self, key: str, value: Any) -> None:
        """Discard the value."""
        return None

    @property
    def stats(self) -> dict[str, float]:
        """Zeroed counters, so metrics stay equal-shaped whether caching is on or off."""
        return {"hits": 0.0, "misses": 0.0, "entries": 0.0, "evictions": 0.0, "expired": 0.0}

    make_key = staticmethod(PerceptCache.make_key)


def build_cache(config: CacheConfig) -> PerceptCache:
    """Build the cache described by ``config``; :class:`NullCache` when disabled.

    The return annotation stays ``PerceptCache`` for the frozen ``AGENTS.md`` §4 seam;
    ``NullCache`` is a duck-typed drop-in rather than a subclass on purpose (it must not
    inherit the LRU bookkeeping it is meant to avoid).
    """
    if not config.enabled:
        return NullCache()
    return PerceptCache(max_entries=config.max_entries, ttl_s=config.ttl_s)
