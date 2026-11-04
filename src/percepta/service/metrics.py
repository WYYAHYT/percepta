"""Prometheus text exposition without ``prometheus_client`` (spec §11.3).

The spec lists the metric *names*, not a client library, and §16.3 forbids new default
dependencies — so this is a small, thread-safe registry plus a renderer for the Prometheus
text format (version 0.0.4). Latency is a histogram (spec §17 quotes p50/p90), the rest are
counters and gauges.

Metric families used by the service:

======================= ======== ====================================================
name                    type     labels
======================= ======== ====================================================
percepta_requests_total counter  tool, status
percepta_latency_ms     histogram tool
percepta_inflight       gauge    —
percepta_cache_hits_total counter —
percepta_http_responses_total counter code
percepta_request_latency_ms histogram —
======================= ======== ====================================================
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from typing import Any

__all__ = [
    "BUCKETS",
    "METRICS",
    "Metrics",
    "escape_label_value",
    "render_prometheus",
]

#: Histogram bucket upper bounds in milliseconds (last one is implicit ``+Inf``).
BUCKETS: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0,
                              2500.0, 5000.0, 10000.0)

_HELP = {
    "percepta_requests_total": "Percepts produced, by tool and status.",
    "percepta_latency_ms": "Percept latency in milliseconds, by tool.",
    "percepta_inflight": "Requests currently inside the pipeline.",
    "percepta_cache_hits_total": "Percepts served from the cache.",
    "percepta_http_responses_total": "HTTP responses by status code.",
    "percepta_request_latency_ms": "End-to-end request latency in milliseconds.",
}

_TYPES = {
    "percepta_requests_total": "counter",
    "percepta_latency_ms": "histogram",
    "percepta_inflight": "gauge",
    "percepta_cache_hits_total": "counter",
    "percepta_http_responses_total": "counter",
    "percepta_request_latency_ms": "histogram",
}

Labels = dict[str, str] | None


def escape_label_value(value: str) -> str:
    """Escape a label value per the Prometheus text format."""
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _key(name: str, labels: Labels) -> tuple[str, tuple[tuple[str, str], ...]]:
    return name, tuple(sorted((str(k), str(v)) for k, v in (labels or {}).items()))


def _format_labels(labels: tuple[tuple[str, str], ...], extra: str | None = None) -> str:
    pairs = [f'{key}="{escape_label_value(value)}"' for key, value in labels]
    if extra is not None:
        pairs.append(f'le="{extra}"')
    return "{" + ",".join(pairs) + "}" if pairs else ""


def _format_value(value: float) -> str:
    if value == float("inf"):
        return "+Inf"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6g}"


class Metrics:
    """Thread-safe in-process metric registry with a Prometheus text renderer."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._histograms: dict[tuple[str, tuple[tuple[str, str], ...]], dict[str, Any]] = {}

    # -- recording -------------------------------------------------------

    def inc(self, name: str, labels: Labels = None, value: float = 1.0) -> None:
        key = _key(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + float(value)

    def set_gauge(self, name: str, value: float, labels: Labels = None) -> None:
        key = _key(name, labels)
        with self._lock:
            self._gauges[key] = float(value)

    def add_gauge(self, name: str, delta: float, labels: Labels = None) -> None:
        key = _key(name, labels)
        with self._lock:
            self._gauges[key] = self._gauges.get(key, 0.0) + float(delta)

    def observe_ms(self, name: str, value: float, labels: Labels = None) -> None:
        """Record one observation of a millisecond histogram."""
        key = _key(name, labels)
        value = float(value)
        with self._lock:
            entry = self._histograms.get(key)
            if entry is None:
                entry = {"buckets": [0] * len(BUCKETS), "sum": 0.0, "count": 0}
                self._histograms[key] = entry
            for index, bound in enumerate(BUCKETS):
                if value <= bound:
                    entry["buckets"][index] += 1
            entry["sum"] += value
            entry["count"] += 1

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()

    # -- introspection ---------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Plain-python copy of everything recorded so far (handy in tests)."""
        with self._lock:
            return {
                "counters": {f"{name}{dict(labels)}": value for (name, labels), value in
                             self._counters.items()},
                "gauges": {f"{name}{dict(labels)}": value for (name, labels), value in
                           self._gauges.items()},
                "histograms": {
                    f"{name}{dict(labels)}": {
                        "count": entry["count"],
                        "sum": entry["sum"],
                    }
                    for (name, labels), entry in self._histograms.items()
                },
            }

    def counter_value(self, name: str, labels: Labels = None) -> float:
        with self._lock:
            return self._counters.get(_key(name, labels), 0.0)

    def gauge_value(self, name: str, labels: Labels = None) -> float:
        with self._lock:
            return self._gauges.get(_key(name, labels), 0.0)

    # -- rendering -------------------------------------------------------

    def render(self) -> str:
        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            histograms = {key: {"buckets": list(value["buckets"]), "sum": value["sum"],
                                "count": value["count"]} for key, value in self._histograms.items()}

        names = sorted({name for name, _ in counters} | {name for name, _ in gauges}
                       | {name for name, _ in histograms})
        lines: list[str] = []
        for name in names:
            lines.append(f"# HELP {name} {_HELP.get(name, name)}")
            lines.append(f"# TYPE {name} {_TYPES.get(name, 'untyped')}")
            lines.extend(self._render_family(name, counters, gauges, histograms))
        return "\n".join(lines) + ("\n" if lines else "")

    @staticmethod
    def _render_family(
        name: str,
        counters: dict[tuple[str, tuple[tuple[str, str], ...]], float],
        gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float],
        histograms: dict[tuple[str, tuple[tuple[str, str], ...]], dict[str, Any]],
    ) -> Iterable[str]:
        out: list[str] = []
        for (family, labels), value in sorted(counters.items()):
            if family == name:
                out.append(f"{name}{_format_labels(labels)} {_format_value(value)}")
        for (family, labels), value in sorted(gauges.items()):
            if family == name:
                out.append(f"{name}{_format_labels(labels)} {_format_value(value)}")
        for (family, labels), entry in sorted(histograms.items()):
            if family != name:
                continue
            # Buckets are stored cumulatively: observe() increments every bucket whose
            # bound is >= the observation.
            for bound, count in zip(BUCKETS, entry["buckets"], strict=True):
                out.append(
                    f"{name}_bucket{_format_labels(labels, extra=_format_value(bound))} "
                    f"{_format_value(count)}"
                )
            out.append(
                f'{name}_bucket{_format_labels(labels, extra="+Inf")} '
                f"{_format_value(entry['count'])}"
            )
            out.append(f"{name}_sum{_format_labels(labels)} {_format_value(entry['sum'])}")
            out.append(f"{name}_count{_format_labels(labels)} {_format_value(entry['count'])}")
        return out


#: Process-wide registry; the service always records, the route decides whether to expose.
METRICS = Metrics()


def render_prometheus() -> str:
    """Render the process-wide registry in the Prometheus text exposition format."""
    return METRICS.render()
