"""Admission slots live as long as actual inference, including detached HTTP requests."""

from __future__ import annotations

import asyncio
import threading

import httpx
import numpy as np
import pytest

from percepta.client import build_perceive_request
from percepta.config import AppConfig
from percepta.registry import ToolRegistry
from percepta.schema import ToolCapabilities
from percepta.service import metrics
from percepta.service.app import create_app
from percepta.tools.base import BaseTool


class GatedTool(BaseTool):
    """Hold a real pipeline call until the test explicitly releases it."""

    name = "gated"
    kind = "fake"

    def __init__(self, fail: bool = False):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.closed = threading.Event()
        self.closed_early = False
        self.calls = 0
        self.fail = fail

    def capabilities(self):
        return ToolCapabilities(kind="fake", prompts={"text"})

    def perceive(self, ctx, prompts):
        self.calls += 1
        self.entered.set()
        try:
            assert self.release.wait(5), "test did not release inference"
            if self.fail:
                raise RuntimeError("detached inference failed")
            return []
        finally:
            self.finished.set()

    def close(self):
        self.closed_early = self.entered.is_set() and not self.finished.is_set()
        self.closed.set()


def _app(tool, *, warmup=False):
    registry = ToolRegistry()
    registry.register(tool.name, tool)
    config = AppConfig.model_validate({
        "pipeline": {"default": [tool.name]}, "tools": {tool.name: {"enabled": True}},
        "server": {"max_queue": 1, "request_timeout_ms": 30},
        "cache": {"enabled": False}, "warmup": {"enabled": warmup, "height": 8, "width": 8},
    })
    return create_app(config, registry=registry)


def _payload():
    return build_perceive_request(
        {"cam": np.zeros((8, 8, 3), dtype=np.uint8)}, ["object"], options={"fuse": False, "cache": False},
    ).model_dump(mode="json")


async def _event(event):
    assert await asyncio.to_thread(event.wait, 2), "test inference did not progress"


async def _inflight(value):
    async def wait():
        while metrics.METRICS.gauge_value("percepta_inflight") != value:
            await asyncio.sleep(0.005)
    await asyncio.wait_for(wait(), 2)


@pytest.mark.parametrize("cancel", [False, True], ids=["timeout", "cancelled_http_request"])
@pytest.mark.parametrize("fail", [False, True], ids=["successful_worker", "failed_worker"])
def test_detached_request_keeps_slot_until_actual_inference_finishes(cancel, fail):
    async def run():
        tool = GatedTool(fail=fail)
        app = _app(tool)
        metrics.METRICS.reset()
        errors = []
        loop = asyncio.get_running_loop()
        original_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: errors.append(context))
        try:
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                    request = asyncio.create_task(client.post("/v1/perceive", json=_payload()))
                    try:
                        await _event(tool.entered)
                        if cancel:
                            request.cancel()
                            with pytest.raises(asyncio.CancelledError):
                                await request
                        else:
                            assert (await request).status_code == 504
                        assert metrics.METRICS.gauge_value("percepta_inflight") == 1
                        for _ in range(2):
                            assert (await client.post("/v1/perceive", json=_payload())).status_code == 503
                        assert tool.calls == 1
                    finally:
                        tool.release.set()
                        await _event(tool.finished)
                    await _inflight(0)
                    app.state.config.server.request_timeout_ms = 2000
                    assert (await client.post("/v1/perceive", json=_payload())).status_code == (500 if fail else 200)
                    await _inflight(0)
                    state = app.state.service
                    assert state.try_acquire()
                    assert not state.try_acquire(), "request released its slot twice"
                    state.release()
                    await asyncio.sleep(0)
            assert not errors, "detached task exception was not retrieved"
        finally:
            tool.release.set()
            loop.set_exception_handler(original_handler)
    asyncio.run(run())


@pytest.mark.parametrize("warmup", [False, True], ids=["timed_out_request", "warmup"])
def test_shutdown_waits_for_inference_before_closing_tools(warmup):
    async def run():
        tool = GatedTool()
        app = _app(tool, warmup=warmup)
        metrics.METRICS.reset()
        shutdown_started = asyncio.Event()

        async def serve():
            async with app.router.lifespan_context(app):
                if not warmup:
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                        assert (await client.post("/v1/perceive", json=_payload())).status_code == 504
                await _event(tool.entered)
                shutdown_started.set()

        serving = asyncio.create_task(serve())
        try:
            await asyncio.wait_for(shutdown_started.wait(), 2)
            await asyncio.sleep(0.02)
            assert not serving.done(), "shutdown finished while inference still held the model"
            assert not tool.closed.is_set()
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                assert (await client.post("/v1/perceive", json=_payload())).status_code == 503
        finally:
            tool.release.set()
            await asyncio.wait_for(serving, 2)
        assert tool.closed.is_set()
        assert not tool.closed_early
        assert metrics.METRICS.gauge_value("percepta_inflight") == 0
    asyncio.run(run())
