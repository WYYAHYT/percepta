"""CPU-only multiview adapter for testing the actual resident worker transport."""

import os
import time

import numpy as np


class FakeMultiViewAdapter:
    def __init__(self, mode="ok"):
        self.mode, self.calls = mode, 0
        print("model logs must stay outside the protocol")
        if mode == "startup_error":
            raise RuntimeError("model assets unavailable")

    def infer(self, images, intrinsics, T_world_cam):
        self.calls += 1
        if self.mode == "crash":
            os._exit(19)
        if self.mode == "hang":
            time.sleep(60)
        depth = np.full(images.shape[:3], 1.25, dtype=np.float32)
        if self.mode == "nan":
            depth[0, 0, 0] = np.nan
        if self.mode == "empty":
            depth.fill(0)
        return {"depth": depth, "confidence": np.full_like(depth, 2), "depth_unit": "relative" if self.mode == "relative" else "m"}

    def describe(self):
        return {"model": "cpu-test", "calls": self.calls, "depth_unit": "m"}

    def close(self):
        pass
