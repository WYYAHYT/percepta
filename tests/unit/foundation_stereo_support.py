"""Importable CPU adapter and deliberate pipe faults for the stereo worker tests only."""

from __future__ import annotations

import os
import struct
import sys
import time

import numpy as np

from percepta.backends.stereo_protocol import (
    MAX_RESPONSE_BYTES,
    decode_response,
    write_frame,
)


class FakeStereoAdapter:
    def __init__(self, mode="normal", offset=0.25):
        self.mode, self.offset, self.calls = mode, offset, 0
        assert "torch" not in sys.modules
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["HF_DATASETS_OFFLINE"] == "1"
        # Exercise Python AND native stdout redirection, and a huge non-newline log.
        print("fake model initialization")
        os.write(1, b"native upstream print\n")
        if mode == "chatty":
            for _ in range(128):
                os.write(2, b"x" * 4096)
        if mode == "startup_crash":
            os._exit(17)
        if mode == "startup_error":
            raise RuntimeError("fake assets unavailable")
        if mode == "startup_hang":
            time.sleep(60)
        # Fault injection remains in this trusted factory, never the runtime request.
        if mode in {
            "partial_ready", "bad_ready", "block_read", "partial_prefix", "partial_payload",
            "corrupt", "oversize", "trickle", "transaction_deadline", "truncated",
        }:
            sys.modules["__main__"].write_frame = self._faulty_write

    def _faulty_write(self, stream, payload):
        ready = decode_response(payload).status == "ready"
        if ready and self.mode == "partial_ready":
            os.write(stream.fileno(), b"\x40\x00")
            time.sleep(60)
        elif ready and self.mode == "bad_ready":
            write_frame(stream, b"not a readiness frame")
        elif ready:
            write_frame(stream, payload)
            if self.mode == "block_read":
                time.sleep(60)
            elif self.mode == "transaction_deadline":
                time.sleep(0.4)
        elif self.mode == "partial_prefix":
            os.write(stream.fileno(), b"\x40\x00")
            time.sleep(60)
        elif self.mode in {"partial_payload", "truncated"}:
            os.write(stream.fileno(), struct.pack("<I", len(payload)) + payload[:25])
            if self.mode == "truncated":
                os._exit(19)
            time.sleep(60)
        elif self.mode == "corrupt":
            write_frame(stream, b"garbage stdout, not disparity")
        elif self.mode == "oversize":
            os.write(stream.fileno(), struct.pack("<I", MAX_RESPONSE_BYTES + 1))
            time.sleep(60)
        elif self.mode == "trickle":
            framed = struct.pack("<I", len(payload)) + payload
            for value in framed:
                os.write(stream.fileno(), bytes([value]))
                time.sleep(0.1)
        else:
            write_frame(stream, payload)

    def infer_pair(self, left, right):
        self.calls += 1
        print("fake inference")
        if self.mode == "crash":
            os._exit(18)
        if self.mode == "error":
            raise RuntimeError("fake inference failure")
        if self.mode == "hang":
            time.sleep(60)
        if self.mode == "transaction_deadline":
            time.sleep(0.4)
        result = (left[..., 0].astype(np.float32) - right[..., 0] + self.offset)
        if self.mode == "wrong_shape":
            return result[:-1]
        if self.mode == "wrong_dtype":
            return result.astype(np.float64)
        if self.mode == "nan":
            result[0, 0] = np.nan
        return result

    def describe(self):
        return {
            "model": "fake-stereo", "device": "cpu", "max_disp": 192, "valid_iters": 32,
            "model_latency_ms": 0.125, "peak_cuda_memory_bytes": None,
            "calls": self.calls, "adapter_pid": os.getpid(), "torch_imported": "torch" in sys.modules,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "hf_hub_offline": os.environ["HF_HUB_OFFLINE"],
            "hf_datasets_offline": os.environ["HF_DATASETS_OFFLINE"],
        }

    def close(self):
        if self.mode == "close_hang":
            time.sleep(60)
