"""Stereo wire and real resident subprocess tests: no GPU, weights, installs or network."""

from __future__ import annotations

import io
import json
import os
import pickle
import struct
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from percepta.backends.base import BackendRequestError, BackendUnavailable
from percepta.backends.foundation_stereo_backend import STDERR_LIMIT_BYTES, FoundationStereoBackend
from percepta.backends.stereo_protocol import (
    HEADER,
    MAX_METADATA_BYTES,
    MAX_PIXELS,
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    PROTOCOL_VERSION,
    REQUEST_MAGIC,
    RESPONSE_MAGIC,
    ProtocolError,
    StereoRequest,
    StereoResponse,
    decode_request,
    decode_response,
    encode_request,
    encode_response,
    read_frame,
    write_frame,
)

ROOT = Path(__file__).resolve().parents[2]
FACTORY = "tests.unit.foundation_stereo_support:FakeStereoAdapter"


def pair():
    left = np.arange(9 * 13 * 3, dtype=np.uint8).reshape(9, 13, 3)
    return left[:, ::-1], np.roll(left, 2, axis=1)[:, ::-1]  # exercise noncontiguous RGB


def packet(meta, *, response=False, height=0, width=0, raw=b""):
    blob = meta if isinstance(meta, bytes) else json.dumps(meta).encode()
    return HEADER.pack(
        RESPONSE_MAGIC if response else REQUEST_MAGIC, PROTOCOL_VERSION, len(blob), height, width,
    ) + blob + raw


def test_wire_roundtrip_exact_bytes_and_no_pickle(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("pickle/numpy.load must not be used by stereo protocol")

    monkeypatch.setattr(pickle, "loads", forbidden)
    monkeypatch.setattr(pickle, "load", forbidden)
    monkeypatch.setattr(np, "load", forbidden)
    left, right = pair()
    payload = encode_request(StereoRequest(left=left, right=right))
    assert HEADER.unpack_from(payload)[:2] == (REQUEST_MAGIC, 1)
    meta_size = HEADER.unpack_from(payload)[2]
    assert len(payload) == HEADER.size + meta_size + left.size * 2
    assert payload[-left.size * 2:] == left.tobytes() + right.tobytes()
    request = decode_request(payload)
    np.testing.assert_array_equal(request.left, left)
    np.testing.assert_array_equal(request.right, right)
    assert request.left.dtype == np.uint8
    assert request.left.flags.writeable
    assert decode_request(encode_request(StereoRequest(op="shutdown"))).op == "shutdown"

    disparity = (left[..., 0].astype(np.float32) - right[..., 0]) / 4
    payload = encode_response(StereoResponse("ok", disparity, {"latency_ms": 2.0}))
    decoded = decode_response(payload)
    np.testing.assert_array_equal(decoded.disparity, disparity)
    assert decoded.disparity.dtype == np.float32
    assert decoded.disparity.shape == left.shape[:2]
    assert payload[-disparity.nbytes:] == disparity.astype("<f4").tobytes()
    assert decoded.diagnostics == {"latency_ms": 2.0}
    for response in (StereoResponse("ready", diagnostics={"model": "fake"}),
                     StereoResponse("error", error="no CUDA")):
        result = decode_response(encode_response(response))
        assert result.status == response.status and result.error == response.error

    framed = io.BytesIO()
    write_frame(framed, payload)
    assert framed.getvalue()[:4] == struct.pack("<I", len(payload))
    framed.seek(0)
    assert read_frame(framed) == payload
    assert read_frame(framed) is None


@pytest.mark.parametrize("response", [False, True])
def test_all_truncations_trailing_bytes_magic_version_are_rejected(response):
    left, right = pair()
    payload = (encode_response(StereoResponse("ok", np.ones((2, 3), np.float32))) if response
               else encode_request(StereoRequest(left=left[:2, :3], right=right[:2, :3])))
    decode = decode_response if response else decode_request
    for cut in range(len(payload)):
        with pytest.raises(ProtocolError):
            decode(payload[:cut])
    for damaged in (payload + b"extra", b"JUNK" + payload[4:],
                    payload[:4] + struct.pack("<I", 999) + payload[8:]):
        with pytest.raises(ProtocolError):
            decode(damaged)
    # The peer must never deserialize an object or guess a dtype from arbitrary bytes.
    with pytest.raises(ProtocolError):
        decode(b"\x80\x04pickle-looking bytes" * 4)


@pytest.mark.parametrize("blob", [b"{", b"[]", b"null", b"\xff", b'{"op":NaN}',
                                   b'{"op":"infer","op":"shutdown"}'])
@pytest.mark.parametrize("response", [False, True])
def test_invalid_json_is_protocol_error(blob, response):
    with pytest.raises(ProtocolError):
        (decode_response if response else decode_request)(packet(blob, response=response))


@pytest.mark.parametrize("meta,height,width,raw", [
    ({"op": "infer", "dtype": "float32"}, 1, 1, b"\0" * 6),
    ({"op": "infer", "dtype": "uint8", "backend_factory": "evil:code"}, 1, 1, b"\0" * 6),
    ({"op": "infer", "dtype": "uint8", "source_dir": "/tmp"}, 1, 1, b"\0" * 6),
    ({"op": "infer", "dtype": "uint8"}, 0, 1, b""),
    ({"op": "infer", "dtype": "uint8"}, MAX_PIXELS + 1, 1, b""),
    ({"op": "infer", "dtype": "uint8"}, 1, 1, b"\0" * 7),
    ({"op": "shutdown"}, 1, 0, b""),
    ({"op": "shutdown"}, 0, 0, b"trailing"),
    ({"op": "unknown"}, 0, 0, b""),
])
def test_bad_request_metadata_dimensions_lengths(meta, height, width, raw):
    with pytest.raises(ProtocolError):
        decode_request(packet(meta, height=height, width=width, raw=raw))


@pytest.mark.parametrize("meta,height,width,raw", [
    ({"status": "ok", "dtype": "float64", "diagnostics": {}}, 1, 1, b"\0" * 8),
    ({"status": "ok", "dtype": "float32_le", "diagnostics": []}, 1, 1, b"\0" * 4),
    ({"status": "ok", "dtype": "float32_le", "diagnostics": {}}, 1, 1, struct.pack("<f", np.nan)),
    ({"status": "ok", "dtype": "float32_le", "diagnostics": {}}, MAX_PIXELS + 1, 1, b""),
    ({"status": "ready", "diagnostics": {}}, 1, 1, b"\0" * 4),
    ({"status": "ready", "diagnostics": []}, 0, 0, b""),
    ({"status": "error", "error": 42}, 0, 0, b""),
    ({"status": "error", "error": ""}, 0, 0, b""),
    ({"status": "unknown"}, 0, 0, b""),
])
def test_bad_response_metadata_dimensions_lengths(meta, height, width, raw):
    with pytest.raises(ProtocolError):
        decode_response(packet(meta, response=True, height=height, width=width, raw=raw))


@pytest.mark.parametrize("shape,dtype", [((2, 3), np.uint8), ((2, 3, 4), np.uint8),
                                        ((2, 3, 3), np.float32), ((0, 3, 3), np.uint8)])
def test_encoder_rejects_invalid_input_array(shape, dtype):
    image = np.zeros(shape, dtype)
    with pytest.raises(ProtocolError):
        encode_request(StereoRequest(left=image, right=image))


def test_encoder_rejects_mismatched_pair_and_invalid_disparity():
    left, right = pair()
    with pytest.raises(ProtocolError, match="match"):
        encode_request(StereoRequest(left=left, right=right[:-1]))
    for result in (np.zeros((2, 3), np.float64), np.zeros((2, 3, 1), np.float32),
                   np.full((2, 3), np.inf, np.float32)):
        with pytest.raises(ProtocolError):
            encode_response(StereoResponse("ok", result))


def test_exact_limits_enforced_before_array_allocation(monkeypatch):
    assert MAX_REQUEST_BYTES == HEADER.size + MAX_METADATA_BYTES + 6 * MAX_PIXELS
    assert MAX_RESPONSE_BYTES == HEADER.size + MAX_METADATA_BYTES + 4 * MAX_PIXELS
    for magic, decode in ((REQUEST_MAGIC, decode_request), (RESPONSE_MAGIC, decode_response)):
        for length in (0, MAX_METADATA_BYTES + 1):
            with pytest.raises(ProtocolError, match="metadata length"):
                decode(HEADER.pack(magic, 1, length, 0, 0))
    image = np.broadcast_to(np.zeros(3, np.uint8), (MAX_PIXELS + 1, 1, 3))
    with pytest.raises(ProtocolError, match="dimensions"):
        encode_request(StereoRequest(left=image, right=image))
    with pytest.raises(ProtocolError, match="metadata"):
        encode_response(StereoResponse("ready", diagnostics={"log": "x" * MAX_METADATA_BYTES}))

    # Reader bounds the outer length before attempting to read its body.
    from percepta.backends.foundation_stereo_worker import _RequestReader

    stream = io.BytesIO(struct.pack("<I", MAX_REQUEST_BYTES + 1))
    with pytest.raises(ProtocolError, match="length limit"):
        read_frame(_RequestReader(stream))
    for raw in (b"\0\0", struct.pack("<I", 8) + b"abc"):
        with pytest.raises(ProtocolError, match="truncated"):
            read_frame(io.BytesIO(raw))


@pytest.fixture
def make_backend(monkeypatch):
    # Parent adds src; explicitly append the repo root so the child can import tests.
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, (existing, str(ROOT)))))
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")  # worker must enforce offline, not setdefault
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "0")
    backends = []
    children = []
    original_popen = subprocess.Popen

    def track_popen(*args, **kwargs):
        assert kwargs["shell"] is False
        assert args[0][1:4] == ["-u", "-m", "percepta.backends.foundation_stereo_worker"]
        process = original_popen(*args, **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", track_popen)

    def make(mode="normal", **kwargs):
        args = {"python": sys.executable, "backend_factory": FACTORY,
                "backend_args": {"mode": mode}, "startup_timeout_s": 5, "timeout_s": 2}
        args.update(kwargs)
        backend = FoundationStereoBackend(**args)
        backends.append(backend)
        return backend

    yield make
    for backend in backends:
        backend.close()
        assert backend._process is None
        if backend._stderr_thread is not None:
            assert not backend._stderr_thread.is_alive()
    for child in children:
        assert child.poll() is not None, "own child was not reaped"
        assert child.wait(timeout=0) == child.returncode
        assert all(stream.closed for stream in (child.stdin, child.stdout, child.stderr))


def test_resident_subprocess_roundtrip_and_description(make_backend):
    backend = make_backend("chatty")
    left, right = pair()
    assert backend.describe()["state"] == "new"
    backend.start()
    info = backend.describe()
    assert info["running"] and info["pid"] != os.getpid()
    assert info["pid"] == info["adapter_pid"]
    assert info["calls"] == 0  # readiness follows initialization, not inference
    backend.start()
    for call in range(1, 4):
        disparity = backend.infer_pair(left, right)
        np.testing.assert_array_equal(disparity, left[..., 0].astype(np.float32) - right[..., 0] + .25)
        assert disparity.shape == left.shape[:2] and disparity.dtype == np.float32
        assert backend.describe()["pid"] == info["pid"]
        assert backend.describe()["calls"] == call
    assert backend.describe()["max_disp"] == 192
    assert backend.describe()["valid_iters"] == 32
    assert backend.describe()["device"] == "cpu"
    assert backend.describe()["model_latency_ms"] == .125
    assert backend.describe()["torch_imported"] is False
    assert backend.describe()["hf_hub_offline"] == "1"
    assert backend.describe()["hf_datasets_offline"] == "1"
    assert len(backend._stderr_tail) <= STDERR_LIMIT_BYTES
    assert b"fake inference" in backend._stderr_tail
    backend.close()
    backend.close()
    assert not backend.describe()["running"]
    with pytest.raises(BackendUnavailable, match="closed"):
        backend.start()


@pytest.mark.parametrize("mode", ["startup_crash", "startup_error", "bad_ready"])
def test_startup_failure_is_terminal_and_reaped(make_backend, mode):
    backend = make_backend(mode)
    with pytest.raises(BackendUnavailable):
        backend.start()
    assert backend._process is None and backend.describe()["state"] == "failed"
    with pytest.raises(BackendUnavailable):
        backend.start()


@pytest.mark.parametrize("mode", ["startup_hang", "partial_ready"])
def test_startup_deadline_includes_partial_handshake(make_backend, mode):
    backend = make_backend(mode, startup_timeout_s=.6)
    started = time.monotonic()
    with pytest.raises(BackendUnavailable, match="deadline"):
        backend.start()
    assert time.monotonic() - started < .6 + 2


@pytest.mark.parametrize("mode", ["crash", "error", "corrupt", "truncated", "oversize",
                                   "wrong_shape", "wrong_dtype", "nan"])
def test_midcall_failure_is_unavailable_and_never_restarts(make_backend, mode):
    backend = make_backend(mode)
    backend.start()
    process = backend._process
    with pytest.raises(BackendUnavailable):
        backend.infer_pair(*pair())
    assert process.poll() is not None
    assert backend._process is None and backend.describe()["state"] == "failed"
    with pytest.raises(BackendUnavailable):
        backend.infer_pair(*pair())


@pytest.mark.parametrize("mode", ["hang", "partial_prefix", "partial_payload", "trickle", "block_read"])
def test_entire_transaction_deadline_covers_partial_pipes(make_backend, mode):
    backend = make_backend(mode, timeout_s=.35)
    backend.start()
    image = np.zeros((512, 512, 3), np.uint8)  # larger than a pipe; exercises blocked send
    started = time.monotonic()
    with pytest.raises(BackendUnavailable, match="deadline"):
        backend.infer_pair(image, image)
    assert time.monotonic() - started < .35 + 2


def test_send_and_read_share_one_deadline(make_backend):
    backend = make_backend("transaction_deadline", timeout_s=.6)
    backend.start()
    image = np.zeros((512, 512, 3), np.uint8)
    started = time.monotonic()
    with pytest.raises(BackendUnavailable, match="deadline"):
        backend.infer_pair(image, image)
    assert time.monotonic() - started < .6 + 2


def test_idle_crash_is_not_restarted_and_does_not_kill_other_worker(make_backend):
    first, other = make_backend(), make_backend()
    first.start()
    other.start()
    process = first._process
    process.kill()
    process.wait(timeout=2)
    with pytest.raises(BackendUnavailable, match="exited"):
        first.infer_pair(*pair())
    assert other.infer_pair(*pair()).dtype == np.float32
    assert other.describe()["running"]


def test_close_is_bounded_even_if_adapter_close_hangs(make_backend):
    backend = make_backend("close_hang")
    backend.start()
    process = backend._process
    started = time.monotonic()
    backend.close()
    backend.close()
    assert time.monotonic() - started < 2
    assert process.poll() is not None


def test_invalid_pair_does_not_spawn_or_poison_worker(make_backend):
    backend = make_backend()
    left, right = pair()
    with pytest.raises(BackendRequestError):
        backend.infer_pair(left.astype(np.float32), right)
    assert backend._process is None and backend.describe()["state"] == "new"
    backend.infer_pair(left, right)
    with pytest.raises(BackendRequestError):
        backend.infer_pair(left, right[:-1])
    assert backend.infer_pair(left, right).dtype == np.float32


@pytest.mark.parametrize("override,expected", [(None, "5"), ("", "5"), ("  ", "5"),
                                               (" 1,2 ", "1,2")])
def test_cuda_environment_inherited_unless_nonblank(make_backend, monkeypatch, override, expected):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    backend = make_backend(cuda_visible_devices=override)
    backend.start()
    assert backend.describe()["visible_devices"] == expected
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "5"


def test_absent_cuda_stays_absent_and_paths_defaults(monkeypatch, tmp_path):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    backend = FoundationStereoBackend(
        checkpoint=str(tmp_path / "model.pth"), hf_cache=str(tmp_path / "hf_cache")
    )
    assert backend._worker_env()["HF_HUB_CACHE"] == str(tmp_path / "hf_cache")
    assert backend._worker_env()["HUGGINGFACE_HUB_CACHE"] == str(tmp_path / "hf_cache")
    assert backend._worker_env()["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert "CUDA_VISIBLE_DEVICES" not in backend._worker_env()
    assert backend.worker_config["model_config"] == str(tmp_path / "cfg.yaml")
    assert backend.worker_config["valid_iters"] == 32
    assert backend.startup_timeout_s == 180 and backend.timeout_s == 120
    assert backend._worker_env()["PYTHONPATH"].split(os.pathsep)[0] == str(ROOT / "src")
    backend.close()


@pytest.mark.parametrize("kwargs", [{"valid_iters": 0}, {"valid_iters": True}, {"valid_iters": 1.5},
                                     {"timeout_s": 0}, {"startup_timeout_s": -1},
                                     {"timeout_s": float("inf")}, {"timeout_s": float("nan")}])
def test_invalid_deployment_numeric_config(kwargs):
    with pytest.raises(BackendRequestError):
        FoundationStereoBackend(**kwargs)


def test_missing_interpreter_and_assets_are_unavailable(make_backend, tmp_path):
    backend = make_backend(python=str(tmp_path / "nonexistent-python"))
    with pytest.raises(BackendUnavailable):
        backend.start()
    backend = make_backend(backend_factory=None, source_dir=str(tmp_path / "missing-source"))
    with pytest.raises(BackendUnavailable, match="source_dir not found"):
        backend.start()


@pytest.mark.parametrize("vit_size,cuda_available", [(None, True), ("vitb", True), (None, False)])
def test_actual_adapter_upstream_contract_with_cpu_stubs(monkeypatch, tmp_path, vit_size, cuda_available):
    from percepta.backends.foundation_stereo_worker import FoundationStereoAdapter

    class Config(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

    class Tensor:
        def __init__(self, array):
            self.array = array
            self.shape = array.shape

        def permute(self, *axes):
            return Tensor(self.array.transpose(axes))

        def __getitem__(self, index):
            return Tensor(self.array[index])

        def float(self):
            return Tensor(self.array.astype(np.float32))

        def cuda(self):
            return self

        cpu = cuda

        def numpy(self):
            return self.array

    class Padder:
        def __init__(self, shape, *, divis_by, force_square):
            assert shape == (1, 3, 9, 13)
            assert divis_by == 32 and force_square is False

        def pad(self, *images):
            return tuple(Tensor(np.pad(image.array, ((0, 0), (0, 0), (0, 23), (0, 19))))
                         for image in images)

        def unpad(self, disparity):
            assert disparity.array.dtype == np.float32
            return disparity[:, :, :9, :13]

    left, right = pair()

    def forward(image1, image2, *, iters, test_mode):
        assert iters == 7 and test_mode is True
        assert image1.shape == image2.shape == (1, 3, 32, 32)
        assert image1.array.dtype == np.float32
        np.testing.assert_array_equal(image1.array[0, :, :9, :13], left.transpose(2, 0, 1))
        np.testing.assert_array_equal(image2.array[0, :, :9, :13], right.transpose(2, 0, 1))
        return Tensor(np.full((1, 1, 32, 32), 3.5, np.float16))

    config = Config(max_disp=192)
    if vit_size is not None:
        config.vit_size = vit_size
    model = Mock()
    model.cuda.return_value = model
    model.eval.return_value = model
    model.forward.side_effect = forward
    constructor = Mock(return_value=model)
    load_config = Mock(return_value=config)
    event = Mock()
    event.elapsed_time.return_value = 1.75
    torch = SimpleNamespace(
        __version__="stub",
        serialization=SimpleNamespace(add_safe_globals=Mock(), clear_safe_globals=Mock()),
        cuda=SimpleNamespace(
            is_available=lambda: cuda_available, get_device_name=lambda i: "stub GPU",
            synchronize=Mock(), reset_peak_memory_stats=Mock(), Event=Mock(return_value=event),
            max_memory_allocated=lambda: 2048, empty_cache=Mock(),
        ),
        load=Mock(return_value={"model": {"weights": "sentinel"}}),
        from_numpy=Tensor, inference_mode=Mock(side_effect=nullcontext),
        autocast=Mock(side_effect=lambda **kwargs: nullcontext()), float16=np.float16,
    )
    modules = {
        "torch": torch,
        "omegaconf": SimpleNamespace(
            OmegaConf=SimpleNamespace(load=load_config), DictConfig=dict, ListConfig=list
        ),
        "omegaconf.base": SimpleNamespace(ContainerMetadata=object, Metadata=object),
        "omegaconf.nodes": SimpleNamespace(AnyNode=object),
        "core": SimpleNamespace(), "core.utils": SimpleNamespace(),
        "core.foundation_stereo": SimpleNamespace(FoundationStereo=constructor),
        "core.utils.utils": SimpleNamespace(InputPadder=Padder),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, "path", sys.path.copy())  # adapter's source injection is isolated
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "0")
    checkpoint = tmp_path / "model.pth"
    checkpoint.touch()  # dummy existence only; torch.load above never reads it
    (tmp_path / "cfg.yaml").touch()
    args = {"source_dir": str(tmp_path), "checkpoint": str(checkpoint), "valid_iters": 7}
    if not cuda_available:
        with pytest.raises(RuntimeError, match="requires CUDA"):
            FoundationStereoAdapter(**args)
        constructor.assert_not_called()
        torch.load.assert_not_called()
        return
    adapter = FoundationStereoAdapter(**args)
    assert os.environ["HF_HUB_OFFLINE"] == os.environ["HF_DATASETS_OFFLINE"] == "1"
    load_config.assert_called_once_with(str(tmp_path / "cfg.yaml"))
    torch.load.assert_called_once_with(str(checkpoint), map_location="cpu", weights_only=True)
    model.load_state_dict.assert_called_once_with({"weights": "sentinel"}, strict=True)
    model.cuda.assert_called_once_with()
    model.eval.assert_called_once_with()
    assert config.vit_size == (vit_size or "vitl") and config.max_disp == 192
    result = adapter.infer_pair(left, right)
    assert result.shape == (9, 13) and result.dtype == np.float32
    assert np.all(result == 3.5)
    torch.inference_mode.assert_called_once_with()
    torch.autocast.assert_called_once_with(device_type="cuda", dtype=np.float16)
    assert adapter.describe()["model_latency_ms"] == 1.75
    assert adapter.describe()["peak_cuda_memory_bytes"] == 2048
    assert adapter.describe()["valid_iters"] == 7
    assert adapter.describe()["max_disp"] == 192
    adapter.close()
    assert adapter.model is None
    torch.cuda.empty_cache.assert_called_once_with()


def test_parent_and_worker_imports_do_not_import_torch():
    # Fresh interpreter avoids being fooled by the rest of pytest's imported modules.
    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'timm', 'omegaconf', 'core'}:
        raise AssertionError('heavy import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from percepta.backends.foundation_stereo_backend import FoundationStereoBackend
from percepta.backends.foundation_stereo_worker import FoundationStereoAdapter
backend = FoundationStereoBackend(backend_factory='tests.unit.foundation_stereo_support:FakeStereoAdapter')
backend.close()
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=5,
                            env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert result.returncode == 0, result.stderr
