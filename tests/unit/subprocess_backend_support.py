"""Real CPU pipe worker with controllable stalls for generic subprocess deadline tests."""

from __future__ import annotations

import json
import struct
import sys
import time

import numpy as np

from percepta.backends.protocol import (
    PROTOCOL_VERSION,
    RESPONSE_MAGIC,
    ResponseFrame,
    decode_request,
    encode_response,
    read_frame,
    write_frame,
)


def _response_packet(metadata, data=b"\0" * 4):
    blob = metadata if isinstance(metadata, bytes) else json.dumps(metadata).encode("utf-8")
    return struct.pack("<III", RESPONSE_MAGIC, PROTOCOL_VERSION, len(blob)) + blob + data


def _partial_response(response: ResponseFrame) -> None:
    payload = encode_response(response)
    sys.stdout.buffer.write(struct.pack("<I", len(payload)) + payload[:2])
    sys.stdout.buffer.flush()
    time.sleep(60)


def _main() -> None:
    config = json.loads(sys.argv[1])
    mode = config.get("mode", "normal")
    if mode == "malformed_startup":
        write_frame(sys.stdout.buffer, bytes.fromhex(config["payload_hex"]))
        return
    if mode == "startup_hang":
        time.sleep(60)
    if mode == "partial_ready":
        _partial_response(ResponseFrame("ready"))
    write_frame(sys.stdout.buffer, encode_response(ResponseFrame("ready")))
    if mode == "write_hang":
        time.sleep(60)
    image = None
    while True:
        if mode == "delayed_roundtrip":
            header = sys.stdin.buffer.read(4)
            if not header:
                return
            time.sleep(0.12)
            payload = sys.stdin.buffer.read(struct.unpack("<I", header)[0])
        else:
            payload = read_frame(sys.stdin.buffer)
        if payload is None:
            return
        request = decode_request(payload)
        if request.shutdown:
            return
        if mode == "malformed_prepare" and request.image is not None or mode == "malformed_segment" and request.image is None:
            write_frame(sys.stdout.buffer, bytes.fromhex(config["payload_hex"]))
            continue
        if mode == "eof_segment" and request.image is None:
            return
        if request.image is not None:
            if mode == "prepare_hang":
                time.sleep(60)
            if mode == "delayed_roundtrip":
                time.sleep(0.12)
            image = request.image
            response = ResponseFrame("ok")
            if mode == "prepare_error":
                response = ResponseFrame("error", error="model failed", error_kind="unavailable")
        else:
            if mode == "segment_hang":
                time.sleep(60)
            if mode == "partial_response":
                _partial_response(ResponseFrame("ok"))
            if mode == "segment_error":
                response = ResponseFrame("error", error="model failed", error_kind="unavailable")
            elif mode == "invalid_prompt" and json.loads(request.prompt_json).get("text") == "invalid":
                response = ResponseFrame("error", error="bad prompt", error_kind="invalid_request")
            else:
                mask = image[:, :, 0] > 128
                response = ResponseFrame("ok", masks=mask[None], scores=np.array([1.], dtype=np.float32))
        write_frame(sys.stdout.buffer, encode_response(response))


if __name__ == "__main__":
    _main()
