#!/usr/bin/env bash
# Launch the camera-rig probe against the real RoboDojo simulator.
#
# Kept as a file rather than an inline ssh command so the environment is set exactly once and
# cannot drift between invocations. Nothing here modifies the RoboDojo checkout: the probe lives
# outside it and only reads.
set -uo pipefail

REPO="${ROBODOJO_DIR:-/home/void/work/projects_new/RoboDojo}"
PROBE="${PROBE_SCRIPT:-/home/void/rigprobe_read_camera_rig.py}"
OUT="${PROBE_OUT:-/home/void/rigprobe_out}"
TAG="${1:-baseline}"
TASK="${PROBE_TASK:-stack_blocks}"

cd "$REPO" || { echo "no such repo: $REPO" >&2; exit 1; }

export ROBODOJO_DIR="$REPO"
export PYTHONPATH="$REPO:$REPO/XPolicyLab"
export OMNI_KIT_ACCEPT_EULA=YES
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$OUT"
echo "[run] repo=$REPO"
echo "[run] task=$TASK tag=$TAG out=$OUT"
echo "[run] gpu=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"

exec "$REPO/.venv/bin/python" -u "$PROBE" \
  --repo "$REPO" \
  --task "$TASK" \
  --out "$OUT" \
  --tag "$TAG" \
  --warmup-steps "${PROBE_STEPS:-8}"
