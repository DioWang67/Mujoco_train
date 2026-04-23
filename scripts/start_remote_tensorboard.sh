#!/usr/bin/env bash
set -eu
set -o pipefail

REMOTE_ROOT="${1:-/root/anaconda3/mujoco-train-system}"
PROJECT_SLUG="${2:-h1}"
JOB_NAME="${3:-$PROJECT_SLUG}"
PORT="${4:-6006}"

LOGDIR="${REMOTE_ROOT}/runs/${PROJECT_SLUG}/logs/tb/${JOB_NAME}"
REMOTE_LOG="/tmp/tb_${PROJECT_SLUG}_${JOB_NAME}_${PORT}.log"

mkdir -p "${LOGDIR}"
pkill -f "tensorboard.*--port ${PORT}" >/dev/null 2>&1 || true

nohup /root/anaconda3/bin/tensorboard \
  --logdir "${LOGDIR}" \
  --port "${PORT}" \
  --host 127.0.0.1 \
  > "${REMOTE_LOG}" 2>&1 < /dev/null &

sleep 2

if ! bash -lc "echo > /dev/tcp/127.0.0.1/${PORT}" >/dev/null 2>&1; then
  echo "TensorBoard failed on remote." >&2
  if [[ -f "${REMOTE_LOG}" ]]; then
    tail -n 20 "${REMOTE_LOG}" >&2
  else
    echo "Remote log not created: ${REMOTE_LOG}" >&2
  fi
  exit 1
fi

echo "TensorBoard ready at 127.0.0.1:${PORT}"
echo "Remote log: ${REMOTE_LOG}"
