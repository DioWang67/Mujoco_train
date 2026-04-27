#!/usr/bin/env bash
set -eu
set -o pipefail

REMOTE_ROOT="${1:-/root/anaconda3/mujoco-train-system}"
PROJECT_SLUG="${2:-h1}"
JOB_NAME="${3:-$PROJECT_SLUG}"
PORT="${4:-6006}"

LOGDIR="${REMOTE_ROOT}/runs/${PROJECT_SLUG}/logs/tb/${JOB_NAME}"
REMOTE_LOG="/tmp/tb_${PROJECT_SLUG}_${JOB_NAME}_${PORT}.log"

pick_python() {
  local candidates=()
  if [[ -n "${MUJOCO_REMOTE_PYTHON:-}" ]]; then
    candidates+=("${MUJOCO_REMOTE_PYTHON}")
  fi
  candidates+=(
    "/root/anaconda3/bin/python3"
    "/root/anaconda3/bin/python"
    "python3"
    "python"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if ! command -v "${candidate}" >/dev/null 2>&1 && [[ ! -x "${candidate}" ]]; then
      continue
    fi
    if "${candidate}" -c "import tensorboard" >/dev/null 2>&1; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

mkdir -p "${LOGDIR}"
pkill -f "tensorboard.*--port ${PORT}" >/dev/null 2>&1 || true

PYTHON_BIN="$(pick_python || true)"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "No usable Python with tensorboard installed was found." >&2
  exit 1
fi

nohup "${PYTHON_BIN}" -m tensorboard.main \
  --logdir "${LOGDIR}" \
  --port "${PORT}" \
  --host 127.0.0.1 \
  > "${REMOTE_LOG}" 2>&1 < /dev/null &
TB_PID="$!"

for _ in $(seq 1 30); do
  if bash -lc "echo > /dev/tcp/127.0.0.1/${PORT}" >/dev/null 2>&1; then
    echo "TensorBoard ready at 127.0.0.1:${PORT}"
    echo "Remote log: ${REMOTE_LOG}"
    exit 0
  fi

  if ! kill -0 "${TB_PID}" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo "TensorBoard failed on remote." >&2
echo "TensorBoard PID: ${TB_PID}" >&2
if ps -p "${TB_PID}" >/dev/null 2>&1; then
  echo "TensorBoard process is still running but port ${PORT} did not become reachable." >&2
else
  echo "TensorBoard process exited before port ${PORT} became reachable." >&2
fi
if [[ -f "${REMOTE_LOG}" ]]; then
  tail -n 80 "${REMOTE_LOG}" >&2
else
  echo "Remote log not created: ${REMOTE_LOG}" >&2
fi
exit 1
