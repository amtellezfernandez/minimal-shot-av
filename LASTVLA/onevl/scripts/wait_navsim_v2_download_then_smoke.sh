#!/usr/bin/env bash
set -euo pipefail

NAVSIM_V2_REPO="${NAVSIM_V2_REPO:-${HOME}/dev/navsim-v2}"
OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${NAVSIM_V2_REPO}/data}"
ONEVL_DIR="${ONEVL_DIR:-${HOME}/dev/minimal-shot-av-codex/LASTVLA/onevl}"
SMOKE_LOG="${SMOKE_LOG:-/tmp/navsim_v2_navhard_smoke.log}"
POLL_SECONDS="${POLL_SECONDS:-60}"

SYNTHETIC_SCENES="${OPENSCENE_DATA_ROOT}/navhard_two_stage/synthetic_scene_pickles"
SYNTHETIC_SENSORS="${OPENSCENE_DATA_ROOT}/navhard_two_stage/sensor_blobs"

echo "[wait] waiting for NAVSIM v2 navhard download to finish"
while pgrep -u "${USER}" -f "download_navhard_two_stage.sh|navsim_v2.2_navhard_two_stage_.*tar.gz|wget .*navsim-v2" >/dev/null; do
  date
  tail -n 4 /tmp/navsim_v2_navhard_download.log 2>/dev/null || true
  sleep "${POLL_SECONDS}"
done

if [[ ! -d "${SYNTHETIC_SCENES}" ]]; then
  echo "[wait] missing synthetic scenes directory: ${SYNTHETIC_SCENES}" >&2
  exit 3
fi
if [[ ! -d "${SYNTHETIC_SENSORS}" ]]; then
  echo "[wait] missing synthetic sensors directory: ${SYNTHETIC_SENSORS}" >&2
  exit 3
fi

echo "[wait] NAVSIM v2 data present"
du -sh "${OPENSCENE_DATA_ROOT}" "${SYNTHETIC_SCENES}" "${SYNTHETIC_SENSORS}" 2>/dev/null || true

echo "[wait] waiting for existing OneVL inference jobs to clear"
while pgrep -u "${USER}" -f "python .*infer_onevl.py" >/dev/null; do
  ps -eo pid,etime,%cpu,%mem,cmd | grep -E "infer_onevl.py" | grep -v grep || true
  sleep "${POLL_SECONDS}"
done

echo "[wait] launching NAVSIM v2 navhard smoke"
cd "${ONEVL_DIR}"
nohup ./scripts/run_navsim_v2_navhard_smoke.sh > "${SMOKE_LOG}" 2>&1 < /dev/null &
echo "[wait] smoke pid=$! log=${SMOKE_LOG}"
