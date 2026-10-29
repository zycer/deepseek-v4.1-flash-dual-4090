#!/usr/bin/env bash
# Host-memory watchdog. MUST run on node A (the node that owns the 189 GiB of pinned
# Engram tables) for the entire lifetime of the deployment. Strongly recommended on node B.
#
#   sudo nohup ./memguard.sh >/dev/null 2>&1 &
#
# Why: cudaHostAlloc pages are accounted as Shmem and cannot be reclaimed when there is
# no swap. Streaming 476 GB of safetensors on top of that trips the kernel OOM killer,
# which kills a VLLM::Worker_PP process with no Python traceback.
#
# `echo 1 > drop_caches` drops clean page cache only: no writeback, no data loss.
# It never touches Shmem, so it cannot undo the Engram allocation either.
#
# Tunables:
#   THRESH_GB  drop cache below this much MemAvailable (default 60)
#   EMERG_GB   also drop dentries/inodes below this (default 15)
#   INTERVAL   poll interval in seconds (default 1 -- do not raise it)
#   LOG        log path (default none; set to a file to record drops)
set -uo pipefail

THRESH_GB="${THRESH_GB:-60}"
EMERG_GB="${EMERG_GB:-15}"
INTERVAL="${INTERVAL:-1}"
LOG="${LOG:-}"

if [ "$(id -u)" -ne 0 ]; then
  echo "error: needs root to write /proc/sys/vm/drop_caches" >&2
  exit 1
fi

if [ -n "${LOG}" ]; then
  echo "$(date '+%F %T') memguard start on $(hostname) thresh=${THRESH_GB}GB interval=${INTERVAL}s" >> "${LOG}"
fi

avail_gb() { awk '/^MemAvailable:/{printf "%d", $2/1048576}' /proc/meminfo; }

n=0
while true; do
  avail=$(avail_gb)
  if [ "${avail}" -lt "${THRESH_GB}" ]; then
    echo 1 > /proc/sys/vm/drop_caches
    [ "${avail}" -lt "${EMERG_GB}" ] && echo 2 > /proc/sys/vm/drop_caches
    n=$((n + 1))
    # Log sparsely: at 1 Hz this fires hundreds of times per minute.
    if [ -n "${LOG}" ] && [ $((n % 60)) -eq 0 ]; then
      echo "$(date '+%F %T') dropped x${n} avail=${avail}GB" >> "${LOG}"
    fi
  fi
  sleep "${INTERVAL}"
done
