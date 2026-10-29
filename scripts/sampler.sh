#!/usr/bin/env bash
# Append a memory line to LOG every INTERVAL seconds. Run it before launch and leave it
# running: when a worker disappears, this file is the only evidence of why.
#
#   ./sampler.sh /var/log/dsv4-mem.log &
#
# Columns: MemTotal, MemAvailable, Cached, Shmem, and per-GPU used MiB.
#   - Shmem plateauing near 190 GiB while MemAvailable stays >25 GiB = healthy
#   - MemAvailable falling monotonically with Cached rising        = OOM incoming
set -uo pipefail

LOG="${1:-/tmp/dsv4-mem.log}"
INTERVAL="${INTERVAL:-3}"

echo "# $(date '+%F %T') sampler start on $(hostname), interval=${INTERVAL}s" >> "${LOG}"

while true; do
  mem=$(awk '/^MemTotal:|^MemAvailable:|^Cached:|^Shmem:/{printf "%s=%.0fMB ", $1, $2/1024}' /proc/meminfo)
  gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd, -)
  echo "$(date '+%F %T') ${mem}| gpu_used=${gpu}" >> "${LOG}"
  sleep "${INTERVAL}"
done
