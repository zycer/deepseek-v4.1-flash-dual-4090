#!/usr/bin/env bash
# Stop the engine on both nodes. Worker first, then driver -- mirrors startup order.
#
#   MASTER_ADDR=... WORKER_ADDR=... ./stop.sh
#
# Set LOCAL_ONLY=1 to only remove the container on this host.
set -uo pipefail

NAME="${NAME:-dsv41}"
LOCAL_ONLY="${LOCAL_ONLY:-0}"

if [ "${LOCAL_ONLY}" = "1" ]; then
  docker rm -f "${NAME}" >/dev/null 2>&1
  echo "stopped ${NAME} on $(hostname)"
  exit 0
fi

: "${MASTER_ADDR:?set MASTER_ADDR}"
: "${WORKER_ADDR:?set WORKER_ADDR}"

for h in "${WORKER_ADDR}" "${MASTER_ADDR}"; do
  if [ "${h}" = "${MASTER_ADDR}" ] && [ "$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -cx "${h}")" -gt 0 ]; then
    docker rm -f "${NAME}" >/dev/null 2>&1 && echo "stopped ${NAME} on local node (${h})"
  else
    ssh -o ConnectTimeout=10 -o BatchMode=yes "root@${h}" \
      "docker rm -f ${NAME} >/dev/null 2>&1; echo stopped ${NAME} on ${h}"
  fi
done
