#!/usr/bin/env bash
# Launch DeepSeek-V4.1-Flash-1017 across two 8x RTX 4090 nodes.
#
#   ./launch.sh 0        # node A: driver, hosts the OpenAI API
#   ./launch.sh 1        # node B: headless worker, no HTTP port
#
# Required environment:
#   MASTER_ADDR   IP/hostname of node A (reachable from both)
#   MODEL_PATH    path to the 476 GB checkpoint on the local filesystem
#   IMAGE         vLLM image, e.g. ghcr.io/yhfgyyf/vllm-deepseek-v4-sm89:...
#
# Optional environment: see the block below. Every value can be overridden.
#
# Presets (MODE):
#   longctx  (default) 512K context, fp8_ds_mla KV, exact KV byte cap, no speculation
#   fast                64K context, tries DSpark -- see docs/06-negative-results.md,
#                       DSpark cannot run with PP=2, so this preset will not boot here
set -euo pipefail

NODE_RANK="${1:?usage: launch.sh <node_rank: 0|1>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"

# ---- required ----
# NB: keep apostrophes out of ${VAR:?...} messages -- the word is quote-parsed by bash.
MASTER_ADDR="${MASTER_ADDR:?set MASTER_ADDR to the IP of node A}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the local checkpoint path}"
IMAGE="${IMAGE:?set IMAGE to the vLLM image}"

# ---- topology ----
NAME="${NAME:-dsv41}"
MASTER_PORT="${MASTER_PORT:-29501}"   # NCCL/gloo rendezvous, unrelated to HTTP
PORT="${PORT:-9000}"                  # OpenAI API, only bound on node A
IFACE="${IFACE:-bond0}"               # NIC used by NCCL/gloo
TP="${TP:-8}"                         # <= 8: the kernel rejects 16 (4 local Q heads)
PP="${PP:-2}"                         # mandatory: 286 GiB of weights > 192 GB per node
MODE="${MODE:-longctx}"

# ---- files ----
PATCH_DIR="${PATCH_DIR:-$ROOT/patched}"    # built by apply-patches.sh
CACHE_DIR="${CACHE_DIR:-$ROOT/.cache}"     # vllm/flashinfer/torch/tilelang compile cache

# ---- serving ----
SNAME="${SNAME:-deepseek-v4.1-flash-1017}"   # space separated if you serve several names
SP=/opt/venv/lib/python3.12/site-packages

case "${MODE}" in
  longctx)
    MAXLEN="${MAXLEN:-524288}"
    MAXSEQS="${MAXSEQS:-1}"       # no headroom for a second KV stream
    MAXBATCH="${MAXBATCH:-1024}"  # 2048 doubles activations and OOMs at 100K prefill
    GMEM="${GMEM:-0.97}"
    KVDTYPE="${KVDTYPE:-fp8_ds_mla}"
    KVB="${KVB:-560M}"            # exact cap; the cap is what frees room for activations
    # Speculative decoding is OFF: it is free in VRAM but loses 22-57% throughput on a
    # 1 Gbps PP link. Set NGRAM=1 to reproduce the experiment. See docs/06.
    if [ "${NGRAM:-0}" = "1" ]; then
      SPEC=(--spec-method ngram --spec-tokens "${SPECT:-5}")
    else
      SPEC=()
    fi
    ;;
  fast)
    MAXLEN="${MAXLEN:-65536}"
    MAXSEQS="${MAXSEQS:-1}"
    MAXBATCH="${MAXBATCH:-1024}"
    GMEM="${GMEM:-0.97}"
    KVDTYPE="${KVDTYPE:-fp8_ds_mla}"
    KVB="${KVB:-}"
    SPEC=(--spec-method dspark --spec-tokens "${SPECT:-5}")
    ;;
  *)
    echo "unknown MODE='${MODE}' (expected longctx|fast)" >&2
    exit 2
    ;;
esac

# ---- assemble ----
EXTRA=()
[ "${NODE_RANK}" != "0" ] && EXTRA+=(--headless)

KVARGS=()
[ -n "${KVB}" ] && KVARGS+=(--kv-cache-memory-bytes "${KVB}")

read -ra SNAMES <<< "${SNAME}"

# Bind-mount each patched file over the image's copy. Missing entries are skipped so you
# can run with a subset (0005 is optional).
PATCH_MOUNTS=()
n_patches=0
while IFS= read -r rel; do
  if [ -f "${PATCH_DIR}/${rel}" ]; then
    PATCH_MOUNTS+=(-v "${PATCH_DIR}/${rel}:${SP}/${rel}:ro")
    n_patches=$((n_patches + 1))
  fi
done <<'PATCHLIST'
vllm/models/deepseek_v4_1/common/engram.py
vllm/models/deepseek_v4/nvidia/model.py
vllm/models/deepseek_v4_1/nvidia/router.py
vllm/v1/worker/utils.py
vllm/v1/worker/gpu_model_runner.py
PATCHLIST

if [ "${n_patches}" -eq 0 ]; then
  echo "WARNING: no patched files found in ${PATCH_DIR}." >&2
  echo "         Run ./scripts/apply-patches.sh first, or the engine will not boot." >&2
fi

mkdir -p "${CACHE_DIR}" "${CACHE_DIR%/}/tilelang"

echo "MODE=${MODE} rank=${NODE_RANK} port=${PORT} len=${MAXLEN} seqs=${MAXSEQS}" \
     "batch=${MAXBATCH} kv=${KVDTYPE} kvbytes=${KVB:-auto}" \
     "spec=${SPEC[*]:-none} names=${SNAMES[*]} patches=${n_patches}"

CMD=(docker run -d --name "${NAME}"
  --gpus all --net=host --ipc=host
  --shm-size 64g
  --ulimit memlock=-1 --ulimit stack=67108864
  --cap-add IPC_LOCK
  -v "${MODEL_PATH}:/model:ro"
  "${PATCH_MOUNTS[@]}"
  -v "${CACHE_DIR}:/root/.cache"
  -v "${CACHE_DIR%/}/tilelang:/root/.tilelang"
  -e "NCCL_SOCKET_IFNAME=${IFACE}"
  -e "GLOO_SOCKET_IFNAME=${IFACE}"
  -e NCCL_IB_DISABLE=1
  -e NCCL_DEBUG=WARN
  -e VLLM_LOGGING_LEVEL=INFO
  -e OMP_NUM_THREADS=8
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
  "${IMAGE}"
  /model
    --port "${PORT}"
    --served-model-name "${SNAMES[@]}"
    --trust-remote-code
    --tensor-parallel-size "${TP}"
    --pipeline-parallel-size "${PP}"
    --distributed-executor-backend mp
    --nnodes 2
    --node-rank "${NODE_RANK}"
    --master-addr "${MASTER_ADDR}"
    --master-port "${MASTER_PORT}"
    --max-model-len "${MAXLEN}"
    --max-num-seqs "${MAXSEQS}"
    --max-num-batched-tokens "${MAXBATCH}"
    --gpu-memory-utilization "${GMEM}"
    --kv-cache-dtype "${KVDTYPE}"
    "${KVARGS[@]}"
    --reasoning-parser deepseek_v41
    --enable-auto-tool-choice
    --tool-call-parser deepseek_v41
    --engram-config '{"cpu_offload": true}'
    "${SPEC[@]}"
    "${EXTRA[@]}")

# DRY_RUN=1 prints the exact argv and starts nothing -- use it to diff two configs.
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "--- docker argv (DRY_RUN=1, nothing started) ---"
  printf ' %q' "${CMD[@]}"
  echo
  exit 0
fi

# Only now is it safe to touch the existing container.
docker rm -f "${NAME}" >/dev/null 2>&1 || true
"${CMD[@]}"

echo "started ${NAME} on $(hostname) rank=${NODE_RANK} mode=${MODE}"
echo "watch:  docker logs -f ${NAME}"
echo "ready:  curl -sf -o /dev/null -w '%{http_code}\\n' http://127.0.0.1:${PORT}/v1/models"
