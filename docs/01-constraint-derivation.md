# 01 · Constraint derivation — why `TP=8 × PP=2` is the only option

This document derives the configuration from first principles and records the exact
error message that enforces each constraint. If you change hardware, re-run this
derivation before changing flags.

---

## The checkpoint

```
476 GB on disk, 48 safetensors shards
```

Byte-level accounting, taken from `model.safetensors.index.json` and from the loaded
state dict (`tools/model_weight_report.py`):

| Component | Size | Where it lives |
|---|---|---|
| Engram n-gram tables (layers 1 and 14) | **189.1 GiB** | pinned host memory, UVA |
| Dense + FP4 expert weights (40 layers) | **286.1 GiB** | GPU HBM |
| MTP / DSpark draft layers (`mtp.*`, 3 stages) | 7.39 GiB | **never loaded** — see `06-negative-results.md` |

The Engram split is the reason this model is "only" 286 GiB of GPU pressure rather than
475 GiB. `--engram-config '{"cpu_offload": true}'` puts the tables in pinned host memory
and the lookup Triton kernel reads them directly over UVA.

---

## Constraint 1 — two machines are mandatory

```
286.1 GiB of weights  >  8 × 24 GB = 192 GB of single-node HBM
```

Ratio: **1.49×**. There is no quantisation escape hatch — routed experts are already
FP4 (`scale_fmt=ue8m0`), dense parts are FP8 block-quantised. Going lower would require
re-quantising the checkpoint, which is a different project.

Also note the 189.1 GiB of Engram must fit in host RAM on whichever node owns layers 1
and 14. With PP=2 splitting 40 layers 20/20, **both layers land in stage 0**, so one node
carries all 189 GiB. That node needs ≥ 230 GiB RAM. This asymmetrical memory profile is
easy to miss and is the root of the memory problems in `03-host-memory.md`.

---

## Constraint 2 — `TP=16` is rejected by the attention kernel

The obvious way to spread 286 GiB over 16 GPUs is tensor parallelism with
`--tensor-parallel-size 16`. It fails immediately:

```
ValueError: DeepSeek V4.1 native attention supports 8, 16, or 32 local query heads, got 4.
```

`num_attention_heads = 64`. Dividing across TP ranks:

| TP | local query heads per rank | legal? |
|---|---|---|
| 2 | 32 | ✅ |
| 4 | 16 | ✅ |
| 8 | 8 | ✅ |
| **16** | **4** | ❌ |
| 32 | 2 | ❌ |

The set {8, 16, 32} is a kernel constraint (warp-level tiling in the native sparse
attention kernel), not a vLLM policy. `TP ≤ 8` is therefore the ceiling on one node.

---

## Constraint 3 — PP must span the node boundary

Given `TP ≤ 8` and 16 GPUs, the remaining axis is pipeline parallelism:
`TP=8 × PP=2`, `world_size = 16`.

The natural question is whether the pipeline cut point is safe. It is, and the config
says why:

```jsonc
"kv_source_layer_ids":      [2, 8, 14, 20],
"index_source_layer_ids":   [2, 8, 14, 20],
"candidate_source_layer_id": 20,
"num_hidden_layers":         40,
```

Layers 2, 8, 14 and 20 are *sources* that later layers reuse KV from (compressed sparse
attention). Cutting at 20/20 puts the last source at the boundary: every consumer is in
the same stage as its source, so no cross-layer KV reuse ever crosses the wire. The
upstream source comments call this the "20+20 layer" architecture — the split was
evidently designed with PP=2 in mind.

**If you ever change `pipeline_parallel_size`, keep the cut point on a
`candidate_source_layer_id` boundary.** Otherwise you get either a hang or wrong
attention outputs, not a clean error.

---

## Constraint 4 — why PP and not TP across nodes

Both nodes are joined by a **2 × 1 Gbps bond**, roughly 250 MB/s, no RDMA, no IB,
`NCCL_IB_DISABLE=1` (NCCL over TCP).

Per decode step, for a batch of 1:

| Strategy | Cross-node traffic per step |
|---|---|
| TP across nodes | one all-reduce of the activation **per layer** — 20 round trips each way |
| PP across nodes | **one** activation handoff at the stage boundary |

PP wins by ~40× on interconnect pressure. This is also why `ngram` speculative decoding
loses here (`06-negative-results.md`): it multiplies the number of forward passes per
step, i.e. it multiplies exactly the quantity this link is worst at.

---

## Constraint 5 — the image is not optional

```bash
docker run --rm vllm/vllm-openai:v0.32.0 python -c "
from vllm.model_executor.models.registry import _VLLM_MODELS
print([k for k in _VLLM_MODELS if 'eepseek' in k])"
```

Stock vLLM has **no `DeepseekV4` architecture registered at all**. The architecture and
its `fp8_ds_mla` KV layout live only in the community SM89 build:

```
ghcr.io/yhfgyyf/vllm-deepseek-v4-sm89:0.28.1rc1-vision11-sm89-sm120-cu130
  → vLLM 0.28.1rc1.dev517+glm53.dsv41.vision11.sm89sm120.cu130
```

It is compiled for SM89 (4090) and SM120 with CUDA 13.0. FP4 experts run through the
MARLIN backend on SM89 — dequantise-on-the-fly inside the GEMM, no FP4→BF16 materialisation
in HBM. That is what makes 286 GiB fit in 192 GB.

---

## Constraint 6 — `fp8_ds_mla` is required for 512K, not a tuning knob

KV cache capacity per worker, measured:

| `--kv-cache-dtype` | capacity | 524,288-token request |
|---|---|---|
| `auto` (bf16) | 187,974 tokens | ❌ needs 2.8× more |
| **`fp8_ds_mla`** | **651,867 tokens** | ✅ 1.24× headroom |

`fp8_ds_mla` is not a generic FP8 KV format — it is a DeepSeek-MLA-specific layout
(FP8 payload plus a shared per-block scale) understood only by the
`DeepseekV41NativeSparseBackend`. `DeepseekV41NativeSparseBackend.supported_kv_cache_dtypes`
includes it and the backend is selected on SM89, which is why this works at all.

Effective cost: **≈1,025 B/token/worker**, so 524,288 tokens ≈ 537 MB, rounded up to a
`560M` cap.

---

## Final configuration

```
--tensor-parallel-size 8       # ≤ 8: 64 Q heads / {8,16,32} local-head constraint
--pipeline-parallel-size 2     # mandatory: 286 GiB > 192 GB single-node HBM
--distributed-executor-backend mp
--nnodes 2 --node-rank 0|1
--max-model-len 524288         # native max is 1M (yarn factor 16 × 65536)
--kv-cache-dtype fp8_ds_mla    # required for 512K
--kv-cache-memory-bytes 560M   # exact cap; leaves ~0.6 GiB for activations
--max-num-seqs 1               # no headroom for a second KV stream
--max-num-batched-tokens 1024  # 2048 OOMs
--engram-config '{"cpu_offload": true}'
```

Anything that deviates from this set failed on real hardware for a reason documented
above. If you have more/faster hardware, `01 → 02 → 03` will re-derive it; the two
numbers to recompute are local-head legality and the PP cut point.
