# Patches

Five unified diffs against the vLLM build baked into
`ghcr.io/yhfgyyf/vllm-deepseek-v4-sm89:0.28.1rc1-vision11-sm89-sm120-cu130`
(`vllm 0.28.1rc1.dev517+glm53.dsv41.vision11.sm89sm120.cu130`).

Paths in the diffs are relative to `site-packages`
(`/opt/venv/lib/python3.12/site-packages`), so `patch -p1` from a directory that mirrors
that layout applies them exactly. `../scripts/apply-patches.sh` does this for you by
extracting the pristine files from the image with `docker cp`, so you never need to rebuild.

## Verification status

All five were verified by applying the diff to the pristine file extracted from the image
and comparing the result **byte-for-byte** against the file running in production:

```
0001-engram-exact-pinned-alloc.patch     APPLY-OK  (identical to patched file)
0002-dummy-input-ids-moe-model.patch     APPLY-OK  (identical to patched file)
0003-dummy-input-ids-router.patch        APPLY-OK  (identical to patched file)
0004-kvcache-skip-orphan-group.patch     APPLY-OK  (identical to patched file)
0005-gpu-model-runner-ngram-pp.patch     APPLY-OK  (identical to patched file)
```

## Required vs optional

| | Patch | Required for boot? |
|---|---|---|
| 0001 | `engram.py` — exact pinned allocation | **Yes.** Without it the host OOMs during weight load. |
| 0002 | `deepseek_v4/nvidia/model.py` — dummy `input_ids` | **Yes.** Fails during memory profiling. |
| 0003 | `deepseek_v4_1/nvidia/router.py` — dummy `input_ids` | **Yes.** Same, V4.1 router. |
| 0004 | `v1/worker/utils.py` — skip orphan KV group | **Yes.** `StopIteration` during KV allocation, empty error message. |
| 0005 | `v1/worker/gpu_model_runner.py` — two latent bugs | No. Only reachable with `--spec-method ngram`, which is off. |

Keep 0005 anyway if you intend to experiment with speculative decoding — it fixes real bugs
(a missed attribute rename and a `self.drafter` that only exists on the last PP rank).

## What each patch does, in one line

- **0001** — PyTorch's pinned allocator rounds every request up to a power of two; an
  11.44 GiB Engram shard reserves 16 GiB, and 16 of them demand 264 GiB on a 251 GiB host.
  Raw `cudaHostAlloc` is exact and still UVA-addressable. Also stops `_storage()` from
  silently re-copying the tables (they lack the pinned flag, so the original code copied
  11 GiB per lookup).
- **0002 / 0003** — the VL MoE routing bias requires `input_ids`, but profiling and CUDA
  graph dummy runs pass `None`. Synthesise all-zero ids for that case only.
- **0004** — the last PP stage also owns draft-layer KV specs that belong to no scheduler
  group, so `next(...)` raises. Warn and skip instead.
- **0005** — `InputBatch.num_computed_tokens` was renamed to `num_computed_tokens_cpu` in
  this fork but one call site kept the old name; and `self.drafter` is only created on the
  last PP rank yet dereferenced unconditionally.

Root cause detail, with source excerpts, is in [`../docs/02-patches.md`](../docs/02-patches.md).

## If a patch does not apply

The hunks carry generous context. Line numbers move between builds, but the anchors are
stable. In order:

```bash
# 1. confirm you have the right image
docker inspect dsv41 --format '{{.Image}}'   # or: docker exec dsv41 python3 -c 'import vllm;print(vllm.__version__)'

# 2. retry with fuzz
( cd patched && patch -p1 --fuzz=2 -i ../patches/0001-engram-exact-pinned-alloc.patch )

# 3. read the hunk and hand-apply -- they are small; 0001 is the only one with real logic
```

Do not ship a silently fuzzed patch. Re-read the resulting code against
`../docs/02-patches.md` and confirm the intent survived.

## Licensing

These diffs are derivative works of [vLLM](https://github.com/vllm-project/vllm), which is
Apache-2.0. See [`../NOTICE`](../NOTICE). They contain no model weights.
