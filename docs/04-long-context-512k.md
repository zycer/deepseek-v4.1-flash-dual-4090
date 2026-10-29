# 04 · 512K context — where the memory actually goes

This is the most counter-intuitive part of the whole deployment, and the one that took
the longest to get right. The short version:

> **Long-context prefill is limited by activation memory, not by KV cache space.**
> Giving KV *less* is what lets you have *more* context.

---

## The budget on one card

| Line item | GiB |
|---|---|
| Nominal HBM per RTX 4090 | 23.99 |
| CUDA context + driver reservation | −1.1 |
| **Usable by the vLLM worker** | **22.9** |
| Model weights (PP0 shard) | −21.68 |
| **Remaining for KV + activations + graphs** | **≈1.22** |

1.22 GiB is the entire working budget. Every decision below is about how to split it.

---

## KV is small; activations are not

With `--kv-cache-dtype fp8_ds_mla`, measured cost is **≈900 B/token/worker**:

| | |
|---|---|
| Cost per token | ≈900 B |
| 524,288 tokens | ≈450 MiB = 0.44 GiB |
| `--kv-cache-memory-bytes 560M` yields | **651,867 tokens** (1.24× the 512K request) |

So 512K of KV costs only 0.44 GiB — about a third of the budget. It looks like there is
room to spare.

The naive move is to let `--gpu-memory-utilization 0.97` decide. Here is what happens:

| Config | KV allocated | Free for activations | 32K prefill |
|---|---|---|---|
| `--gpu-memory-utilization 0.97` | 0.79 GiB | ~0.01 GiB | ❌ OOM |
| `--kv-cache-memory-bytes 560M` | 0.56 GiB | ~0.24 GiB | ✅ |
| `--gpu-memory-utilization 0.94` | 0.15 GiB | ~0.65 GiB | ❌ (different failure) |

With a memory-utilization target, vLLM gives **everything left over** to KV, because that
is normally the right call: KV is what limits concurrency. Here the leftover is 1.2 GiB and
KV greedily takes 0.79 of it, leaving activations nothing.

### The actual OOM

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 16.00 MiB.
  ... in project_query_and_cache_kv()
```

A **16 MiB** allocation fails. That is the whole story: an allocation three orders of
magnitude smaller than the model. Chasing it with `--gpu-memory-utilization` makes it
worse, which is why this burns so much time — the flag that *seems* to control memory
makes the symptom worse, not better.

---

## Why activations scale with sequence length here

For most models, prefill activation is dominated by `max_num_batched_tokens × hidden_size`,
which is a constant. This model has two extra length-dependent consumers:

1. **DSA compressed sparse attention** builds per-request index structures whose size
   tracks the sequence length. `index_source_layer_ids` means those structures are shared
   across layers, but they are still O(n) per request.
2. **Chunked prefill workspace.** With `enable_chunked_prefill=True` the prefill is split
   into `--max-num-batched-tokens` chunks, and each chunk's attention must read the whole
   prefix built so far. The workspace for that is O(prefix) and does not shrink.

That is the mechanism behind the `--max-num-batched-tokens` rule below.

---

## The `1024` vs `2048` cliff

```
--max-num-batched-tokens 2048  →  100K prefill OOM at ~32K
--max-num-batched-tokens 1024  →  498K prefill succeeds
```

Doubling the chunk size doubles the activation working set of the chunk while the O(prefix)
part is unchanged — and the former was already the binding constraint. There is no upside
to 2048 at this memory level: prefill throughput measured 1.6–2.2k tok/s either way, since
it is bandwidth-bound on 16 GPUs, not chunk-size-bound.

> **`--max-num-batched-tokens 1024` is not a preference. It is a hard requirement.**

### A failure mode that looks like success

When prefill OOMs, the CUDA error takes down `EngineCore`, but the HTTP server process is a
sibling, so the container stays `Up`:

```
$ docker ps --filter name=dsv41 --format '{{.Status}}'
Up 6 minutes                    # looks fine

$ curl -s -o /dev/null -w '%{http_code}' localhost:9000/v1/models
000                             # actually dead
```

**Always health-check with `/v1/models`, never with `docker ps`.** Recovering requires a
full engine restart (8–10 min cold). Because there is no retry, tune in small steps:
32K → 100K → 250K → 500K.

---

## Context length ceiling

`--max-model-len 524288` is a choice, not the limit. Native `max_position_embeddings`
supports 1M with yarn factor 16 × 65536, and the sparse attention backend handles it.

Capacity ceiling given the 560M KV cap is 651,867 tokens. To go further:

| Target context | KV cap needed | Notes |
|---|---|---|
| 512K (current) | `560M` | 1.24× headroom |
| 640K | `~700M` | prefill >6 min; activations get ~0.1 GiB, risky |
| 1M | `~1.1G` | not viable — exceeds the 1.22 GiB budget entirely |

Pushing past ~640K requires either a second 4090 node (more PP stages) or
`--kv-cache-dtype` beyond FP8, which this backend does not offer. **512K is the practical
ceiling on this hardware**, which is why it was chosen over 1M.

---

## Verification procedure

Use `bench/longctx_needle.py`, which buries a secret at the midpoint of a filler document
and asks for it back. Recall proves the full prefix was actually attended to, not truncated.

```bash
python3 bench/longctx_needle.py 32000 100000 500000
```

Measured on this deployment:

| prompt_tokens | TTFT | prefill tok/s | decode tok/s | recall |
|---|---|---|---|---|
| 18,621 | 11.4 s | 1,630 | 56.9 | ✅ |
| 101,306 | 46.4 s | 2,183 | 58.6 | ✅ |
| 498,846 | 286.4 s | 1,742 | 66.0 | ✅ |

Two things worth noting from that table:

- **Decode gets *faster* at long context** (56.9 → 66.0 tok/s). This is DSA working as
  intended: sparse attention reads a bounded subset of the prefix, so step cost is roughly
  flat while the measurement window absorbs prefill tail effects.
- **Prefill is ~1.7k tok/s regardless of length.** Linear, no degradation — a good sign
  that chunked prefill is doing its job and nothing is spilling to host.

For 500K, TTFT is ~4.8 minutes. If that matters to you, `--enable-prefix-caching` (already
on) is the effective mitigation: repeat requests against the same document get near-zero
TTFT because the KV blocks are reused. For agent workloads that re-read a large context,
this is worth more than any speculative decoding possibly could be.
