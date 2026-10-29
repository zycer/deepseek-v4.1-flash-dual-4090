# 05 · Benchmarks — method and numbers

All numbers below were measured on the hardware described in the README, on a single
engine serving one request at a time unless stated otherwise. No numbers are projected or
derived; if a figure is an estimate it says so.

---

## Method

**Latency.** Measured client-side over streaming SSE. `TTFT` is the wall-clock time from
`POST` to the **first non-empty delta** (content or reasoning). Decode rate is
`completion_tokens / (total − TTFT)`, using the server's own `usage.completion_tokens`
rather than counting chunks — chunks are not tokens. `usage` is obtained by sending
`stream_options: {"include_usage": true}`.

**Prefill rate.** `prompt_tokens / TTFT`. This is a *lower bound*: TTFT includes
scheduling, the first decode step and a KV-block write, so short prefills read
optimistically high. Treat the 1.7–2.9k tok/s spread in the table as measurement noise
around ~2k, not as a real length effect.

**Recall.** `bench/longctx_needle.py` places a unique secret at the midpoint of a filler
document and asks for it. Recall is a binary HIT/MISS. This is the only check that
distinguishes "the context window is 512K" from "the context window is 512K *and the model
actually attends to all of it*".

**Environment.** `temperature=0`, engine idle before each run, page cache warm after the
first. GPU clocks are not locked; expect ±5% run-to-run.

---

## Single-stream latency vs context length

| prompt_tokens | TTFT | prefill tok/s | decode tok/s | recall |
|---|---|---|---|---|
| 18,621 | 8.68 s | 2,146 | 56.9 | HIT |
| 31,869 | 10.86 s | 2,934 | 56.9 | HIT |
| 57,789 | 25.86 s | 2,235 | 57.8 | HIT |
| 101,306 | 46.4 s | 2,183 | 58.6 | HIT |
| **498,846** | **286.4 s** | 1,742 | **66.0** | HIT |

Two features of this table are worth more than the absolute numbers:

**Decode gets faster as context grows** (56.9 → 66.0 tok/s). That is compressed sparse
attention behaving as designed: the attention read set stays roughly bounded, so per-step
cost does not grow with the prefix. On a dense model you would see decode degrade.

**Prefill is flat at ~2k tok/s** regardless of length. Linear scaling with no knee means
chunked prefill is working and nothing is spilling to host memory. If you see prefill rate
drop at a specific length, you have crossed the activation cliff from
[`04-long-context-512k.md`](04-long-context-512k.md) and are about to OOM.

### Practical read

- 32K is interactive: ~11 s to first token.
- 100K is a coffee break: ~46 s.
- 500K is a batch job: ~4.8 minutes. Not interactive, and not intended to be.

If your workload re-uses the same long document, `--enable-prefix-caching` (on by default)
turns the second and subsequent requests into near-zero TTFT. For agent loops that re-read
a large context this is the single highest-value feature in the whole configuration —
worth more than any speculative decoding could be.

---

## Concurrency: it does not scale, and that is expected

`--max-num-seqs 1` means the scheduler admits one sequence at a time. Four simultaneous
requests therefore **queue**, they do not batch:

| | single stream | 4 concurrent (each 220 tokens) |
|---|---|---|
| wall | 5.59 s (300 tok) | **15.66 s** (829 tok total) |
| aggregate | 55.5 tok/s | **52.9 tok/s** |
| spread | — | 11.5 s between slowest and fastest |

Aggregate throughput is the same as one stream. This is not a bug and not a tuning
problem — it is the budget. With ~1.2 GiB of headroom there is no room for a second KV
stream plus the CUDA graphs it needs (see `04-long-context-512k.md`).

Practical consequences:

- Concurrency is a **latency** feature here, not a throughput feature: N users each get
  1/N of the speed. If you need service-level throughput, run multiple engines on separate
  hardware.
- Do not set `--max-num-seqs 2` expecting a gain without also raising
  `--kv-cache-memory-bytes`; you will instead OOM during CUDA graph capture.
- Earlier experiments on this hardware with `--max-num-seqs 8` and bf16 KV did reach
  ~104 tok/s aggregate at 4-way — the throughput is achievable, it just costs context
  length. That is the trade this deployment deliberately made.

---

## Resource usage in steady state

| | node A (stage 0) | node B (stage 1) |
|---|---|---|
| GPU memory per card | 23,906 MiB | 22,998 MiB |
| GPU memory free per card | ~200 MiB | ~200 MiB |
| `MemAvailable` | ~30 GiB | ~228 GiB |
| `Shmem` (pinned Engram) | ~189 GiB | ~1.6 GiB |

KV cache: `651,867 tokens` per worker. Note that with `kv_source_layer_ids` sharing,
KV is **fully replicated across all 8 TP ranks**, so capacity is a per-card quantity, not
a distributed pool — 651,867 is the number of tokens one card can hold, and every card
holds the same.

Startup timeline (warm page cache; cold adds ~4 minutes of disk read):

```
t+0:00  API server starts
t+0:14  EngineCore init, world_size=16 joins
t+0:33  Engram token map built (129,280 → 99,092, layers 1 and 14)
t+2:55  weights loaded (225 s cold, 30 s warm)
t+3:16  KV cache allocated: 651,867 tokens
t+3:36  CUDA graphs captured (8 s, 0.08 GiB)
t+3:40  engine ready
t+3:43  first request triggers Triton JIT of _engram_lookup_kernel
```

The first request after a start is slow (~8 s instead of 0.7 s) because the Engram lookup
kernel JITs on demand. Warm it before you benchmark, or include the compilation in your
SLO budget.

---

## What limits what

| Bottleneck | Evidence | Remedy |
|---|---|---|
| Cross-node link (250 MB/s) | ngram speculation lost 41% *because* it sextuples cross-node traffic | faster NIC — the highest-value upgrade |
| Activation workspace (~1.2 GiB) | 16 MiB alloc fails at 32K when KV is greedy | `--kv-cache-memory-bytes`, keep batch tokens at 1024 |
| Host RAM on node A (~30 GiB) | `Shmem` 189 GiB, no swap, OOM-killed workers | `memguard.sh`, more RAM |
| Concurrency budget | 4-way = 1-way aggregate | more context ⇒ fewer sequences, by design |
| Not GPU FLOPs | GPU util ~100% during decode, and decode is 57 tok/s | — |

The GPU compute is fully occupied; there is no headroom being wasted. Every remaining
improvement is a bandwidth problem, on one of three buses: the network, the PCIe host
link, or the HBM-to-SM path.

---

## Reproducing

```bash
# latency + recall at several sizes
python3 bench/longctx_needle.py 32000 100000 500000

# functional checks (models, stream, reasoning split, tools, concurrency)
python3 bench/smoke_test.py --concurrency 4

# speculative decoding A/B, run once off and once on
python3 bench/ngram_ab.py --label "off"
``` 

Each script is stdlib-only and takes `--base` / `--model`, so they run from any host with
network access to the driver node — no GPU-side tooling required.
