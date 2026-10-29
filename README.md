# DeepSeek-V4.1-Flash-1017 on 2 × 8 × RTX 4090

![GPUs](https://img.shields.io/badge/GPUs-16_×_RTX_4090_24GB-informational)
![Host RAM](https://img.shields.io/badge/host_RAM-2_×_251_GiB-informational)
![Context](https://img.shields.io/badge/context-512K_tokens-success)
![vLLM](https://img.shields.io/badge/vLLM-0.28.1rc1_sm89_cu130-blueviolet)
![License](https://img.shields.io/badge/license-MIT_%2B_Apache--2.0-blue)

Serving DeepSeek-V4.1-Flash-1017, a 475 GiB MoE checkpoint, on two machines with
8× RTX 4090 (24 GB) each, at a 512K-token context. The two boxes are connected by a
plain 2×1 Gbps Ethernet bond: no InfiniBand, no NVLink, no NVSwitch.

Every number below was measured on real hardware. The approaches that failed are
written down too, so nobody has to redo them.

> When this was written, there was no public record of the model running on 4090s.
> Everything available assumed 8×H100/H200 with NVLink and 1–2 TB of host RAM. Once
> we started, it turned out there is only one workable configuration, and several
> default behaviours in PyTorch and vLLM work directly against you at this scale.
> This repo is that write-up.

---

## Numbers

| | |
|---|---|
| Checkpoint on disk | 476 GB (48 safetensors shards) |
| Parallelism | TP=8 × PP=2, `world_size=16`, one engine, `mp` executor |
| Max context | 524,288 tokens |
| KV cache capacity | 651,867 tokens (`fp8_ds_mla`) |
| Prefill throughput | 1.6k–2.2k tok/s |
| Decode throughput | 56–66 tok/s (single stream) |
| Long-context recall | needle found at 498,846 tokens |
| Steady-state GPU memory | 23.9 / 24.0 GB per card |
| Cold start | ~5m20s (warm page cache: ~1 min) |
| Uptime | 44 h continuous, no restart |

End-to-end latency, single stream, `temperature=0`:

| prompt_tokens | TTFT | prefill tok/s | decode tok/s | needle recall |
|---|---|---|---|---|
| 18,621 | 11.4 s | 1,630 | 56.9 | yes |
| 101,306 | 46.4 s | 2,183 | 58.6 | yes |
| **498,846** | **286.4 s** | 1,742 | 66.0 | yes |

---

## Hardware

| | Node A (driver) | Node B (worker) |
|---|---|---|
| GPUs | 8 × RTX 4090 24 GB | 8 × RTX 4090 24 GB |
| GPU total | 192 GB | 192 GB |
| CPU | 96 threads | 96 threads |
| Host RAM | 251 GiB, no swap | 251 GiB |
| NVIDIA driver | 595.84 | 595.84 |
| Interconnect | 2 × 1 Gbps bond (~250 MB/s) | same link |

That last row is the main design input. It rules out tensor parallelism across nodes,
and that in turn leaves exactly one layout.

---

## Why this configuration is forced

Each step is a hard constraint, not a tuning preference. The full derivation, with the
error messages, is in [`docs/01-constraint-derivation.md`](docs/01-constraint-derivation.md).

```
475 GiB checkpoint
├── 189.1 GiB  Engram n-gram tables (layers 1, 14)  → pinned host memory, UVA lookups
├── 286.1 GiB  dense + FP4 expert weights           → GPU HBM
└──   7.39 GiB MTP/DSpark draft layers              → never loaded (see below)
```

1. **Two machines are mandatory.** 286 GiB of weights do not fit in 192 GB of HBM, and
   there is no quantisation escape because the experts are already FP4.
2. **`TP=16` is illegal.** `num_attention_heads=64`, and the native attention kernel
   accepts only 8, 16 or 32 local query heads. At TP=16 each rank keeps 4:
   `ValueError: DeepSeek V4.1 native attention supports 8, 16, or 32 local query heads, got 4.`
3. **So `TP=8 × PP=2`.** TP=8 gives 8 local heads, which is legal. PP=2 splits the 40
   layers 20/20, and that boundary is safe: `kv_source_layer_ids=[2,8,14,20]` and
   `candidate_source_layer_id=20`, so cross-layer KV reuse never crosses a stage.
4. **PP across the wire, not TP.** PP sends activations once per forward pass; TP would
   all-reduce on every layer. At 250 MB/s that difference is too large to absorb.
5. **`--kv-cache-dtype fp8_ds_mla` is required.** It is the only reason 512K fits at all.
   bf16 KV tops out at 187,974 tokens, a 3.5× shortfall.

---

## The four patches you will need

The image's DeepSeek-V4 support does not boot on 24 GB cards or 256 GiB hosts. All four
patches are in [`patches/`](patches/) as `git apply`-able unified diffs, verified to apply
cleanly and to produce files byte-identical to what runs in production.

| # | Target | Problem | Fix |
|---|---|---|---|
| 0001 | `deepseek_v4_1/common/engram.py` | PyTorch's pinned allocator rounds up to the next power of two. An 11.44 GiB Engram shard reserves 16 GiB, so 8 ranks × 2 layers waste 73 GiB and the host runs out of memory. | Allocate with raw `cudaHostAlloc` (exact, still UVA-addressable) |
| 0002 | `deepseek_v4/nvidia/model.py` | VL MoE routing bias demands `input_ids`; memory-profiling and CUDA-graph dummy runs have none, so it raises `ValueError`. | Synthesise all-zero token ids for dummy runs only |
| 0003 | `deepseek_v4_1/nvidia/router.py` | Same branch in the V4.1 router. | Same |
| 0004 | `v1/worker/utils.py` | The last PP stage also owns MTP draft-layer KV specs that belong to no scheduler group, so `next(...)` raises `StopIteration`. This shows up as a silent boot failure. | Warn and skip instead of raising |

A fifth patch, `0005`, fixes two latent bugs in `gpu_model_runner.py` that only trigger
under speculative decoding. It is optional and not needed for the shipped config.

Apply them with [`scripts/apply-patches.sh`](scripts/apply-patches.sh). It extracts the
target files from the image, patches them, and writes an overlay directory that
`launch.sh` bind-mounts. No image rebuild.

---

## Things that cost us time

### 1. The long-context limit is activation memory, not KV cache

This one is counter-intuitive. Weights take 21.68 GiB of each 24 GB card, leaving about
1.2 GiB for KV, activations and CUDA graphs. `fp8_ds_mla` KV is only ~1 KB/token, so 512K
tokens needs ~0.54 GiB, which is small. But if you let `--gpu-memory-utilization` fill the
card, KV takes 0.79 GiB and activations get a few tens of MB, and then a 32K prefill OOMs
inside `project_query_and_cache_kv()` on a 16 MiB allocation.

The fix is backwards: cap KV with `--kv-cache-memory-bytes 560M` and leave ~0.6 GiB for
activations. Giving KV less is what buys you more context. See
[`docs/04-long-context-512k.md`](docs/04-long-context-512k.md).

### 2. `--max-num-batched-tokens` must stay at 1024

Doubling it doubles activation memory. A 100K prefill then OOMs and takes the whole
`EngineCore` down. The awkward part is that the container stays `Up` while the API is
dead, which looks like nothing happened. Check `/v1/models`, not `docker ps`.

### 3. Engram's pinned tables land in `Shmem` and cannot be reclaimed without swap

189 GiB of pinned host memory is counted as `Shmem` in `/proc/meminfo`, and node A has no
swap, so it is never reclaimed. Add the page cache from streaming 476 GB of safetensors and
the kernel OOM killer fires, which looks like a worker silently disappearing. You need a
1 Hz watchdog that drops clean page cache (`echo 1 > /proc/sys/vm/drop_caches`) below 60 GiB
available. See [`scripts/memguard.sh`](scripts/memguard.sh).

### 4. DSpark speculative decoding and PP=2 cannot be combined

V4.1's speculative decoding is DSpark, not classic MTP; the `mtp.*` weights are a red
herring. It cannot be enabled here, for a structural reason:

- `DSparkDeepseekV4ForCausalLM` does not implement `SupportsPP`
  (`class DSparkDeepseekV4ForCausalLM(nn.Module)`, and `forward()` has no
  `intermediate_tensors`), so `draft_model_config.verify_with_parallel_config()` raises
  `NotImplementedError: Pipeline parallelism is not supported for this model.`
- `draft_parallel_config.pipeline_parallel_size` is hardcoded from the target config, so
  there is no flag to run the draft at PP=1.
- PP=2 is itself mandatory (see above).

Also, `grep -nE 'pp_group|PPMissingLayer|intermediate_tensors' dspark.py` returns nothing.
There is no pipeline-parallel code in it at all. That is an authoring decision, not a
missing flag. Details in [`docs/06-negative-results.md`](docs/06-negative-results.md).

### 5. `ngram` speculative decoding makes things slower here

It really is free in GPU memory: pure CPU `numpy`+`numba`, `additional_slots=0`, and KV
capacity even rises to 683,749 tokens. It still loses throughput everywhere:

| workload | ngram off | ngram on | delta |
|---|---|---|---|
| pure generation | 30.0 tok/s | 23.3 tok/s | −22% |
| verbatim restatement (best case) | 37.0 tok/s | 21.8 tok/s | −41% |
| extractive summarisation | 47.0 tok/s | 20.4 tok/s | −57% |

Acceptance rate was 16.6% (75 / 451, K=5, about 0.8 tokens/step). Two causes add up: each
verify step pushes K+1=6 tokens' activations across a 1 Gbps link, and ngram forces a
fallback from model runner V2 to V1. Even the workload it is designed for lost 41%. The
shipped config has it off.

---

## Quick start

```bash
# 0. On both nodes: 8 free GPUs, the image, and the 476 GB checkpoint
export MASTER_ADDR=10.0.0.1          # node A
export WORKER_ADDR=10.0.0.2          # node B
export MODEL_PATH=/models/DeepSeek-V4.1-Flash-1017
export IMAGE=ghcr.io/yhfgyyf/vllm-deepseek-v4-sm89:0.28.1rc1-vision11-sm89-sm120-cu130

# 1. Build the patch overlay (once, on both nodes)
./scripts/apply-patches.sh

#    preview the exact docker argv first -- starts nothing
DRY_RUN=1 ./scripts/launch.sh 0

# 2. Start the host-memory watchdog on BOTH nodes (node A will OOM without it)
sudo nohup ./scripts/memguard.sh >/dev/null 2>&1 &

# 3. Start node A first (it is the master), then node B
./scripts/launch.sh 0
ssh $WORKER_ADDR ./scripts/launch.sh 1     # auto-adds --headless

# 4. Wait for readiness (~5m20s cold)
until curl -sf -o /dev/null http://$MASTER_ADDR:9000/v1/models; do sleep 10; done

# 5. Verify
./bench/smoke_test.py
./bench/longctx_needle.py 32000 100000
```

Call it:

```bash
curl http://10.0.0.1:9000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash-1017",
       "messages":[{"role":"user","content":"hello"}],
       "max_tokens":256}'
```

### Reasoning and tool calls

`--reasoning-parser deepseek_v41` and `--tool-call-parser deepseek_v41` are both enabled.

```jsonc
{
  "message": {
    "content":   "final answer only",
    "reasoning": "chain of thought",        // not "reasoning_content"
    "tool_calls": [ /* finish_reason = "tool_calls" */ ]
  }
}
```

The field is `reasoning`, not `reasoning_content` as in most other vLLM builds. If your
client reads `reasoning_content`, change that one line.

---

## Repository layout

```
.
├── README.md                  # this file
├── README.zh-CN.md            # Chinese version
├── docs/
│   ├── 01-constraint-derivation.md   # why TP=8×PP=2, with the error messages
│   ├── 02-patches.md                 # root cause of each patch
│   ├── 03-host-memory.md             # Shmem, pinned pages, no-swap OOM, memguard
│   ├── 04-long-context-512k.md       # activation vs KV budget arithmetic
│   ├── 05-benchmarks.md              # measurement method and raw numbers
│   ├── 06-negative-results.md        # DSpark, ngram, MTP results
│   └── 07-troubleshooting.md         # symptom → cause → fix
├── patches/                   # 5 verified unified diffs
├── scripts/
│   ├── launch.sh              # parametrised launcher, longctx | fast presets
│   ├── apply-patches.sh       # image → overlay dir, no rebuild
│   ├── memguard.sh            # page-cache watchdog (always run on node A)
│   ├── stop.sh
│   └── sampler.sh             # 3 s memory sampler for post-mortems
├── bench/
│   ├── smoke_test.py          # models / non-stream / stream / tools / reasoning
│   ├── longctx_needle.py      # needle-in-a-haystack with TTFT and decode timing
│   └── ngram_ab.py            # 3-workload A/B for speculative decoding
└── tools/
    ├── probe_config.py        # dump parsers, KV dtypes, spec methods, model config
    └── model_weight_report.py # per-layer / per-shard weight accounting
```

---

## Limitations

- **One concurrent sequence.** `--max-num-seqs 1`, and it is genuinely serial: four
  concurrent requests take about 4× the wall time and aggregate throughput does not rise
  (52.9 vs 55.5 tok/s measured). With ~1.2 GiB of headroom there is no room for a second
  KV stream plus its CUDA graphs. Concurrency here buys fairness, not throughput. See
  [`docs/05-benchmarks.md`](docs/05-benchmarks.md#concurrency-it-does-not-scale-and-that-is-expected).
- **`pp=2` costs about 2× decode latency** versus a hypothetical single-node TP=8, because
  every layer pair crosses the wire. At 250 MB/s that is unavoidable. A 10 GbE link is the
  upgrade with the best return.
- **`--kv-cache-memory-bytes` disables memory profiling**, so `--gpu-memory-utilization`
  is silently ignored. The log says so: *"skipped memory profiling, does not respect
  gpu_memory_utilization"*. Do not tune util while a KV byte cap is set.
- **No process supervision.** `restart=no`, no systemd unit. After a reboot you have to
  bring things up by hand in the order memguard → node A → node B. See
  [`docs/07-troubleshooting.md`](docs/07-troubleshooting.md#recovery-after-reboot).
- **One image verified.** The official `vllm/vllm-openai:v0.32.0` does not have a
  `DeepseekV4` architecture in its registry at all. This is built against the community
  SM89 build above; porting the patches to a newer vLLM means re-locating the hunks.

---

## Contributing

If you get this running on other hardware — 3090s, A6000s, a faster interconnect, a newer
vLLM — please open an issue with your numbers. The most useful variable is interconnect
bandwidth, since it decides whether speculative decoding helps or hurts.

## License

Scripts and documentation: MIT.

Contents of `patches/` are modifications of [vLLM](https://github.com/vllm-project/vllm)
(Apache-2.0); see [`NOTICE`](NOTICE). DeepSeek-V4.1-Flash weights are under their own
license and are not included here.

## Acknowledgements

The SM89/SM120 image and the `fp8_ds_mla` KV layout come from
[`yhfgyyf`](https://github.com/yhfgyyf)'s community build; none of this runs on stock
vLLM. The Engram architecture is DeepSeek's.
