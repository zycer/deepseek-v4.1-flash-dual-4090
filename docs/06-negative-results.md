# 06 · Negative results — what does not work, and why

Everything in this file was measured or read out of the actual source. It is here because
each item cost hours, and because "we tried X and it lost" is information you cannot get
from a success story.

Contents: DSpark speculation, ngram speculation, the MTP weight misconception, and five
dead-end flags.

---

## 1 · DSpark speculative decoding vs PP=2: mutually exclusive

### What DSpark is

DeepSeek-V4.1's speculative decoding is **DSpark**, not classic MTP. The source says so
directly:

> `DeepSeek V4.1 checkpoints contain DSpark stages, not classic MTP. Use speculative method 'dspark'.`

Correct flags are `--spec-method dspark --spec-tokens 5` (`dspark_block_size = 5`, and
`--spec-tokens` must be explicit or you get
`num_speculative_tokens must be provided with speculative model`).

### The failure

```
NotImplementedError: Pipeline parallelism is not supported for this model.
```

Thrown from `vllm/config/speculative.py:1765` →
`draft_model_config.verify_with_parallel_config()` → `config/model.py:1434`:

```python
if pipeline_parallel_size > 1 and not self.registry.is_pp_supported_model(
    self.architectures, self
):
    raise NotImplementedError("Pipeline parallelism is not supported for this model.")
```

`is_pp_supported_model` resolves `model_info.supports_pp`, which `interfaces.py` decides by
two paths — and DSpark satisfies neither:

```python
def _supports_pp_attributes(model):
    return SupportsPP in model.__mro__ or isinstance(model, _SupportsPPType)

def _supports_pp_inspect(model):
    return supports_kw(model_forward, "intermediate_tensors")
```

```python
class DSparkDeepseekV4ForCausalLM(nn.Module):          # path 1: no SupportsPP
    def forward(self, input_ids, positions, inputs_embeds=None):   # path 2: no intermediate_tensors
```

### Why it is not a missing one-line decorator

The decisive evidence is that **there is no pipeline-parallel code in `dspark.py` at all**:

```bash
grep -nE 'pp_group|get_pp_group|PPMissingLayer|intermediate_tensors|make_empty_intermediate|is_first_rank|is_last_rank|pipeline' dspark.py
# (no matches)
```

The layers are built unconditionally on every rank:

```python
self.layers = nn.ModuleList([
    DeepseekV4DecoderLayer(...) for i in range(self.num_dspark_layers)   # always 3
])
```

Compare the main model, which uses `make_layers()` + `PPMissingLayer` so each stage only
materialises the layers it owns. DSpark does the opposite. Add the design premises and the
intent is unmistakable:

```python
class DSparkDeepseekV4ForCausalLM(nn.Module):
    has_own_embed_tokens = False    # borrows the target's
    has_own_lm_head = False         # borrows the target's
```

It is an MTP-style draft head that consumes the target's last hidden states via
`combine_hidden_states(aux_hidden_states)` and hangs off the target's KV via
`get_draft_kv_cache_layer_names()`. Sharing a stage with the target is not an oversight in
its design — it *is* the design.

### Is there a configuration escape?

No. The tempting idea is "run the draft with PP=1 while the target uses PP=2", especially
since the verification call passes `draft_parallel_config` rather than the target's. But:

```python
# speculative.py:1705
draft_parallel_config = ParallelConfig(
    pipeline_parallel_size=target_parallel_config.pipeline_parallel_size,  # hardcoded copy
    tensor_parallel_size=speculative_draft_tensor_parallel_size,           # only TP is tunable
    ...
)
```

`pipeline_parallel_size` is **hardcoded from the target**, and the only draft-side CLI knob
is `--spec-draft-tensor-parallel-size`. There is no draft PP flag to set.

Since PP=2 is mandatory (constraint 1 in `01-constraint-derivation.md`), **DSpark cannot be
enabled on this hardware, at all.** This is a design boundary, not a bug.

### What it would cost even if it worked

Worth recording, because it explains why nobody upstream rushed to implement it:

| Item | Cost |
|---|---|
| Weights | `mtp.*` is 7.39 GiB; ÷2 PP stages ≈ 0.92 GiB/card at TP=8 |
| Available headroom | **~1.2 GiB/card total**, of which KV already uses 0.56 |
| Verdict | would force cutting KV → trading context length for speed |
| Load balance | 3 draft layers over 2 stages = 2+1, so stage 2 idles |
| Cross-node | +1 activation handoff per verify step on a 250 MB/s link |

Marginal at best, and it would have to be paid for with context. The authors' choice not to
support PP here is defensible.

---

## 2 · ngram speculative decoding: free of VRAM, negative on throughput

`--spec-method ngram` is the obvious fallback once DSpark is ruled out: it is model
agnostic, needs no draft weights, and therefore cannot violate the PP constraint.

### It really is free of GPU memory

Confirmed on three axes:

- `NgramProposer` is pure CPU (`numpy` + `numba`); its preallocated buffers are two
  `int32` arrays with 6 and 1 elements — 28 bytes.
- The capability table declares `N-gram | ngram | parallel_drafting=No | additional_slots=0`.
- Measured KV capacity **rose** from 651,867 to 683,749 tokens after enabling it (the V1
  runner it forces has slightly lower overhead).

So the premise was right. The performance was not.

### It is slower on every workload

Three workloads, same script, same prompts, `temperature=0`, K=5:

| workload | why it should help | ngram off | ngram on | delta |
|---|---|---|---|---|
| A pure generation | ✗ nothing to match | 9.77 s / 30.0 tok/s | 13.73 s / 23.3 tok/s | **−22%** |
| B verbatim restatement | ✓✓ output ≈ input | 6.78 s / 37.0 tok/s | 14.70 s / 21.8 tok/s | **−41%** |
| C extractive summary | ✓ some phrases reused | 6.81 s / 47.0 tok/s | 15.69 s / 20.4 tok/s | **−57%** |

Long-context decode fell from 56.4 to ~20–23 tok/s.

**Workload B is the killer datapoint.** It is the textbook best case for ngram — the model
is asked to repeat the input verbatim — and it still lost 41%. When the best case is
strongly negative, the feature is not a tuning problem.

### Why: 16.6% acceptance on a 1 Gbps link

```
vllm:spec_decode_num_draft_tokens     451
vllm:spec_decode_num_accepted_tokens   75     → 16.6%, ≈0.8 accepted tokens/step
```

Each verify step must forward `K+1 = 6` tokens instead of 1, and with PP=2 **every one of
those steps pushes activations across the node boundary**. So the cost side multiplies
cross-node traffic by ~6, while the benefit side yields 0.8 extra tokens. On the 250 MB/s
link that arithmetic cannot win.

A second, independent penalty: ngram forces a runner downgrade.

```
Model Runner V2 does not yet support ngram/ngram_gpu speculative decoding;
using the V1 model runner instead.
```

### When it would be worth revisiting

- On a single node (no PP), where verify steps cost no network at all.
- On ≥10 GbE between nodes — the cost side drops ~10×.
- With a workload whose acceptance rate is much higher, e.g. code editing where the model
  emits long copies of the input verbatim. Even then, measure first.

Shipped configuration: **off** (`NGRAM=0`), and the two image bugs found while trying it are
preserved as optional patch 0005 so the experiment is repeatable.

---

## 3 · The MTP "dead weight" misconception

An early reading of this deployment concluded that `mtp.*` was 7.39 GiB of dead weight that
could be trimmed to reclaim VRAM. **That was wrong**, and the correction matters because it
is the kind of thing people repeat.

What was actually true: 7.39 GiB of `mtp.*` tensors exist **on disk** in the checkpoint
index. What was not true: that any of it is loaded.

**It is not loaded.** The model instantiates `num_hidden_layers = 40` layers; the MTP heads
are only constructed when speculative decoding is enabled, which it is not. Runtime evidence:
during the DSpark attempt, the failure happened in `EngineArgs.create_engine_config()` —
*before* the executor existed, *before* any worker started, and therefore **before a single
safetensors shard was read**. A successful start logs 48 shard progress bars; the DSpark
attempt logged none.

So there is nothing to reclaim, and no VRAM saving available from "trimming MTP". Verify on
your own checkpoint with `tools/model_weight_report.py`.

---

## 4 · Five flags that look helpful and are not

### `--gpu-memory-utilization 0.97` for more context

Does the opposite. vLLM gives all leftover memory to KV, starving activations, so 32K
prefill OOMs on a 16 MiB allocation. Use `--kv-cache-memory-bytes` instead, which caps KV
absolutely. When a byte cap is set, the util value is **silently ignored** — the log says
*"skipped memory profiling, does not respect gpu_memory_utilization"*. Do not tune util
while a KV cap is active.

### `PYTORCH_CUDA_ALLOC_CONF=pinned_use_cuda_host_register:True`

Does **not** stop PyTorch's pinned allocator from rounding requests to powers of two. This
was measured directly: the 11.44 GiB shard still reserved 16 GiB. It changes the
registration path, not the size class. Only raw `cudaHostAlloc` (patch 0001) fixes it.

### `--kv-offloading-size` / `--offload-backend uva`

Does not extend `max_model_len`. These enable `OffloadingConnector`, which uses host memory
as a **second-tier prefix cache** — it lets you keep *more distinct prefixes* resident, not a
*longer single sequence*. A 500K single request still needs 500K of KV in HBM at once. Wrong
tool for this problem, though useful for many-tenant workloads.

### `--tensor-parallel-size 16`

Rejected by the attention kernel: `got 4` local query heads where only 8/16/32 are legal.
The only TP values available are 1, 2, 4, 8.

### `--max-num-batched-tokens 2048` (or 4096) for faster prefill

Doubles activation working set; 100K prefill OOMs and kills `EngineCore` while the container
stays `Up`. Prefill throughput is bandwidth-bound across 16 GPUs, so there was nothing to
gain anyway: measured 1.7–2.9k tok/s at both 1024 and 2048.

---

## Summary table

| Attempt | Outcome | Root cause |
|---|---|---|
| `--spec-method dspark` | ❌ cannot start | draft model has no PP implementation; PP=2 mandatory |
| DSpark with draft PP=1 | ❌ no such flag | `draft_parallel_config.pp` hardcoded from target |
| `--spec-method ngram` | ❌ −22…−57% | 16.6% acceptance × 6× cross-node traffic, plus runner downgrade |
| Trim `mtp.*` to save VRAM | ❌ nothing to trim | never loaded in the first place |
| `gpu-memory-utilization` ↑ | ❌ OOM at 32K | KV takes all slack, activations starved |
| `pinned_use_cuda_host_register` | ❌ no effect | rounding happens in the size class, not the register path |
| `--kv-offloading-size` | ❌ wrong axis | extends prefix cache breadth, not sequence length |
| `TP=16` | ❌ kernel rejects | 4 local Q heads, legal set is {8,16,32} |
| `max-num-batched-tokens 2048` | ❌ OOM | activation working set doubles |
| Prefix caching (already on) | ✅ large win | re-reads of a long context become ~free |
| `fp8_ds_mla` | ✅ 3.5× KV | the reason 512K fits at all |
| Exact `cudaHostAlloc` | ✅ −73 GiB | the reason the host does not OOM |
