# 02 · Patch notes — root cause of each fix

Five patches, all in [`../patches/`](../patches/). Each was verified by extracting the
target file from the image, applying the diff with `patch -p1`, and comparing byte-for-byte
against the file running in production. All five are clean.

Line numbers refer to `vllm 0.28.1rc1.dev517+glm53.dsv41.vision11.sm89sm120.cu130`. On a
different build they will move; the surrounding context in each hunk is the reliable anchor.

---

## 0001 · `engram.py` — PyTorch's pinned allocator rounds to powers of two

**This is the patch that decides whether the deployment works at all.**

### Symptom

The engine dies during weight loading with the worker process simply disappearing.
No Python traceback. If you are watching `dmesg` you see:

```
Out of memory: Killed process 12345 (VLLM::Worker_PP) total-vm:... 
```

`free -g` shows `Shmem` climbing toward and past the machine's physical RAM.

### Root cause

The Engram tables are allocated with `torch.empty(..., pin_memory=True)`:

```python
kwargs = {"device": "cpu", "pin_memory": True} if cpu_offload else {}
self.weight = nn.Parameter(
    torch.empty(self.part_num_embeddings, dim, dtype=torch.float8_e4m3fn, **kwargs),
    requires_grad=False,
)
```

PyTorch's caching pinned allocator rounds every request up to the **next power of two**.
An Engram shard is 11.44 GiB:

| | |
|---|---|
| Requested per shard | 11.44 GiB |
| Actually reserved | **16 GiB** |
| Waste per shard | 4.56 GiB |
| Shards (8 TP ranks × 2 Engram layers) | 16 |
| **Total waste** | **≈73 GiB** |
| `cudaHostAlloc` demand | 8 × 2 × 16 = **264 GiB** > 251 GiB physical |

`PYTORCH_CUDA_ALLOC_CONF=pinned_use_cuda_host_register:True` does **not** help — that
switches the registration path, not the size rounding. We measured it. It still rounds.

### Fix

Bypass the allocator entirely with raw `cudaHostAlloc`, which is exact:

```python
_CUDART = None

def _exact_pinned(numel: int, dtype: torch.dtype, keepalive: list) -> torch.Tensor:
    global _CUDART
    import ctypes

    if _CUDART is None:
        _CUDART = ctypes.CDLL("libcudart.so")
    nbytes = numel * dtype.itemsize
    ptr = ctypes.c_void_p()
    rc = _CUDART.cudaHostAlloc(
        ctypes.byref(ptr), ctypes.c_size_t(nbytes), ctypes.c_uint(0)
    )
    if rc != 0:
        raise RuntimeError(f"cudaHostAlloc({nbytes}) failed with rc={rc}")
    buf = (ctypes.c_char * nbytes).from_address(ptr.value)
    keepalive.append(buf)     # CPython would otherwise free memory the tensor points at
    keepalive.append(ptr)
    return torch.frombuffer(buf, dtype=torch.uint8).view(dtype)
```

`keepalive` is a list held on the module (`self._keepalive`) for the parameter's whole
lifetime. Dropping the ctypes buffer would let CPython free a block a live tensor still
references — a use-after-free that shows up much later as garbage KV or a segfault.

Result, measured: allocation ratio **1.00** (exact), `Shmem` drops from 264 GiB to
**189 GiB**. Verified with a standalone `cudaHostAlloc` probe before touching the model.

### Second half of the same patch: `_storage()` must not re-copy

The original `_storage()` builds *accelerator views* of the pinned tensors:

```python
# BEFORE
if not self.cpu_offload:
    return self.weight.data, self.weight_scale_inv.data
src = (self.weight.data_ptr(), self.weight_scale_inv.data_ptr())
if self._view_src != src:
    self._views = (
        get_accelerator_view_from_cpu_tensor(self.weight.data),
        get_accelerator_view_from_cpu_tensor(self.weight_scale_inv.data),
    )
    self._view_src = src
return self._views
```

`get_accelerator_view_from_cpu_tensor()` re-copies any tensor whose pinned flag is not
set. A tensor built with `torch.frombuffer` over a `cudaHostAlloc` block **has no such
flag**, so every lookup would silently copy 11 GiB to the GPU. The patch returns the host
tensors directly:

```python
# AFTER
return self.weight.data, self.weight_scale_inv.data
```

This is safe because on this driver the host and device pointers of a `cudaHostAlloc`
block are identical (UVA), which the Triton lookup kernel relies on anyway. Verified
separately with a UVA probe: the pointer equality test passed and a Triton kernel wrote
into a `cudaHostAlloc` buffer from device code.

> **If you skip this half of the patch** you get an OOM inside the first lookup, not at
> startup. Do not split them.

---

## 0002 / 0003 · dummy `input_ids` for VL MoE routing

Two patches, same bug in two files (`model.py` for V4, `router.py` for V4.1).

### Symptom

```
ValueError: DeepSeek V4 mixed text/image MoE routing requires input_ids.
```

during **memory profiling or CUDA graph capture**, before any request has arrived.

### Root cause

The MoE gate carries a vision-language correction bias. When it is present, the routing
code needs one token id per row so it can tell text tokens from image placeholder tokens:

```python
if self.gate.e_score_correction_bias_vl is not None and input_ids is None:
    raise ValueError("DeepSeek V4 mixed text/image MoE routing requires input_ids.")
```

Dummy runs — the memory-profiling forward pass and CUDA graph capture — synthesise
`input_ids` as `None` because they only need shapes. So a model that supports images
cannot be profiled.

### Fix

Synthesise all-zero token ids *only* for that case:

```python
input_ids = torch.zeros(
    hidden_states.shape[0], dtype=torch.long, device=hidden_states.device
)
```

Token id 0 never maps to an image placeholder, so this is equivalent to "every dummy row
is text". Real requests always carry `input_ids`, so this branch is dead while serving.
The `tid2eid` (hash MoE) branch is deliberately left raising — that one needs genuine
token identity and a dummy pass would produce meaningless experts.

---

## 0004 · `worker/utils.py` — a silently fatal `StopIteration`

### Symptom

Startup hangs then fails with an **empty** exception. This is the single most misleading
failure in the whole deployment: no message, no traceback, and every log line before it
says the model loaded fine. It looks exactly like a memory problem, and you will waste
hours re-tuning `--gpu-memory-utilization` (we did).

### Root cause

`allocate_kv_cache()` resolves each KV tensor's owning scheduler group with `next()`:

```python
group_id, group = next(
    (group_id, group)
    for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
    if layer_name in group.layer_names
)
```

The last pipeline stage also owns the MTP/DSpark draft layers. Their KV specs exist but
belong to **no** scheduler-facing group, so the generator is empty and `next()` raises
`StopIteration`. Inside a multiprocessing RPC, that surfaces as an empty error string.

### Fix

```python
_matches = [
    (group_id, group)
    for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
    if layer_name in group.layer_names
]
if not _matches:
    logger.warning(
        "No KV cache group covers layer %s (draft layer on the last PP "
        "stage?). Skipping its KV cache. Group layer counts: %s",
        layer_name,
        [len(g.layer_names) for g in kv_cache_config.kv_cache_groups],
    )
    continue
group_id, group = _matches[0]
```

Those layers are unused without speculative decoding, so skipping their cache is correct
*and* saves memory. The warning is deliberate: if you later enable speculation and it
misbehaves, this line tells you why.

---

## 0005 · `gpu_model_runner.py` — two latent bugs (optional, speculation-only)

Not needed for the shipped config. Both are only reachable when
`speculative_config` is set, and both are genuine bugs in the image rather than in
upstream vLLM.

### 0005a — the `num_computed_tokens` rename was missed in one place

The `InputBatch` class in this fork renamed the attribute to `num_computed_tokens_cpu`,
but `_prepare_lookback_token_ids()` kept the old name:

```
AttributeError: 'InputBatch' object has no attribute 'num_computed_tokens'
```

A full-tree scan for `input_batch.num_computed_tokens\b` found exactly **one** hit;
lines 4067 and 4423 in the same file already use `_cpu`. So this is a missed rename, not
an API change. The fix is one word, and it matches what the callee already expects
(`_build_lookback_token_ids` wants an `np.ndarray`):

```diff
-            self.input_batch.num_computed_tokens[:num_reqs],
+            self.input_batch.num_computed_tokens_cpu[:num_reqs],
```

### 0005b — `self.drafter` only exists on the last PP rank

```python
# BEFORE
self.use_aux_hidden_state_outputs = False
# Set up speculative decoding.
# NOTE(Jiayi): currently we put the entire draft model on the last PP rank.
if self.speculative_config and get_pp_group().is_last_rank:
    self.drafter = ...
```

`_build_attention_metadata()` then dereferences `self.drafter` unconditionally, so **PP0
always crashes** when speculation is enabled:

```
AttributeError: 'GPUModelRunner' object has no attribute 'drafter'
```

Two changes:

```diff
 self.use_aux_hidden_state_outputs = False
+# PATCHED: non-last PP ranks never created `drafter`, yet
+# _build_attention_metadata() dereferences it unconditionally.
+self.drafter = None
 ...
-if self.speculative_config and get_pp_group().is_last_rank:
+if self.speculative_config and (
+    get_pp_group().is_last_rank
+    or self.speculative_config.method == "ngram"
+):
```

Creating the drafter on every rank is only sensible for `ngram`, which is a pure-CPU
proposer (two `int32` arrays); it would be wrong for DSpark, whose draft model is 7.39 GiB.
That asymmetry is why the condition names `ngram` explicitly instead of dropping the rank
check.

Note that DSpark still cannot run here for an unrelated structural reason — see
`06-negative-results.md`. Patch 0005 does not unlock DSpark.

---

## Applying them

```bash
./scripts/apply-patches.sh          # image -> ./patched/ overlay, no rebuild
```

The script creates a throwaway container, `docker cp`s the five target files out,
patches them, and leaves them at their site-packages-relative paths. `launch.sh` then
bind-mounts each one read-only over the original. Nothing in the image is modified, so
`docker rmi` + re-pull always gives you a clean baseline.

### Verifying a patch before you trust it

```bash
# forward-apply against the pristine file and compare to what is running
docker create --name tmp $IMAGE
docker cp tmp:/opt/venv/lib/python3.12/site-packages/vllm/models/deepseek_v4_1/common/engram.py /tmp/a.py
docker rm -f tmp
cp /tmp/a.py /tmp/b.py
patch -p1 -d /tmp < patches/0001-engram-exact-pinned-alloc.patch  # adjust -p as needed
cmp /tmp/b.py "$(pwd)/patched/vllm/models/deepseek_v4_1/common/engram.py"
```

All five patches in this repo passed exactly this check against the production files.
