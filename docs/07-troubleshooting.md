# 07 · Troubleshooting

Every entry below was hit for real. The first column is what you see, not what is wrong —
on this deployment the visible symptom is frequently a lie.

---

## The most important one: the container is `Up` but the API is dead

```bash
$ docker ps --filter name=dsv41 --format '{{.Status}}'
Up About an hour                  # ← meaningless

$ curl -s -o /dev/null -w '%{http_code}\n' localhost:9000/v1/models
000                               # ← truth
```

`EngineCore` and the HTTP server are separate processes. A CUDA OOM kills the engine and
leaves the server process alive, so Docker sees a healthy container. **Always health-check
with `/v1/models`.** A readiness probe that uses `docker ps` will silently pass a dead
service.

---

## Startup failures

### `Worker disappeared`, no traceback, container restarting

Host OOM kill, not a vLLM error. Confirm:

```bash
dmesg -T | grep -iE 'killed process|out of memory'
# ... Killed process 4471 (VLLM::Worker_PP) total-vm:... 
```

Cause: 189 GiB of pinned Engram in `Shmem` (unreclaimable without swap) + streaming 476 GB
of safetensors fills node A. Fix: run `scripts/memguard.sh` **before** launching, 1 s
interval, 60 GiB threshold. See [`03-host-memory.md`](03-host-memory.md).

### `ValueError: DeepSeek V4 mixed text/image MoE routing requires input_ids.`

Patches 0002/0003 not applied (or applied to only one of the two files). This fires during
memory profiling, before any request.

### Empty error / silent hang during startup, then engine dies

This is the `StopIteration` from patch 0004, and it is the most misleading failure here:
**the exception message is empty**, and the log up to that point looks perfectly healthy.
If the KV cache allocation line never appears, suspect this before suspecting memory. Symptom
of a wasted afternoon: you re-tune `--gpu-memory-utilization` repeatedly, which changes
nothing.

### `ValueError: DeepSeek V4.1 native attention supports 8, 16, or 32 local query heads, got 4.`

`--tensor-parallel-size 16`. Legal TP values are 1, 2, 4, 8. See
[`01-constraint-derivation.md`](01-constraint-derivation.md).

### `NotImplementedError: Pipeline parallelism is not supported for this model.`

Speculative decoding is on. The draft model (DSpark) has no PP support, and PP=2 is
mandatory. Set `NGRAM=0` and remove `--spec-method`. See
[`06-negative-results.md`](06-negative-results.md).

### `num_speculative_tokens must be provided with speculative model`

`--spec-method dspark` without `--spec-tokens`. DSpark's draft block is 5 tokens, so pass
`--spec-tokens 5`. (Still won't boot on PP=2 — this is only relevant if you have removed PP.)

### `ValueError: No available memory for the cache blocks.`

Usually genuine, sometimes a decoy. Check the reported number:

```
Available KV cache memory: -0.47 GiB     → genuinely over-committed
Available KV cache memory:  0.51 GiB     → enough; look elsewhere
```

If it reports positive but the engine still dies, it is almost certainly patch 0004
(`StopIteration`), not memory. This exact misdiagnosis cost two tuning rounds.

**Fix if genuine:** lower `--kv-cache-memory-bytes`, or lower `--max-num-batched-tokens`
to 1024, or accept a shorter `--max-model-len`.

### `CUDA out of memory ... Tried to allocate 16.00 MiB ... in project_query_and_cache_kv()`

Activation starvation, not KV shortage. A *16 MiB* allocation failing tells you the card is
completely full. Reduce the KV budget so activations have room:

```
--kv-cache-memory-bytes 560M     (not --gpu-memory-utilization 0.97)
--max-num-batched-tokens 1024    (not 2048)
```

See [`04-long-context-512k.md`](04-long-context-512k.md).

### `ValueError: Model architectures ['DeepseekV4ForCausalLM'] are not supported`

You are on stock vLLM. Official `vllm/vllm-openai` has no `DeepseekV4` in its registry at
all. Use the SM89 community image.

### Neither node joins / `world_size` mismatch / NCCL timeout

- Startup order wrong: node A must be up first, node B second. Node B needs `--headless`.
- Wrong NIC: `NCCL_SOCKET_IFNAME` must name the interface that actually carries the peer IP.
  Check with `ip -br a`.
- `--nnodes 2 --node-rank 0|1 --master-addr` must agree on both sides, and
  `--master-port` must be reachable (it is not the HTTP port).

---

## Runtime problems

### `message.reasoning_content` is `None`

In this build the chain of thought is in **`message.reasoning`**. Most other vLLM versions
use `reasoning_content`, and `--reasoning-parser` alone does not decide the output key. Fix
your client, or normalise server-side.

### `model` name not found (404)

The served ids are whatever `--served-model-name` listed. Check:

```bash
curl -s localhost:9000/v1/models | tr ',' '\n' | grep '"id"'
```

Default: `deepseek-v4.1-flash-1017`. Note the response body always echoes the **primary**
name, even if you requested an alias — a few clients validate this and will fail.

### `address already in use`

On the driver node only. On this hardware port 9000 is commonly taken by another inference
service, and on a worker node 9000 may belong to MinIO. Only the driver binds the API port;
change it with `PORT=9100`. Do not change `MASTER_PORT` to solve this — that is a separate
port.

### Decode suddenly drops to ~20 tok/s

Speculative decoding got enabled. Check `/metrics` for `spec_decode` series; if
`accepted/drafted` is low, turn it off. Also note ngram forces the V1 model runner, which is
slower by itself. See [`06-negative-results.md`](06-negative-results.md).

### Long prefill takes minutes, or the request hangs

Expected above ~300K tokens: 498,846 tokens needs 286 s. This is not a hang. If it exceeds
~6 minutes at 500K, verify `--enable-chunked-prefill` is on (it is by default and is only
disabled for encoder-decoder models).

---

## Recovery after reboot

There is no supervisor (`restart=no`, no systemd unit). The order is not optional:

```bash
# 1. on BOTH nodes, first -- node A will be OOM-killed without it
sudo nohup ./scripts/memguard.sh >/dev/null 2>&1 &

# 2. driver first
MASTER_ADDR=... MODEL_PATH=... IMAGE=... ./scripts/launch.sh 0

# 3. then the worker (it will block on rendezvous until node A is ready)
ssh root@$WORKER_ADDR 'MASTER_ADDR=... MODEL_PATH=... IMAGE=... ./scripts/launch.sh 1'

# 4. wait for the API, not for docker
until curl -sf -o /dev/null -w '' http://$MASTER_ADDR:9000/v1/models; do sleep 10; done
```

Cold start is 8–10 minutes if the page cache is gone (476 GB of disk reads), ~1 minute warm.
Warm the Triton kernel with one throwaway request before you benchmark, or your first
measurement will be 8 s instead of 0.7 s.

### Clean shutdown

```bash
MASTER_ADDR=... WORKER_ADDR=... ./scripts/stop.sh
```

Worker first, then driver. `memguard.sh` can keep running; it is harmless when nothing is
allocated.

> **Do not stop memguard with `pkill -f memguard.sh`.** The pattern matches your own SSH
> command line and kills your session (exit 255). Use
> `pgrep -f 'memguard[.]sh'` to find it, then `kill <pid>`.

---

## Diagnostics worth running before you change anything

```bash
# What the engine actually received (not what you think you passed)
docker inspect dsv41 --format '{{json .Args}}' | python3 -m json.tool

# What launch.sh *would* pass -- diff two configs without restarting anything
DRY_RUN=1 ./scripts/launch.sh 0
DRY_RUN=1 KVB=700M NGRAM=1 ./scripts/launch.sh 0

# Did CUDA graph capture and KV allocation succeed?
docker logs dsv41 2>&1 | grep -E 'GPU KV cache size|Available KV|graph capturing|init engine'

# Was it host OOM?
dmesg -T | grep -iE 'killed process|out of memory'

# What does the memory profile look like right now?
./scripts/sampler.sh /tmp/mem.log &   # then watch /tmp/mem.log

# Which files are actually bind-mounted from the overlay?
docker inspect dsv41 --format '{{range .Mounts}}{{.Source}} -> {{.Destination}}{{println}}{{end}}'

# Is the running engine the one you think it is?
curl -s localhost:9000/v1/models | tr ',' '\n' | grep -E '"id"|max_model_len'
```

That last one catches a surprising number of problems: a stale container with the previous
config answers on the port perfectly happily.
