# 03 · Host memory — the part nobody warns you about

GPU memory is the obvious budget. On this deployment the *host* memory budget is the one
that kills you, and the failure mode is deceptive: a worker process vanishes with no
Python traceback.

---

## The memory profile is asymmetric

After patch 0001, pinned Engram tables total 189 GiB. Both Engram layers (1 and 14) fall
in PP stage 0, i.e. **node A**, so node A carries all of it:

| | node A (stage 0) | node B (stage 1) |
|---|---|---|
| Total RAM | 251 GiB | 251 GiB |
| Pinned Engram tables | **189 GiB** | 0 |
| Weights on GPU | ~20.7 GiB/card | ~20.1 GiB/card |
| Swap | **none** | 7 GiB |
| `MemAvailable` in steady state | **~30 GiB** | ~228 GiB |

Node B is almost idle. Node A has ~30 GiB of usable headroom, cannot swap, and still has
to stream 476 GB of safetensors through the page cache during startup.

---

## Why `Shmem` is a trap here

Pinned memory allocated via `cudaHostAlloc` / `cudaHostRegister` is accounted as
**`Shmem`** in `/proc/meminfo`, not as `AnonPages`:

```
$ awk '/^MemTotal|^MemAvailable|^Cached|^Shmem/{printf "%-14s %6.1f GiB\n", $1, $2/1048576}' /proc/meminfo
MemTotal        251.5 GiB
MemAvailable     30.2 GiB
Cached          192.0 GiB
Shmem           189.0 GiB      <-- the Engram tables
```

The consequences:

- **`Shmem` is not reclaimable** when there is no swap. With swap it could be written out;
  without, those pages are pinned forever by definition (the GPU holds a mapping to them).
- It is **not** part of `Cached`, so `drop_caches` will never touch it.
- Therefore node A's **real** budget for everything else is
  `251 − 189 = 62 GiB`, minus the CUDA context, python heaps, and NCCL buffers.

The trap: reading `free -g` shows `Cached 192 GiB` and it *looks* like there is plenty to
reclaim. Most of that "cache" is actually the unreclaimable pinned table.

---

## The OOM kill sequence

What actually happens on a cold start without mitigation:

1. Engram tables allocate → `Shmem` reaches 189 GiB.
2. Weight loading `mmap`s all 48 shards and streams them; page cache grows toward 476 GB
   worth of file reads (bounded by RAM, so it fills whatever is left).
3. `MemAvailable` falls below the kernel's watermark.
4. The kernel OOM killer picks the largest RSS process — `VLLM::Worker_PP` — and kills it.
5. The container may stay `Up` (the engine core is a child), and Docker logs show a
   traceback from the *surviving* parent, not the cause.

Confirm after the fact:

```bash
dmesg -T | grep -iE 'killed process|out of memory'
# [Tue Oct 27 09:14:22 2026] Out of memory: Killed process 4471 (VLLM::Worker_PP) ...
```

If you see `VLLM::Worker_PP` in `dmesg`, it was host OOM — not a vLLM bug, not a CUDA
error, and not something a flag will fix.

---

## Mitigation: drop clean page cache, continuously

`scripts/memguard.sh` — run it on **both** nodes, mandatory on node A:

```bash
THRESH_GB="${THRESH_GB:-60}"
while true; do
  avail=$(awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo)
  if [ "${avail}" -lt "${THRESH_GB}" ]; then
    echo 1 > /proc/sys/vm/drop_caches
    if [ "${avail}" -lt 15 ]; then echo 2 > /proc/sys/vm/drop_caches; fi
  fi
  sleep 1
done
```

Why this is safe and why the parameters matter:

- `echo 1 > drop_caches` drops **clean** page cache only — no writeback, no data loss,
  no effect on running processes. `echo 2` additionally drops dentries/inodes, which is
  why it is reserved for the emergency branch (`< 15 GiB`).
- **1 second, not 4.** Our first version polled every 4 s with a `sync` before dropping.
  It still OOM'd: page cache can grow ~1 GB/s while streaming shards, so 4 s is 4 GB of
  headroom you don't have. Do not add `sync` — it forces writeback of unrelated dirty pages
  and makes the pressure worse.
- **60 GiB threshold, not 70.** At 70 GiB you start dropping cache while the loader is
  still streaming and thrash the read path; at 60 GiB you intervene just before the kernel
  does, and startup stays fast.

Without the watchdog, cold start OOMs reliably. With it, 44 h of uptime.

> `memguard.sh` needs root and must outlive your SSH session:
> `sudo nohup ./scripts/memguard.sh >/dev/null 2>&1 &`
> Do **not** use `pkill -f memguard.sh` to stop it — the pattern matches your own SSH
> command line and kills your session (exit 255). Match a bracket pattern instead:
> `pgrep -f 'memguard[.]sh'`.

---

## Diagnosing memory pressure

Run `scripts/sampler.sh` (3 s interval) from before launch until steady state:

```
10:32:11 MemTotal=257473MB MemAvailable=30112MB Cached=196608MB Shmem=193536MB
```

The signature of a healthy start is `Shmem` plateauing at ~189 GiB while
`MemAvailable` stays above 25 GiB. A monotonic decline in `MemAvailable` with `Cached`
growing is the OOM precursor.

Delta check between the two nodes:

```bash
for h in nodeA nodeB; do
  ssh root@$h 'printf "%-8s avail=%s GiB  shmem=%s GiB\n" \
    "$(hostname)" \
    "$(awk "/^MemAvailable:/{print int(\$2/1048576)}" /proc/meminfo)" \
    "$(awk "/^Shmem:/{print int(\$2/1048576)}" /proc/meminfo)"'
done
```

Expected: node A `avail≈30 / shmem≈189`, node B `avail≈228 / shmem≈1.6`.

---

## If you have more RAM

Everything above is a consequence of 251 GiB. With ≥ 320 GiB on node A, the watchdog
becomes a safety net rather than a requirement, and you can raise
`--kv-cache-memory-bytes` for a larger context. Note that adding RAM to node B does
nothing — the asymmetric profile is structural. If you want symmetry, you would need to
put one Engram layer in each pipeline stage, which requires a config where
`kv_source_layer_ids` splits the Engram layers across the cut point. Not attempted here.
