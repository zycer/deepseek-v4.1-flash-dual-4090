#!/usr/bin/env python3
"""Account for every byte in a safetensors checkpoint, per shard and per layer.

This is how the MTP/DSpark question gets settled with data instead of folklore: read the
tensor index, bucket by layer, and see whether the `mtp.*` tensors are 7.39 GiB of dead
weight or simply never loaded (they are never loaded -- see docs/06).

    ./model_weight_report.py /path/to/checkpoint
    ./model_weight_report.py /path/to/checkpoint --mtp-from 40

Reads only the safetensors headers, so it is fast and needs no GPU.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import struct
import sys

GB = 1024 ** 3


def read_header(path: str) -> dict:
    """First 8 bytes are a little-endian u64 header length, then JSON."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("--mtp-from", type=int, default=None,
                    help="first layer index considered MTP/draft (default: num_hidden_layers)")
    ap.add_argument("--quiet-layers", action="store_true", help="skip the per-layer dump")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.model_dir, "*.safetensors")))
    if not files:
        print(f"no *.safetensors in {args.model_dir}", file=sys.stderr)
        return 1

    # layer count of the main model, so we know where "draft" starts
    mtp_from = args.mtp_from
    if mtp_from is None:
        try:
            with open(os.path.join(args.model_dir, "config.json")) as f:
                mtp_from = json.load(f).get("num_hidden_layers", 10 ** 6)
        except OSError:
            mtp_from = 10 ** 6

    print(f"shards       : {len(files)}")
    print(f"main layers  : {mtp_from}")

    total = 0
    by_layer: collections.Counter = collections.Counter()
    by_bucket: collections.Counter = collections.Counter()
    samples: dict[str, list[str]] = collections.defaultdict(list)
    per_shard: list[tuple[str, int]] = []

    for path in files:
        hdr = read_header(path)
        shard_bytes = 0
        for key, meta in hdr.items():
            if key == "__metadata__":
                continue
            offs = meta.get("data_offsets")
            if not offs:
                continue
            nbytes = offs[1] - offs[0]
            total += nbytes
            shard_bytes += nbytes

            m = re.search(r"layers\.(\d+)\.", key)
            if m:
                layer = int(m.group(1))
                by_layer[layer] += nbytes
                bucket = f"layer{layer}" if layer >= mtp_from else "main-layers"
            else:
                bucket = "non-layer:" + key.split(".")[0]
            by_bucket[bucket] += nbytes
            if len(samples[bucket]) < 2:
                samples[bucket].append(key)
        per_shard.append((os.path.basename(path), shard_bytes))

    print(f"total        : {total / GB:.2f} GiB ({total / 10**9:.1f} GB)")
    print()

    print("=== buckets ===")
    for bucket, nbytes in sorted(by_bucket.items(), key=lambda kv: -kv[1]):
        share = nbytes / total * 100
        print(f"  {bucket:<32} {nbytes / GB:>8.2f} GiB  {share:>5.1f}%  e.g. {samples[bucket][0]}")

    draft = sum(n for ln, n in by_layer.items() if ln >= mtp_from)
    if draft:
        print()
        print(f"=== MTP / draft layers (>= {mtp_from}) ===")
        for ln in sorted(l for l in by_layer if l >= mtp_from):
            print(f"  layer {ln:<4} {by_layer[ln] / GB:>7.3f} GiB   e.g. {samples[f'layer{ln}'][0]}")
        print(f"  >>> total {draft / GB:.2f} GiB ({draft / 1024 ** 2:.0f} MiB)")
        print("  NOTE: these are only instantiated when speculative decoding is enabled.")
        print("        With dspark off they are never read -- not 'wasted', just unused.")

    if not args.quiet_layers:
        print()
        print("=== per-layer weight (main model) ===")
        main_layers = [(ln, by_layer[ln]) for ln in sorted(by_layer) if ln < mtp_from]
        for ln, nbytes in main_layers:
            print(f"  layer {ln:<4} {nbytes / GB:>7.3f} GiB")
        if main_layers:
            avg = sum(n for _, n in main_layers) / len(main_layers)
            print(f"  average {avg / GB:.3f} GiB over {len(main_layers)} layers")

    print()
    print("=== largest shards ===")
    for name, nbytes in sorted(per_shard, key=lambda kv: -kv[1])[:5]:
        print(f"  {name:<44} {nbytes / GB:>8.2f} GiB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
