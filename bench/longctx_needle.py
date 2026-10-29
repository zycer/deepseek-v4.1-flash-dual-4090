#!/usr/bin/env python3
"""Needle-in-a-haystack with latency measurement, for validating long-context claims.

Buries a secret at a chosen position inside a filler document, asks for it back, and
reports TTFT / prefill / decode. Recall proves the whole prefix was attended to, not
silently truncated -- which is exactly what a nominal `max_model_len` does not prove.

    ./longctx_needle.py 32000 100000 500000
    ./longctx_needle.py 500000 --position 0.9 --base http://10.0.0.1:9000/v1

Sizes are given in *tokens*. Because this vocabulary tokenises the Chinese filler at about
0.58 tokens per character, the document is built with an empirically calibrated
characters-per-token factor; the server-reported `prompt_tokens` is authoritative and is
what you should quote. Expect ±3% versus the requested size.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

SECRET = "XK-7391-QZ"

# Calibrated on this checkpoint: the filler below tokenises at ~0.58 tokens/char.
CHARS_PER_TOKEN = 1.72

FILLER = (
    "第{t}节。在分布式训练系统里，梯度同步、通信重叠与显存管理是三个互相牵制的因素。"
    "工程师通常在吞吐和延迟之间做权衡，并通过分片、重计算与流水线并行来缓解压力。"
    "本节讨论这些机制在实际部署中的相互作用，以及若干常见的失效模式。"
)

QUESTION = "\n\n问题：上文提到的机密代号是什么？只回答代号本身。"


def build_doc(n_tokens: int, position: float) -> str:
    """Filler document of roughly `n_tokens` tokens with the secret at `position`."""
    n_chars = int(n_tokens * CHARS_PER_TOKEN)
    needle = f"本次实验的机密代号是 {SECRET}，请务必记住。"
    parts: list[str] = []
    chars = 0
    idx = 0
    needle_at = int(n_chars * position)
    placed = False

    while chars < n_chars:
        if not placed and chars >= needle_at:
            parts.append(needle)
            chars += len(needle)
            placed = True
        para = FILLER.format(t=idx)
        parts.append(para)
        chars += len(para)
        idx += 1

    if not placed:
        parts.insert(0, needle)
    return "".join(parts)


def run_one(base: str, model: str, n_target: int, position: float,
            max_tokens: int, timeout: float) -> dict | None:
    doc = build_doc(n_target, position)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": doc + QUESTION}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    t0 = time.time()
    ttft = None
    content: list[str] = []
    reasoning: list[str] = []
    usage: dict = {}

    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                if not raw.startswith(b"data: "):
                    continue
                body = raw[6:].strip()
                if body == b"[DONE]":
                    break
                try:
                    obj = json.loads(body)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices") or []:
                    delta = ch.get("delta") or {}
                    if delta.get("content") or delta.get("reasoning"):
                        if ttft is None:
                            ttft = time.time() - t0
                    if delta.get("content"):
                        content.append(delta["content"])
                    if delta.get("reasoning"):
                        reasoning.append(delta["reasoning"])
    except (urllib.error.URLError, OSError) as e:
        print(f"  ERROR: {e!r}")
        return None

    total = time.time() - t0
    out = "".join(content)
    pt = usage.get("prompt_tokens")
    ct = usage.get("completion_tokens")
    decode_t = max(total - (ttft or 0), 1e-6)

    res = {
        "target": n_target,
        "prompt_tokens": pt,
        "ttft": ttft,
        "total": total,
        "completion_tokens": ct,
        "decode_tok_s": (ct / decode_t) if ct else 0.0,
        "prefill_tok_s": (pt / ttft) if (pt and ttft) else None,
        "answer": out.strip(),
        "hit": SECRET in out,
    }

    print(f"--- target {n_target:,} (needle @ {position:.0%}) ---")
    print(f"  prompt_tokens  = {pt:,}" if pt else "  prompt_tokens  = n/a")
    print(f"  TTFT           = {ttft:.2f} s" if ttft else "  TTFT           = n/a")
    print(f"  total          = {total:.2f} s")
    print(f"  completion_tok = {ct}")
    print(f"  decode tok/s   = {res['decode_tok_s']:.1f}")
    if res["prefill_tok_s"]:
        print(f"  prefill tok/s  = {res['prefill_tok_s']:.0f}")
    print(f"  answer         = {res['answer'][:120]!r}")
    print(f"  RECALL         = {'HIT' if res['hit'] else 'MISS'}")
    print()
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sizes", nargs="*", type=int, default=[100_000],
                    help="approximate prompt sizes in tokens")
    ap.add_argument("--base", default="http://127.0.0.1:9000/v1")
    ap.add_argument("--model", default="deepseek-v4.1-flash-1017")
    ap.add_argument("--position", type=float, default=0.5, help="needle position, 0..1")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--timeout", type=float, default=3600.0)
    args = ap.parse_args()

    sizes = args.sizes or [100_000]
    print(f"base={args.base} model={args.model}\n")

    rows = []
    for n in sizes:
        r = run_one(args.base, args.model, n, args.position, args.max_tokens, args.timeout)
        if r:
            rows.append(r)

    if len(rows) > 1:
        print("=" * 74)
        print(f"{'target':>9} {'prompt':>9} {'TTFT(s)':>9} {'prefill':>9} {'decode':>8} recall")
        for r in rows:
            print(f"{r['target']:>9,} {r['prompt_tokens'] or 0:>9,} "
                  f"{(r['ttft'] or 0):>9.2f} {(r['prefill_tok_s'] or 0):>9.0f} "
                  f"{r['decode_tok_s']:>8.1f} {'HIT' if r['hit'] else 'MISS'}")

    return 0 if all(r["hit"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
