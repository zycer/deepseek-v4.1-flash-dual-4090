#!/usr/bin/env python3
"""A/B benchmark for speculative decoding across three workloads.

Speculative decoding should help when the output reuses the input (restatement,
extractive summarisation) and be neutral otherwise. This script measures all three and
also reports the acceptance rate from `/metrics`, which is what actually decides the
outcome.

    # run once with the feature off, once with it on, then compare:
    ./ngram_ab.py --label "ngram off"
    ./ngram_ab.py --label "ngram on"  --base http://10.0.0.1:9000/v1

Reference result on a 2x1 Gbps PP link (K=5, acceptance 16.6%) -- every workload got
slower, including the one the feature is designed for:

    workload                     off        on      delta
    pure generation            30.0      23.3      -22%
    verbatim restatement       37.0      21.8      -41%
    extractive summary         47.0      20.4      -57%
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request

DOC = (
    "边缘计算节点的部署需要综合考虑网络时延、带宽成本和数据合规三个维度。"
    "在时延敏感的工业质检场景中，建议将推理服务下沉到产线侧，端到端时延可控制在二十毫秒以内。"
    "对于带宽受限的偏远站点，应采用增量同步策略，仅在夜间回传摘要数据。"
    "数据合规方面，涉及个人信息的数据必须在本地完成脱敏后再上行，"
    "原始日志的保留周期不应超过七十二小时，审计记录需独立存储并留存六个月。"
)

CASES = [
    ("A pure generation",
     "请写一段约 250 字的技术说明，介绍分布式推理中张量并行的原理。"),
    ("B verbatim restatement",
     f"请把下面这段文字一字不改地完整复述一遍，不要加任何解释：\n\n{DOC}"),
    ("C extractive summary",
     f"从下面文字中原样摘出所有含数字的句子，逐条列出，保持原文措辞：\n\n{DOC}"),
]


def scrape_metrics(base: str) -> dict[str, float]:
    """Pull spec_decode counters from the Prometheus endpoint, if exposed."""
    root = base.rstrip("/").removesuffix("/v1")
    try:
        with urllib.request.urlopen(f"{root}/metrics", timeout=10) as r:
            text = r.read().decode()
    except (urllib.error.URLError, OSError):
        return {}

    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = re.match(r"(vllm:spec_decode_\w+?)(?:_total)?(?:\{[^}]*\})?\s+([\d.eE+]+)", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def call(base: str, model: str, prompt: str, max_tokens: int,
         temperature: float, timeout: float) -> dict | None:
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode()
    except (urllib.error.URLError, OSError) as e:
        print(f"  ERROR: {e!r}")
        return None
    dt = time.time() - t0
    d = json.loads(body)
    if "error" in d:
        print(f"  ERROR: {str(d['error'])[:200]}")
        return None
    u = d["usage"]
    msg = d["choices"][0]["message"]
    ct = u["completion_tokens"]
    return {
        "prompt_tokens": u["prompt_tokens"],
        "completion_tokens": ct,
        "wall": dt,
        "tok_per_s": ct / dt if dt else 0.0,
        "content": msg.get("content") or "",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:9000/v1")
    ap.add_argument("--model", default="deepseek-v4.1-flash-1017")
    ap.add_argument("--label", default="run")
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=1800.0)
    args = ap.parse_args()

    before = scrape_metrics(args.base)
    results: list[tuple[str, float]] = []

    for name, prompt in CASES:
        print("=" * 70)
        print(f"[{args.label}] {name}")
        r = call(args.base, args.model, prompt, args.max_tokens,
                 args.temperature, args.timeout)
        if not r:
            results.append((name, 0.0))
            continue
        print(f"  prompt_tokens     = {r['prompt_tokens']}")
        print(f"  completion_tokens = {r['completion_tokens']}")
        print(f"  wall              = {r['wall']:.2f} s")
        print(f"  overall tok/s     = {r['tok_per_s']:.1f}")
        print(f"  output[:60]       = {r['content'][:60]!r}")
        results.append((name, r["tok_per_s"]))
        print()

    after = scrape_metrics(args.base)
    if after:
        drafted = after.get("vllm:spec_decode_num_draft_tokens", 0) - before.get("vllm:spec_decode_num_draft_tokens", 0)
        accepted = after.get("vllm:spec_decode_num_accepted_tokens", 0) - before.get("vllm:spec_decode_num_accepted_tokens", 0)
        if drafted > 0:
            print(f"spec decode: drafted={drafted:.0f} accepted={accepted:.0f} "
                  f"acceptance={accepted / drafted:.1%}")
        else:
            print("spec decode: no drafts issued (feature off, or no match found)")

    print()
    print("=" * 70)
    print(f"summary [{args.label}]:")
    for name, tps in results:
        print(f"  {name:<26} {tps:>6.1f} tok/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
