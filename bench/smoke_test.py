#!/usr/bin/env python3
"""End-to-end smoke test for the DeepSeek-V4.1-Flash endpoint.

Checks, in order: model listing, non-stream chat, stream timing (TTFT + decode),
reasoning/answer split, tool calling, and optional N-way concurrency.

    ./smoke_test.py
    ./smoke_test.py --base http://10.0.0.1:9000/v1 --concurrency 4

Exit code is non-zero if any check fails, so it is safe to use as a readiness gate.
Stdlib only -- no pip install needed.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request

FAILED: list[str] = []


def post(base: str, payload: dict, timeout: float = 3600):
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def stream(base: str, payload: dict, timeout: float = 3600) -> dict:
    """Consume an SSE stream. Returns timings and concatenated content/reasoning."""
    payload = {**payload, "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.time()
    ttft = None
    n_content = n_reasoning = 0
    content: list[str] = []
    reasoning: list[str] = []
    usage = {}

    with post(base, payload, timeout) as r:
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
                if delta.get("content"):
                    if ttft is None:
                        ttft = time.time() - t0
                    content.append(delta["content"])
                    n_content += 1
                if delta.get("reasoning"):
                    if ttft is None:
                        ttft = time.time() - t0
                    reasoning.append(delta["reasoning"])
                    n_reasoning += 1

    total = time.time() - t0
    ct = usage.get("completion_tokens") or (n_content + n_reasoning)
    decode_t = max(total - (ttft or 0), 1e-6)
    return {
        "ttft": ttft,
        "total": total,
        "content": "".join(content),
        "reasoning": "".join(reasoning),
        "n_content": n_content,
        "n_reasoning": n_reasoning,
        "completion_tokens": ct,
        "usage": usage,
        "tok_per_s": ct / decode_t if ct else 0.0,
    }


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def test_models(base: str) -> None:
    print("== /v1/models ==")
    try:
        with urllib.request.urlopen(f"{base}/models", timeout=30) as r:
            data = json.load(r)
    except (urllib.error.URLError, OSError) as e:
        check("GET /v1/models", False, repr(e))
        return
    ids = [m["id"] for m in data.get("data", [])]
    print(f"  ids: {ids}")
    check("at least one model served", bool(ids))
    for m in data.get("data", []):
        if m.get("max_model_len"):
            print(f"  {m['id']}: max_model_len={m['max_model_len']}")


def test_nonstream(base: str, model: str) -> None:
    print("\n== non-stream chat ==")
    t0 = time.time()
    try:
        with post(base, {
            "model": model,
            "messages": [{"role": "user", "content": "用 Python 写一个快速排序，一句话说明时间复杂度。"}],
            "max_tokens": 400,
            "temperature": 0.6,
        }) as r:
            j = json.load(r)
    except (urllib.error.URLError, OSError) as e:
        check("POST /chat/completions", False, repr(e))
        return
    dt = time.time() - t0
    if "error" in j:
        check("chat completion", False, str(j["error"])[:200])
        return
    u = j.get("usage", {})
    msg = j["choices"][0]["message"]
    print(f"  wall {dt:.2f}s  prompt={u.get('prompt_tokens')}  completion={u.get('completion_tokens')}")
    print(f"  content[:200]: {msg.get('content', '')[:200]!r}")
    check("non-stream returned text", bool((msg.get("content") or msg.get("reasoning"))))


def test_stream(base: str, model: str) -> None:
    print("\n== stream timing ==")
    try:
        r = stream(base, {
            "model": model,
            "messages": [{"role": "user", "content": "用大约 200 字介绍中国高速铁路的发展。"}],
            "max_tokens": 300,
            "temperature": 0.6,
        })
    except (urllib.error.URLError, OSError) as e:
        check("streaming request", False, repr(e))
        return
    ttft = f"{r['ttft']:.3f}s" if r["ttft"] else "n/a"
    print(f"  TTFT {ttft}  total {r['total']:.2f}s  "
          f"tokens {r['completion_tokens']}  decode {r['tok_per_s']:.1f} tok/s")
    check("stream produced content", r["n_content"] > 0 or r["n_reasoning"] > 0)
    check("throughput above 20 tok/s", r["tok_per_s"] > 20, f"{r['tok_per_s']:.1f}")


def test_reasoning_split(base: str, model: str) -> None:
    print("\n== reasoning / answer split ==")
    try:
        with post(base, {
            "model": model,
            "messages": [{"role": "user", "content": "9.11 和 9.9 哪个大？只回答结论。"}],
            "max_tokens": 400,
            "temperature": 0.0,
        }) as r:
            j = json.load(r)
    except (urllib.error.URLError, OSError) as e:
        check("reasoning request", False, repr(e))
        return
    msg = j["choices"][0]["message"]
    keys = list(msg.keys())
    print(f"  message keys: {keys}")
    has_reasoning = bool(msg.get("reasoning"))
    print(f"  reasoning[:120]: {str(msg.get('reasoning'))[:120]!r}")
    print(f"  content[:120]  : {str(msg.get('content'))[:120]!r}")
    check("reasoning is in `reasoning` (not reasoning_content)", has_reasoning)
    if "reasoning_content" in msg:
        print("  note: this build also exposes reasoning_content")


def test_tool_call(base: str, model: str) -> None:
    print("\n== tool calling ==")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "北京现在天气怎么样？用工具查一下。"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "查询指定城市的实时天气",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string", "description": "城市名"}},
                    "required": ["city"],
                },
            },
        }],
        "tool_choice": "auto",
        "max_tokens": 400,
        "temperature": 0.3,
    }
    try:
        with post(base, payload) as r:
            j = json.load(r)
    except (urllib.error.URLError, OSError) as e:
        check("tool-call request", False, repr(e))
        return
    if "error" in j:
        check("tool call", False, str(j["error"])[:200])
        return
    ch = j["choices"][0]
    calls = ch["message"].get("tool_calls") or []
    print(f"  finish_reason: {ch.get('finish_reason')}")
    print(f"  tool_calls   : {json.dumps(calls, ensure_ascii=False)[:300]}")
    ok = bool(calls) and ch.get("finish_reason") == "tool_calls"
    if ok:
        try:
            json.loads(calls[0]["function"]["arguments"])
            check("tool call arguments are valid JSON", True)
        except json.JSONDecodeError:
            check("tool call arguments are valid JSON", False, calls[0]["function"]["arguments"])
    check("tool call emitted", ok)


def test_concurrency(base: str, model: str, n: int) -> None:
    """Note: with --max-num-seqs 1 these requests queue. Aggregate throughput is still
    meaningful (decode batching), but latency per request is not."""
    print(f"\n== {n}-way concurrency ==")
    results: list[dict | None] = [None] * n
    errors: list[str] = []

    def worker(i: int) -> None:
        try:
            results[i] = stream(base, {
                "model": model,
                "messages": [{"role": "user", "content": "讲一个 150 字左右的中国古代寓言故事。"}],
                "max_tokens": 220,
                "temperature": 0.7,
            })
        except (urllib.error.URLError, OSError) as e:
            errors.append(repr(e))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0

    ok = [r for r in results if r]
    if errors:
        check("all concurrent requests succeeded", False, errors[0][:150])
        return
    total_tokens = sum(r["completion_tokens"] for r in ok)
    print(f"  wall {wall:.2f}s  {len(ok)}/{n} ok  "
          f"{total_tokens} tokens  aggregate {total_tokens / wall:.1f} tok/s")
    if len(ok) > 1:
        spread = max(r["total"] for r in ok) - min(r["total"] for r in ok)
        print(f"  slowest-fastest spread: {spread:.2f}s (queueing is expected at max_num_seqs=1)")
    check("concurrency completed", len(ok) == n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:9000/v1")
    ap.add_argument("--model", default="deepseek-v4.1-flash-1017")
    ap.add_argument("--concurrency", type=int, default=0, help="also run an N-way test")
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["models", "nonstream", "stream", "reasoning", "tools", "concurrency"])
    args = ap.parse_args()

    print(f"base={args.base} model={args.model}\n")
    if "models" not in args.skip:
        test_models(args.base)
    if "nonstream" not in args.skip:
        test_nonstream(args.base, args.model)
    if "stream" not in args.skip:
        test_stream(args.base, args.model)
    if "reasoning" not in args.skip:
        test_reasoning_split(args.base, args.model)
    if "tools" not in args.skip:
        test_tool_call(args.base, args.model)
    if args.concurrency and "concurrency" not in args.skip:
        test_concurrency(args.base, args.model, args.concurrency)

    print()
    if FAILED:
        print(f"FAILED ({len(FAILED)}): {', '.join(FAILED)}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
