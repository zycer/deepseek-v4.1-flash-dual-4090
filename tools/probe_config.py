#!/usr/bin/env python3
"""Dump the facts you need before choosing flags: which parsers exist, which KV cache
dtypes the build accepts, which speculative methods are registered, and what the model
config actually says.

Run INSIDE the container (it imports vllm):

    docker cp probe_config.py dsv41:/tmp/
    docker exec dsv41 python3 /tmp/probe_config.py /model

Guessing these from memory is how people end up passing --reasoning-parser with a name
that does not exist in that build.
"""
from __future__ import annotations

import json
import os
import sys

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/model"


def section(title: str) -> None:
    print(f"\n=== {title} ===")


section("vllm version")
try:
    import vllm
    print(vllm.__version__)
except Exception as e:  # noqa: BLE001
    print("ERR", e)

section("reasoning parsers")
try:
    from vllm.reasoning import ReasoningParserManager
    print(sorted(ReasoningParserManager.list_registered()))
except Exception as e:  # noqa: BLE001
    print("ERR", e)

section("tool parsers")
try:
    from vllm.tool_parsers import ToolParserManager
    print(sorted(ToolParserManager.list_registered()))
except Exception as e:  # noqa: BLE001
    print("ERR", e)

section("speculative methods")
try:
    from vllm.config.speculative import SpeculativeMethod
    print([m.value for m in SpeculativeMethod])
except Exception as e:  # noqa: BLE001
    print("ERR", e)

section("kv cache dtypes")
try:
    from vllm.config.cache import CacheDType
    print([m.value for m in CacheDType])
except Exception as e:  # noqa: BLE001
    print("ERR", e)

section("registered model architectures (deepseek only)")
try:
    from vllm.model_executor.models.registry import _VLLM_MODELS
    hits = [k for k in _VLLM_MODELS if "eepseek" in k or "spark" in k.lower()]
    print(sorted(hits))
except Exception as e:  # noqa: BLE001
    print("ERR", e)

section("model config")
cfg_path = os.path.join(MODEL, "config.json")
try:
    with open(cfg_path) as f:
        cfg = json.load(f)
except OSError as e:
    print("ERR", e)
    sys.exit(0)

KEYS = [
    "architectures", "num_hidden_layers", "max_position_embeddings",
    "num_attention_heads", "num_key_value_heads",
    "kv_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim", "v_head_dim",
    "kv_source_layer_ids", "index_source_layer_ids", "candidate_source_layer_id",
    "num_nextn_predict_layers", "n_routed_experts", "num_experts_per_tok",
    "torch_dtype", "quantization_config",
]
for k in KEYS:
    if k in cfg:
        v = cfg[k]
        if len(str(v)) > 240:
            v = str(v)[:240] + "..."
        print(f"{k} = {v}")

print("\n--- other top-level keys ---")
print(sorted(set(cfg) - set(KEYS)))

# The two numbers that decide the parallel layout.
heads = cfg.get("num_attention_heads")
if heads:
    print(f"\n--- local query heads per TP rank (legal set is 8/16/32) ---")
    for tp in (1, 2, 4, 8, 16, 32):
        per = heads // tp if heads % tp == 0 else None
        ok = "OK" if per in (8, 16, 32) else ("--" if per is not None else "not divisible")
        print(f"  TP={tp:<3} heads/rank={per if per is not None else '-':<4} {ok}")

# Whether the draft model can be pipelined -- decides DSpark feasibility.
section("pipeline-parallel support of draft models")
try:
    from vllm.model_executor.models.registry import ModelRegistry
    reg = ModelRegistry()
    for arch in ("DSparkDeepseekV4ForCausalLM", "DeepseekV4ForCausalLM"):
        try:
            info = reg.resolve_model_cls([arch])
            supported = getattr(info[1] if isinstance(info, tuple) else info, "supports_pp", "?")
            print(f"  {arch}: supports_pp={supported}")
        except Exception as e:  # noqa: BLE001
            print(f"  {arch}: ERR {e}")
except Exception as e:  # noqa: BLE001
    print("ERR", e)
