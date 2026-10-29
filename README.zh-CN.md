# 用 2×8 张 RTX 4090 跑 DeepSeek-V4.1-Flash-1017

![GPUs](https://img.shields.io/badge/GPU-16_×_RTX_4090_24GB-informational)
![Host RAM](https://img.shields.io/badge/内存-2_×_251_GiB-informational)
![Context](https://img.shields.io/badge/上下文-512K_tokens-success)
![vLLM](https://img.shields.io/badge/vLLM-0.28.1rc1_sm89_cu130-blueviolet)
![License](https://img.shields.io/badge/license-MIT_%2B_Apache--2.0-blue)

一套**已经跑通、可复现**的部署方案：把 475 GiB 的 **DeepSeek-V4.1-Flash-1017**
跨两台各 8×RTX 4090（24 GB）的机器跑起来，支持 **512K 上下文**，而机器之间只有
**2×1 Gbps 普通以太网 bond**——没有 InfiniBand，没有 NVLink，没有 NVSwitch。

这里所有数据都是实机实测的。**负面结果也一并记录**，因为那才是最贵的部分。

> **为什么建这个仓库。** 写这份文档的时候，我在整个 GitHub 上找不到任何一条这个模型
> 跑在 4090 上的公开记录。所有教程都假定 8×H100/H200 + NVLink + 1~2 TB 主机内存。
> 而下面这条约束链极其苛刻：它只留了唯一一个合法配置，并且 PyTorch 和 vLLM 在这个
> 量级上的若干默认行为会主动跟你对着干。这是我当时最想看到的那份地图。

---

## TL;DR

| 项 | 数值 |
|---|---|
| Checkpoint 体积 | 476 GB（48 个 safetensors 分片） |
| 并行方式 | **TP=8 × PP=2**，`world_size=16`，单引擎，`mp` executor |
| 最大上下文 | **524,288** tokens |
| KV cache 容量 | **651,867** tokens（`fp8_ds_mla`） |
| Prefill 吞吐 | 1.6k–2.2k tok/s |
| Decode 吞吐 | 56–66 tok/s（单流） |
| 长上下文召回 | ✅ 498,846 tokens 处埋针命中 |
| 稳态显存 | 每卡 23.9 / 24.0 GB |
| 冷启动 | **约 5 分 20 秒**（页缓存热时约 1 分钟） |
| 撰写时已连续运行 | 44 小时未重启 |

单流实测延迟（`temperature=0`）：

| prompt_tokens | TTFT | prefill tok/s | decode tok/s | 埋针召回 |
|---|---|---|---|---|
| 18,621 | 11.4 s | 1,630 | 56.9 | ✅ |
| 101,306 | 46.4 s | 2,183 | 58.6 | ✅ |
| **498,846** | **286.4 s** | 1,742 | 66.0 | ✅ |

---

## 硬件

| | 节点 A（driver） | 节点 B（worker） |
|---|---|---|
| GPU | 8 × RTX 4090 24 GB | 8 × RTX 4090 24 GB |
| 显存合计 | 192 GB | 192 GB |
| CPU | 96 线程 | 96 线程 |
| 主机内存 | 251 GiB，**无 swap** | 251 GiB |
| 驱动 | 595.84 | 595.84 |
| 机器间互联 | 2 × 1 Gbps bond（≈250 MB/s） | 同上 |

**这个互联带宽是整个方案最重要的设计输入。** 它直接否掉了跨机张量并行，
进而把配置唯一地锁死成下面这个样子。

---

## 为什么只能是这套配置

配置空间里只有一个合法点。每一步都是硬约束而非偏好——完整的推导过程和对应的
报错原文见 [`docs/01-constraint-derivation.md`](docs/01-constraint-derivation.md)。

```
475 GiB checkpoint
├── 189.1 GiB  Engram n-gram 表（第 1、14 层）→ pinned 主机内存，UVA 查表
├── 286.1 GiB  稠密 + FP4 专家权重            → GPU 显存
└──   7.39 GiB MTP/DSpark 草稿层              → 从未加载（见负面结果）
```

1. **必须双机。** 286 GiB 权重塞不进单机 192 GB 显存。也没有量化退路——专家已经是 FP4 了。
2. **`TP=16` 不合法。** `num_attention_heads=64`，而原生 attention kernel 只接受
   8/16/32 个 local query head。TP=16 时每卡只剩 4 个：
   `ValueError: DeepSeek V4.1 native attention supports 8, 16, or 32 local query heads, got 4.`
3. **所以是 `TP=8 × PP=2`。** TP=8 → 每卡 8 个 head（合法），PP=2 把 40 层切成 20/20。
   切在 20 层是安全的：`kv_source_layer_ids=[2,8,14,20]`、`candidate_source_layer_id=20`，
   跨层 KV 复用不会跨越 stage 边界。
4. **跨机用 PP 而不是 TP。** PP 每次前向只传一次激活；TP 要每层 all-reduce。
   在 250 MB/s 上这个差别是致命的。
5. **`--kv-cache-dtype fp8_ds_mla` 是必需项，不是优化项。** 它才是 512K 能成立的原因；
   bf16 KV 只有 187,974 tokens，差 3.5 倍。

---

## 你一定会需要的四个补丁

镜像自带的 DeepSeek-V4 支持在 24 GB 卡 / 256 GiB 主机上起不来。四个补丁都在
[`patches/`](patches/) 下，是 `git apply` 可用的 unified diff，已验证能干净应用，
且应用结果与线上生产文件**逐字节一致**。

| # | 目标文件 | 问题 | 改法 |
|---|---|---|---|
| 0001 | `deepseek_v4_1/common/engram.py` | **PyTorch 的 pinned 分配器会向上取整到 2 的幂。** 11.44 GiB 的表实占 16 GiB → 8 rank × 2 层白扔 **73 GiB** → 主机必然 OOM | 改用裸 `cudaHostAlloc` 精确分配（仍可 UVA 寻址） |
| 0002 | `deepseek_v4/nvidia/model.py` | VL MoE 路由偏置强制要求 `input_ids`，而显存 profiling / CUDA graph 的 dummy run 不带 → `ValueError` | 仅对 dummy run 合成全 0 token id |
| 0003 | `deepseek_v4_1/nvidia/router.py` | 同上，V4.1 router 里的同类分支 | 同上 |
| 0004 | `v1/worker/utils.py` | PP 末段还持有 MTP 草稿层的 KV spec，但它不属于任何 scheduler group → `next(...)` 抛 `StopIteration` → **静默启动失败** | 改为 warning + 跳过，而不是抛异常 |

第五个补丁（`0005`）修的是 `gpu_model_runner.py` 里两个只在投机解码下才触发的潜伏 bug。
它是**可选**的，当前配置用不到。

用 [`scripts/apply-patches.sh`](scripts/apply-patches.sh) 应用：它从镜像里抽出目标文件、
打上补丁、生成一个 overlay 目录，再由 `launch.sh` bind-mount 进去。**不需要重建镜像。**

---

## 五个值得一读的发现

这几条花的时间最多，而且每一条都反直觉到你可能也会踩。

### 1. 长上下文的瓶颈是**激活显存**，不是 KV cache

这一条颠覆常识。每张 24 GB 卡上权重吃掉 21.68 GiB，只剩 **约 1.2 GiB** 留给
KV + 激活 + CUDA graph。而 `fp8_ds_mla` 的 KV 只要约 1 KB/token，512K tokens 仅需
约 0.54 GiB——其实很小。但如果你让 `--gpu-memory-utilization` 把卡吃满，
KV 会抢走 0.79 GiB，激活只剩几十 MB，**32K prefill 就会 OOM**，炸在
`project_query_and_cache_kv()` 里一个 16 MiB 的分配上。

解法是反过来：用 **`--kv-cache-memory-bytes 560M`** 给 KV 设精确上限，
把约 0.6 GiB 留给激活。**减少 KV 预算，反而解锁了更长的上下文。**
详见 [`docs/04-long-context-512k.md`](docs/04-long-context-512k.md)。

### 2. `--max-num-batched-tokens` 必须是 1024，不能 2048

翻倍会让激活也翻倍，100K prefill 直接 OOM，并把整个 `EngineCore` 打死。
更阴的是：**容器仍然是 `Up` 状态，但 API 已经死了**，看起来就像"没崩"。
永远用 `/v1/models` 判断存活，不要用 `docker ps`。

### 3. Engram 的 pinned 表落在 `Shmem` 里，无 swap 时不可回收

189 GiB pinned 主机内存在 `/proc/meminfo` 里计为 `Shmem`，而节点 A 没有 swap——
所以永远回收不了。再叠上流式读 476 GB safetensors 产生的 page cache，就会触发内核
OOM killer，表现是**worker 无声消失**。必须有一个 1 Hz 的看门狗，在可用内存低于
60 GiB 时丢干净页缓存（`echo 1 > /proc/sys/vm/drop_caches`）。
见 [`scripts/memguard.sh`](scripts/memguard.sh)。

### 4. DSpark 投机解码与 PP=2 **互斥**

V4.1 的投机解码是 **DSpark**，不是经典 MTP（`mtp.*` 是个障眼法）。
这里开不了，原因是结构性的：

- `DSparkDeepseekV4ForCausalLM` 没有实现 `SupportsPP`
  （`class DSparkDeepseekV4ForCausalLM(nn.Module)`，`forward()` 没有
  `intermediate_tensors`），所以
  `draft_model_config.verify_with_parallel_config()` 直接抛
  `NotImplementedError: Pipeline parallelism is not supported for this model.`
- `draft_parallel_config.pipeline_parallel_size` 是**从 target 配置硬编码复制**的，
  没有任何 CLI 参数能让 draft 单独跑 PP=1。
- 而 PP=2 本身是必需的（见上）。

另外，对 `dspark.py` 执行
`grep -nE 'pp_group|PPMissingLayer|intermediate_tensors'` **一行都匹配不到**：
里面根本没有流水线并行的代码。这是作者的设计取舍，不是漏了一个开关。
详见 [`docs/06-negative-results.md`](docs/06-negative-results.md)。

### 5. `ngram` 投机解码在这里是**负优化**

它确实不占 GPU 显存（纯 CPU 的 `numpy`+`numba`，`additional_slots=0`，
KV 容量甚至涨到 683,749 tokens）。但它让一切都更慢了：

| 场景 | 关闭 ngram | 开启 ngram | 变化 |
|---|---|---|---|
| 纯生成 | 30.0 tok/s | 23.3 tok/s | **−22%** |
| 原样复述（最有利场景） | 37.0 tok/s | 21.8 tok/s | **−41%** |
| 抽取式摘要 | 47.0 tok/s | 20.4 tok/s | **−57%** |

接受率只有 **16.6%**（75 / 451，K=5 约 0.8 token/步）。两个叠加原因：
verify 阶段每一步要把 `K+1=6` 个 token 的激活推过 1 Gbps 链路；
并且 ngram 会强制从 model runner V2 回退到 V1。连它最擅长的场景都掉了 41%。
出货配置里是**关闭**的。

---

## 快速开始

```bash
# 0. 两台机器前置：8 张空卡、镜像、476 GB checkpoint
export MASTER_ADDR=10.0.0.1          # 节点 A
export WORKER_ADDR=10.0.0.2          # 节点 B
export MODEL_PATH=/models/DeepSeek-V4.1-Flash-1017
export IMAGE=ghcr.io/yhfgyyf/vllm-deepseek-v4-sm89:0.28.1rc1-vision11-sm89-sm120-cu130

# 1. 生成补丁 overlay（两台各做一次）
./scripts/apply-patches.sh

#    先预览真正的 docker argv，不会启动任何东西
DRY_RUN=1 ./scripts/launch.sh 0

# 2. 在两台机器上都起内存看门狗（节点 A 没有它会 OOM）
sudo nohup ./scripts/memguard.sh >/dev/null 2>&1 &

# 3. 先起节点 A（它是 master），再起节点 B
./scripts/launch.sh 0
ssh $WORKER_ADDR ./scripts/launch.sh 1     # 自动附加 --headless

# 4. 等待就绪（冷启动约 5m20s）
until curl -sf -o /dev/null http://$MASTER_ADDR:9000/v1/models; do sleep 10; done

# 5. 验证
./bench/smoke_test.py
./bench/longctx_needle.py 32000 100000
```

调用示例：

```bash
curl http://10.0.0.1:9000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash-1017",
       "messages":[{"role":"user","content":"你好"}],
       "max_tokens":256}'
```

同一个引擎对外提供两个模型名：`deepseek-v4.1-flash-1017` 和 `/qwen-32b`
（后者是为了让原先指向 SGLang 那套服务的客户端零改动切换，模型 id 里带斜杠是合法的）。

### 思维链与工具调用

`--reasoning-parser deepseek_v41` 与 `--tool-call-parser deepseek_v41` 都已开启。

```jsonc
{
  "message": {
    "content":   "只有最终答案",
    "reasoning": "思维链",                  // ← 不是 "reasoning_content"
    "tool_calls": [ /* finish_reason = "tool_calls" */ ]
  }
}
```

⚠️ 字段名是 **`reasoning`**，不是多数 vLLM 构建里的 `reasoning_content`。
如果你的客户端读的是 `reasoning_content`，改那一行。

---

## 目录结构

```
.
├── README.md                  # 英文
├── README.zh-CN.md            # 本文件
├── docs/
│   ├── 01-constraint-derivation.md   # 为什么是 TP=8×PP=2，含报错原文
│   ├── 02-patches.md                 # 每个补丁的根因分析
│   ├── 03-host-memory.md             # Shmem / pinned 页 / 无 swap OOM / memguard
│   ├── 04-long-context-512k.md       # 激活 vs KV 的预算算术
│   ├── 05-benchmarks.md              # 完整测量方法与原始数据
│   ├── 06-negative-results.md        # DSpark、ngram、MTP——什么行不通
│   └── 07-troubleshooting.md         # 现象 → 根因 → 解法
├── patches/                   # 5 个已验证补丁 + 说明
├── scripts/
│   ├── launch.sh              # 参数化启动脚本，longctx | fast 两套预设
│   ├── apply-patches.sh       # 镜像 → overlay 目录，无需重建
│   ├── memguard.sh            # 页缓存看门狗（节点 A 必须常驻）
│   ├── stop.sh
│   └── sampler.sh             # 3 秒采样，用于事后复盘
├── bench/
│   ├── smoke_test.py          # models / 非流式 / 流式 / 工具调用 / 思维链
│   ├── longctx_needle.py      # 埋针召回 + TTFT / decode 计时
│   └── ngram_ab.py            # 投机解码三场景 A/B
└── tools/
    ├── probe_config.py        # 打印 parser、KV dtype、投机方法、模型 config
    └── model_weight_report.py # 逐层 / 逐分片权重账目
```

---

## 已知局限（老实交代）

- **单序列并发。** `--max-num-seqs 1`，而且是真正的串行：4 路并发请求的墙钟时间约为 4 倍，
  聚合吞吐**不会**上升（实测 52.9 vs 55.5 tok/s）。只有约 1.2 GiB 余量，放不下第二条
  KV 流加它的 CUDA graph。这里的并发买的是公平性，不是吞吐——见
  [`docs/05-benchmarks.md`](docs/05-benchmarks.md#concurrency-it-does-not-scale-and-that-is-expected)。
- **`pp=2` 带来约 2 倍的 decode 延迟惩罚**（相比假想的单机 TP=8），因为每两层都要过网。
  在 250 MB/s 上无法避免；**换 10 GbE 是收益最高的硬件升级**。
- **`--kv-cache-memory-bytes` 会禁用显存 profiling**，所以 `--gpu-memory-utilization`
  被静默忽略（日志会写：*"skipped memory profiling, does not respect
  gpu_memory_utilization"*）。设了 KV 字节上限就别再调 util。
- **没有进程守护。** `restart=no`，无 systemd unit。机器重启后需要手动按
  memguard → 节点 A → 节点 B 的顺序拉起。见
  [`docs/07-troubleshooting.md`](docs/07-troubleshooting.md#recovery-after-reboot)。
- **只验证了一个镜像。** 官方 `vllm/vllm-openai:v0.32.0` 的模型注册表里完全没有
  `DeepseekV4` 架构。本方案基于上面那个社区 SM89 构建；把这些补丁移植到新版 vLLM
  需要自己重定位行号。

---

## 参与

如果你在不同硬件上跑通了——3090、A6000、更快的互联、或更新的 vLLM——欢迎开 issue
贴出你的数字。最有意思的变量是互联带宽：它决定了投机解码到底是帮忙还是添乱。

## 许可

脚本与文档：**MIT**。
`patches/` 下的内容是对 [vLLM](https://github.com/vllm-project/vllm)（Apache-2.0）的
衍生修改，见 [`NOTICE`](NOTICE)。DeepSeek-V4.1-Flash 的模型权重适用其自身许可，
**不在本仓库内**。

## 致谢

SM89/SM120 镜像和 `fp8_ds_mla` KV 布局来自 [`yhfgyyf`](https://github.com/yhfgyyf)
的社区构建——原版 vLLM 跑不了这个模型。Engram 架构是 DeepSeek 的。
