# 两台 8 卡 4090 跑 DeepSeek-V4.1-Flash-1017

![GPU](https://img.shields.io/badge/GPU-16_×_RTX_4090_24GB-informational)
![内存](https://img.shields.io/badge/内存-2_×_251_GiB-informational)
![上下文](https://img.shields.io/badge/上下文-512K_tokens-success)
![vLLM](https://img.shields.io/badge/vLLM-0.28.1rc1_sm89_cu130-blueviolet)
![License](https://img.shields.io/badge/license-MIT_%2B_Apache--2.0-blue)

把 475 GiB 的 DeepSeek-V4.1-Flash-1017 跑在两台各 8×RTX 4090（24 GB）的机器上，
上下文开到 512K。两台机器之间只有 2×1 Gbps 的普通以太网 bond，没有 InfiniBand，
也没有 NVLink。

下面的数字都是实机测出来的。跑不通的路子也写在里面了，省得别人再走一遍。

> 写这份文档的时候，GitHub 上找不到这个模型跑在 4090 上的公开记录，能查到的方案
> 都假设 8×H100/H200 + NVLink + 1~2 TB 主机内存。实际动手之后发现配置只有一种
> 可能，而且 PyTorch 和 vLLM 在这个量级上有好几处默认行为正好跟你反着来。所以整理成仓库。

---

## 结果概览

| 项 | 数值 |
|---|---|
| Checkpoint 体积 | 476 GB（48 个 safetensors 分片） |
| 并行方式 | TP=8 × PP=2，`world_size=16`，单引擎，`mp` executor |
| 最大上下文 | 524,288 tokens |
| KV cache 容量 | 651,867 tokens（`fp8_ds_mla`） |
| Prefill 吞吐 | 1.6k–2.2k tok/s |
| Decode 吞吐 | 56–66 tok/s（单流） |
| 长上下文召回 | 498,846 tokens 处埋针命中 |
| 稳态显存 | 每卡 23.9 / 24.0 GB |
| 冷启动 | 约 5 分 20 秒（页缓存热时约 1 分钟） |
| 持续运行 | 44 小时未重启 |

单流延迟实测（`temperature=0`）：

| prompt_tokens | TTFT | prefill tok/s | decode tok/s | 埋针召回 |
|---|---|---|---|---|
| 18,621 | 11.4 s | 1,630 | 56.9 | 是 |
| 101,306 | 46.4 s | 2,183 | 58.6 | 是 |
| **498,846** | **286.4 s** | 1,742 | 66.0 | 是 |

---

## 硬件

| | 节点 A（driver） | 节点 B（worker） |
|---|---|---|
| GPU | 8 × RTX 4090 24 GB | 8 × RTX 4090 24 GB |
| 显存合计 | 192 GB | 192 GB |
| CPU | 96 线程 | 96 线程 |
| 主机内存 | 251 GiB，无 swap | 251 GiB |
| 驱动 | 595.84 | 595.84 |
| 机器间互联 | 2 × 1 Gbps bond（约 250 MB/s） | 同上 |

最后那行的带宽基本决定了整套方案。它把跨机张量并行排除掉了，配置也就只剩一种。

---

## 为什么配置只能这么选

每一步都是硬约束，不是调优偏好。完整推导和对应的报错原文见
[`docs/01-constraint-derivation.md`](docs/01-constraint-derivation.md)。

```
475 GiB checkpoint
├── 189.1 GiB  Engram n-gram 表（第 1、14 层）→ pinned 主机内存，UVA 查表
├── 286.1 GiB  稠密 + FP4 专家权重            → GPU 显存
└──   7.39 GiB MTP/DSpark 草稿层              → 从未加载（见后文）
```

1. **必须双机。** 286 GiB 权重塞不进单机 192 GB 显存，也没有量化退路，专家已经是 FP4。
2. **TP=16 不合法。** `num_attention_heads=64`，原生 attention kernel 只接受 8/16/32
   个 local query head。TP=16 时每卡只剩 4 个，直接报
   `ValueError: DeepSeek V4.1 native attention supports 8, 16, or 32 local query heads, got 4.`
3. **所以是 TP=8 × PP=2。** TP=8 时每卡 8 个 head，合法；PP=2 把 40 层切成 20/20。
   切在 20 层是安全的，`kv_source_layer_ids=[2,8,14,20]`、`candidate_source_layer_id=20`，
   跨层 KV 复用不会越过 stage 边界。
4. **跨机用 PP 不用 TP。** PP 每次前向只传一次激活，TP 要每层 all-reduce。
   250 MB/s 的链路上这个差别太大。
5. **`--kv-cache-dtype fp8_ds_mla` 是必需项。** 512K 能成立靠的就是它，
   bf16 KV 只能到 187,974 tokens，差 3.5 倍。

---

## 用的哪个镜像

官方 vLLM 跑不了这个模型，用的是社区构建的 SM89 镜像：

```
ghcr.io/yhfgyyf/vllm-deepseek-v4-sm89:0.28.1rc1-vision11-sm89-sm120-cu130
```

先把 tag 拆开看：`0.28.1rc1` 是 vLLM 基线版本，`vision11` 表示带视觉输入支持，
`sm89-sm120` 是编译覆盖的算力架构（4090 是 SM89，Blackwell 是 SM120），`cu130`
是 CUDA 13.0。

线上实际用的那份镜像，元数据如下：

| 项 | 值 |
|---|---|
| digest | `sha256:89d95823…`，11.9 GB，2026-09-16 构建 |
| vLLM 版本 | `0.28.1rc1.dev517+glm53.dsv41.vision11.sm89sm120.cu130` |
| PyTorch | `2.13.0+cu130` |
| 基础镜像 | Ubuntu 24.04 + NVIDIA CUDA 13.0.3（devel） |
| 构建源码 | <https://github.com/yhfgyyf/vllm-deepseek-v4-sm89> |
| ENTRYPOINT | `vllm serve`，所以命令行直接接模型路径 |

为什么只能用这个：官方 `vllm/vllm-openai:v0.32.0` 的模型注册表里根本没有
`DeepseekV4` 这个架构（在 `_VLLM_MODELS` 里 grep 不到），也没有 `fp8_ds_mla`
这个 KV 布局，`--kv-cache-dtype` 这个参数等于不存在。按镜像标签的说明，这个构建
还一起带了 DeepSeek-V4-Flash/Vision 和 GLM-5.3-Flash，一份镜像同时覆盖 SM89 和
SM120 两种卡。

需要提醒的是，`patches/` 里的补丁是针对这个镜像里的文件写的。换成别的构建，
行号会全部错位，得重新定位。

---

## 需要的四个补丁

镜像自带的 DeepSeek-V4 支持在 24 GB 显存 / 256 GiB 内存上起不来。四个补丁在
[`patches/`](patches/) 下，是 `git apply` 能用的 unified diff，已验证可以干净应用，
打完之后和线上跑的文件逐字节一致。

| # | 目标文件 | 问题 | 改法 |
|---|---|---|---|
| 0001 | `deepseek_v4_1/common/engram.py` | PyTorch 的 pinned 分配器会向上取整到 2 的幂。11.44 GiB 的表实际占 16 GiB，8 rank × 2 层多占 73 GiB，主机必然 OOM | 改用裸 `cudaHostAlloc` 精确分配，仍可 UVA 寻址 |
| 0002 | `deepseek_v4/nvidia/model.py` | VL MoE 路由偏置要求传 `input_ids`，但显存 profiling 和 CUDA graph 的 dummy run 不带，抛 `ValueError` | 只在 dummy run 时合成全 0 token id |
| 0003 | `deepseek_v4_1/nvidia/router.py` | 同上，V4.1 router 里是同一个分支 | 同上 |
| 0004 | `v1/worker/utils.py` | PP 末段还持有 MTP 草稿层的 KV spec，但它不属于任何 scheduler group，`next(...)` 抛 `StopIteration`，表现为静默启动失败 | 改成 warning 后跳过 |

第五个补丁 `0005` 修 `gpu_model_runner.py` 里两个只在投机解码下才触发的潜伏 bug，
可选，当前配置用不到。

用 [`scripts/apply-patches.sh`](scripts/apply-patches.sh) 应用：从镜像里抽出目标文件、
打补丁、生成 overlay 目录，`launch.sh` 再 bind-mount 进去，不用重建镜像。

---

## 几个踩坑记录

### 1. 长上下文卡的是激活显存，不是 KV cache

这一条跟直觉相反。每张卡上权重占掉 21.68 GiB，只剩约 1.2 GiB 给 KV、激活和 CUDA graph。
`fp8_ds_mla` 的 KV 只要约 1 KB/token，512K tokens 也就 0.54 GiB，其实很小。但如果让
`--gpu-memory-utilization` 把卡吃满，KV 会占掉 0.79 GiB，激活只剩几十 MB，32K prefill
就会 OOM，炸在 `project_query_and_cache_kv()` 里一个 16 MiB 的分配上。

解法是反过来：用 `--kv-cache-memory-bytes 560M` 给 KV 设精确上限，把约 0.6 GiB 留给激活。
减少 KV 预算，反而能开更长的上下文。详见
[`docs/04-long-context-512k.md`](docs/04-long-context-512k.md)。

### 2. `--max-num-batched-tokens` 只能是 1024，不能 2048

翻倍会让激活也翻倍，100K prefill 直接 OOM，整个 `EngineCore` 被打死。麻烦的是
容器还是 `Up` 状态，API 已经没了，看着像是没崩。判断存活要用 `/v1/models`，别用 `docker ps`。

### 3. Engram 的 pinned 表算在 `Shmem` 里，没 swap 就回收不掉

189 GiB pinned 主机内存在 `/proc/meminfo` 里计为 `Shmem`，节点 A 没有 swap，所以永远
回收不了。再叠上流式读 476 GB safetensors 产生的 page cache，就会触发内核 OOM killer，
表现是 worker 无声消失。需要一个 1 Hz 的看门狗，可用内存低于 60 GiB 时丢干净页缓存
（`echo 1 > /proc/sys/vm/drop_caches`）。见 [`scripts/memguard.sh`](scripts/memguard.sh)。

### 4. DSpark 投机解码和 PP=2 不能同时用

V4.1 的投机解码是 DSpark，不是经典 MTP（`mtp.*` 那个是障眼法）。这里开不了，
原因是结构性的：

- `DSparkDeepseekV4ForCausalLM` 没有实现 `SupportsPP`
  （`class DSparkDeepseekV4ForCausalLM(nn.Module)`，`forward()` 没有
  `intermediate_tensors`），`draft_model_config.verify_with_parallel_config()` 直接抛
  `NotImplementedError: Pipeline parallelism is not supported for this model.`
- `draft_parallel_config.pipeline_parallel_size` 是从 target 配置硬编码复制的，
  没有 CLI 参数能让 draft 单独跑 PP=1。
- 而 PP=2 本身是必需的，见上面。

另外对 `dspark.py` 执行 `grep -nE 'pp_group|PPMissingLayer|intermediate_tensors'`
一行都匹配不到，里面没有流水线并行的代码。这是作者的设计取舍，不是漏了开关。
详见 [`docs/06-negative-results.md`](docs/06-negative-results.md)。

### 5. ngram 投机解码在这里是负优化

它确实不占 GPU 显存（纯 CPU 的 `numpy`+`numba`，`additional_slots=0`，KV 容量还涨到
683,749 tokens），但让一切都更慢了：

| 场景 | 关闭 ngram | 开启 ngram | 变化 |
|---|---|---|---|
| 纯生成 | 30.0 tok/s | 23.3 tok/s | −22% |
| 原样复述（最有利场景） | 37.0 tok/s | 21.8 tok/s | −41% |
| 抽取式摘要 | 47.0 tok/s | 20.4 tok/s | −57% |

接受率只有 16.6%（75 / 451，K=5 约 0.8 token/步）。两个原因叠加：verify 阶段每一步要把
K+1=6 个 token 的激活推过 1 Gbps 链路，另外 ngram 会强制从 model runner V2 退回 V1。
连它最擅长的场景都掉了 41%。默认配置里是关的。

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

#    先预览 docker argv，不会启动任何东西
DRY_RUN=1 ./scripts/launch.sh 0

# 2. 两台都起内存看门狗（节点 A 没有它会 OOM）
sudo nohup ./scripts/memguard.sh >/dev/null 2>&1 &

# 3. 先起节点 A（master），再起节点 B
./scripts/launch.sh 0
ssh $WORKER_ADDR ./scripts/launch.sh 1     # 自动附加 --headless

# 4. 等就绪（冷启动约 5m20s）
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

### 思维链和工具调用

`--reasoning-parser deepseek_v41` 和 `--tool-call-parser deepseek_v41` 都已开启。

```jsonc
{
  "message": {
    "content":   "只有最终答案",
    "reasoning": "思维链",                  // 不是 "reasoning_content"
    "tool_calls": [ /* finish_reason = "tool_calls" */ ]
  }
}
```

字段名是 `reasoning`，不是多数 vLLM 构建里的 `reasoning_content`。客户端读
`reasoning_content` 的话改这一行就行。

---

## 目录结构

```
.
├── README.md                  # 英文
├── README.zh-CN.md            # 本文件
├── docs/
│   ├── 01-constraint-derivation.md   # 为什么是 TP=8×PP=2，含报错原文
│   ├── 02-patches.md                 # 每个补丁的根因
│   ├── 03-host-memory.md             # Shmem / pinned 页 / 无 swap OOM / memguard
│   ├── 04-long-context-512k.md       # 激活和 KV 的预算算术
│   ├── 05-benchmarks.md              # 测量方法和原始数据
│   ├── 06-negative-results.md        # DSpark、ngram、MTP 实测结果
│   └── 07-troubleshooting.md         # 现象 → 根因 → 解法
├── patches/                   # 5 个已验证补丁
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

## 已知限制

- **只支持单序列并发。** `--max-num-seqs 1`，而且是真串行：4 路并发请求的墙钟时间
  约为 4 倍，聚合吞吐不涨（实测 52.9 vs 55.5 tok/s）。只剩约 1.2 GiB 余量，放不下
  第二条 KV 流和它的 CUDA graph。这里的并发买的是公平性，不是吞吐，见
  [`docs/05-benchmarks.md`](docs/05-benchmarks.md#concurrency-it-does-not-scale-and-that-is-expected)。
- **`pp=2` 带来约 2 倍的 decode 延迟**，相比假想的单机 TP=8，因为每两层都要过网。
  250 MB/s 上没法避免。换 10 GbE 是收益最高的硬件升级。
- **`--kv-cache-memory-bytes` 会让显存 profiling 失效**，所以 `--gpu-memory-utilization`
  被静默忽略（日志里会写 *"skipped memory profiling, does not respect
  gpu_memory_utilization"*）。设了 KV 字节上限就别再调 util。
- **没有进程守护。** `restart=no`，没有 systemd unit。机器重启后要手动按
  memguard → 节点 A → 节点 B 的顺序拉起来，见
  [`docs/07-troubleshooting.md`](docs/07-troubleshooting.md#recovery-after-reboot)。
- **只在 SM89 上验证过。** 镜像虽然也编了 SM120，但我们手上没有 Blackwell 的机器
  实测。换新版 vLLM 的话，补丁得重新定位行号。

---

## 参与

如果在别的硬件上跑通了，比如 3090、A6000、更快的互联或者更新的 vLLM，欢迎开 issue
贴一下数字。最有参考价值的是互联带宽，它决定了投机解码到底帮忙还是添乱。

## 许可

脚本和文档：MIT。

`patches/` 下的内容是对 [vLLM](https://github.com/vllm-project/vllm)（Apache-2.0）的
修改，见 [`NOTICE`](NOTICE)。DeepSeek-V4.1-Flash 的模型权重适用其自身许可，不在本仓库内。

## 致谢

SM89/SM120 镜像和 `fp8_ds_mla` KV 布局来自 [`yhfgyyf`](https://github.com/yhfgyyf)
的社区构建，原版 vLLM 跑不了这个模型。Engram 架构是 DeepSeek 的。
