# SGLang BytePS All-Reduce 实现说明

## Status

Accepted

## Date

2026-05-23

## Context

本次修改为 SGLang 增加一个可选的 BytePS All-Reduce 后端，用于第一阶段替换模型计算主路径中的 All-Reduce 通信。目标是复用 Megatron-DPU 中 BytePS wrapper 的实现思路，在 SGLang model worker 内完成 BytePS local 环境设置和初始化，并通过 `GroupCoordinator.all_reduce()` 的统一分发入口接管模型主路径 All-Reduce。

第一阶段范围收敛如下：

- 只替换模型计算主路径 All-Reduce。
- 不替换 scheduler、cache、speculative 等控制面直接调用的 `torch.distributed.all_reduce()`。
- 暂不替换 reduce-scatter、all-gather、broadcast、send/recv。
- 暂不替换 `quant_all_reduce()`。
- 首测拓扑为单机多 GPU。
- BytePS 第一阶段不支持 CUDA graph capture。开启 BytePS 后，遇到 CUDA graph / piecewise CUDA graph 必须显式报错，不允许 fallback。

## Decision Drivers

- **显式开关**：通过 `--use-byteps-all-reduce` 控制，默认行为不变。
- **无静默 fallback**：开启 BytePS 后，模型主路径 All-Reduce 要么走 BytePS，要么抛出明确错误。
- **稳定命名**：BytePS tensor name 必须跨 rank 一致，不能使用 `id(self)` 等进程本地值。
- **局部改动**：优先复用 SGLang 现有 `GroupCoordinator` 路由模型，不重构无关路径。
- **单机优先**：第一阶段只校验单机多 GPU，避免引入外层 `bpslaunch` 包 SGLang server。

## Decision

新增 `--use-byteps-all-reduce` 和 `--byteps-all-reduce-debug` 两个 SGLang server 参数。开启后，在 `ModelRunner` 分布式初始化流程中设置 BytePS local env，初始化 BytePS，并打开 `parallel_state.py` 中的 BytePS All-Reduce 全局开关。

模型主路径 All-Reduce 仍从 SGLang 原有 wrapper 进入：

```text
communication_op.py / model layers
    -> GroupCoordinator.all_reduce(logical_name=...)
    -> byteps_collectives.byteps_allreduce_inplace(...)
    -> byteps.torch.ops.push_pull_async_inplace(...)
    -> byteps.torch.ops.synchronize(...)
```

BytePS 使用 SUM 语义，即 `average=False`，以匹配 SGLang TP / attention TP / MoE All-Reduce 对部分结果求和的语义。

## Modified Files

| 文件 | 修改内容 |
|---|---|
| `sglang/python/sglang/srt/distributed/byteps_collectives.py` | 新增 BytePS wrapper，包含幂等初始化、declare 缓存、group name 构造、`push_pull_async_inplace + synchronize`。 |
| `sglang/python/sglang/srt/server_args.py` | 新增 `use_byteps_all_reduce`、`byteps_all_reduce_debug` dataclass 字段和 CLI 参数。 |
| `sglang/python/sglang/srt/model_executor/model_runner.py` | 在 model worker 初始化中设置 BytePS local env，初始化 BytePS，并调用 `set_byteps_all_reduce(...)`。 |
| `sglang/python/sglang/srt/distributed/parallel_state.py` | 新增 BytePS 全局开关；`GroupCoordinator.all_reduce()` 增加 `logical_name`；开启 BytePS 后进入硬约束 BytePS 路由。 |
| `sglang/python/sglang/srt/distributed/communication_op.py` | TP、attention TP、MoE All-Reduce wrapper 增加可选 `logical_name`。 |
| `sglang/python/sglang/srt/layers/linear.py` | RowParallelLinear 主路径 All-Reduce 传递稳定 logical name。 |
| `sglang/python/sglang/srt/layers/vocab_parallel_embedding.py` | VocabParallelEmbedding 主路径 All-Reduce 传递稳定 logical name。 |

## Implementation Notes

### BytePS 初始化

`initialize_byteps_for_sglang(local_rank, local_size, debug)` 负责设置 BytePS 必需环境变量并执行幂等初始化：

- `BYTEPS_LOCAL_RANK`
- `BYTEPS_LOCAL_SIZE`
- `DMLC_ROLE=worker`
- `DMLC_WORKER_ID=0`
- `DMLC_NUM_WORKER=1`
- `DMLC_NUM_SERVER=0`

初始化后会校验：

- `bps.local_rank()` 是否等于 SGLang worker 的 `gpu_id`。
- `bps.local_size()` 是否等于 `tp_size * pp_size`。
- `bps.size()` 是否等于 `tp_size * pp_size`，以限制第一阶段单机多 GPU 范围。

### BytePS name 规则

BytePS group name 由三部分组成：

```text
sglang.{group.unique_name}.r{rank_fingerprint}.{logical_name}
```

- `group.unique_name`：SGLang 现有 group 唯一名，如 TP / attention TP group。
- `rank_fingerprint`：对 group ranks 做 SHA1 短指纹，避免不同 group 使用相同 logical name 时冲突。
- `logical_name`：模型层传入的稳定名称。

`declare_and_cache_byteps_group()` 会缓存 name 和 `expected_workers`，若同名 tensor 被不同 group size 声明，会抛出错误。

### All-Reduce 路由

未启用 `--use-byteps-all-reduce` 时，`GroupCoordinator.all_reduce()` 保持原有 SGLang 路径不变。

启用后，`GroupCoordinator.all_reduce()` 在任何 custom / PyNccl / MSCCLPP / TorchSymmMem / torch.distributed 路由之前进入 BytePS guard：

```text
world_size == 1
    -> 直接返回 input

use_byteps_all_reduce == true
    -> 校验 CUDA graph、tensor/device/contiguous、特殊 communicator、symmetric memory
    -> byteps_allreduce_inplace(input, group, logical_name)

use_byteps_all_reduce == false
    -> 继续原 SGLang backend selection
```

### logical_name 生成

`linear.py` 和 `vocab_parallel_embedding.py` 优先使用已有 `prefix`：

```text
linear.{prefix}.all_reduce
embedding.{prefix}.all_reduce
```

没有 `prefix` 时使用模块级 deterministic counter：

```text
linear.{ClassName}.{counter}.all_reduce
embedding.{ClassName}.{counter}.all_reduce
```

进入不同通信 group 时再加 scope 前缀：

- 普通 TP：`tp.{...}`
- attention TP：`attn_tp.{...}`

这样可以避免 DP-Attention 下 attention TP group 与普通 TP group 复用同一个 BytePS logical name。

## Explicit Error Paths

开启 `--use-byteps-all-reduce` 后，以下模型主路径 All-Reduce 场景会直接抛出 `RuntimeError`，不允许 fallback：

- 正在进行 CUDA graph capture：提示使用 `--disable-cuda-graph`。
- 正在进行 piecewise CUDA graph：提示使用 `--disable-piecewise-cuda-graph`。
- 输入不是 CUDA tensor，例如 CPU tensor。
- 输入 tensor 非 contiguous。
- HPU / XPU / NPU communicator 路径处于激活状态。
- symmetric memory 路径处于激活状态。
- `quant_all_reduce()` 被调用：提示 BytePS 第一阶段暂不支持量化通信。

开启 BytePS 后，custom all-reduce、PyNccl、MSCCLPP、TorchSymmMem、torch.distributed fallback 等非 BytePS 路径不会被继续尝试。

## Consequences

### Positive

- BytePS All-Reduce 可通过单一 CLI 开关启用，默认 SGLang 行为不变。
- 模型主路径通信后端切换集中在 `GroupCoordinator.all_reduce()`，调用侧改动较小。
- BytePS name 具备 group 隔离和稳定 logical name，便于 declare 缓存和调试。
- 不满足 BytePS 条件时直接报错，有利于首测阶段尽早暴露不支持路径。

### Negative

- 第一阶段不支持 CUDA graph，会要求启动参数禁用 CUDA graph / piecewise CUDA graph。
- `quant_all_reduce()` 暂不支持，启用量化通信时会报错。
- 当前只按单机多 GPU 初始化 BytePS，多机扩展需要后续补充 DMLC worker/server 编排。

### Risks

- 没有 prefix 的模块依赖构造顺序一致来保证 deterministic counter 跨 rank 一致。
- 仍有部分模型文件直接调用 `tensor_model_parallel_all_reduce()` 且未提供业务级 logical name，此类调用会使用 `GroupCoordinator` 的 generic fallback name。后续如发现 name 复用风险，应继续补稳定名称。
- BytePS phase 1 与 SGLang CUDA graph 路径互斥，性能对比需要在 eager 模式下进行。

## Verification

本轮只做静态/语法检查，没有运行功能测试、集成测试，也没有启动 SGLang 服务。

已执行：

```bash
env PYTHONPYCACHEPREFIX=/private/tmp/sglang_byteps_pycache python3 -m py_compile \
  sglang/python/sglang/srt/distributed/byteps_collectives.py \
  sglang/python/sglang/srt/server_args.py \
  sglang/python/sglang/srt/model_executor/model_runner.py \
  sglang/python/sglang/srt/distributed/parallel_state.py \
  sglang/python/sglang/srt/distributed/communication_op.py \
  sglang/python/sglang/srt/layers/linear.py \
  sglang/python/sglang/srt/layers/vocab_parallel_embedding.py
```

结果：语法检查通过。

未运行：

- SGLang server 启动。
- 功能测试。
- 集成测试。
- BytePS 多 GPU 实测。
- CUDA graph / piecewise CUDA graph 实测。

## Related Work

- `docs/superpowers/plans/2026-05-23-sglang-byteps-allreduce.md`
- `docs/byteps_megatron_changes.md`
- `sglang/docs/allreduce_communication.md`
