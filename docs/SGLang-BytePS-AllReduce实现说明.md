# SGLang BytePS All-Reduce 实现说明

## 状态

已实现第一阶段代码改动。当前改动只覆盖 SGLang 模型计算主路径 All-Reduce，不包含功能测试、集成测试或服务启动验证。

## 日期

2026-05-25

## 变更范围

本轮为 `sglang-0.5.10.post1` 增加一个显式开启的 BytePS All-Reduce 路径。开关关闭时，SGLang 原有通信路径保持不变；开关开启后，模型主路径经 `GroupCoordinator.all_reduce()` 进入 BytePS，不允许静默退回 custom All-Reduce、PyNccl、MSCCLPP、TorchSymmMem 或 `torch.distributed.all_reduce()`。

第一阶段只处理这些内容：

- 替换模型计算主路径 All-Reduce。
- 保留 scheduler、cache、speculative 等控制面中直接调用的 `torch.distributed.all_reduce()`。
- 保留 reduce-scatter、all-gather、broadcast、send/recv。
- 不实现 CUDA graph capture 支持。遇到 CUDA graph 或 piecewise CUDA graph 时直接报错。
- 首测拓扑按单机多 GPU 设计。
- 不使用外层 `bpslaunch` 包 SGLang server，而是在 SGLang model worker 内设置 BytePS local env 并初始化 BytePS。

## 修改文件

| 文件 | 修改内容 |
| --- | --- |
| `sglang-0.5.10.post1/python/sglang/srt/distributed/byteps_collectives.py` | 新增 BytePS wrapper，包含幂等初始化、declare 缓存、group name 构造、`push_pull_async_inplace + synchronize`。 |
| `sglang-0.5.10.post1/python/sglang/srt/server_args.py` | 新增 `use_byteps_all_reduce`、`byteps_all_reduce_debug` 字段和对应 CLI 参数。 |
| `sglang-0.5.10.post1/python/sglang/srt/model_executor/model_runner.py` | 在 model worker 分布式初始化流程中设置 BytePS local env、初始化 BytePS，并调用 `set_byteps_all_reduce(...)`。 |
| `sglang-0.5.10.post1/python/sglang/srt/distributed/parallel_state.py` | 新增 BytePS 全局开关；`GroupCoordinator.all_reduce()` 增加 `logical_name` 参数；开启 BytePS 后执行强制 BytePS 路由和错误检查。 |
| `sglang-0.5.10.post1/python/sglang/srt/distributed/communication_op.py` | TP、attention TP、MoE All-Reduce wrapper 增加可选 `logical_name` 参数。 |
| `sglang-0.5.10.post1/python/sglang/srt/layers/linear.py` | `RowParallelLinear` 主路径 All-Reduce 传递稳定 BytePS logical name。 |
| `sglang-0.5.10.post1/python/sglang/srt/layers/vocab_parallel_embedding.py` | `VocabParallelEmbedding` 主路径 All-Reduce 传递稳定 BytePS logical name。 |

## CLI 参数

新增两个参数：

```bash
--use-byteps-all-reduce
--byteps-all-reduce-debug
```

`--use-byteps-all-reduce` 控制是否启用 BytePS All-Reduce。默认关闭。

`--byteps-all-reduce-debug` 用于输出 BytePS 初始化、declare 和路由相关日志。默认关闭。

## 初始化流程

BytePS 初始化发生在 `ModelRunner` 的分布式初始化流程中。

当前顺序是：

```text
set_custom_all_reduce(...)
set_mscclpp_all_reduce(...)
set_torch_symm_mem_all_reduce(...)
set_byteps_all_reduce(False)

如果开启 --use-byteps-all-reduce:
    设置 BYTEPS_LOCAL_RANK = gpu_id
    设置 BYTEPS_LOCAL_SIZE = tp_size * pp_size

init_distributed_environment(...)
initialize_model_parallel(...)

如果开启 --use-byteps-all-reduce:
    initialize_byteps_for_sglang(...)
    set_byteps_all_reduce(True, debug=...)
```

`initialize_byteps_for_sglang()` 当前会通过 `setdefault` 设置：

```text
BYTEPS_LOCAL_RANK
BYTEPS_LOCAL_SIZE
DMLC_ROLE=worker
DMLC_NUM_WORKER=1
DMLC_WORKER_ID=0
```

随后调用 `bps.init()`。该初始化是进程内幂等的，由 `_BPS_INITIALIZED` 保护。

初始化后会校验：

- `bps.local_rank()` 是否等于当前 SGLang worker 的 `gpu_id`。
- `bps.local_size()` 是否等于 `tp_size * pp_size`。

当前代码没有设置 `DMLC_NUM_SERVER`，也没有校验 `bps.size()`。

## BytePS collective wrapper

新增文件 `byteps_collectives.py` 提供三个核心能力。

**1. declare 缓存**

`declare_and_cache_byteps_group(name, expected_workers)` 会记录 BytePS tensor name 对应的 worker 数量。第一次出现时调用：

```python
bps.declare(name, expected_workers=expected_workers)
```

后续如果同名 tensor 使用了不同 `expected_workers`，直接抛出 `RuntimeError`。

**2. 稳定 group name**

BytePS name 由 SGLang group 信息和 logical name 组成：

```text
sglang.{group.unique_name}.r{rank_fingerprint}.{logical_name}
```

其中：

- `group.unique_name` 来自 SGLang 的 `GroupCoordinator`。
- `rank_fingerprint` 是 group ranks 的 SHA1 短指纹。
- `logical_name` 来自上层调用点。

这样普通 TP group、attention TP group、MoE group 即使使用相同 logical name，也会因为 group name 或 ranks 指纹不同而隔离。

**3. SUM 语义**

BytePS All-Reduce 使用：

```python
bps_ops.push_pull_async_inplace(
    tensor,
    average=False,
    name=name,
    version=0,
    priority=0,
)
bps_ops.synchronize(handle)
```

`average=False` 保持 SUM 语义，匹配 SGLang TP / attention TP / MoE All-Reduce 对部分结果求和的行为。

## All-Reduce 路由

`GroupCoordinator.all_reduce()` 增加了可选参数：

```python
def all_reduce(self, input_: torch.Tensor, logical_name: Optional[str] = None)
```

未开启 `--use-byteps-all-reduce` 时，原有 SGLang 逻辑不变：

```text
CPU / shared memory
HPU / XPU / NPU communicator
symmetric memory + PyNccl
custom All-Reduce / quick All-Reduce
MSCCLPP
TorchSymmMem
piecewise CUDA graph + PyNccl outplace
PyNccl / TorchSymmMem / torch.distributed fallback
```

开启 `--use-byteps-all-reduce` 后，`world_size == 1` 仍直接返回输入；其他模型主路径 All-Reduce 先进入 BytePS guard。满足条件时调用：

```python
byteps_allreduce_inplace(input_, self, logical_name)
```

不满足条件时直接报错，不再尝试原有 fallback 路径。

## 显式报错路径

开启 `--use-byteps-all-reduce` 后，以下情况会抛出 `RuntimeError`：

- CUDA graph capture 被请求或正在进行。错误信息提示使用 `--disable-cuda-graph`。
- piecewise CUDA graph 正在进行。错误信息提示使用 `--disable-piecewise-cuda-graph`。
- 输入是 CPU tensor。
- 输入不是 CUDA tensor。
- 输入 tensor 非 contiguous。
- HPU、XPU、NPU communicator 路径处于激活状态。
- symmetric memory 路径处于激活状态。
- fused All-Reduce + RMSNorm 路径被调用。
- FlashInfer / AITER all-reduce fusion 与 BytePS 同时开启。
- 代码试图进入 custom、quick、PyNccl、MSCCLPP、TorchSymmMem 或 `torch.distributed` fallback。
- BytePS All-Reduce 被调用时 BytePS 尚未初始化。
- 同一个 BytePS tensor name 被不同 `expected_workers` 重复声明。
- BytePS 初始化后的 `local_rank` 或 `local_size` 与 SGLang worker 元数据不一致。

当前源码中没有发现 `quant_all_reduce()` 符号，因此没有新增专门的 `quant_all_reduce()` 报错分支。若后续引入该路径，需要在进入默认通信后端前显式报错，保持“不 fallback”的约束。

## logical_name 规则

`communication_op.py` 中这些 wrapper 现在都支持 `logical_name`：

- `tensor_model_parallel_all_reduce`
- `attention_tensor_model_parallel_all_reduce`
- `moe_tensor_model_parallel_all_reduce`
- `moe_expert_parallel_all_reduce`

`RowParallelLinear` 使用：

```text
row_parallel_linear.{prefix}
```

没有 `prefix` 时使用模块级 deterministic counter：

```text
row_parallel_linear.unnamed_{counter}
```

`VocabParallelEmbedding` 使用：

```text
vocab_parallel_embedding.{prefix}
```

没有 `prefix` 时使用：

```text
vocab_parallel_embedding.unnamed_{counter}
```

没有使用 `id(self)`。counter 依赖各 rank 模块构造顺序一致，这是第一阶段接受的约束。

仍有一些模型文件直接调用 `tensor_model_parallel_all_reduce()`，没有传业务级 logical name。这些调用会落到 `GroupCoordinator` 的 generic name：

```text
generic.{dtype}.{shape}
```

这个兜底名称适合同步、有序调用。后续如果某个模型路径出现同 shape name 复用风险，需要继续给具体调用点补稳定 logical name。

## 与 CUDA graph 的关系

BytePS 第一阶段不支持 CUDA graph capture。

当前实现做了两层保护：

- `GroupCoordinator.graph_capture()` 中，如果 BytePS 开关已启用，直接报错。
- `GroupCoordinator.all_reduce()` 中，如果检测到当前 CUDA stream 正在 capture，或处于 piecewise CUDA graph 上下文，直接报错。

因此，开启 BytePS 后不会退回 PyNccl、custom All-Reduce 或 `torch.distributed` 来完成 graph 内通信。

## NCCL warmup

`ModelRunner` 中的 NCCL/RCCL pre-warm 在 BytePS 开启时会跳过：

```text
pre_warm_nccl and not use_byteps_all_reduce
```

这样可以避免 BytePS 模式启动时额外触发 NCCL warmup All-Reduce。

## 当前未覆盖内容

本轮没有改这些路径：

- scheduler、cache、speculative 等控制面直接 `torch.distributed.all_reduce()`。
- reduce-scatter。
- all-gather。
- broadcast。
- send/recv。
- BytePS 多机 worker/server 编排。
- CUDA graph capture 支持。
- 手工或自动化测试文档。

## 静态检查

本轮只做了语法级检查，没有启动服务，没有运行功能测试或集成测试。

已执行：

```bash
PYTHONPYCACHEPREFIX=/private/tmp/sglang-byteps-pycache python3 -m py_compile \
  sglang-0.5.10.post1/python/sglang/srt/distributed/byteps_collectives.py \
  sglang-0.5.10.post1/python/sglang/srt/distributed/parallel_state.py \
  sglang-0.5.10.post1/python/sglang/srt/distributed/communication_op.py \
  sglang-0.5.10.post1/python/sglang/srt/server_args.py \
  sglang-0.5.10.post1/python/sglang/srt/model_executor/model_runner.py \
  sglang-0.5.10.post1/python/sglang/srt/layers/linear.py \
  sglang-0.5.10.post1/python/sglang/srt/layers/vocab_parallel_embedding.py
```

结果：通过。

未运行：

- SGLang server 启动。
- BytePS 多 GPU 正确性测试。
- SGLang 功能测试。
- 集成测试。
- CUDA graph / piecewise CUDA graph 实测。

## 后续建议

- 首次功能验证使用单机多 GPU，并显式关闭 CUDA graph / piecewise CUDA graph。
- 若目标模型触发 generic logical name，优先给具体调用点补稳定业务名。
- 如果后续要支持多机，需要把 `DMLC_NUM_WORKER`、`DMLC_WORKER_ID`、server/scheduler 编排从单机默认值扩展为外部可配置。
- 如果后续引入或定位到 `quant_all_reduce()`，开启 BytePS 时应先显式报错，再考虑单独支持。
