# SGLang BytePS All-Reduce 实现说明

## 状态

已实现第一阶段代码改动，包含 TP 多 rank 初始化修复和跨 batch size 的 BytePS tensor name 隔离修复。Qwen2.5-0.5B-Instruct `--tp-size 2` + `--use-byteps-all-reduce` 端到端推理验证通过。当前改动只覆盖 SGLang 模型计算主路径 All-Reduce。

## 日期

2026-06-04

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
| `byteps/byteps/common/operations.cc` | 修复 `DMLC_USE_GDR` 未设置时 `byteps_init()` 通过空指针构造 `std::string` 崩溃的问题，默认按 `DMLC_USE_GDR=0` 处理。 |
| `byteps/3rdparty/ps-lite/src/van.cc` | BytePS scheduler ADD_NODE 广播健壮性修复：`ready_` 改为广播完整送达后才设置；广播前 null-check 并 `Connect()`；`Send()` 对 ADD_NODE 失败优雅降级（打印 warning 跳过而非崩溃）；ZMQ 调度器增加回环连接以支持 `RequestLocalStop()` 唤醒接收线程；新增 `BYTEPS_DEBUG_NODE_REGISTRATION` 环境变量用于节点注册调试日志。 |
| `byteps/3rdparty/ps-lite/src/zmq_van.h` | ZMQ 套接字错误信息增强：连接成功时打印 node id/addr/my_node/target node；无套接字时列出当前所有已知 sender ID；scheduler ADD_NODE 缺失 socket 时返回 -1 替代 `LOG(FATAL)`。 |
| `byteps/byteps/server/server.cc` | 调度器纯角色模式支持：`is_server_=false` 路径（`DMLC_ROLE=scheduler` 的进程）正确启动 PS、执行 barrier 等待所有节点注册、干净调用 `Finalize()` 关闭，不再 crash 或 leak。`BytePSServerEngineThread` 中修复 `COPY_FIRST`/`SUM_RECV` 操作的 `msg.src` 从 shared array 重新解释赋值。 |

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

注意：`initialize_byteps_for_sglang()` **不会设置** `BYTEPS_SOCKET_PATH`、`DMLC_INTERFACE`、`DMLC_NODE_HOST`。在 ZMQ 通信路径下，`BYTEPS_SOCKET_PATH`（通常设为 `/tmp/byteps_socket`）是必需的；本地 loopback 测试时，`DMLC_INTERFACE=lo` 和 `DMLC_NODE_HOST=127.0.0.1` 也应在启动 SGLang 前由外部显式设置。

初始化后会校验：

- `bps.local_rank()` 是否等于当前 SGLang worker 的 `gpu_id`。
- `bps.local_size()` 是否等于 `tp_size * pp_size`。

当前代码没有设置 `DMLC_NUM_SERVER`，也没有校验 `bps.size()`。

BytePS C++ 初始化中，`DMLC_USE_GDR` 未设置时现在按 `0` 处理。也就是说，普通 TCP/ZMQ smoke test 不需要额外开启 GDR；测试命令仍建议显式设置：

```text
DMLC_USE_GDR=0
```

如果显式设置 `DMLC_USE_GDR=1`，仍会走原有 GDR 初始化路径。

## BytePS collective wrapper

新增文件 `byteps_collectives.py` 提供三个核心能力。

**1. declare 缓存**

`declare_and_cache_byteps_group(name, expected_workers)` 会记录 BytePS tensor name 对应的 worker 数量。第一次出现时调用：

```python
bps.declare(name, expected_workers=expected_workers)
```

后续如果同名 tensor 使用了不同 `expected_workers`，直接抛出 `RuntimeError`。

`expected_workers` 的值通过 `_byteps_expected_workers()` 辅助函数计算：

```python
def _byteps_expected_workers() -> int:
    import byteps.torch as bps

    local_size = max(1, bps.local_size())
    return max(1, bps.size() // local_size)
```

该函数用 `max(1, ...)` 保护除零和空环境，确保在没有 BytePS root worker 的退化场景（如单 rank 测试）下也能正常声明 tensor。

这里的 `expected_workers` 是 BytePS PS server 等待的 worker 节点数，不是 SGLang TP group 的 rank 数。单机多 GPU 首测时：

```text
bps.size() = 2
bps.local_size() = 2
local_size = max(1, 2) = 2
expected_workers = max(1, 2 // 2) = 1
```

如果把 TP rank 数 `2` 传给 BytePS server，server 会等待两个 worker 节点的 push，但单机只有一个 BytePS root worker 会向 server push，导致 `push_pull_async_inplace + synchronize` 卡住。

调试模式下，`byteps_allreduce_inplace()` 会同时打印 `group_world_size`（SGLang TP group 的 rank 数）和 `expected_workers`（BytePS server 等待的 worker 节点数），方便排查 declaration 不匹配导致的 hang 问题。

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

## BytePS scheduler/server 健壮性修复

本轮的 BytePS C++ 代码变更解决了三个单机多 GPU 运行时发现的实际问题，使 scheduler/server 在小规模部署下更稳定。

### van.cc — ADD_NODE 广播健壮性

`byteps/3rdparty/ps-lite/src/van.cc` 中 `ProcessAddNodeCommandAtScheduler()` 做了以下修复：

1. **调试日志开关**：新增 `BYTEPS_DEBUG_NODE_REGISTRATION` 环境变量。设置后 scheduler 会在收到每个 ADD_NODE、分配 id、连接节点、广播 ADD_NODE 时打印 `INFO` 级别日志，显示 sender、`collected_nodes`、`expected_nodes` 和完整 node 列表。

2. **`ready_` 广播完成保护**：原先 `ready_ = true` 无条件设置，现在改为只有 ADD_NODE 广播成功送达所有 worker 和 server 节点后才设 `ready_=true`。如果广播不完整（部分节点不可达），会打印 warning 并延迟 readiness，避免在未确认所有节点可达时过早进入调度状态。

3. **广播前冗连接**：在向每个节点发送 ADD_NODE 之前，先检查是否已经有活跃连接；如果没有则调用 `Connect()` 建立连接。同时增加 null-check，找不到目标 node 时打印 warning 而不是崩溃。

4. **`Send()` 优雅降级**：原先 `Send()` 失败会直接触发 `CHECK_NE(send_bytes, -1)` 崩溃。现在如果 `is_scheduler_` 且消息类型是 `ADD_NODE`，`SendMsg()` 返回 -1 时会打印 warning 并返回 0（跳过），而不是崩溃。这避免了单节点不可达时整个调度器崩溃。

5. **ZMQ 调度器需要的回环连接**：原先注释 "The scheduler does not need a loopback RDMA connection to itself"，但忽略了 ZMQ 模式下 scheduler 需要通过回环向自身发送 `TERMINATE` 消息来唤醒接收线程。修复为：`!is_scheduler_ || GetType() == "zeromq"` 时也建立到 scheduler 的连接。

### zmq_van.h — ZMQ 套接字错误信息增强

`byteps/3rdparty/ps-lite/src/zmq_van.h` 做了以下改进：

1. **连接调试日志**：新增 `BYTEPS_DEBUG_NODE_REGISTRATION` 支持，连接成功后打印 node id、address、my_node 和 target node 信息。

2. **无套接字错误信息**：当找不到目标节点的 socket 时，原先只打印 `LOG(FATAL) "there is no socket to node <id>"`。现在会列出当前所有已知 sender 的 ID 列表（如 `current senders=[4,5,8,9]`），方便排查。对于 scheduler ADD_NODE 场景，以 `LOG(WARNING)` 替代 `LOG(FATAL)`，返回 -1 让上层 `Send()` 优雅处理。

### server.cc — 调度器纯角色模式和 COPY_FIRST/SUM_RECV 修复

`byteps/byteps/server/server.cc` 做了以下修复：

1. **调度器纯角色模式**：`byteps_server()` 函数现在正确处理 `is_server_=false`（即 `DMLC_ROLE=scheduler` 的进程）。调度器进程不启动 engine thread 或 KVServer，而是调用 `StartPS()` 启动 PS，执行 barrier 等待所有节点注册，然后干净调用 `Finalize()` 关闭。这确保 scheduler 进程不会 leak 或异常退出。

2. **COPY_FIRST/SUM_RECV 操作的 `msg.src` 修复**：`BytePSServerEngineThread()` 中，对于 `COPY_FIRST` 和 `SUM_RECV` 操作，当数据通过 shared array (`sarray`) 传递时，需要将 `msg.src` 重新解释为 `sarray.vals.data()` 的地址，避免空指针解引用。

### 新增环境变量

| 变量 | 用途 |
| --- | --- |
| `BYTEPS_DEBUG_NODE_REGISTRATION` | 设置为 1 时在 van.cc 和 zmq_van.h 输出更详细的节点注册日志。首次排障时建议开启。 |
| `BYTEPS_SOCKET_PATH` | BytePS 本地 ZMQ socket 文件目录，建议设为 `/tmp/byteps_socket`。 |
| `DMLC_INTERFACE` | 网络接口名，本地 loopback 测试设为 `lo`。 |
| `DMLC_NODE_HOST` | 节点 IP 地址，本地 loopback 测试设为 `127.0.0.1`。 |

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
- BytePS 多机 worker/server 编排（虽已改进单机 scheduler/server 稳定性和调度器纯角色模式支持，但多机网络拓扑、故障恢复、跨节点 barrier 仍未覆盖）。
- CUDA graph capture 支持。

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

服务器侧已手工验证 BytePS import、单 worker init smoke test 和单机双 rank `float16 push_pull` smoke test。验证环境使用外部 BytePS scheduler/server，worker 环境显式设置 `DMLC_USE_GDR=0`、`DMLC_NUM_SERVER=1`、`DMLC_ENABLE_RDMA=0`、`DMLC_INTERFACE=lo`、`DMLC_NODE_HOST=127.0.0.1`、`BYTEPS_SOCKET_PATH=/tmp/byteps_socket`，输出：

```text
rank: 0
size: 1
local_rank: 0
local_size: 1
```

双 rank push/pull smoke test 通过：两个本地 rank 均以 `expected_workers=1` 完成 float16 tensor All-Reduce，求和结果 `[3.0, 3.0, ...]` 符合预期。

未运行：

- SGLang server 启动。
- BytePS 多 GPU 正确性测试。
- SGLang 功能测试。
- 集成测试。
- CUDA graph / piecewise CUDA graph 实测。

## 后续建议

- 首次功能验证使用单机多 GPU，并显式关闭 CUDA graph / piecewise CUDA graph。
- 若目标模型触发 generic logical name，优先给具体调用点补稳定业务名。
- 如果后续要支持多机，需要把 `DMLC_NUM_WORKER`、`DMLC_WORKER_ID`、server/scheduler 编排从单机默认值扩展为外部可配置。当前 scheduler/server 已在 `server.cc` 中支持调度器纯角色模式，但多机 worker 还需要对应的初始化参数注入和网络拓扑配置。
- 如果后续引入或定位到 `quant_all_reduce()`，开启 BytePS 时应先显式报错，再考虑单独支持。

## Bug 修复记录

以下两个 Bug 在 2026-06-04 的端到端测试中发现并修复。

### Bug 1 — TP 多 rank 初始化 hang

**现象**：`--tp-size 2` 时只有 TP0 完成 `bps.init()` 并打印 `BytePS initialized for SGLang`，TP1 的 init 永远不返回，导致 SGLang 启动超时。

**根因**：`byteps_collectives.py` 中 `initialize_byteps_for_sglang()` 调用 `bps.init()`。BytePS 内部只有 root device（`local_rank=0`）向 scheduler 注册。TP0（local_rank=0）先执行到 `bps.init()` 并注册完成，scheduler 收到 1 个 worker + 1 个 server 后达到 `expected_nodes=2` 完成 barrier，设置 `ready_=true`。TP1（local_rank=1）随后到达 `bps.init()` 时 scheduler 已越过注册阶段，TP1 的 init 里等待的注册完成信号永远不会到达，永久 hang。

**修复**：[byteps_collectives.py:29-31](sglang-0.5.10.post1/python/sglang/srt/distributed/byteps_collectives.py) 在 `bps.init()` 前添加 `torch.distributed.barrier()`，确保所有 TP rank 同时到达 init 调用点再一起向 scheduler 注册。

```python
if not _BPS_INITIALIZED:
    if local_size > 1:
        import torch.distributed
        torch.distributed.barrier()
    bps.init()
    _BPS_INITIALIZED = True
```

### Bug 2 — 推理请求 hang（不同 batch size 复用同名 BytePS tensor）

**现象**：Warmup 的第一个 forward pass（prefill，如 shape=6×896）顺利完成所有 BytePS all-reduce，`Application startup complete` 正常打印。但后续 decode step（shape 变为 1×896）进入 `model_runner.forward()` 后再也不返回，HTTP 请求 30 秒后超时。`/health` 端点返回 503。

**排查过程**：在 `tp_worker.py` 的 `forward_batch_generation()` 入口/出口添加 `logger.info()` 日志，观察到：
- Warmup 第 1 次 forward（prefill）：`Entering → returned → sample → returning` 全部通过
- Warmup 第 2 次 forward（decode）：`Entering` 后永不 `returned`

根因确认：decode 的 BytePS all-reduce 调用了与 prefill 相同的 `bps.declare()` name（已在 declare cache 中），但 tensor **shape 不同**。BytePS server 在第一次 push_pull 时按 prefill shape（6×896）分配了内部缓冲区，第二次 push_pull 用 decode shape（1×896）时缓冲区大小不匹配，`push_pull_async_inplace + synchronize` 永久 hang。

**修复**：[byteps_collectives.py:116-120](sglang-0.5.10.post1/python/sglang/srt/distributed/byteps_collectives.py) 在构造 BytePS tensor name 时拼接 tensor 的 shape 维度，确保不同 batch size 使用独立的 BytePS 声明和服务端缓冲区。

```python
# 修复前
name = build_byteps_group_name(group, logical_name)

# 修复后
shape_suffix = "x".join(str(d) for d in tensor.shape)
name = build_byteps_group_name(
    group, f"{logical_name}.{shape_suffix}"
)
```

每个新的 (logical_name, shape) 组合会首次触发 `bps.declare()`，后续相同 name+shape 直接从 `_DECLARED_BPS_GROUPS` 缓存命中。SGLang 推理的 batch size 种类有限（warmup prefill 的 6、decode 的 1 等），不会产生无限增长的声明。

### 测试验证

环境：单机 2×GPU，sgl-dev2 conda 环境，Qwen2.5-0.5B-Instruct，float16。

```bash
# 启动命令
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-0.5B-Instruct --tp-size 2 \
  --host 0.0.0.0 --port 30000 --log-level info --dtype float16 \
  --disable-custom-all-reduce --disable-cuda-graph \
  --disable-piecewise-cuda-graph \
  --enforce-disable-flashinfer-allreduce-fusion \
  --use-byteps-all-reduce --byteps-all-reduce-debug
```

请求验证：

```python
import requests
r = requests.post('http://127.0.0.1:30000/generate', json={
    'text': 'The capital of France is',
    'sampling_params': {'temperature': 0, 'max_new_tokens': 1}
})
# 结果: {'text': ' Paris', 'e2e_latency': 0.419s}
```

- ✅ 模型正确输出 `Paris`
- ✅ 端到端延迟 0.419s
- ✅ 所有 24 层 RowParallelLinear + VocabParallelEmbedding 的 BytePS all-reduce 正常完成
- ✅ Prefill（shape 6×896）+ Decode（shape 1×896）两种 batch size 的 push_pull 均正常
