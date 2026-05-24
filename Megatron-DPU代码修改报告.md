# Megatron-DPU 代码修改报告

> 基于官方 Megatron-LM 与 BytePS 代码，将 Megatron 的 DP（数据并行）与 TP（张量并行）默认 NCCL All-Reduce 通信替换为 BytePS push/pull 通信，并在 BytePS 侧进行 RDMA 带宽与多线程优化。

---

## 一、修改概览

| 模块 | 修改内容 | 影响范围 |
|---|---|---|
| **Megatron-LM 配置** | 新增 `use_dpu_reduce` / `use_dpu_tp_reduce` 开关 | `arguments.py`, `distributed_data_parallel_config.py`, `model_parallel_config.py` |
| **Megatron-LM 初始化** | 统一入口 `bps.init()` | `initialize.py` |
| **Megatron-LM DP 梯度同步** | NCCL all-reduce → BytePS push_pull（in-place 同步/异步） | `param_and_grad_buffer.py`, `distributed_data_parallel.py` |
| **Megatron-LM TP All-Reduce** | RowParallelLinear forward 输出 NCCL all-reduce → BytePS push_pull | `mappings.py`, `layers.py` |
| **Megatron-LM BytePS 封装** | **新文件**：group 命名、declare 缓存、预声明机制 | `byteps_collectives.py` |
| **BytePS RDMA 双通道** | 控制面 + 数据面分离，push/pull 数据走 data endpoint | `rdma_van.h`, `rdma_transport.h`, `rdma_utils.h` |
| **BytePS Push/Pull 多线程** | `BYTEPS_PUSH_THREAD` 环境变量控制并发 | `global.cc`, `operations.cc` |
| **BytePS Hash 策略** | 默认 `djb2` → `raw` | `global.cc` |
| **BytePS 可观测性** | PUSH 队列 debug 日志、`get_pushpull_speed` 接口、benchmark | `scheduled_queue.cc`, `ops.py` |

---

## 二、Megatron-LM 侧逐文件修改详情

### 2.1 配置层：开关与参数

#### `megatron/training/arguments.py`（行 2565-2570）

新增两个 CLI 参数：

```python
group.add_argument('--use-dpu-reduce', action='store_true',
                   default=False, help='If set, use DPU for grad reduce.')
group.add_argument('--use-dpu-tp-reduce', action='store_true',
                   default=False, help='If set, use BytePS for TP all-reduce.')
```

- `--use-dpu-reduce`：启用 BytePS 替代 DP 梯度 all-reduce
- `--use-dpu-tp-reduce`：启用 BytePS 替代 TP 层内 all-reduce

#### `megatron/core/distributed/distributed_data_parallel_config.py`（行 10）

```python
@dataclass
class DistributedDataParallelConfig:
    use_dpu_reduce: bool = False
    ...
```

#### `megatron/core/model_parallel_config.py`（行 19-23）

```python
@dataclass
class ModelParallelConfig:
    use_dpu_reduce: bool = False
    """Use BytePS for data-parallel gradient all-reduce."""
    use_dpu_tp_reduce: bool = False
    """Use BytePS for tensor-parallel all-reduce."""
    ...
```

两个配置类均新增对应字段，确保配置可贯穿 DDP 和 TP 各层。

---

### 2.2 初始化层：统一 `bps.init()` 入口

#### `megatron/training/initialize.py`（行 107-109）

```python
args = get_args()
if args.use_dpu_reduce or args.use_dpu_tp_reduce:
    bps.init()
```

**设计要点**：
- 在 `initialize_rerun_state_machine` 之前完成 BytePS 初始化
- 只要 DP 或 TP 任一启用 BytePS，即执行初始化
- 避免在训练脚本中各自 `bps.init()`，统一入口便于维护

---

### 2.3 BytePS 通信封装层（新文件）

#### `megatron/core/distributed/byteps_collectives.py`（全新文件，145 行）

提供三个核心接口和一整套命名/声明机制：

##### 2.3.1 Group 命名函数

```python
def build_byteps_group_name(scope: str, logical_name: str) -> str:
```

- `scope='dp'`：返回 `dp.tp{tp_rank}.pp{pp_rank}.{logical_name}`
  - DP group 由相同 tp_rank 的 rank 组成，因此用 tp_rank 区分不同 DP group
- `scope='tp'`：返回 `tp.dp{dp_rank}.pp{pp_rank}.cp{cp_rank}.{logical_name}`
  - TP group 由相同 dp_rank 的 rank 组成，因此用 dp_rank 区分不同 TP group

**命名中排除 context_parallel (CP) rank**（DP 场景）：因为 CP rank 会在 collective 内部变化，不应成为 BytePS tensor key 的一部分。

##### 2.3.2 Declare & Cache 机制

```python
_DECLARED_BPS_GROUPS: Dict[str, int] = {}

def declare_and_cache_byteps_group(name: str, expected_workers: int) -> None:
```

- 每个 BytePS tensor 只 declare 一次（幂等）
- 缓存 expected_workers，重复调用时校验一致性
- 解决 `bps.declare()` 的全局单次调用约束

##### 2.3.3 三个 wrapper 函数

| 函数 | 行为 | 使用场景 |
|---|---|---|
| `byteps_allreduce(tensor, ...)` | `push_pull` → 返回新 tensor（copy 模式） | 兼容旧代码 |
| `byteps_allreduce_inplace(tensor, ...)` | `push_pull_async_inplace` + `synchronize` → 就地修改 | DP 同步梯度 |
| `byteps_allreduce_async_inplace(tensor, ...)` | `push_pull_async_inplace` → 返回 handle | DP 异步梯度（overlap 模式） |

所有函数均自动完成 `declare_and_cache_byteps_group` 调用。

##### 2.3.4 预声明函数

```python
def pre_declare_all_byteps_groups(
    num_tp_layers, num_dp_buckets, use_dpu_tp=True, use_dpu_dp=True
):
```

- 在训练开始前统一按确定性顺序预声明所有 BytePS tensor
- 确保所有 worker 给相同 tensor 分配相同的 declared_key
- 声明完成后 `dist.barrier()` 保证全局一致

---

### 2.4 DP 梯度同步替换

#### `megatron/core/distributed/param_and_grad_buffer.py`

##### 导入（行 15, 17）

```python
from byteps.torch import ops as bps_ops
from .byteps_collectives import byteps_allreduce_async_inplace, byteps_allreduce_inplace
```

##### 核心替换逻辑（行 388-424）

在 `start_grad_sync()` 方法中，当满足条件 `not use_distributed_optimizer AND use_dpu_reduce` 时：

```python
if (not self.ddp_config.use_distributed_optimizer
        and getattr(self.ddp_config, "use_dpu_reduce", False)):

    byteps_average = self.ddp_config.average_in_collective

    if async_op:
        # 异步模式：为每个 bucket 发起 push_pull_async_inplace
        handles = []
        for _, bucket in enumerate(self.buckets):
            handle = byteps_allreduce_async_inplace(
                bucket.grad_data,
                group=communication_group,
                scope='dp',
                logical_name=f"bucket_{bucket.bucket_id}",
                average=byteps_average,
                version=0, priority=0,
            )
            handles.append(handle)
        self.grad_reduce_handle = handles
        return
    else:
        # 同步模式：逐个 bucket 同步完成 push_pull_inplace
        for _, bucket in enumerate(self.buckets):
            byteps_allreduce_inplace(
                bucket.grad_data,
                group=communication_group,
                scope='dp',
                logical_name=f"bucket_{bucket.bucket_id}",
                average=byteps_average,
                version=0, priority=0,
            )
        self.grad_reduce_handle = None
        return
```

**关键差异 vs NCCL 路径**：
- NCCL 使用 `_coalescing_manager` 批量合并通信操作，BytePS 路径逐个 bucket 处理
- BytePS 路径直接在 bucket.grad_data 上 in-place 操作（零额外拷贝）
- BytePS 路径不进入 `stream_context`（通信流），直接在默认流完成

##### 完成同步逻辑（行 509-525）

```python
if not getattr(self.ddp_config, "use_dpu_reduce", False):
    # NCCL 路径
    self.grad_reduce_handle.wait()
else:
    # BytePS 路径：逐个 handle 调用 bps_ops.synchronize()
    handles = self.grad_reduce_handle
    for handle in handles:
        bps_ops.synchronize(handle)
```

NCCL 使用单个 `Work.wait()`，BytePS 使用 `bps_ops.synchronize(handle)` 逐个等待。

#### `megatron/core/distributed/distributed_data_parallel.py`（行 312-317）

```python
if self.ddp_config.use_dpu_reduce and not self.ddp_config.use_distributed_optimizer:
    if bps.size() < self.intra_dp_cp_group.size():
        raise RuntimeError(
            f"BytePS world size is smaller than Megatron data-parallel size: "
            f"byteps.size()={bps.size()} vs dp_size={self.intra_dp_cp_group.size()}."
        )
```

在 DDP 构造函数中添加 BytePS worker 数量校验，确保 BytePS 有足够的 worker 覆盖 DP group 规模。

---

### 2.5 TP All-Reduce 替换

#### `megatron/core/tensor_parallel/mappings.py`

##### 新增 `_bps_reduce` 函数（行 36-62）

```python
def _bps_reduce(input_, group, name):
    assert group is not None
    assert name is not None
    if group.size() == 1:
        return input_

    from byteps.torch import ops as bps_ops
    from megatron.core.distributed.byteps_collectives import byteps_allreduce_async_inplace

    handle = byteps_allreduce_async_inplace(
        input_, group=group, scope='tp', logical_name=name,
        average=False, version=0, priority=0,
    )
    return bps_ops.synchronize(handle)
```

**注意**：TP all-reduce 的 `average=False`（保持 SUM 语义），因为 TP all-reduce 就是 SUM。

##### 新增 `_BpsReduceFromModelParallelRegion` autograd function（行 264-280）

```python
class _BpsReduceFromModelParallelRegion(torch.autograd.Function):
    """Use DPU All-reduce the input from the model parallel region."""

    @staticmethod
    def forward(ctx, input_, group, name):
        return _bps_reduce(input_, group, name)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None  # backward 是 identity（copy）
```

**设计解析**：
- `forward`：跨 TP group 做 BytePS all-reduce（SUM）
- `backward`：梯度直接透传（identity），因为 forward 的 all-reduce 对 input 的梯度是 1（每个 rank 贡献相同的梯度）

##### 新增 wrapper 函数（行 527-529）

```python
def bps_reduce_from_tensor_model_parallel_region(input_, group=None, name=None):
    group = get_tensor_model_parallel_group_if_none(group)
    return _BpsReduceFromModelParallelRegion.apply(input_, group, name)
```

#### `megatron/core/tensor_parallel/layers.py`

##### 新增导入（行 39）

```python
from .mappings import (
    ...
    bps_reduce_from_tensor_model_parallel_region,
)
```

##### 新增 comm name 生成器（行 66-74）

```python
_DPU_TP_COMM_NAME_COUNTER = count()

def _build_dpu_tp_comm_name(module_kind: str, tp_comm_buffer_name=None) -> str:
    parts = ["tp", module_kind]
    if tp_comm_buffer_name:
        parts.append(tp_comm_buffer_name)
    parts.append(str(next(_DPU_TP_COMM_NAME_COUNTER)))
    return "_".join(parts)
```

- 使用全局递增计数器确保每个 TP comm name 唯一
- name 格式：`tp_row_parallel_linear_0`, `tp_row_parallel_linear_1`, ...

##### RowParallelLinear 构造函数（行 1145-1147）

```python
self.dpu_tp_comm_name = _build_dpu_tp_comm_name(
    "row_parallel_linear", tp_comm_buffer_name
)
```

##### RowParallelLinear.forward 中的分叉逻辑（行 1289-1296）

```python
if self.config.use_dpu_tp_reduce:
    output_ = bps_reduce_from_tensor_model_parallel_region(
        output_parallel,
        group=self.tp_group,
        name=self.dpu_tp_comm_name,
    )
else:
    output_ = reduce_from_tensor_model_parallel_region(output_parallel, group=self.tp_group)
```

**仅影响 RowParallelLinear**：ColumnParallelLinear 的 forward 输出使用的是 all-gather（不是 all-reduce），因此无需 BytePS 替换。

---

## 三、BytePS 侧优化详情

### 3.1 RDMA 双通道（控制面 / 数据面分离）

#### `byteps/3rdparty/ps-lite/src/rdma_utils.h`（行 171-176）

```cpp
struct RequestContext {
  uint32_t node;
  uint16_t port;
  uint8_t isDataPlane;   // 新增字段
  char hostname[kMaxHostnameLength];
};
```

- 在 RDMA 连接建立时传递 `isDataPlane` 标志，标识此连接是数据面还是控制面

#### `byteps/3rdparty/ps-lite/src/rdma_transport.h`（行 38-81）

```cpp
struct Endpoint {
  ...
  bool isDataPlane;  // 新增字段

  Endpoint(bool data_plane = false)
      : ..., isDataPlane(data_plane), ... {
    // 数据面用 BYTEPS_RDMA_RX_DEPTH，控制面用 BYTEPS_RDMA_CTRL_RX_DEPTH
    if (!isDataPlane) {
      kRxDepth = byteps_ctrl_rx_depth
                     ? atoi(byteps_ctrl_rx_depth)
                     : std::min(default_rx_depth, 128);
    } else {
      kRxDepth = default_rx_depth;
    }
  }
};
```

#### `byteps/3rdparty/ps-lite/src/rdma_van.h`

##### 新增 data_endpoints 容器

```cpp
std::unordered_map<int, std::unique_ptr<Endpoint>> data_endpoints_;
```

##### 新增 Connect2Node 方法（行 211-270）

```cpp
void Connect2Node(const Node& node, bool dataPlane = false) {
    auto& whichEndpoints = dataPlane ? data_endpoints_ : endpoints_;
    // ... 建立连接，Endpoint 构造时传入 dataPlane 标志
    whichEndpoints[node.id] = std::make_unique<Endpoint>(dataPlane);
}

void Connect(const Node& node) override {
    Connect2Node(node, false);   // 控制面连接
    Connect2Node(node, true);    // 数据面连接
}
```

##### SendMsg 数据路径分离（行 445-554）

```cpp
int SendMsg(Message& msg) override {
    bool is_pushpull = IsValidPushpull(msg);  // 判断是否为 push/pull 消息

    // 分别查找控制和数据 endpoint
    Endpoint* endpoint = endpoints_[remote_id];
    Endpoint* dataEndpoint = is_pushpull ? data_endpoints_[remote_id] : nullptr;

    auto trans = endpoint->GetTransport();         // 控制面 transport
    std::shared_ptr<Transport> dataTrans;
    if (is_pushpull) {
        dataTrans = dataEndpoint->GetTransport();  // 数据面 transport
    }

    // Push 请求 / Pull 响应走 dataTrans（数据面）
    if (msg.meta.push && msg.meta.request) {
        dataTrans->SendPushRequest(msg, msg_buf, addr_tuple);   // worker push→server
    } else if (!msg.meta.push && !msg.meta.request) {
        dataTrans->SendPullResponse(msg, msg_buf, addr_tuple, ...); // server pull→worker
    }
    // 控制消息仍走 trans（控制面）
    else { ... }
}
```

**核心逻辑**：`IsValidPushpull(msg)` 判断消息类型，若为 push/pull 数据消息则路由至数据面 endpoint。

---

### 3.2 Push/Pull 多线程

#### `byteps/byteps/common/global.cc`（行 123-124）

```cpp
_push_thread = getenv("BYTEPS_PUSH_THREAD") ? atoi(getenv("BYTEPS_PUSH_THREAD")) : 1;
```

- 新增环境变量 `BYTEPS_PUSH_THREAD`，默认值 1（保持向后兼容）
- 可通过设置更大的值提升并发度

#### `byteps/byteps/common/operations.cc`（行 53-58, 74-76, 92-93）

```cpp
// PullLoop 多线程
if (BytePSGlobal::IsRootDevice()) {
    for (int i = 0; i < BytePSGlobal::GetPushThread(); i++) {
        func.push_back(PullLoop);
    }
}

// PushLoop 多线程
if (BytePSGlobal::IsRootDevice()) {
    for (int i = 0; i < BytePSGlobal::GetPushThread(); i++) {
        func.push_back(PushLoop);
    }
}
```

- 原始 BytePS 只启动 1 个 PushLoop 和 1 个 PullLoop
- 修改后按 `BYTEPS_PUSH_THREAD` 的值启动多个实例

---

### 3.3 Hash 策略调整

#### `byteps/byteps/common/global.cc`（行 165-166）

```cpp
// 修改前：默认 "djb2"
_hash_knob = std::string(getenv("BYTEPS_KEY_HASH_FN") ? ... : "djb2");

// 修改后：默认 "raw"
_hash_knob = std::string(getenv("BYTEPS_KEY_HASH_FN") ? ... : "raw");
```

#### 新增 Hash_Raw 分发逻辑（行 680-681）

```cpp
} else if (!_hash_knob.compare(std::string("raw"))) {
    server = key % num_servers;
}
```

- `raw` 策略：直接 `key % num_servers`，无哈希计算开销
- `djb2` 策略：保留支持，可通过 `BYTEPS_KEY_HASH_FN=djb2` 切换回去

#### Distributed job 判断条件调整（行 158）

```cpp
// 修改前：
_is_distributed_job = (_num_worker > 1) ? true : _is_distributed_job;
// 修改后：
_is_distributed_job = (_num_worker > 0) ? true : _is_distributed_job;
```

- 允许单 worker 情况也能进入 distributed 模式（便于测试和调试）

---

### 3.4 可观测性增强

#### `byteps/byteps/common/scheduled_queue.cc`（行 101-105）

```cpp
if (getQueueType() == PUSH) {
    BPS_LOG(DEBUG) << "Queue " << LogStrings[_qt]
                   << " addTask: " << entry->tensor_name
                   << " key: " << entry->key
                   << " rank: " << BytePSGlobal::GetLocalRank();
}
```

- 新增 PUSH 队列 DEBUG 级别日志，便于排查 push/pull 执行时序

#### `byteps/byteps/torch/ops.py`（行 46）

```python
get_pushpull_speed = _basics.get_pushpull_speed
```

- 导出 `get_pushpull_speed` 接口，可在训练脚本中获取 push/pull 吞吐统计

#### `byteps/example/pytorch/pushpull_bench.py`（新文件）

- 新增 push/pull 性能 benchmark，用于评估不同参数下 BytePS 通信带宽

---

## 四、修改前后对比（核心路径）

### 4.1 DP 梯度同步

| 维度 | 修改前（NCCL） | 修改后（BytePS） |
|---|---|---|
| 通信原语 | `torch.distributed.all_reduce` | `bps.push_pull_async_inplace` |
| 数据拷贝 | 0（NCCL in-place） | 0（in-place，零额外拷贝） |
| 批量合并 | `_coalescing_manager` 批量合并 | 逐 bucket 独立发起 |
| 异步支持 | `async_op=True` + `Work.wait()` | `push_pull_async_inplace` + `bps_ops.synchronize()` |
| 通信后端 | NCCL (GPU-direct) | BytePS (RDMA + PS 调度) |
| 控制开关 | 无 | `--use-dpu-reduce` |

### 4.2 TP All-Reduce

| 维度 | 修改前（NCCL） | 修改后（BytePS） |
|---|---|---|
| 通信原语 | `torch.distributed.all_reduce` | `bps.push_pull_async_inplace` |
| Autograd 函数 | `_ReduceFromModelParallelRegion` | `_BpsReduceFromModelParallelRegion` |
| 语义 | SUM（forward all-reduce，backward identity） | SUM（保持一致） |
| 控制开关 | 无 | `--use-dpu-tp-reduce` |
| 作用层 | 所有 TP 层的 all-reduce | RowParallelLinear.forward 输出 reduce |

### 4.3 BytePS Push/Pull 性能

| 维度 | 修改前（原始 BytePS） | 修改后（优化版） |
|---|---|---|
| RDMA 通道 | 单一 endpoint（控制与数据混合） | 双通道（控制面 + 数据面分离） |
| Push/Pull 线程数 | 1 | 可配置（`BYTEPS_PUSH_THREAD`，默认 1） |
| Key Hash 策略 | `djb2` | `raw`（`key % num_servers`） |
| 可观测性 | 无 push/pull 速度接口 | `get_pushpull_speed` + PUSH 队列日志 |

---

## 五、数据流与调用链路

### 5.1 DP 梯度同步完整链路

```
训练 backward 完成
    ↓
param_and_grad_buffer.start_grad_sync()
    ↓
[use_dpu_reduce=True and not use_distributed_optimizer]
    ↓
for each bucket:
    byteps_allreduce_inplace / byteps_allreduce_async_inplace
        ↓
    _declare_group_and_get_name(scope='dp', logical_name='bucket_{id}')
        ↓ build_byteps_group_name → "dp.tp{tp_rank}.pp{pp_rank}.bucket_{id}"
        ↓ declare_and_cache_byteps_group (幂等)
    bps_ops.push_pull_async_inplace(tensor, name=...)
        ↓
    [async]: 收集 handle → self.grad_reduce_handle
    [sync]: bps_ops.synchronize(handle)
    ↓
finish_grad_sync():
    for handle in handles: bps_ops.synchronize(handle)
```

### 5.2 TP All-Reduce 完整链路

```
RowParallelLinear.forward()
    ↓
output_parallel = XW^T (local matmul)
    ↓
[use_dpu_tp_reduce=True]
    ↓
bps_reduce_from_tensor_model_parallel_region(output_parallel, group, name)
    ↓ _BpsReduceFromModelParallelRegion.apply()
        ↓ forward: _bps_reduce(input_, group, name)
            ↓ build_byteps_group_name(scope='tp', logical_name=name)
                → "tp.dp{dp_rank}.pp{pp_rank}.cp{cp_rank}.{name}"
            ↓ byteps_allreduce_async_inplace(scope='tp', ...)
            ↓ bps_ops.synchronize(handle)
        ↓ backward: identity (grad_output 直接透传)
    ↓
output = reduced_output + bias
```

### 5.3 BytePS SendMsg RDMA 路径

```
Van::SendMsg(msg)
    ↓
IsValidPushpull(msg) → true (push/pull 数据消息)
    ↓
查找 control endpoint + data endpoint
    ↓
[push request]:   dataTrans->SendPushRequest()
[pull response]:  dataTrans->SendPullResponse()
[control msg]:    trans->SendRendezvousBegin() / SendPushResponse() / SendPullRequest()
```

---

## 六、设计决策与注意事项

### 6.1 为什么 DP group 命名用 tp_rank？

DP group 由相同 `tp_rank` 的 ranks 组成。例如 TP=2, DP=2：
- DP group 0：rank 0 (tp_rank=0) 和 rank 2 (tp_rank=0)
- DP group 1：rank 1 (tp_rank=1) 和 rank 3 (tp_rank=1)

因此 `dp.tp{tp_rank}` 能唯一标识一个 DP group。

### 6.2 为什么 TP group 命名不用 tp_rank？

TP group 由相同 `dp_rank` 的 ranks 组成。例如：
- TP group 0：rank 0 (dp_rank=0) 和 rank 1 (dp_rank=0)
- TP group 1：rank 2 (dp_rank=1) 和 rank 3 (dp_rank=1)

因此 `tp.dp{dp_rank}` 能唯一标识一个 TP group。

### 6.3 为什么 DP scope 的命名排除 cp_rank？

CP ranks 在 DP 通信集体操作中会变化，不应固定到 BytePS tensor key 中。

### 6.4 为什么 TP all-reduce 的 average=False？

TP all-reduce 的数学语义是 SUM。每个 TP rank 持有张量的一个分片，forward 输出需要 SUM 还原。而 DP all-reduce 需要 AVG（`average_in_collective=True`），因为每个 DP rank 持有完整的梯度副本。

### 6.5 为什么 ColumnParallelLinear 不需要 BytePS 替换？

ColumnParallelLinear 的 forward 输出使用 all-gather（非 all-reduce），因此 BytePS 替换仅需在 RowParallelLinear 中应用。

### 6.6 bps.init() 的位置考量

放在 `initialize.py` 中，在 `initialize_rerun_state_machine` 之前，确保：
- 所有 rank 初始化顺序一致
- 避免训练脚本漏初始化
- 方便 rerun 机制复用初始化状态

---

## 七、使用方式

```bash
# 仅 DP 使用 BytePS
python pretrain_gpt.py --use-dpu-reduce ...

# 仅 TP 使用 BytePS
python pretrain_gpt.py --use-dpu-tp-reduce ...

# DP + TP 均使用 BytePS
python pretrain_gpt.py --use-dpu-reduce --use-dpu-tp-reduce ...

# 性能调优环境变量
export BYTEPS_PUSH_THREAD=4           # push/pull 线程数
export BYTEPS_KEY_HASH_FN=raw         # hash 策略（raw/djb2/naive/sdbm）
export BYTEPS_RDMA_RX_DEPTH=2048      # 数据面 RX depth
export BYTEPS_RDMA_CTRL_RX_DEPTH=128  # 控制面 RX depth
```

---

## 八、版本历史

| 提交 | 日期 | 内容 |
|---|---|---|
| `d72d072` | 2025-11-25 | 首次 DP 梯度同步替换为 BytePS push_pull |
| `dd15a04` | 2025-11-26 | 新增 `--use-dpu-reduce` 参数，`bps.init()` 迁移至 `initialize.py` |
| `a90a8ca` | 2025-11-29 | DP 梯度同步改为 in-place（减少拷贝） |
| `e23c5f4` | 2026-01-27 | RDMA 双通道 + push/pull 多线程 + hash 策略等带宽优化 |
| 后续提交 | 2026 Q1-Q2 | 新增 `--use-dpu-tp-reduce`、`byteps_collectives.py` 封装、TP all-reduce 替换、pre_declare 机制 |
