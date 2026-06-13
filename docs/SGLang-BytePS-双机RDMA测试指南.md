# SGLang BytePS 双机 RDMA 推理指南

本文档记录在 gpu04 + gpu03 双机上使用 BytePS（ibverbs RDMA）跑通 SGLang tp=4 跨机推理的完整流程和踩坑记录。

## 1. 架构概述

```
gpu04 (node_rank=0)                    gpu03 (node_rank=1)
┌──────────────────────┐              ┌──────────────────────┐
│ BytePS Scheduler     │              │                      │
│ BytePS Server        │◄── RDMA ───►│ SGLang TP2, TP3      │
│ SGLang TP0, TP1      │   ibverbs   │                      │
└──────────────────────┘              └──────────────────────┘
```

- **BytePS Scheduler**: 节点注册协调，运行在 gpu04
- **BytePS Server**: 张量聚合（push/pull），运行在 gpu04
- **4 个 TP Rank**: TP0/TP1 在 gpu04，TP2/TP3 在 gpu03
- **RDMA 网卡**: Mellanox ConnectX-6 Dx, RoCEv2, 单口绑定 `mlx5_1` → `ens39f1np1` → 192.168.1.x

关键 BytePS 参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| `DMLC_NUM_WORKER` | 2 | 两个节点，每节点一个 root worker |
| `DMLC_WORKER_ID` | 0 或 1 | 节点索引 |
| `BYTEPS_LOCAL_SIZE` | 2 | 每节点的 GPU 数（非总 TP size） |
| `expected_workers` | 2 | `bps.declare()` 中每个 key 期望的 push 数 |
| `DMLC_ENABLE_RDMA` | ibverbs | 走 verbs RDMA 路径 |

## 2. 代码修改

### 2.1 修改前的问题

原代码有三个阻止多机运行的 bug：

**Bug 1 — `DMLC_NUM_WORKER` 硬编码为 1**

`byteps_collectives.py` 中原代码：
```python
os.environ.setdefault("DMLC_NUM_WORKER", "1")
```
多机场景下 BytePS scheduler 需要知道总共等待多少个 worker 节点才能完成 barrier。单机是 1，双机应该是 2。

**Bug 2 — `DMLC_WORKER_ID` 总是 0**

```python
os.environ.setdefault("DMLC_WORKER_ID", os.environ.get("DMLC_WORKER_ID", "0"))
```
每个节点的 root worker 需要唯一的 worker ID 向 scheduler 注册。不改的话 gpu03 也用 ID=0，scheduler 看到两个相同 ID 的 worker 会出问题。

**Bug 3 — `BYTEPS_LOCAL_SIZE` 错误**

`model_runner.py` 中：
```python
os.environ.setdefault("BYTEPS_LOCAL_SIZE", str(self.tp_size * self.pp_size))
```
多机时 `tp_size=4` 会把 `BYTEPS_LOCAL_SIZE` 设为 4，但 gpu04 只有 2 张 GPU。BytePS root device 会等待 4 个本机进程，但只有 2 个存在，导致 hang。正确的值是 `tp_size // nnodes = 2`。

### 2.2 修改内容

**`byteps_collectives.py`：**

1. 新增 `_BPS_NNODES` 模块变量
2. `initialize_byteps_for_sglang()` 增加 `nnodes` 和 `node_rank` 参数，用它们设置 `DMLC_NUM_WORKER` 和 `DMLC_WORKER_ID`
3. `_byteps_expected_workers()` 在多机模式下直接返回 `_BPS_NNODES`

```python
# 新增
_BPS_NNODES = 1

def initialize_byteps_for_sglang(
    local_rank: int, local_size: int,
    nnodes: int = 1, node_rank: int = 0,  # 新增参数
    debug: bool = False,
) -> None:
    global _BPS_NNODES
    _BPS_NNODES = nnodes
    # ...
    os.environ.setdefault("DMLC_NUM_WORKER", str(nnodes))    # 改
    os.environ.setdefault("DMLC_WORKER_ID", str(node_rank))  # 改

def _byteps_expected_workers() -> int:
    if _BPS_NNODES > 1:          # 多机：直接用节点数
        return _BPS_NNODES
    # 单机：原有逻辑兜底
    local_size = max(1, bps.local_size())
    return max(1, bps.size() // local_size)
```

**`model_runner.py`：**

1. 计算 `gpus_per_node = tp_size // nnodes` 作为 `BYTEPS_LOCAL_SIZE`
2. 传递 `nnodes` 和 `node_rank` 给 `initialize_byteps_for_sglang`

```python
gpus_per_node = max(1, self.server_args.tp_size // self.server_args.nnodes)
os.environ.setdefault("BYTEPS_LOCAL_SIZE", str(gpus_per_node * self.pp_size))

initialize_byteps_for_sglang(
    local_rank=self.gpu_id,
    local_size=gpus_per_node * self.pp_size,
    nnodes=self.server_args.nnodes,
    node_rank=self.server_args.node_rank,
    debug=self.server_args.byteps_all_reduce_debug,
)
```

### 2.3 无需修改

- `server_args.py` — `--nnodes`、`--node-rank`、`--dist-init-addr` 已存在
- `parallel_state.py` / `engine.py` — 多机 TP rank 分区逻辑已正确

## 3. 启动流程

### 先决条件

- 两边都已安装含 RDMA 支持的 BytePS（`BYTEPS_WITH_UCX=1 BYTEPS_UCX_HOME=/usr`）
- 两边 conda 环境 `sgl-dev` 已安装 SGLang（`pip install -e "python"`）
- gpu04 → gpu03 SSH 免密
- 模型（如 Qwen3-32B）已在两边缓存

### Step 1 — 启动前清理（两边都要做）

```bash
# gpu04
kill $(ps aux | grep -E "byteps|sglang" | grep -v grep | awk '{print $2}') 2>/dev/null
rm -rf /tmp/byteps_socket
rm -f /dev/shm/BytePS_* /dev/shm/ps_* /dev/shm/sem.BytePS_*

# gpu03
ssh -p 10022 root@172.16.41.73 'kill $(ps aux | grep sglang | grep -v grep | awk "{print \$2}") 2>/dev/null'
ssh -p 10022 root@172.16.41.73 'rm -rf /tmp/byteps_socket; rm -f /dev/shm/BytePS_* /dev/shm/ps_* /dev/shm/sem.BytePS_*'
```

### Step 2 — gpu04 启动 BytePS Scheduler

```bash
conda activate sgl-dev
cd /tmp
export DMLC_ENABLE_RDMA=ibverbs
export DMLC_NUM_WORKER=2
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=192.168.1.13
export DMLC_PS_ROOT_PORT=9000
export DMLC_INTERFACE=ens39f1np1
export DMLC_NODE_HOST=192.168.1.13
export DMLC_ROLE=scheduler
export BYTEPS_FORCE_DISTRIBUTED=1
bpslaunch
```

期望输出：`Creating Van: ibverbs. group_size=1`

### Step 3 — gpu04 启动 BytePS Server

```bash
conda activate sgl-dev
cd /tmp
export DMLC_ENABLE_RDMA=ibverbs
export DMLC_NUM_WORKER=2
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=192.168.1.13
export DMLC_PS_ROOT_PORT=9000
export DMLC_INTERFACE=ens39f1np1
export DMLC_NODE_HOST=192.168.1.13
export DMLC_ROLE=server
export BYTEPS_FORCE_DISTRIBUTED=1
bpslaunch
```

期望输出：`Connect to Node 1 with Transport=RDMA`

### Step 4 — gpu04 启动 SGLang (node_rank=0)

```bash
conda activate sgl-dev
mkdir -p /tmp/byteps_socket
rm -f /dev/shm/BytePS_* /dev/shm/ps_* /dev/shm/sem.BytePS_*

unset BYTEPS_LOCAL_RANK BYTEPS_LOCAL_SIZE
export DMLC_ENABLE_RDMA=ibverbs
export DMLC_INTERFACE=ens39f1np1
export DMLC_NODE_HOST=192.168.1.13
export DMLC_PS_ROOT_URI=192.168.1.13
export DMLC_PS_ROOT_PORT=9000
export DMLC_NUM_SERVER=1
export BYTEPS_FORCE_DISTRIBUTED=1
export BYTEPS_KEY_HASH_FN=raw
export BYTEPS_PUSH_THREAD=1
export BYTEPS_SOCKET_PATH=/tmp/byteps_socket
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_1
export NCCL_IB_GID_INDEX=0
export NCCL_SOCKET_IFNAME=ens39f1np1
export GLOO_SOCKET_IFNAME=ens39f1np1

CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
  --model-path Qwen/Qwen3-32B \
  --tp-size 4 --nnodes 2 --node-rank 0 \
  --dist-init-addr 192.168.1.13:25000 \
  --host 0.0.0.0 --port 30000 --log-level info --dtype float16 \
  --disable-custom-all-reduce --disable-cuda-graph --disable-piecewise-cuda-graph \
  --enforce-disable-flashinfer-allreduce-fusion \
  --use-byteps-all-reduce --byteps-all-reduce-debug
```

等日志中出现 `Init torch distributed begin` 后执行 Step 5。

### Step 5 — gpu03 启动 SGLang (node_rank=1)

```bash
conda activate sgl-dev
mkdir -p /tmp/byteps_socket
rm -f /dev/shm/BytePS_* /dev/shm/ps_* /dev/shm/sem.BytePS_*

unset BYTEPS_LOCAL_RANK BYTEPS_LOCAL_SIZE
export DMLC_ENABLE_RDMA=ibverbs
export DMLC_INTERFACE=ens39f1np1
export DMLC_NODE_HOST=192.168.1.12
export DMLC_PS_ROOT_URI=192.168.1.13
export DMLC_PS_ROOT_PORT=9000
export DMLC_NUM_SERVER=1
export BYTEPS_FORCE_DISTRIBUTED=1
export BYTEPS_KEY_HASH_FN=raw
export BYTEPS_PUSH_THREAD=1
export BYTEPS_SOCKET_PATH=/tmp/byteps_socket
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_1
export NCCL_IB_GID_INDEX=0
export NCCL_SOCKET_IFNAME=ens39f1np1
export GLOO_SOCKET_IFNAME=ens39f1np1

CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
  --model-path Qwen/Qwen3-32B \
  --tp-size 4 --nnodes 2 --node-rank 1 \
  --dist-init-addr 192.168.1.13:25000 \
  --host 0.0.0.0 --port 30001 --log-level info --dtype float16 \
  --disable-custom-all-reduce --disable-cuda-graph --disable-piecewise-cuda-graph \
  --enforce-disable-flashinfer-allreduce-fusion \
  --use-byteps-all-reduce --byteps-all-reduce-debug
```

### gpu04 vs gpu03 差异对照

| 环境变量 / 参数 | gpu04 | gpu03 |
|----------------|-------|-------|
| `DMLC_NODE_HOST` | `192.168.1.13` | `192.168.1.12` |
| `--node-rank` | `0` | `1` |
| `--port` | `30000` | `30001` |
| Scheduler/Server | 需要启动 | 不需要 |

其他环境变量完全相同。

## 4. 验证

```bash
curl http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"temperature":0,"max_new_tokens":4}}'
```

期望所有四个 TP rank 日志中均出现：
```
Creating Van: ibverbs
BytePS initialized for SGLang: rank=X local_rank=Y size=4 local_size=2
Declared BytePS tensor name=... expected_workers=2
```

## 5. 踩坑记录

### 坑 1：tp_size 必须能整除模型的 attention heads 数

**现象**：gpu03 SGLang 启动后立即报
```
AssertionError: assert self.total_num_heads % tp_size == 0
```

**原因**：Qwen2.5-0.5B 只有 14 个 attention heads，`14 % 4 = 2 ≠ 0`。SGLang 要求 tp_size 能整除模型的总 heads 数。

**解决**：换用 heads 数能被 4 整除的模型。Qwen3-32B 有 64 heads（`64 % 4 = 0`），符合要求。

> 选择模型时注意：`num_attention_heads % tp_size == 0` 且 `num_key_value_heads % tp_size == 0`。

### 坑 2：跨机时 `BYTEPS_LOCAL_SIZE` 必须等于本机 GPU 数

**现象**：原代码把 `BYTEPS_LOCAL_SIZE` 设为 `tp_size * pp_size`（双机 tp=4 时为 4），导致 BytePS root device 等待 4 个本机进程，但只有 2 个存在，rank 1 卡在 `bps.init()` 中不出来。

**原因**：BytePS 的 `BYTEPS_LOCAL_SIZE` 含义是**本节点的进程数**，不是全局 TP size。跨机时每节点只有 `tp_size // nnodes` 个 GPU。

**解决**：修改 `model_runner.py`，计算 `gpus_per_node = tp_size // nnodes`，作为 `BYTEPS_LOCAL_SIZE`。

### 坑 3：`DMLC_NUM_WORKER` 和 `DMLC_WORKER_ID` 硬编码

**现象**：BytePS scheduler 只等待 1 个 worker，gpu03 的 worker 无法注册，导致整个集群 hang。

**原因**：`byteps_collectives.py` 中 `DMLC_NUM_WORKER` 硬编码为 `"1"`，`DMLC_WORKER_ID` 硬编码为 `"0"`。

**解决**：从 `--nnodes` 和 `--node-rank` 分别获取真实节点数和当前节点编号。

### 坑 4：`expected_workers` 计算不适用多机

**现象**：原公式 `max(1, bps.size() // bps.local_size())` 在多机下计算错误。BytePS 中 `bps.size()` 和 `bps.local_size()` 在多机场景下语义和单机不同，`bps.size()` 可能返回总进程数而非节点数。

**解决**：多机模式（`nnodes > 1`）下直接用 `nnodes` 作为 `expected_workers`。单机保持原逻辑兜底。

### 坑 5：`/dev/shm` 共享内存残留导致 hang

**现象**：第二次启动时 TP1（或跨机时任意 rank）卡在 `bps.init()`，只看到一个 rank 的 `BytePS initialized` 日志。

**原因**：BytePS 使用 POSIX 共享内存（`/dev/shm/BytePS_*`）在同机 rank 间传递节点信息。上次进程异常退出后，残留的共享内存段让新进程读到脏数据。

**解决**：每次启动前清理：
```bash
rm -f /dev/shm/BytePS_* /dev/shm/ps_* /dev/shm/sem.BytePS_*
```

### 坑 6：BytePS server 收到垃圾 `expected_workers` 值

**现象**：Server 启动后报
```
Check failed: (state->expected_workers) == (expected_workers)
Key X registered with inconsistent expected_workers: 1897029193 vs 898307597
```
（数值是随机垃圾，非正常的 1 或 2）

**原因**：BytePS server 的 C++ 代码中，`expected_workers` 通过 key-value 序列化在 worker 和 server 之间传输。如果两边 BytePS 的 C++ 扩展编译版本不同（struct 布局 / 序列化格式不一致），server 端反序列化时会读到错误偏移位置的数据。

**解决**：确保两台机器的 BytePS 使用**完全相同的源码和编译参数**构建。具体来说：(1) 两边仓库代码一致，(2) 编译参数一致（都是 `BYTEPS_WITH_UCX=1 BYTEPS_UCX_HOME=/usr`），(3) 系统依赖一致（`libucx-dev`、`rdma-core` 等均安装）。

## 6. 快速参考：RDMA 网卡绑定

```
NCCL_IB_HCA=mlx5_1
    → /sys/class/infiniband/mlx5_1/device/net/
    → ens39f1np1
    → 192.168.1.13 (gpu04) / 192.168.1.12 (gpu03)
```

详见 `gpu-cluster-env.md` 和 Megatron-LM Qwen 训练脚本中的自动发现模式。
