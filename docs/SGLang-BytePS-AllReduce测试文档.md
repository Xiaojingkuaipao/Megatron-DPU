# SGLang BytePS All-Reduce 测试文档

## 1. 约束与前置条件

- 只替换模型计算主路径 `GroupCoordinator.all_reduce()`，不替换 reduce-scatter/all-gather/broadcast/send/recv。
- 不支持 CUDA graph / piecewise CUDA graph，必须加 `--disable-cuda-graph --disable-piecewise-cuda-graph`。
- 不支持 FlashInfer/AITER all-reduce fusion，必须加 `--enforce-disable-flashinfer-allreduce-fusion`。
- 不支持 BF16，必须加 `--dtype float16`。
- 不使用外层 `bpslaunch` 包裹 SGLang server；BytePS scheduler/server 单独启动。
- 需要 2 张 GPU。

## 2. 环境准备

使用已有 conda 环境 `sgl-dev`（Python 3.11, PyTorch 2.9.1+cu128）。

```bash
conda activate sgl-dev
```

### 2.1 安装系统依赖（仅 UCX 模式需要）

```bash
sudo apt-get update
sudo apt-get install -y libucx-dev libucx0 ucx-utils rdma-core libibverbs-dev librdmacm-dev
```

验证：

```bash
ucx_info -v          # 期望: UCT version=1.12.1 ...
ucx_info -d | grep -A2 "Transport: posix"  # 期望: Type: intra-node
```

### 2.2 安装 BytePS

```bash
conda activate sgl-dev

# 卸载旧版本 + 清理
python -m pip uninstall -y byteps
rm -rf /workspace/Megatron-DPU/byteps/build
rm -rf /workspace/Megatron-DPU/byteps/dist
rm -rf /workspace/Megatron-DPU/byteps/byteps.egg-info

# 强制重建 ps-lite（避免复用旧的 ZMQ-only libps.a）
rm -f /workspace/Megatron-DPU/byteps/3rdparty/ps-lite/build/*.o
rm -f /workspace/Megatron-DPU/byteps/3rdparty/ps-lite/build/libps.a

# 编译安装（同时启用 TCP + UCX + RDMA 三种传输）
cd /workspace/Megatron-DPU/byteps
BYTEPS_WITH_UCX=1 \
BYTEPS_UCX_HOME=/usr \
BYTEPS_WITHOUT_TENSORFLOW=1 \
BYTEPS_WITHOUT_MXNET=1 \
BYTEPS_WITH_PYTORCH=1 \
python setup.py install
```

验证导入和 UCX 链接：

```bash
cd /tmp && python - <<'PY'
import byteps.torch as bps
from byteps.torch import ops as bps_ops
print("byteps import ok:", bps.__file__)
PY

# 确认 server .so 链接了 UCX 库
ldd $(python -c "import byteps.server; print(byteps.server.__file__.replace('__init__.py','c_lib' + __import__('sysconfig').get_config_var('EXT_SUFFIX')))") 2>/dev/null \
  | grep -E "ucp|uct|ucs|ucm|rdmacm"
# 期望: libucp.so.0, libuct.so.0, libucs.so.0, libucm.so.0, librdmacm.so.1
```

### 2.3 安装 SGLang

```bash
cd /workspace/Megatron-DPU/sglang-0.5.10.post1
pip install -e "python"
```

## 3. 启动服务

启动顺序：**Scheduler → Server → SGLang**。需要三个终端。

### 3.1 启动前清理

每次重新启动前，先彻底清理残留。**UCX 模式下 `/dev/shm` 清理尤其重要**，否则旧共享内存段会导致 TP1 在 `bps.init()` 中 hang：

```bash
kill $(ps aux | grep -E "byteps|sglang" | grep -v grep | awk '{print $2}') 2>/dev/null
sleep 1
rm -rf /tmp/byteps_socket
rm -f /dev/shm/BytePS_* /dev/shm/ps_* /dev/shm/sem.BytePS_*
```

### 3.2 公共环境变量

三个终端都需要 `conda activate sgl-dev` 和以下公共变量：

```bash
export DMLC_NUM_WORKER=1
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=127.0.0.1
export DMLC_PS_ROOT_PORT=9000
export BYTEPS_FORCE_DISTRIBUTED=1
export BYTEPS_KEY_HASH_FN=raw
export BYTEPS_PUSH_THREAD=1
export DMLC_INTERFACE=lo
export DMLC_NODE_HOST=127.0.0.1
```

### 3.3 TCP 模式

**终端 1 — Scheduler：**

```bash
conda activate sgl-dev
cd /tmp
export DMLC_NUM_WORKER=1 DMLC_NUM_SERVER=1 DMLC_PS_ROOT_URI=127.0.0.1 DMLC_PS_ROOT_PORT=9000
export BYTEPS_FORCE_DISTRIBUTED=1 BYTEPS_KEY_HASH_FN=raw BYTEPS_PUSH_THREAD=1
export DMLC_ENABLE_RDMA=0 DMLC_INTERFACE=lo DMLC_NODE_HOST=127.0.0.1 DMLC_USE_GDR=0
export DMLC_ROLE=scheduler
bpslaunch
```

**终端 2 — Server：**

```bash
conda activate sgl-dev
cd /tmp
export DMLC_NUM_WORKER=1 DMLC_NUM_SERVER=1 DMLC_PS_ROOT_URI=127.0.0.1 DMLC_PS_ROOT_PORT=9000
export BYTEPS_FORCE_DISTRIBUTED=1 BYTEPS_KEY_HASH_FN=raw BYTEPS_PUSH_THREAD=1
export DMLC_ENABLE_RDMA=0 DMLC_INTERFACE=lo DMLC_NODE_HOST=127.0.0.1 DMLC_USE_GDR=0
export DMLC_ROLE=server
bpslaunch
```

**终端 3 — SGLang：**

```bash
conda activate sgl-dev
mkdir -p /tmp/byteps_socket

unset BYTEPS_LOCAL_RANK BYTEPS_LOCAL_SIZE
export DMLC_ROLE=worker DMLC_NUM_WORKER=1 DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=127.0.0.1 DMLC_PS_ROOT_PORT=9000 DMLC_WORKER_ID=0
export BYTEPS_FORCE_DISTRIBUTED=1 BYTEPS_KEY_HASH_FN=raw BYTEPS_PUSH_THREAD=1
export BYTEPS_SOCKET_PATH=/tmp/byteps_socket
export DMLC_ENABLE_RDMA=0 DMLC_INTERFACE=lo DMLC_NODE_HOST=127.0.0.1 DMLC_USE_GDR=0

CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --tp-size 2 --host 0.0.0.0 --port 30000 --log-level info --dtype float16 \
  --disable-custom-all-reduce --disable-cuda-graph --disable-piecewise-cuda-graph \
  --enforce-disable-flashinfer-allreduce-fusion \
  --use-byteps-all-reduce --byteps-all-reduce-debug
```

期望日志：两端都出现 `BytePS initialized for SGLang: rank=0/1 ... size=2 local_size=2`，然后 `Uvicorn running on http://0.0.0.0:30000`。

### 3.4 UCX 模式（单机推荐）

与 TCP 的唯一区别：三端都用 `DMLC_ENABLE_UCX=1` 替代 `DMLC_ENABLE_RDMA=0`。

**终端 1 — Scheduler：**

```bash
conda activate sgl-dev
cd /tmp
export DMLC_NUM_WORKER=1 DMLC_NUM_SERVER=1 DMLC_PS_ROOT_URI=127.0.0.1 DMLC_PS_ROOT_PORT=9000
export BYTEPS_FORCE_DISTRIBUTED=1 BYTEPS_KEY_HASH_FN=raw BYTEPS_PUSH_THREAD=1
export DMLC_ENABLE_UCX=1 DMLC_INTERFACE=lo DMLC_NODE_HOST=127.0.0.1
export DMLC_ROLE=scheduler
bpslaunch
```

期望：`enable UCX for networking. group_size=1`

**终端 2 — Server：**

```bash
conda activate sgl-dev
cd /tmp
export DMLC_NUM_WORKER=1 DMLC_NUM_SERVER=1 DMLC_PS_ROOT_URI=127.0.0.1 DMLC_PS_ROOT_PORT=9000
export BYTEPS_FORCE_DISTRIBUTED=1 BYTEPS_KEY_HASH_FN=raw BYTEPS_PUSH_THREAD=1
export DMLC_ENABLE_UCX=1 DMLC_INTERFACE=lo DMLC_NODE_HOST=127.0.0.1
export DMLC_ROLE=server
bpslaunch
```

期望：`UCX connect ... to remote [role=scheduler]`

**终端 3 — SGLang：**

```bash
conda activate sgl-dev
mkdir -p /tmp/byteps_socket

unset BYTEPS_LOCAL_RANK BYTEPS_LOCAL_SIZE
export DMLC_ROLE=worker DMLC_NUM_WORKER=1 DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=127.0.0.1 DMLC_PS_ROOT_PORT=9000 DMLC_WORKER_ID=0
export BYTEPS_FORCE_DISTRIBUTED=1 BYTEPS_KEY_HASH_FN=raw BYTEPS_PUSH_THREAD=1
export BYTEPS_SOCKET_PATH=/tmp/byteps_socket
export DMLC_ENABLE_UCX=1 DMLC_INTERFACE=lo DMLC_NODE_HOST=127.0.0.1

CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --tp-size 2 --host 0.0.0.0 --port 30000 --log-level info --dtype float16 \
  --disable-custom-all-reduce --disable-cuda-graph --disable-piecewise-cuda-graph \
  --enforce-disable-flashinfer-allreduce-fusion \
  --use-byteps-all-reduce --byteps-all-reduce-debug
```

期望日志：

```
[TP0] BytePS initialized for SGLang: rank=0 local_rank=0 size=2 local_size=2
[TP1] BytePS initialized for SGLang: rank=1 local_rank=1 size=2 local_size=2
enable UCX for networking. group_size=1
UCX connect ... to remote [role=scheduler,...]
UCX connect ... to remote [role=server,...]
Application startup complete.
Uvicorn running on http://0.0.0.0:30000
```

> **如果 TP1 没有出现 `BytePS initialized` 日志**：说明 TP1 卡在 `bps.init()`。这是因为上次 UCX 运行的共享内存残留。执行 3.1 节的清理步骤后重试。

## 4. 请求验证

```bash
curl http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"temperature":0,"max_new_tokens":4}}'
```

期望：`" Paris. It is"`，HTTP 200，不 hang。

首次请求后日志中出现 `Declared BytePS tensor name=sglang.tp:0.rXXXX.row_parallel_linear.model.layers.0.self_attn.o_proj.6x896 expected_workers=1` 说明主路径已进入 BytePS All-Reduce（tensor name 末尾的 `6x896` 是自动拼接的 shape 后缀）。

## 5. 传输模式

安装后 BytePS 支持三种模式，通过环境变量切换（三端必须一致）：

| 环境变量 | 传输方式 | 适用场景 |
|----------|---------|---------|
| `DMLC_ENABLE_RDMA=0` | ZMQ/TCP loopback | 基础验证 |
| `DMLC_ENABLE_UCX=1` | UCX 共享内存 (posix/sysv/cma) | 单机推荐 |
| `DMLC_ENABLE_RDMA=ibverbs` | InfiniBand/RoCE RDMA | 多机硬件 RDMA |

UCX 在同机上自动选择共享内存传输，无需 RDMA 硬件，比 TCP 快约 10~13%。

## 6. NCCL Baseline 对照

```bash
conda activate sgl-dev
cd /workspace/Megatron-DPU/sglang-0.5.10.post1
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --tp-size 2 --host 0.0.0.0 --port 30001 --log-level info --dtype float16 \
  --disable-custom-all-reduce --disable-cuda-graph --disable-piecewise-cuda-graph \
  --enforce-disable-flashinfer-allreduce-fusion
```

```bash
curl http://127.0.0.1:30001/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"temperature":0,"max_new_tokens":1}}'
```

## 7. 延迟 Benchmark

Benchmark 脚本位于 `/tmp/benchmark_sglang.py`。运行方式：

```bash
conda activate sgl-dev
python /tmp/benchmark_sglang.py "BytePS UCX All-Reduce"
```

脚本自动测 5 个 max_new_tokens 档位（1/4/16/64/128），每档 3 轮 warmup + 10 轮 benchmark，结果存为 `/tmp/benchmark_*.json`。

参考结果（Qwen2.5-0.5B-Instruct, tp-size=2, float16）：

| max_tokens | NCCL | BytePS TCP | BytePS UCX | UCX 提升 |
|:----------:|:----:|:----------:|:----------:|:--------:|
| 1 | 0.071s | 0.199s | 0.180s | 10% |
| 4 | 0.142s | 0.463s | 0.439s | 5% |
| 16 | 0.352s | 1.540s | 1.375s | 11% |
| 64 | 1.129s | 5.970s | 5.186s | 13% |
| 128 | 2.108s | 11.778s | 10.329s | 12% |

## 8. 常见问题

### 启动 / 连接

**`ModuleNotFoundError: No module named 'byteps'`**
重新执行 2.2 节 BytePS 安装。

**找不到 `--use-byteps-all-reduce`**
重新执行 2.3 节 SGLang 安装。检查 `python -c "import sglang; print(sglang.__file__)"`。

**`Missing ./3rdparty/ps-lite`**
```bash
cd /workspace/Megatron-DPU/byteps
git submodule update --init --recursive
# 或兜底:
# git clone -b byteps https://github.com/bytedance/ps-lite 3rdparty/ps-lite
```

**`Address already in use. errno = 98`**
端口被旧进程占用，执行 3.1 节清理步骤。

**`Tensor type torch.cuda.BFloat16Tensor is not supported`**
确认加了 `--dtype float16`。

**`basic_string::_M_construct null not valid`**
scheduler/server/worker 三端都加 `export DMLC_USE_GDR=0`。

### CUDA Graph / Fusion 报错

确认命令包含：
```
--disable-cuda-graph --disable-piecewise-cuda-graph
--enforce-disable-flashinfer-allreduce-fusion
```
且没有 `--enable-aiter-allreduce-fusion`。

### BytePS local rank/size mismatch

启动 SGLang 前确保：
```bash
unset BYTEPS_LOCAL_RANK
unset BYTEPS_LOCAL_SIZE
```

### All-Reduce Hang

**Basic 排查**：先用 TCP 模式跑通，确认 TCP 正常后再切 UCX。

**TP1 不出现 `BytePS initialized`（UCX 特有问题）**：
TP1 卡在 `bps.init()`。原因：`/dev/shm` 中残留了上次 UCX 运行的共享内存段。
解决：执行完整清理：
```bash
kill $(ps aux | grep -E "byteps|sglang" | grep -v grep | awk '{print $2}') 2>/dev/null
sleep 1
rm -rf /tmp/byteps_socket
rm -f /dev/shm/BytePS_* /dev/shm/ps_* /dev/shm/sem.BytePS_*
```

**启动成功但首次 `curl /generate` 卡住**：
如果两个 TP rank 都有 `BytePS initialized` 但请求不返回：
1. 确认 warmup 已完成。若 prefill 有 `Declared` 但 decode 没有，说明 tensor name shape 未隔离（检查 `byteps_collectives.py` Bug #2 修复是否已包含）。
2. 确认两个 TP rank 都声明了同样的 tensor name。
3. 查看 scheduler/server 终端有无日志。
4. 用 `--log-level debug` 启动观察 Entering/returned 日志定位卡住的层。

### UCX 特有

**`unsupported van type: ucx`**
BytePS 编译时未启用 UCX 或 server `.so` 未重新链接。确认：
1. 已删除 `3rdparty/ps-lite/build/*.o` 和 `libps.a` 后重新编译
2. `ldd` 验证 `libucp.so.0` 已链接（见 2.2 节）

**scheduler Aborted (core dumped)**
旧 ZMQ-only `van.o` 被复用。删除 `3rdparty/ps-lite/build/*.o`、`libps.a`、`byteps/build/`，重新编译。

**三端连接不匹配**
scheduler、server、worker 必须使用相同的 `DMLC_ENABLE_UCX` / `DMLC_ENABLE_RDMA` 设置。

**UCX hang 排查**
1. 回退 TCP 确认正常：`DMLC_ENABLE_RDMA=0`
2. 检查 UCX 已安装：`dpkg -l | grep libucx`
3. 检查 BytePS 链接 UCX：`ldd`（见 2.2 节）
4. 开启 UCX 日志：`export UCX_LOG_LEVEL=info`
5. 清理 `/dev/shm` 残留（见上）
