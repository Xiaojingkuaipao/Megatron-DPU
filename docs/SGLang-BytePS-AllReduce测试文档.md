# SGLang BytePS All-Reduce 测试文档

本文档按当前二合一仓库编写。服务器上只需要一份 `Megatron-DPU` 仓库，BytePS 从 `byteps/` 安装，SGLang 从 `sglang-0.5.10.post1/python/` 安装。

## 1. 测试目标

验证 `--use-byteps-all-reduce` 能把 SGLang 模型计算主路径 All-Reduce 路由到本仓库里的 BytePS。

当前第一阶段约束：

- 只替换模型计算主路径 `GroupCoordinator.all_reduce()`。
- 不替换 scheduler、cache、speculative 等控制面直接 `torch.distributed.all_reduce()`。
- 不替换 reduce-scatter、all-gather、broadcast、send/recv。
- 当前源码没有发现 `quant_all_reduce()` 符号；如果后续路径进入量化 All-Reduce，应先显式报错，不允许 fallback。
- 不支持 CUDA graph / piecewise CUDA graph；BytePS 测试必须加 `--disable-cuda-graph --disable-piecewise-cuda-graph`。
- 首测建议单机多 GPU，先用 `--tp-size 2`。
- 不要用外层 `bpslaunch` 包 SGLang server。SGLang model worker 会在进程内设置 BytePS local rank/local size 并初始化 BytePS。

## 2. 拉取仓库代码

```bash
# 首次拉取
git clone <Megatron-DPU-git-url> /workspace/Megatron-DPU
cd /workspace/Megatron-DPU
git checkout <包含本次修改的分支>

# 已有仓库则更新
cd /workspace/Megatron-DPU
git fetch --all --prune
git checkout <包含本次修改的分支>
git pull --ff-only
```

BytePS 依赖 `3rdparty/ps-lite`，拉取后补齐子模块：

```bash
cd /workspace/Megatron-DPU/byteps
git submodule update --init --recursive
```

如果 submodule 命令不能识别 `ps-lite`，用兜底方式：

```bash
rm -rf /workspace/Megatron-DPU/byteps/3rdparty/ps-lite
git clone -b byteps https://github.com/bytedance/ps-lite /workspace/Megatron-DPU/byteps/3rdparty/ps-lite
```

## 3. 进入 Python 环境

服务器使用已有 conda 环境 `sgl-dev2`，Python 版本为 3.11。

```bash
conda activate sgl-dev2
python --version
python -m pip install --upgrade pip setuptools wheel
```

建议先确认 PyTorch 和 CUDA 可用：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
PY
```

`--tp-size 2` 至少需要当前环境可见 2 张 GPU。

## 4. 更新并安装 BytePS

每次服务器拉取了新的 BytePS 代码后，都建议重新安装 BytePS，避免 Python 环境里残留旧扩展。

先卸载旧包并清理构建产物：

```bash
conda activate sgl-dev2
python -m pip uninstall -y byteps
rm -rf /workspace/Megatron-DPU/byteps/build
rm -rf /workspace/Megatron-DPU/byteps/dist
rm -rf /workspace/Megatron-DPU/byteps/byteps.egg-info
```

安装普通 TCP/本机测试版。首测建议先用这个，不要一开始就引入 RDMA/UCX：

```bash
conda activate sgl-dev2
cd /workspace/Megatron-DPU/byteps
BYTEPS_WITHOUT_TENSORFLOW=1 \
BYTEPS_WITHOUT_MXNET=1 \
BYTEPS_WITH_PYTORCH=1 \
python setup.py install
```

如果服务器 NCCL 不在默认路径，安装前加上真实 NCCL 目录，例如：

```bash
export BYTEPS_NCCL_HOME=/usr/local/nccl
```

如果普通 TCP 路径已经验证通过，再安装 RDMA/UCX 版本：

```bash
conda activate sgl-dev2
python -m pip uninstall -y byteps
rm -rf /workspace/Megatron-DPU/byteps/build
rm -rf /workspace/Megatron-DPU/byteps/dist
rm -rf /workspace/Megatron-DPU/byteps/byteps.egg-info
cd /workspace/Megatron-DPU/byteps
BYTEPS_WITH_UCX=1 \
BYTEPS_WITHOUT_TENSORFLOW=1 \
BYTEPS_WITHOUT_MXNET=1 \
BYTEPS_WITH_PYTORCH=1 \
python setup.py install
```

BytePS 安装检查。不要在 BytePS 源码目录里做 import 检查，否则当前目录下的 `byteps/` 可能遮蔽已安装包，导致 `c_lib` 导入异常。建议在 `/tmp` 运行：

```bash
cd /tmp
conda activate sgl-dev2
python - <<'PY'
import byteps.torch as bps
from byteps.torch import ops as bps_ops
print("byteps import ok")
print("byteps module:", bps.__file__)
print("byteps ops import ok")
PY
```

做 BytePS init smoke test 时，当前 BytePS 构建建议使用外部 scheduler/server。不要用 `bpslaunch` 包裹 SGLang server；这里只单独启动 BytePS scheduler/server 来验证 BytePS 本身。

终端 1，启动 scheduler：

```bash
pkill -f "byteps.server" || true
pkill -f "bpslaunch" || true
pkill -f "sglang.launch_server" || true
pkill -f "sglang.srt" || true
rm -rf /tmp/byteps_socket
mkdir -p /tmp/byteps_socket
```

```bash
cd /tmp
conda activate sgl-dev2
export DMLC_NUM_WORKER=1
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=127.0.0.1
export DMLC_PS_ROOT_PORT=9000
export BYTEPS_FORCE_DISTRIBUTED=1
export BYTEPS_KEY_HASH_FN=raw
export BYTEPS_PUSH_THREAD=1
export BYTEPS_LOG_LEVEL=INFO
export DMLC_ENABLE_RDMA=0
export DMLC_INTERFACE=lo
export DMLC_NODE_HOST=127.0.0.1
export DMLC_USE_GDR=0
export DMLC_ROLE=scheduler
bpslaunch
```

终端 2，启动 server：

```bash
cd /tmp
conda activate sgl-dev2
export DMLC_NUM_WORKER=1
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=127.0.0.1
export DMLC_PS_ROOT_PORT=9000
export BYTEPS_FORCE_DISTRIBUTED=1
export BYTEPS_KEY_HASH_FN=raw
export BYTEPS_PUSH_THREAD=1
export BYTEPS_LOG_LEVEL=INFO
export DMLC_ENABLE_RDMA=0
export DMLC_INTERFACE=lo
export DMLC_NODE_HOST=127.0.0.1
export DMLC_USE_GDR=0
export DMLC_ROLE=server
bpslaunch
```

终端 3，启动 worker smoke test：

```bash
cd /tmp
conda activate sgl-dev2
mkdir -p /tmp/byteps_socket

DMLC_ROLE=worker \
DMLC_NUM_WORKER=1 \
DMLC_NUM_SERVER=1 \
DMLC_PS_ROOT_URI=127.0.0.1 \
DMLC_PS_ROOT_PORT=9000 \
DMLC_WORKER_ID=0 \
DMLC_ENABLE_RDMA=0 \
DMLC_USE_GDR=0 \
DMLC_INTERFACE=lo \
DMLC_NODE_HOST=127.0.0.1 \
BYTEPS_FORCE_DISTRIBUTED=1 \
BYTEPS_KEY_HASH_FN=raw \
BYTEPS_PUSH_THREAD=1 \
BYTEPS_SOCKET_PATH=/tmp/byteps_socket \
BYTEPS_LOCAL_RANK=0 \
BYTEPS_LOCAL_SIZE=1 \
timeout 30s python - <<'PY'
import byteps.torch as bps
bps.init()
print("rank:", bps.rank())
print("size:", bps.size())
print("local_rank:", bps.local_rank())
print("local_size:", bps.local_size())
bps.shutdown()
PY
```

通过标准：

```text
rank: 0
size: 1
local_rank: 0
local_size: 1
```

> 注意：本 smoke test 显式设置 `BYTEPS_LOCAL_SIZE=1`，仅用于验证 BytePS 基础可用性。在 SGLang 实际运行时，`initialize_byteps_for_sglang()` 会按 `tp_size * pp_size` 设置 `BYTEPS_LOCAL_SIZE`（例如 `--tp-size 2` 时为 2）。两者在各自场景下都是正确的，不要将 smoke test 的 `1` 作为 SGLang 运行时的值。

如果没有拉取包含 `byteps/byteps/common/operations.cc` 修复的代码，`DMLC_USE_GDR` 未设置时可能出现 `basic_string::_M_construct null not valid`。临时绕过方式是在 scheduler、server、worker 环境中都显式设置 `DMLC_USE_GDR=0`；拉取修复后，未设置时默认也按 `0` 处理。

可选再做一个单机双 rank `float16 push_pull` smoke test。

**关于 `expected_workers`**：这里 `expected_workers=1`，不是 `2`。`byteps_collectives.py` 中的 `_byteps_expected_workers()` 辅助函数计算方式为 `max(1, bps.size() // bps.local_size())`。单机两个本地 rank 时 `bps.size()=2, bps.local_size()=2`，结果为 `max(1, 2 // 2) = 1`。BytePS server 等待的是 worker 节点数（跨机器），不是本地 GPU rank 数。如果误写成 `expected_workers=2`，server 会等待两个 worker 节点的 push，但单机只有一个 BytePS root worker 会向 server push，导致 `push_pull_async_inplace + synchronize` 卡住。

```bash
cat > /tmp/byteps_tp2_pushpull.py <<'PY'
import os
import multiprocessing as mp


def run(local_rank):
    os.environ["DMLC_ROLE"] = "worker"
    os.environ["DMLC_NUM_WORKER"] = "1"
    os.environ["DMLC_NUM_SERVER"] = "1"
    os.environ["DMLC_PS_ROOT_URI"] = "127.0.0.1"
    os.environ["DMLC_PS_ROOT_PORT"] = "9000"
    os.environ["DMLC_WORKER_ID"] = "0"
    os.environ["DMLC_ENABLE_RDMA"] = "0"
    os.environ["DMLC_USE_GDR"] = "0"
    os.environ["DMLC_INTERFACE"] = "lo"
    os.environ["DMLC_NODE_HOST"] = "127.0.0.1"
    os.environ["BYTEPS_FORCE_DISTRIBUTED"] = "1"
    os.environ["BYTEPS_KEY_HASH_FN"] = "raw"
    os.environ["BYTEPS_PUSH_THREAD"] = "1"
    os.environ["BYTEPS_SOCKET_PATH"] = "/tmp/byteps_socket"
    os.environ["BYTEPS_LOCAL_RANK"] = str(local_rank)
    os.environ["BYTEPS_LOCAL_SIZE"] = "2"

    import torch
    import byteps.torch as bps
    from byteps.torch import ops as bps_ops

    torch.cuda.set_device(local_rank)
    bps.init()
    print("init", local_rank, "rank", bps.rank(), "size", bps.size(), flush=True)

    x = torch.full((8,), local_rank + 1, device="cuda", dtype=torch.float16)
    name = "debug.tp2.float16"
    bps_ops.declare(name, expected_workers=1)
    h = bps_ops.push_pull_async_inplace(x, average=False, name=name)
    bps_ops.synchronize(h)
    torch.cuda.synchronize()

    print("done", local_rank, x.cpu().tolist(), flush=True)
    bps.shutdown()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    ps = [mp.Process(target=run, args=(i,)) for i in range(2)]
    for p in ps:
        p.start()
    for p in ps:
        p.join(60)
    for p in ps:
        print("pid", p.pid, "exitcode", p.exitcode, flush=True)
        if p.exitcode is None:
            p.terminate()
PY

rm -rf /tmp/byteps_socket
mkdir -p /tmp/byteps_socket
CUDA_VISIBLE_DEVICES=0,1 timeout 120s python /tmp/byteps_tp2_pushpull.py
```

通过标准：

```text
done 0 [3.0, 3.0, ...]
done 1 [3.0, 3.0, ...]
```

## 5. 安装 SGLang

从同一个二合一仓库安装 SGLang：

```bash
conda activate sgl-dev2
cd /workspace/Megatron-DPU/sglang-0.5.10.post1
python -m pip install -e "python"
```

确认安装的是当前仓库里的 SGLang：

```bash
conda activate sgl-dev2
python - <<'PY'
import sglang
from sglang.srt.server_args import ServerArgs
from sglang.srt.distributed.byteps_collectives import initialize_byteps_for_sglang
print("sglang import ok")
print("sglang module:", sglang.__file__)
print("use_byteps_all_reduce default:", ServerArgs.use_byteps_all_reduce)
print("byteps wrapper import ok")
PY
```

语法检查：

```bash
conda activate sgl-dev2
PYTHONPYCACHEPREFIX=/tmp/sglang-byteps-pycache python -m py_compile \
  /workspace/Megatron-DPU/sglang-0.5.10.post1/python/sglang/srt/distributed/byteps_collectives.py \
  /workspace/Megatron-DPU/sglang-0.5.10.post1/python/sglang/srt/distributed/parallel_state.py \
  /workspace/Megatron-DPU/sglang-0.5.10.post1/python/sglang/srt/distributed/communication_op.py \
  /workspace/Megatron-DPU/sglang-0.5.10.post1/python/sglang/srt/server_args.py \
  /workspace/Megatron-DPU/sglang-0.5.10.post1/python/sglang/srt/model_executor/model_runner.py \
  /workspace/Megatron-DPU/sglang-0.5.10.post1/python/sglang/srt/layers/linear.py \
  /workspace/Megatron-DPU/sglang-0.5.10.post1/python/sglang/srt/layers/vocab_parallel_embedding.py
```

## 6. 启动 BytePS 版 SGLang

首测使用 2 张 GPU。下面示例模型路径是 `/data/models/Qwen2.5-0.5B-Instruct`，测试时替换成服务器真实模型目录。

启动前先确认 BytePS scheduler/server 已按第 10 节方式保持运行。不要用 `bpslaunch` 包裹 SGLang server；SGLang server 使用常规入口 `python -m sglang.launch_server` 启动。

不要手动设置 `BYTEPS_LOCAL_RANK` 和 `BYTEPS_LOCAL_SIZE`，这两个值由 SGLang model worker 内部按 `gpu_id` 和 `tp_size * pp_size` 设置。

```bash
cd /workspace/Megatron-DPU/sglang-0.5.10.post1
conda activate sgl-dev2
mkdir -p /tmp/byteps_socket

unset BYTEPS_LOCAL_RANK
unset BYTEPS_LOCAL_SIZE
export DMLC_ROLE=worker
export DMLC_NUM_WORKER=1
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=127.0.0.1
export DMLC_PS_ROOT_PORT=9000
export DMLC_WORKER_ID=0
export BYTEPS_FORCE_DISTRIBUTED=1
export BYTEPS_KEY_HASH_FN=raw
export BYTEPS_PUSH_THREAD=1
export BYTEPS_LOG_LEVEL=INFO
export BYTEPS_SOCKET_PATH=/tmp/byteps_socket
export DMLC_ENABLE_RDMA=0
export DMLC_INTERFACE=lo
export DMLC_NODE_HOST=127.0.0.1
export DMLC_USE_GDR=0
```

启动 BytePS All-Reduce 服务：

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --tp-size 2 \
  --host 0.0.0.0 \
  --port 30000 \
  --log-level info \
  --dtype float16 \
  --disable-custom-all-reduce \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph \
  --enforce-disable-flashinfer-allreduce-fusion \
  --use-byteps-all-reduce \
  --byteps-all-reduce-debug
```

`--dtype float16` 是当前首测必需项。Qwen2.5 默认会使用 BF16，而当前 BytePS PyTorch op 不支持 `torch.cuda.BFloat16Tensor`，否则第一次 BytePS All-Reduce 会报：

```text
Tensor type torch.cuda.BFloat16Tensor is not supported.
```

期望日志：

- `BytePS initialized for SGLang: rank=0 local_rank=0 size=2 local_size=2`
- `BytePS initialized for SGLang: rank=1 local_rank=1 size=2 local_size=2`
- `Application startup complete.`
- `Uvicorn running on http://0.0.0.0:30000`
- `Declared BytePS tensor name=...`
- 如果日志级别允许 debug，还会看到 `Routing all_reduce through BytePS`

启动成功只表示 HTTP server 和两个 TP worker 已起来。第一次请求后，日志中出现 `Declared BytePS tensor name=...` 才说明模型主路径已经进入 BytePS All-Reduce。

如果启动时报 CUDA graph 或 piecewise CUDA graph 相关错误，确认命令里已经包含：

```text
--disable-cuda-graph
--disable-piecewise-cuda-graph
```

如果启动时报 FlashInfer/AITER all-reduce fusion 与 BytePS 不兼容，确认命令里包含：

```text
--enforce-disable-flashinfer-allreduce-fusion
```

并且不要添加：

```text
--enable-aiter-allreduce-fusion
```

`scheduler/server 节点注册异常（启动后无任何 declare 日志）`：
如果 SGLang 启动成功但首次请求后只看到 `BytePS initialized for SGLang` 而没有 `Declared BytePS tensor name=`，说明 BytePS worker 节点未能成功注册到 scheduler。开启节点注册调试日志排查：

```bash
export BYTEPS_DEBUG_NODE_REGISTRATION=1
export BYTEPS_LOG_LEVEL=INFO
```

设置后 scheduler 终端会打印每次 ADD_NODE 的 collected_nodes/expected_nodes 和完整 node 列表，ZMQ 连接建立时也会打印 node id/addr。常见原因是 scheduler 未先于 SGLang worker 启动、`DMLC_PS_ROOT_PORT` 端口不一致或 `BYTEPS_SOCKET_PATH` 目录未创建。

## 7. 请求验证

请求 BytePS 服务：

```bash
curl http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"temperature":0,"max_new_tokens":1}}'
```

通过标准：

- 请求能正常返回。
- 日志中出现 BytePS 初始化和 declare 信息。
- 首次请求至少应看到 `Declared BytePS tensor name=sglang.tp:0...`。
- 没有 hang。
- 没有 tensor name mismatch。
- 没有 group size mismatch。
- 没有 fallback 到 custom All-Reduce、PyNccl、MSCCLPP、TorchSymmMem 或 `torch.distributed` 的报错。

如果服务已经打印 `Application startup complete`，但 `curl /generate` 卡住不返回，说明启动成功但首次 BytePS collective 可能没有完成。先查看 SGLang、scheduler、server 三个终端是否都有后续日志，再检查两个 TP rank 是否都声明了同一个 BytePS tensor name。

## 8. NCCL baseline 对照

停止 30000 端口上的 BytePS 服务后，启动不带 BytePS 的 baseline。baseline 仍关闭 custom All-Reduce 和 CUDA graph，减少变量。

```bash
conda activate sgl-dev2
cd /workspace/Megatron-DPU/sglang-0.5.10.post1
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --tp-size 2 \
  --host 0.0.0.0 \
  --port 30001 \
  --log-level info \
  --dtype float16 \
  --disable-custom-all-reduce \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph \
  --enforce-disable-flashinfer-allreduce-fusion
```

请求 baseline：

```bash
curl http://127.0.0.1:30001/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"temperature":0,"max_new_tokens":1}}'
```

对比要点：

- BytePS 服务和 baseline 服务都能返回。
- `temperature=0` 下输出应基本一致，允许停止符或截断位置有小差异。
- BytePS 服务日志中能确认主路径 All-Reduce 走了 BytePS declare/push-pull。

## 9. BytePS 参数扫测

正确性跑通后再扫参数。每次改环境变量后重启 SGLang 服务。

```bash
export BYTEPS_PUSH_THREAD=1
```

```bash
export BYTEPS_PUSH_THREAD=2
```

```bash
export BYTEPS_PUSH_THREAD=4
```

```bash
export BYTEPS_PARTITION_BYTES=4096000
```

```bash
export BYTEPS_KEY_HASH_FN=raw
```

如果要切到 RDMA/UCX：

```bash
export DMLC_ENABLE_RDMA=1
```

RDMA/UCX 出问题时，先回退：

```bash
export DMLC_ENABLE_RDMA=0
```

## 10. 启动外部 BytePS scheduler/server

当前测试流程推荐使用这一模式。只单独启动 BytePS scheduler/server，SGLang 仍直接用 `python -m sglang.launch_server` 启动，不要用 `bpslaunch` 包裹 SGLang server。

终端 1，启动 scheduler：

```bash
cd /tmp
conda activate sgl-dev2
export DMLC_NUM_WORKER=1
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=127.0.0.1
export DMLC_PS_ROOT_PORT=9000
export BYTEPS_FORCE_DISTRIBUTED=1
export BYTEPS_KEY_HASH_FN=raw
export BYTEPS_PUSH_THREAD=1
export BYTEPS_LOG_LEVEL=INFO
export DMLC_ENABLE_RDMA=0
export DMLC_INTERFACE=lo
export DMLC_NODE_HOST=127.0.0.1
export DMLC_USE_GDR=0
export DMLC_ROLE=scheduler
bpslaunch
```

终端 2，启动 server：

```bash
cd /tmp
conda activate sgl-dev2
export DMLC_NUM_WORKER=1
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=127.0.0.1
export DMLC_PS_ROOT_PORT=9000
export BYTEPS_FORCE_DISTRIBUTED=1
export BYTEPS_KEY_HASH_FN=raw
export BYTEPS_PUSH_THREAD=1
export BYTEPS_LOG_LEVEL=INFO
export DMLC_ENABLE_RDMA=0
export DMLC_INTERFACE=lo
export DMLC_NODE_HOST=127.0.0.1
export DMLC_USE_GDR=0
export DMLC_ROLE=server
bpslaunch
```

终端 3，启动 SGLang worker。这里仍然不要设置 `BYTEPS_LOCAL_RANK` 和 `BYTEPS_LOCAL_SIZE`：

```bash
cd /workspace/Megatron-DPU/sglang-0.5.10.post1
conda activate sgl-dev2
mkdir -p /tmp/byteps_socket

unset BYTEPS_LOCAL_RANK
unset BYTEPS_LOCAL_SIZE
export DMLC_NUM_WORKER=1
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=127.0.0.1
export DMLC_PS_ROOT_PORT=9000
export BYTEPS_FORCE_DISTRIBUTED=1
export BYTEPS_KEY_HASH_FN=raw
export BYTEPS_PUSH_THREAD=1
export BYTEPS_LOG_LEVEL=INFO
export BYTEPS_SOCKET_PATH=/tmp/byteps_socket
export DMLC_ENABLE_RDMA=0
export DMLC_INTERFACE=lo
export DMLC_NODE_HOST=127.0.0.1
export DMLC_USE_GDR=0
export DMLC_ROLE=worker
export DMLC_WORKER_ID=0

CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --tp-size 2 \
  --host 0.0.0.0 \
  --port 30000 \
  --log-level info \
  --dtype float16 \
  --disable-custom-all-reduce \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph \
  --enforce-disable-flashinfer-allreduce-fusion \
  --use-byteps-all-reduce \
  --byteps-all-reduce-debug
```

## 11. 常见问题

`ModuleNotFoundError: No module named 'byteps'`：
进入 `/workspace/Megatron-DPU/byteps`，在 `sgl-dev2` 环境中重新执行 `python setup.py install`。

找不到 `--use-byteps-all-reduce`：
服务器安装的不是当前仓库里的 SGLang。检查 `python -c "import sglang; print(sglang.__file__)"`，并重新执行 `python -m pip install -e "python"`。

`Missing ./3rdparty/ps-lite`：
进入 `/workspace/Megatron-DPU/byteps`，执行 `git submodule update --init --recursive`。如果仍不行，按第 2 节兜底方式 clone `ps-lite`。

CUDA graph 报错：
这是预期保护。BytePS phase 1 必须禁用 CUDA graph 和 piecewise CUDA graph。

FlashInfer/AITER all-reduce fusion 报错：
BytePS phase 1 不支持 fused All-Reduce + RMSNorm。添加 `--enforce-disable-flashinfer-allreduce-fusion`，并确认没有使用 `--enable-aiter-allreduce-fusion`。

BytePS local rank/local size mismatch：
通常是外部手动设置了 `BYTEPS_LOCAL_RANK` 或 `BYTEPS_LOCAL_SIZE`。启动 SGLang 前执行：

```bash
unset BYTEPS_LOCAL_RANK
unset BYTEPS_LOCAL_SIZE
```

All-Reduce hang：
先用 `DMLC_ENABLE_RDMA=0` 和 `BYTEPS_PUSH_THREAD=1` 跑 TCP/本机路径。确认普通路径正常后，再测试 RDMA/UCX。

另外，开启 `--byteps-all-reduce-debug` 后 `byteps_allreduce_inplace()` 会打印每条 All-Reduce 的 `group_world_size`（SGLang TP group rank 数）和 `expected_workers`（BytePS server 等待的 worker 节点数）。如果 hang 时日志显示两个 TP rank 的 `expected_workers` 不一致，说明 declaration 前后不匹配，检查 BytePS scheduler/server 的 `DMLC_NUM_WORKER` 配置。

启动成功但首次 `curl /generate` 卡住：
如果日志已经出现 `Application startup complete` 和两个 TP rank 的 `BytePS initialized for SGLang`，说明 SGLang 服务启动成功。若第一次请求后停在 `Declared BytePS tensor name=...` 或 `tensor size=...`，说明请求进入了 BytePS All-Reduce，但 collective 没有完成。先确认两个 TP rank 都声明了同一个 tensor name，再看 BytePS scheduler/server 终端是否还有后续日志。

RDMA 连接失败：
先回退 `DMLC_ENABLE_RDMA=0`。如果 TCP 正常，再检查网卡、端口、防火墙、UCX/RDMA runtime，并确认 BytePS 是用 `BYTEPS_WITH_UCX=1` 编译安装的。

`Address already in use. errno = 98`：
这是 `DMLC_PS_ROOT_PORT` 被旧 BytePS scheduler/server 或其他进程占用。清理旧进程后再启动：

```bash
pkill -f "byteps.server" || true
pkill -f "bpslaunch" || true
pkill -f "sglang.launch_server" || true
pkill -f "sglang.srt" || true
rm -rf /tmp/byteps_socket
mkdir -p /tmp/byteps_socket
```

或者把 scheduler、server、SGLang worker 三边的 `DMLC_PS_ROOT_PORT` 一起改成未占用端口。

`Tensor type torch.cuda.BFloat16Tensor is not supported`：
当前 BytePS PyTorch op 不支持 BF16 tensor。首测启动 SGLang 时使用：

```bash
--dtype float16
```

`DMLC_NUM_SERVER` 相关错误：
当前 SGLang wrapper 会设置 `DMLC_ROLE`、`DMLC_NUM_WORKER`、`DMLC_WORKER_ID` 的默认值，但不会设置 `DMLC_NUM_SERVER`、`DMLC_PS_ROOT_URI` 和 `DMLC_PS_ROOT_PORT`。首测建议按第 10 节启动外部 scheduler/server，并在启动 SGLang 前显式设置：

```bash
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=127.0.0.1
export DMLC_PS_ROOT_PORT=9000
```

`basic_string::_M_construct null not valid`：
旧代码在 `byteps_init()` 中直接读取 `DMLC_USE_GDR`，如果未设置会崩溃。拉取包含 `byteps/byteps/common/operations.cc` 修复的代码并重新安装 BytePS；临时绕过是在 scheduler、server、worker 环境中都显式设置：

```bash
export DMLC_USE_GDR=0
```
