# SGLang BytePS All-Reduce 测试文档

本文档按当前二合一仓库编写。服务器上只需要一份 `Megatron-DPU` 仓库，BytePS 从 `byteps/` 安装，SGLang 从 `sglang-0.5.10.post1/python/` 安装。

下面命令统一假设服务器代码目录是：

```text
/home/xzj/Megatron-DPU
```

如果服务器实际目录不同，直接把命令里的 `/home/xzj/Megatron-DPU` 替换成真实目录。本文不再使用 `SGLANG_REPO`、`MEGATRON_DPU_REPO` 这类路径变量。

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

## 2. 提交并更新服务器代码

本地提交并推送：

```bash
cd /Users/zhijingxin/Megatron-DPU
git status --short
git add docs/SGLang-BytePS-AllReduce实现说明.md \
  docs/SGLang-BytePS-AllReduce测试文档.md \
  sglang-0.5.10.post1/python/sglang/srt/distributed/byteps_collectives.py \
  sglang-0.5.10.post1/python/sglang/srt/distributed/communication_op.py \
  sglang-0.5.10.post1/python/sglang/srt/distributed/parallel_state.py \
  sglang-0.5.10.post1/python/sglang/srt/layers/linear.py \
  sglang-0.5.10.post1/python/sglang/srt/layers/vocab_parallel_embedding.py \
  sglang-0.5.10.post1/python/sglang/srt/model_executor/model_runner.py \
  sglang-0.5.10.post1/python/sglang/srt/server_args.py
git commit -m "Add SGLang BytePS all-reduce path"
git push
```

服务器如果还没有仓库：

```bash
git clone <Megatron-DPU-git-url> /home/xzj/Megatron-DPU
cd /home/xzj/Megatron-DPU
git checkout <包含本次修改的分支>
```

服务器如果已经有仓库：

```bash
cd /home/xzj/Megatron-DPU
git fetch --all --prune
git checkout <包含本次修改的分支>
git pull --ff-only
```

确认服务器代码包含本次修改：

```bash
cd /home/xzj/Megatron-DPU
test -f /home/xzj/Megatron-DPU/sglang-0.5.10.post1/python/sglang/srt/distributed/byteps_collectives.py
grep -R "use-byteps-all-reduce" -n /home/xzj/Megatron-DPU/sglang-0.5.10.post1/python/sglang/srt/server_args.py
grep -R "set_byteps_all_reduce" -n /home/xzj/Megatron-DPU/sglang-0.5.10.post1/python/sglang/srt/model_executor/model_runner.py
```

BytePS 依赖 `3rdparty/ps-lite`。如果服务器上这个目录不完整，先补齐：

```bash
cd /home/xzj/Megatron-DPU/byteps
git submodule update --init --recursive
test -d /home/xzj/Megatron-DPU/byteps/3rdparty/ps-lite/src
```

如果上面的 submodule 命令不能识别 `ps-lite`，用下面的兜底方式：

```bash
rm -rf /home/xzj/Megatron-DPU/byteps/3rdparty/ps-lite
git clone -b byteps https://github.com/bytedance/ps-lite /home/xzj/Megatron-DPU/byteps/3rdparty/ps-lite
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
rm -rf /home/xzj/Megatron-DPU/byteps/build
rm -rf /home/xzj/Megatron-DPU/byteps/dist
rm -rf /home/xzj/Megatron-DPU/byteps/byteps.egg-info
```

安装普通 TCP/本机测试版。首测建议先用这个，不要一开始就引入 RDMA/UCX：

```bash
conda activate sgl-dev2
cd /home/xzj/Megatron-DPU/byteps
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
rm -rf /home/xzj/Megatron-DPU/byteps/build
rm -rf /home/xzj/Megatron-DPU/byteps/dist
rm -rf /home/xzj/Megatron-DPU/byteps/byteps.egg-info
cd /home/xzj/Megatron-DPU/byteps
BYTEPS_WITH_UCX=1 \
BYTEPS_WITHOUT_TENSORFLOW=1 \
BYTEPS_WITHOUT_MXNET=1 \
BYTEPS_WITH_PYTORCH=1 \
python setup.py install
```

BytePS 安装检查：

```bash
conda activate sgl-dev2
python - <<'PY'
import byteps.torch as bps
from byteps.torch import ops as bps_ops
print("byteps import ok")
print("byteps module:", bps.__file__)
print("byteps ops import ok")
PY
```

可选做一个单进程 BytePS init smoke test。这里用 `timeout` 防止环境变量不匹配时一直卡住：

```bash
conda activate sgl-dev2
DMLC_ROLE=worker \
DMLC_NUM_WORKER=1 \
DMLC_NUM_SERVER=0 \
DMLC_WORKER_ID=0 \
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

如果这个 smoke test 因 `DMLC_NUM_SERVER`、scheduler 或 server 相关环境报错，不要用 `bpslaunch` 包裹 SGLang server，先继续按后文 SGLang 启动方式测试；必要时再使用“外部 BytePS scheduler/server 兜底模式”。

## 5. 安装 SGLang

从同一个二合一仓库安装 SGLang：

```bash
conda activate sgl-dev2
cd /home/xzj/Megatron-DPU/sglang-0.5.10.post1
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
  /home/xzj/Megatron-DPU/sglang-0.5.10.post1/python/sglang/srt/distributed/byteps_collectives.py \
  /home/xzj/Megatron-DPU/sglang-0.5.10.post1/python/sglang/srt/distributed/parallel_state.py \
  /home/xzj/Megatron-DPU/sglang-0.5.10.post1/python/sglang/srt/distributed/communication_op.py \
  /home/xzj/Megatron-DPU/sglang-0.5.10.post1/python/sglang/srt/server_args.py \
  /home/xzj/Megatron-DPU/sglang-0.5.10.post1/python/sglang/srt/model_executor/model_runner.py \
  /home/xzj/Megatron-DPU/sglang-0.5.10.post1/python/sglang/srt/layers/linear.py \
  /home/xzj/Megatron-DPU/sglang-0.5.10.post1/python/sglang/srt/layers/vocab_parallel_embedding.py
```

## 6. 启动 BytePS 版 SGLang

首测使用 2 张 GPU。下面示例模型路径是 `/data/models/Qwen2.5-0.5B-Instruct`，测试时替换成服务器真实模型目录。

启动前先设置环境。不要手动设置 `BYTEPS_LOCAL_RANK` 和 `BYTEPS_LOCAL_SIZE`，这两个值由 SGLang model worker 内部按 `gpu_id` 和 `tp_size * pp_size` 设置。

```bash
conda activate sgl-dev2
unset BYTEPS_LOCAL_RANK
unset BYTEPS_LOCAL_SIZE
export DMLC_ROLE=worker
export DMLC_NUM_WORKER=1
export DMLC_NUM_SERVER=0
export DMLC_WORKER_ID=0
export BYTEPS_KEY_HASH_FN=raw
export BYTEPS_PUSH_THREAD=1
export BYTEPS_LOG_LEVEL=INFO
export DMLC_ENABLE_RDMA=0
```

启动 BytePS All-Reduce 服务：

```bash
conda activate sgl-dev2
cd /home/xzj/Megatron-DPU/sglang-0.5.10.post1
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.srt.entrypoints.http_server \
  --model-path /data/models/Qwen2.5-0.5B-Instruct \
  --tp-size 2 \
  --host 127.0.0.1 \
  --port 30000 \
  --disable-custom-all-reduce \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph \
  --enforce-disable-flashinfer-allreduce-fusion \
  --use-byteps-all-reduce \
  --byteps-all-reduce-debug
```

期望日志：

- `BytePS initialized for SGLang`
- `Declared BytePS tensor name=...`
- 如果日志级别允许 debug，还会看到 `Routing all_reduce through BytePS`

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

## 7. 请求验证

请求 BytePS 服务：

```bash
curl http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"temperature":0,"max_new_tokens":16}}'
```

通过标准：

- 请求能正常返回。
- 日志中出现 BytePS 初始化和 declare 信息。
- 没有 hang。
- 没有 tensor name mismatch。
- 没有 group size mismatch。
- 没有 fallback 到 custom All-Reduce、PyNccl、MSCCLPP、TorchSymmMem 或 `torch.distributed` 的报错。

## 8. NCCL baseline 对照

停止 30000 端口上的 BytePS 服务后，启动不带 BytePS 的 baseline。baseline 仍关闭 custom All-Reduce 和 CUDA graph，减少变量。

```bash
conda activate sgl-dev2
cd /home/xzj/Megatron-DPU/sglang-0.5.10.post1
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.srt.entrypoints.http_server \
  --model-path /data/models/Qwen2.5-0.5B-Instruct \
  --tp-size 2 \
  --host 127.0.0.1 \
  --port 30001 \
  --disable-custom-all-reduce \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph \
  --enforce-disable-flashinfer-allreduce-fusion
```

请求 baseline：

```bash
curl http://127.0.0.1:30001/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"temperature":0,"max_new_tokens":16}}'
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

## 10. 外部 BytePS scheduler/server 兜底模式

首测不建议使用这一模式。只有当 `DMLC_NUM_SERVER=0` 的本机模式在当前 BytePS 构建中不可用，或需要验证 BytePS scheduler/server 链路时再用。

注意：即使使用外部 scheduler/server，也不要用 `bpslaunch` 包裹 SGLang server。只单独启动 BytePS scheduler/server，SGLang 仍直接用 `python -m sglang...` 启动。

终端 1，启动 scheduler：

```bash
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
export DMLC_ROLE=scheduler
bpslaunch
```

终端 2，启动 server：

```bash
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
export DMLC_ROLE=server
bpslaunch
```

终端 3，启动 SGLang worker。这里仍然不要设置 `BYTEPS_LOCAL_RANK` 和 `BYTEPS_LOCAL_SIZE`：

```bash
conda activate sgl-dev2
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
export DMLC_ENABLE_RDMA=0
export DMLC_ROLE=worker
export DMLC_WORKER_ID=0

cd /home/xzj/Megatron-DPU/sglang-0.5.10.post1
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.srt.entrypoints.http_server \
  --model-path /data/models/Qwen2.5-0.5B-Instruct \
  --tp-size 2 \
  --host 127.0.0.1 \
  --port 30000 \
  --disable-custom-all-reduce \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph \
  --enforce-disable-flashinfer-allreduce-fusion \
  --use-byteps-all-reduce \
  --byteps-all-reduce-debug
```

## 11. 常见问题

`ModuleNotFoundError: No module named 'byteps'`：
进入 `/home/xzj/Megatron-DPU/byteps`，在 `sgl-dev2` 环境中重新执行 `python setup.py install`。

找不到 `--use-byteps-all-reduce`：
服务器安装的不是当前仓库里的 SGLang。检查 `python -c "import sglang; print(sglang.__file__)"`，并重新执行 `python -m pip install -e "python"`。

`Missing ./3rdparty/ps-lite`：
进入 `/home/xzj/Megatron-DPU/byteps`，执行 `git submodule update --init --recursive`。如果仍不行，按第 2 节兜底方式 clone `ps-lite`。

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

RDMA 连接失败：
先回退 `DMLC_ENABLE_RDMA=0`。如果 TCP 正常，再检查网卡、端口、防火墙、UCX/RDMA runtime，并确认 BytePS 是用 `BYTEPS_WITH_UCX=1` 编译安装的。

`DMLC_NUM_SERVER` 相关错误：
当前 SGLang wrapper 会设置 `DMLC_ROLE`、`DMLC_NUM_WORKER`、`DMLC_WORKER_ID` 的默认值，但不会设置 `DMLC_NUM_SERVER`。首测建议在启动 SGLang 前显式设置：

```bash
export DMLC_NUM_SERVER=0
```

如果当前 BytePS 构建不接受 `DMLC_NUM_SERVER=0`，使用第 10 节的外部 scheduler/server 兜底模式。
