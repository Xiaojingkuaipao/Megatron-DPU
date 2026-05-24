# SGLang BytePS All-Reduce 测试文档

本文档按服务器上的实际部署方式编写：**SGLang 和 Megatron-DPU 是两个独立仓库**，SGLang 从 SGLang 仓库安装，BytePS 从 Megatron-DPU 仓库安装；Python 环境使用已有 conda 环境 `sgl-dev2`。

命令里的 `<...>` 按服务器实际情况填写。

## 1. 测试目标

验证 SGLang 新增的 `--use-byteps-all-reduce` 能把模型主路径 All-Reduce 路由到 Megatron-DPU 仓库里的 BytePS。

当前第一阶段约束：

- 只替换模型计算主路径 `GroupCoordinator.all_reduce()`。
- 不替换 scheduler/cache/speculative/disaggregation 等控制面直接 `torch.distributed.all_reduce()`。
- 不替换 reduce-scatter、all-gather、broadcast、send/recv。
- 不支持 `quant_all_reduce()`；触发后会显式报错。
- 不支持 CUDA graph / piecewise CUDA graph；启动 BytePS 测试时必须加 `--disable-cuda-graph --disable-piecewise-cuda-graph`。
- 首测只建议单机多 GPU，先用 `--tp-size 2`。
- 不要用 `bpslaunch` 包 SGLang server，SGLang 自己会启动 model worker。

## 2. 仓库准备

设置两个仓库路径：

```bash
export SGLANG_REPO=/path/to/sglang
export MEGATRON_DPU_REPO=/path/to/Megatron-DPU
export SGLANG_BRANCH=<sglang-branch-or-commit>
export MEGATRON_DPU_BRANCH=<megatron-dpu-branch-or-commit>
```

如果服务器上还没有仓库：

```bash
git clone <sglang-git-url> "$SGLANG_REPO"
git clone <Megatron-DPU-git-url> "$MEGATRON_DPU_REPO"
```

更新 SGLang 仓库：

```bash
cd "$SGLANG_REPO"
git fetch --all --prune
git checkout "$SGLANG_BRANCH"
git pull --ff-only
git submodule update --init --recursive
```

更新 Megatron-DPU 仓库：

```bash
cd "$MEGATRON_DPU_REPO"
git fetch --all --prune
git checkout "$MEGATRON_DPU_BRANCH"
git pull --ff-only
git submodule update --init --recursive
```

如果 checkout 的是具体 commit hash，不需要执行 `git pull --ff-only`，确认 `git rev-parse HEAD` 是目标提交即可。

确认 SGLang 仓库里包含 BytePS All-Reduce 代码：

```bash
cd "$SGLANG_REPO"
test -f python/sglang/srt/distributed/byteps_collectives.py
grep -R "use-byteps-all-reduce" -n python/sglang/srt/server_args.py
```

确认 Megatron-DPU 仓库里有 BytePS：

```bash
cd "$MEGATRON_DPU_REPO"
test -f byteps/setup.py
```

## 3. 安装

进入已有 conda 环境：

```bash
conda activate sgl-dev2
python -m pip install --upgrade pip setuptools wheel
```

安装 SGLang：

```bash
cd "$SGLANG_REPO"
python -m pip install -e "python"
```

安装 Megatron-DPU 里的 BytePS。先用普通 TCP/本机路径测试：

```bash
cd "$MEGATRON_DPU_REPO/byteps"
python setup.py install
```

如果要测 RDMA/UCX，再重新编译安装：

```bash
cd "$MEGATRON_DPU_REPO/byteps"
BYTEPS_WITH_UCX=1 python setup.py install
```

如果服务器 NCCL 不在默认路径，安装 BytePS 前先设置：

```bash
export BYTEPS_NCCL_HOME=/path/to/nccl
```

## 4. 安装检查

基础导入检查：

```bash
conda activate sgl-dev2
python - <<'PY'
import torch
import sglang
import byteps.torch as bps
from sglang.srt.distributed.byteps_collectives import initialize_byteps_for_sglang

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("sglang import ok")
print("byteps import ok")
print("sglang byteps wrapper import ok")
PY
```

语法检查：

```bash
cd "$SGLANG_REPO"
env PYTHONPYCACHEPREFIX=/tmp/sglang_byteps_pycache python -m py_compile \
  python/sglang/srt/distributed/byteps_collectives.py \
  python/sglang/srt/server_args.py \
  python/sglang/srt/model_executor/model_runner.py \
  python/sglang/srt/distributed/parallel_state.py \
  python/sglang/srt/distributed/communication_op.py \
  python/sglang/srt/layers/linear.py \
  python/sglang/srt/layers/vocab_parallel_embedding.py
```

## 5. 启动 BytePS 版 SGLang

首测不需要单独启动 BytePS scheduler/server。当前 SGLang 代码会在 model worker 内部设置 BytePS local env：

```text
BYTEPS_LOCAL_RANK=<gpu_id>
BYTEPS_LOCAL_SIZE=<tp_size * pp_size>
DMLC_ROLE=worker
DMLC_WORKER_ID=0
DMLC_NUM_WORKER=1
DMLC_NUM_SERVER=0
```

推荐首测环境变量：

```bash
conda activate sgl-dev2
export BYTEPS_KEY_HASH_FN=raw
export BYTEPS_PUSH_THREAD=1
export BYTEPS_LOG_LEVEL=INFO
export DMLC_ENABLE_RDMA=0
```

启动服务：

```bash
cd "$SGLANG_REPO"
python -m sglang.srt.entrypoints.http_server \
  --model-path <model-path> \
  --tp-size 2 \
  --host 127.0.0.1 \
  --port 30000 \
  --disable-custom-all-reduce \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph \
  --use-byteps-all-reduce \
  --byteps-all-reduce-debug
```

期望日志里能看到：

- `BytePS initialized for SGLang`
- `Routing All-Reduce to BytePS`
- `BytePS All-Reduce completed`

## 6. 请求验证

请求 BytePS 服务：

```bash
curl http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"temperature":0,"max_new_tokens":16}}'
```

通过标准：

- 请求成功返回。
- 日志中出现 BytePS All-Reduce 路由信息。
- 无 hang、shape mismatch、tensor name mismatch、group size mismatch。

## 7. NCCL baseline 对照

另起一个不用 BytePS 的 baseline 服务：

```bash
cd "$SGLANG_REPO"
python -m sglang.srt.entrypoints.http_server \
  --model-path <model-path> \
  --tp-size 2 \
  --host 127.0.0.1 \
  --port 30001 \
  --disable-custom-all-reduce \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph
```

请求 baseline：

```bash
curl http://127.0.0.1:30001/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"temperature":0,"max_new_tokens":16}}'
```

对比：

- BytePS 和 baseline 都能正常返回。
- `temperature=0` 下输出 token 基本一致，或只存在停止符差异。
- BytePS 日志确认主路径 All-Reduce 没有走 NCCL/custom fallback。

## 8. 常见问题

`ModuleNotFoundError: No module named 'byteps'`：
只安装 SGLang 不会安装 BytePS。进入 `$MEGATRON_DPU_REPO/byteps` 执行 `python setup.py install`。

找不到 `--use-byteps-all-reduce`：
当前安装的不是包含 BytePS 修改的 SGLang 仓库或分支。检查 `$SGLANG_REPO/python/sglang/srt/server_args.py`。

CUDA graph 报错：
这是预期行为。BytePS phase 1 必须加 `--disable-cuda-graph --disable-piecewise-cuda-graph`。

`quant_all_reduce()` 报错：
这是预期行为。BytePS phase 1 不支持量化通信路径，首测关闭量化通信相关参数。

BytePS local rank/local size mismatch：
检查 `--tp-size`、`CUDA_VISIBLE_DEVICES`、实际 GPU 数，以及是否手动设置了 `BYTEPS_LOCAL_RANK` 或 `BYTEPS_LOCAL_SIZE` 覆盖了 SGLang 内部设置。

All-Reduce hang：
先保持 `DMLC_ENABLE_RDMA=0` 用 TCP/本机路径测试；确认普通 TP 路径正常后，再考虑 RDMA 和外部 scheduler/server。

RDMA 连接失败：
先回退 `DMLC_ENABLE_RDMA=0`。如果 TCP 正常，再检查 `DMLC_INTERFACE`、网卡、端口、防火墙、UCX/RDMA runtime，并确认 BytePS 是用 `BYTEPS_WITH_UCX=1` 编译的。

## 9. 可选：外部 BytePS scheduler/server

首测不建议使用这一模式。只有需要验证 BytePS 分布式 scheduler/server 路径时再用。

公共环境：

```bash
conda activate sgl-dev2
export DMLC_NUM_WORKER=1
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=<scheduler_ip>
export DMLC_PS_ROOT_PORT=9000
export BYTEPS_FORCE_DISTRIBUTED=1
export BYTEPS_KEY_HASH_FN=raw
export BYTEPS_PUSH_THREAD=1
export BYTEPS_LOG_LEVEL=INFO
export DMLC_ENABLE_RDMA=0
```

终端 1：

```bash
conda activate sgl-dev2
# 先执行上面的公共环境 export
export DMLC_ROLE=scheduler
bpslaunch
```

终端 2：

```bash
conda activate sgl-dev2
# 先执行上面的公共环境 export
export DMLC_ROLE=server
bpslaunch
```

终端 3 启动 SGLang，不要用 `bpslaunch` 包裹：

```bash
conda activate sgl-dev2
# 先执行上面的公共环境 export
export DMLC_ROLE=worker
export DMLC_WORKER_ID=0

cd "$SGLANG_REPO"
python -m sglang.srt.entrypoints.http_server \
  --model-path <model-path> \
  --tp-size 2 \
  --host 127.0.0.1 \
  --port 30000 \
  --disable-custom-all-reduce \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph \
  --use-byteps-all-reduce \
  --byteps-all-reduce-debug
```
