# SGLang BytePS All-Reduce Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in BytePS backend for SGLang model-path All-Reduce collectives, using the Megatron-DPU BytePS integration as the reference pattern.

**Architecture:** Keep SGLang's existing `GroupCoordinator` routing model and add BytePS as a guarded backend behind a new server flag. Initialize BytePS inside each SGLang model worker, set required BytePS local-rank environment from SGLang rank metadata, and route eligible GPU All-Reduce calls through BytePS `push_pull_async_inplace` with SUM semantics.

**Tech Stack:** SGLang Python runtime, PyTorch distributed, BytePS PyTorch ops, existing DPU BytePS fork in `byteps/`.

---

## Confirmed Phase 1 Scope

These choices are confirmed for the first implementation phase:

1. Replacement scope: only model compute-path All-Reduce through `GroupCoordinator.all_reduce()` and direct model-layer wrappers. Leave scheduler/cache/speculative control-plane `torch.distributed.all_reduce()` unchanged.
2. First test topology: single-node multi-GPU.
3. CLI name: `--use-byteps-all-reduce`.
4. CUDA graph support: BytePS does not need graph-capture support in phase 1. If graph capture is active or fails eligibility checks, fall back to the normal SGLang execution path.

Open implementation choice:

1. Launch model: let SGLang keep spawning TP workers internally and set BytePS env inside each worker, or run SGLang workers through `bpslaunch`. The safer first plan below uses SGLang internal workers.

## Files

- Create: `sglang/python/sglang/srt/distributed/byteps_collectives.py`
- Modify: `sglang/python/sglang/srt/server_args.py`
- Modify: `sglang/python/sglang/srt/model_executor/model_runner.py`
- Modify: `sglang/python/sglang/srt/distributed/parallel_state.py`
- Modify: `sglang/python/sglang/srt/distributed/communication_op.py`
- Modify: `sglang/python/sglang/srt/layers/linear.py`
- Modify: `sglang/python/sglang/srt/layers/vocab_parallel_embedding.py`
- Test: `sglang/test/manual/test_byteps_allreduce.py`
- Docs: `docs/SGLang-BytePS-AllReduce测试文档.md`

## Task 1: Add Server Flag And Runtime State

**Files:**
- Modify: `sglang/python/sglang/srt/server_args.py`
- Modify: `sglang/python/sglang/srt/model_executor/model_runner.py`

- [ ] Add dataclass fields near existing all-reduce flags:

```python
use_byteps_all_reduce: bool = False
byteps_all_reduce_debug: bool = False
```

- [ ] Add CLI args near `--disable-custom-all-reduce`:

```python
parser.add_argument(
    "--use-byteps-all-reduce",
    action="store_true",
    help="Use BytePS push/pull for eligible model-path All-Reduce collectives.",
)
parser.add_argument(
    "--byteps-all-reduce-debug",
    action="store_true",
    help="Log BytePS All-Reduce routing decisions.",
)
```

- [ ] In `ModelRunner.init_torch_distributed()`, before initializing model parallel groups, set BytePS local env from SGLang worker rank metadata when the flag is enabled:

```python
if self.server_args.use_byteps_all_reduce:
    os.environ.setdefault("BYTEPS_LOCAL_RANK", str(self.gpu_id))
    os.environ.setdefault("BYTEPS_LOCAL_SIZE", str(self.tp_size * self.pp_size))
```

- [ ] After `initialize_model_parallel(...)`, initialize BytePS from a new helper:

```python
if self.server_args.use_byteps_all_reduce:
    from sglang.srt.distributed.byteps_collectives import initialize_byteps_for_sglang

    initialize_byteps_for_sglang(
        local_rank=self.gpu_id,
        local_size=self.tp_size * self.pp_size,
        debug=self.server_args.byteps_all_reduce_debug,
    )
```

- [ ] If `pre_warm_nccl` is enabled, keep the existing NCCL warmup only for NCCL fallback. For BytePS testing, launch with `--disable-custom-all-reduce --disable-cuda-graph` and leave `--pre-warm-nccl` unset unless comparing fallback behavior.

## Task 2: Add BytePS Collective Wrapper

**Files:**
- Create: `sglang/python/sglang/srt/distributed/byteps_collectives.py`

- [ ] Implement an idempotent wrapper modeled after `Megatron-LM/megatron/core/distributed/byteps_collectives.py`:

```python
import hashlib
import logging
import os
from typing import Dict

import torch

logger = logging.getLogger(__name__)

_BPS_INITIALIZED = False
_BPS_DEBUG = False
_DECLARED_BPS_GROUPS: Dict[str, int] = {}


def initialize_byteps_for_sglang(local_rank: int, local_size: int, debug: bool = False) -> None:
    global _BPS_INITIALIZED, _BPS_DEBUG
    _BPS_DEBUG = debug
    os.environ.setdefault("BYTEPS_LOCAL_RANK", str(local_rank))
    os.environ.setdefault("BYTEPS_LOCAL_SIZE", str(local_size))
    os.environ.setdefault("DMLC_WORKER_ID", os.environ.get("DMLC_WORKER_ID", "0"))
    import byteps.torch as bps

    if not _BPS_INITIALIZED:
        bps.init()
        _BPS_INITIALIZED = True
    if _BPS_DEBUG:
        logger.info(
            "BytePS initialized for SGLang: rank=%s local_rank=%s size=%s local_size=%s",
            bps.rank(),
            bps.local_rank(),
            bps.size(),
            bps.local_size(),
        )


def _group_fingerprint(ranks) -> str:
    text = ",".join(str(rank) for rank in ranks)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def build_byteps_group_name(group, logical_name: str) -> str:
    if not logical_name:
        raise ValueError("logical_name must be non-empty")
    return f"sglang.{group.unique_name}.r{_group_fingerprint(group.ranks)}.{logical_name}"


def declare_and_cache_byteps_group(name: str, expected_workers: int) -> None:
    import byteps.torch as bps

    cached_workers = _DECLARED_BPS_GROUPS.get(name)
    if cached_workers is None:
        bps.declare(name, expected_workers=expected_workers)
        _DECLARED_BPS_GROUPS[name] = expected_workers
        return
    if cached_workers != expected_workers:
        raise RuntimeError(
            f"BytePS tensor name {name} was declared with inconsistent group size: "
            f"{cached_workers} vs {expected_workers}"
        )


def byteps_allreduce_inplace(tensor: torch.Tensor, group, logical_name: str) -> torch.Tensor:
    from byteps.torch import ops as bps_ops

    name = build_byteps_group_name(group, logical_name)
    declare_and_cache_byteps_group(name, group.world_size)
    handle = bps_ops.push_pull_async_inplace(
        tensor,
        average=False,
        name=name,
        version=0,
        priority=0,
    )
    bps_ops.synchronize(handle)
    return tensor
```

- [ ] Keep SUM semantics with `average=False`; SGLang TP/attention/MoE All-Reduce paths sum partial results.

## Task 3: Route Eligible `GroupCoordinator.all_reduce`

**Files:**
- Modify: `sglang/python/sglang/srt/distributed/parallel_state.py`

- [ ] Add global enable state:

```python
_ENABLE_BYTEPS_ALL_REDUCE = False
_ENABLE_BYTEPS_ALL_REDUCE_DEBUG = False


def set_byteps_all_reduce(enable: bool, debug: bool = False):
    global _ENABLE_BYTEPS_ALL_REDUCE, _ENABLE_BYTEPS_ALL_REDUCE_DEBUG
    _ENABLE_BYTEPS_ALL_REDUCE = enable
    _ENABLE_BYTEPS_ALL_REDUCE_DEBUG = debug
```

- [ ] Update imports in `model_runner.py` and call:

```python
set_byteps_all_reduce(
    self.server_args.use_byteps_all_reduce,
    self.server_args.byteps_all_reduce_debug,
)
```

- [ ] Add an optional `logical_name` parameter:

```python
def all_reduce(self, input_: torch.Tensor, logical_name: Optional[str] = None) -> torch.Tensor:
```

- [ ] After CPU/HPU/XPU/NPU checks and before custom/PyNccl/TorchSymmMem routing, use BytePS only when it is safe:

```python
if (
    _ENABLE_BYTEPS_ALL_REDUCE
    and input_.is_cuda
    and input_.is_contiguous()
    and not is_in_piecewise_cuda_graph()
    and not self.is_symmetric_memory_enabled()
):
    from sglang.srt.distributed.byteps_collectives import byteps_allreduce_inplace

    name = logical_name or f"generic.{input_.dtype}.{tuple(input_.shape)}"
    return byteps_allreduce_inplace(input_, self, name)
```

- [ ] If the tensor is non-contiguous, CPU, or inside graph capture, fall back to current SGLang behavior. Do not make BytePS the fallback for `quant_all_reduce()` in this phase.

## Task 4: Add Stable Names To Main Model Call Sites

**Files:**
- Modify: `sglang/python/sglang/srt/distributed/communication_op.py`
- Modify: `sglang/python/sglang/srt/layers/linear.py`
- Modify: `sglang/python/sglang/srt/layers/vocab_parallel_embedding.py`

- [ ] Let communication wrappers pass optional names:

```python
def tensor_model_parallel_all_reduce(input_: torch.Tensor, logical_name: Optional[str] = None) -> torch.Tensor:
    return get_tp_group().all_reduce(input_, logical_name=logical_name)


def attention_tensor_model_parallel_all_reduce(input_: torch.Tensor, logical_name: Optional[str] = None) -> torch.Tensor:
    return get_attn_tp_group().all_reduce(input_, logical_name=logical_name)
```

- [ ] Store stable module names:

```python
_BYTEPS_ROW_PARALLEL_NAME_COUNTER = count()

self.byteps_all_reduce_name = (
    f"row_parallel_linear.{prefix}"
    if prefix
    else f"row_parallel_linear.unnamed_{next(_BYTEPS_ROW_PARALLEL_NAME_COUNTER)}"
)
```

and:

```python
_BYTEPS_VOCAB_PARALLEL_NAME_COUNTER = count()

self.byteps_all_reduce_name = (
    f"vocab_parallel_embedding.{prefix}"
    if prefix
    else f"vocab_parallel_embedding.unnamed_{next(_BYTEPS_VOCAB_PARALLEL_NAME_COUNTER)}"
)
```

The fallback counters must be module-level counters and must advance in deterministic module construction order on every rank. Do not use `id(self)`, because object addresses differ by process and would produce different BytePS tensor names.

- [ ] In `RowParallelLinear.forward()`, pass the name:

```python
output = tensor_model_parallel_all_reduce(
    output_parallel,
    logical_name=self.byteps_all_reduce_name,
)
```

- [ ] In `VocabParallelEmbedding.forward()`, pass the name to TP or attention-TP reduce:

```python
output_parallel = tensor_model_parallel_all_reduce(
    output_parallel,
    logical_name=self.byteps_all_reduce_name,
)
```

- [ ] Keep `quant_all_reduce` unchanged initially. For tests, do not enable `--enable-quant-communications`.

## Task 5: Cover DP-Attention And MoE All-Reduce Paths

**Files:**
- Modify: `sglang/python/sglang/srt/layers/dp_attention.py`
- Modify: `sglang/python/sglang/srt/layers/communicator.py`
- Review: `sglang/python/sglang/srt/layers/moe/**`

- [ ] For DP-Attention `SUM_LEN`, pass a stable logical name when using `tensor_model_parallel_all_reduce(global_tokens)`.
- [ ] For `LayerCommunicator` attention group All-Reduce calls, pass names such as:

```python
"layer_communicator.attn_tp_gather_hidden_states"
"layer_communicator.tp_all_reduce_scattered_residual"
```

- [ ] For MoE post-expert combine paths that call `moe_tensor_model_parallel_all_reduce()` or `moe_expert_parallel_all_reduce()`, leave them on the central generic path first, then add explicit names after functional tests identify active call sites for the target model.

## Task 6: Add Manual BytePS Correctness Test

**Files:**
- Create: `sglang/test/manual/test_byteps_allreduce.py`

- [ ] Add a small script that initializes PyTorch distributed, initializes SGLang model parallel groups, initializes BytePS, and compares BytePS SUM against `torch.distributed.all_reduce` for float32 and float16 tensors.
- [ ] Run this only under a BytePS scheduler/server/worker environment. It is a manual integration test, not a normal unit test.
- [ ] Expected checks:

```python
assert torch.allclose(byteps_result.float(), torch_result.float(), rtol=1e-3, atol=1e-3)
```

## Task 7: End-To-End Serving Test

**Files:**
- Docs: `docs/SGLang-BytePS-AllReduce测试文档.md`

- [ ] Install the updated SGLang package:

```bash
cd /Users/zhijingxin/Megatron-DPU/sglang
pip install -e "python"
```

- [ ] Build/install BytePS from the local fork:

```bash
cd /Users/zhijingxin/Megatron-DPU/byteps
BYTEPS_WITH_UCX=1 python setup.py install
```

- [ ] Start BytePS scheduler and server with matching DMLC variables.
- [ ] Start SGLang with:

```bash
python -m sglang.srt.entrypoints.http_server \
  --model-path <model-path> \
  --tp-size 2 \
  --disable-custom-all-reduce \
  --disable-cuda-graph \
  --use-byteps-all-reduce \
  --byteps-all-reduce-debug
```

- [ ] Compare deterministic outputs against baseline SGLang without `--use-byteps-all-reduce`.

## Task 8: Performance And Stability Pass

**Files:**
- Modify as needed after correctness is established.

- [ ] Measure TTFT, decode throughput, and BytePS `get_pushpull_speed`.
- [ ] Sweep:

```bash
BYTEPS_PUSH_THREAD=1
BYTEPS_PUSH_THREAD=2
BYTEPS_PUSH_THREAD=4
BYTEPS_PARTITION_BYTES=4096000
BYTEPS_KEY_HASH_FN=raw
```

- [ ] Re-enable CUDA graph only after proving BytePS routing falls back correctly inside graph capture.
- [ ] Decide whether direct control-plane `torch.distributed.all_reduce()` calls should remain on PyTorch distributed. The current recommendation is to leave them unchanged unless profiling shows they matter.

## Known Risks

- BytePS requires stable DMLC and local-rank environment. SGLang's internal worker launcher does not naturally provide `BYTEPS_LOCAL_RANK`, so the implementation must set it before `bps.init()`.
- Generic names based only on dtype/shape can reuse a BytePS key for multiple same-shape collectives. That is acceptable only with synchronous, ordered calls. Explicit module-level names are preferred for hot paths.
- BytePS is not assumed to be CUDA graph capture safe in this plan. The first implementation should fallback during graph capture and tests should use `--disable-cuda-graph`.
- Long-running servers can create many BytePS declarations if dynamic shapes are included in names. Use explicit logical names for stable model layers and add metrics/logs to monitor declaration count.
