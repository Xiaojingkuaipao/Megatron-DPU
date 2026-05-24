# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Repository Overview

This is a multi-component monorepo for distributed LLM training and serving, combining three subsystems:

- **Megatron-LM/** — NVIDIA's Megatron Core + Megatron-LM for distributed transformer training at scale. GPU-optimized with TP/PP/DP/CP/EP parallelism strategies.
- **byteps/** — BytePS parameter server framework (ByteDance). Provides high-performance all-reduce via push/pull semantics over TCP or RDMA. Supports TensorFlow, PyTorch, MXNet, Keras.
- **sglang/** — SGLang high-performance LLM serving/inference engine. Provides RadixAttention, continuous batching, prefill-decode disaggregation, speculative decoding, and broad model support.

### Key Modification: DPU Communication Integration

This repo integrates BytePS as the communication backend for Megatron-LM, replacing NCCL all-reduce for DP and TP:

| Layer | What Changed |
|-------|-------------|
| **Megatron config** | `--use-dpu-reduce` / `--use-dpu-tp-reduce` flags in `arguments.py`, `distributed_data_parallel_config.py`, `model_parallel_config.py` |
| **Megatron init** | Unified `bps.init()` in `megatron/training/initialize.py` |
| **Megatron DP** | `param_and_grad_buffer.py`, `distributed_data_parallel.py` — NCCL all-reduce → BytePS push_pull |
| **Megatron TP** | `megatron/core/tensor_parallel/mappings.py`, `layers.py` — RowParallelLinear output all-reduce → BytePS push_pull |
| **Megatron BP wrapper** | `megatron/core/distributed/byteps_collectives.py` — group naming, declare caching, pre-declare mechanism |
| **BytePS RDMA** | Dual-channel (control + data plane separation) in `rdma_van.h`, `rdma_transport.h`, `rdma_utils.h` |
| **BytePS push/pull** | Multi-threaded via `BYTEPS_PUSH_THREAD` env var in `global.cc`, `operations.cc` |
| **BytePS hash** | Default strategy changed from `djb2` to `raw` |

## Build and Development

### Megatron-LM

```bash
# Install (uses uv for dependency management)
cd Megatron-LM
pip install -e ".[dev]"          # development install
uv sync                          # sync dependencies via uv

# Run tests
uv run --no-sync python -m torch.distributed.run \
    --nproc_per_node 8 --nnodes 1 \
    -m pytest -xvs megatron/core/tests/unit_tests/

# Run a single test file
pytest tests/unit_tests/test_basic.py

# Lint/format (pre-commit)
pre-commit run --files megatron/core/*.py
# Auto-formatting: black, isort (configured in .pre-commit-config.yaml)
```

### BytePS

```bash
# Build from source
cd byteps
python setup.py install

# With RDMA support
BYTEPS_WITH_UCX=1 python setup.py install

# Run tests
cd byteps/tests && python test_byteps.py

# Launch distributed training
bpslaunch python your_script.py
```

### SGLang

```bash
# Install
cd sglang
pip install -e "python"

# Launch server
python -m sglang.srt.entrypoints.http_server --model-path <model> --tp-size 2

# Run tests
python3 test/registered/core/test_srt_endpoint.py                    # single file
python3 test/registered/core/test_srt_endpoint.py TestSRTEndpoint.test_simple_decode  # single test
python3 test/run_suite.py --hw cuda --suite stage-a-test-1-gpu-small  # suite

# Lint/format
pre-commit run --all-files
# Uses: ruff (select=F401,F821), black, isort, codespell, clang-format (for CUDA/C++)
```

## Architecture

### Megatron-LM Structure

```
Megatron-LM/
├── megatron/
│   ├── core/              # Megatron Core library (pip package: megatron-core)
│   │   ├── distributed/   # DP/TP/FSDP configs, BytePS collectives
│   │   ├── tensor_parallel/  # TP layer implementations
│   │   ├── pipeline_parallel/
│   │   ├── fusions/       # Fused CUDA kernels
│   │   ├── models/        # Model definitions (GPT, BERT, T5, etc.)
│   │   ├── datasets/
│   │   └── inference/     # Inference utilities
│   ├── training/          # Training loop, args, checkpointing, tokenizer
│   ├── inference/         # Standalone inference
│   ├── rl/                # RL training integration
│   └── post_training/     # Quantization, export
├── pretrain_gpt.py        # Main GPT pretraining entry point
├── pretrain_vlm.py        # Vision-language model pretraining
├── train_rl.py            # RL training entry point
├── gpt_builders.py        # GPT model builder
└── examples/              # Model-specific training scripts
```

### BytePS Structure

```
byteps/
├── byteps/
│   ├── common/            # Core C++ implementation (global state, operations, comm)
│   ├── torch/             # PyTorch bindings (ops.py, csrc)
│   ├── tensorflow/        # TF bindings
│   ├── mxnet/             # MXNet bindings
│   ├── server/            # Parameter server binary
│   ├── keras/             # Keras bindings
│   └── misc/              # bpslaunch CLI
├── 3rdparty/ps-lite/      # PS-Lite dependency (RDMA/TCP comm layer)
├── launcher/              # Task launcher scripts
└── docker/                # Dockerfiles
```

### SGLang Structure

```
sglang/python/sglang/
├── srt/                   # Serving RunTime (main LLM engine)
│   ├── models/            # Model implementations (llama, qwen, deepseek, ...)
│   ├── layers/            # Custom layers (attention, MoE, quantization)
│   ├── model_executor/    # Model runner, forward batch
│   ├── managers/          # Scheduler, tokenizer manager
│   ├── distributed/       # Communication ops, group coordinator
│   ├── speculative/       # Eagle speculative decoding
│   ├── disaggregation/    # Prefill-decode disaggregation
│   ├── mem_cache/         # Memory pool, RadixAttention cache
│   ├── server_args.py     # All CLI/config parameters (large, 4000+ lines)
│   └── entrypoints/       # HTTP/gRPC server entry points
├── multimodal_gen/        # Diffusion/multimodal generation (separate from LLM)
├── jit_kernel/            # Lightweight JIT CUDA kernels
├── lang/                  # SGLang DSL (frontend programming model)
├── cli/                   # CLI tools
├── test/                  # Test suites
└── sgl-kernel/            # Heavyweight AOT CUDA/C++ kernels
```

### Key Data Flow: Megatron-LM DP Gradient Sync with BytePS

```
Megatron-LM DDP (distributed_data_parallel.py)
    → param_and_grad_buffer.py (grad copying)
    → byteps_collectives.py (declare + push_pull)
    → byteps torch ops (byteps/torch/ops.py)
    → BytePS C++ core (push → server aggregate → pull)
    → RDMA dual-channel (rdma_transport.h, rdma_van.h)
```

### Key Data Flow: SGLang Request Processing

```
HTTP request → Tokenizer → Scheduler (batch formation, KV cache)
    → TPWorker → ModelRunner → Model.forward()
    → Sampling → Response
```

Communication during forward pass goes through `GroupCoordinator` objects:
- `get_tp_group()` — TP all-reduce for RowParallelLinear, MLP, embeddings
- `get_attn_tp_group()` — DP-Attention communication
- `get_moe_ep_group()` — MoE expert parallelism

### Naming Conventions (SGLang Speculative Decoding)

When working with code under `sglang/python/sglang/srt/speculative/` or related attention backends, follow these rules (see `.Codex/rules/speculative-naming.md` for full details):

- Use verb form: `accept_tokens` not `accepted_tokens`
- Bonus token: always `bonus_token` / `bonus_tokens`
- `accept_*` includes bonus; `correct_*` excludes bonus
- `num_` for counts, `_ct` for counters, `_rate` for rates
- Avoid `length` / `lens` internally (exception: Triton kernel signatures)
- Drop redundant `_token_id` suffix in spec scope
- Plural for tensors, singular for scalars

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `BYTEPS_PUSH_THREAD` | BytePS push/pull concurrency level |
| `BYTEPS_WITH_UCX` | Enable UCX/RDMA transport in BytePS build |
| `NCCL_MAX_NCHANNELS` | Limit NCCL channels (set to 1 for tests to reduce memory) |

## Build Artifacts

- `Megatron-LM/uv.lock` — uv lockfile for dependency pinning
- BytePS 3rdparty deps in `byteps/3rdparty/ps-lite/deps/` (generated, `gitignore`d)
- Dockerfiles in `Dockerfile.*` at repo root and within each component's `docker/`
