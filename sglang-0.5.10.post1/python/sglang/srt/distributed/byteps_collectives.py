import hashlib
import logging
import os
from typing import Dict

import torch

logger = logging.getLogger(__name__)

_BPS_INITIALIZED = False
_BPS_DEBUG = False
_BPS_NNODES = 1
_DECLARED_BPS_GROUPS: Dict[str, int] = {}


def initialize_byteps_for_sglang(
    local_rank: int,
    local_size: int,
    nnodes: int = 1,
    node_rank: int = 0,
    debug: bool = False,
) -> None:
    global _BPS_INITIALIZED, _BPS_DEBUG, _BPS_NNODES

    _BPS_DEBUG = debug
    _BPS_NNODES = nnodes
    os.environ.setdefault("BYTEPS_LOCAL_RANK", str(local_rank))
    os.environ.setdefault("BYTEPS_LOCAL_SIZE", str(local_size))
    os.environ.setdefault("DMLC_ROLE", "worker")
    os.environ.setdefault("DMLC_NUM_WORKER", str(nnodes))
    os.environ.setdefault("DMLC_WORKER_ID", str(node_rank))

    import byteps.torch as bps

    if not _BPS_INITIALIZED:
        # Ensure all local ranks reach init simultaneously so the BytePS
        # root device (local_rank=0) can register before the scheduler
        # finalizes the ADD_NODE barrier.
        if local_size > 1:
            import torch.distributed
            torch.distributed.barrier()
        bps.init()
        _BPS_INITIALIZED = True

    if bps.local_rank() != local_rank:
        raise RuntimeError(
            "BytePS local rank mismatch for --use-byteps-all-reduce: "
            f"expected {local_rank}, got {bps.local_rank()}."
        )
    if bps.local_size() != local_size:
        raise RuntimeError(
            "BytePS local size mismatch for --use-byteps-all-reduce: "
            f"expected {local_size}, got {bps.local_size()}."
        )

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
        raise ValueError("logical_name must be a stable non-empty string")
    return (
        f"sglang.{group.unique_name}.r{_group_fingerprint(group.ranks)}."
        f"{logical_name}"
    )


def declare_and_cache_byteps_group(name: str, expected_workers: int) -> None:
    import byteps.torch as bps

    cached_workers = _DECLARED_BPS_GROUPS.get(name)
    if cached_workers is None:
        bps.declare(name, expected_workers=expected_workers)
        _DECLARED_BPS_GROUPS[name] = expected_workers
        if _BPS_DEBUG:
            logger.info(
                "Declared BytePS tensor name=%s expected_workers=%s",
                name,
                expected_workers,
            )
        return
    if cached_workers != expected_workers:
        raise RuntimeError(
            f"BytePS tensor name {name} was declared with inconsistent group size: "
            f"{cached_workers} vs {expected_workers}"
        )


def _byteps_expected_workers() -> int:
    if _BPS_NNODES > 1:
        return _BPS_NNODES
    import byteps.torch as bps

    local_size = max(1, bps.local_size())
    return max(1, bps.size() // local_size)


def byteps_allreduce_inplace(
    tensor: torch.Tensor,
    group,
    logical_name: str,
) -> torch.Tensor:
    if not _BPS_INITIALIZED:
        raise RuntimeError(
            "BytePS All-Reduce was requested before BytePS was initialized. "
            "Ensure --use-byteps-all-reduce is initialized in the model worker."
        )

    from byteps.torch import ops as bps_ops

    # Include tensor shape in the BytePS name so that different batch
    # sizes (e.g. prefill vs decode) use independent server-side
    # buffers. BytePS does not support reusing the same name with
    # different tensor sizes across forward passes.
    shape_suffix = "x".join(str(d) for d in tensor.shape)
    name = build_byteps_group_name(
        group, f"{logical_name}.{shape_suffix}"
    )
    expected_workers = _byteps_expected_workers()
    declare_and_cache_byteps_group(name, expected_workers)
    if _BPS_DEBUG:
        logger.debug(
            "BytePS all-reduce: name=%s shape=%s dtype=%s device=%s "
            "group_world_size=%s expected_workers=%s",
            name,
            tuple(tensor.shape),
            tensor.dtype,
            tensor.device,
            group.world_size,
            expected_workers,
        )
    handle = bps_ops.push_pull_async_inplace(
        tensor,
        average=False,
        name=name,
        version=0,
        priority=0,
    )
    bps_ops.synchronize(handle)
    return tensor
