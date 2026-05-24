# SGLang 中 All-Reduce 通信调研

SGLang 的推理核心代码集中在 `python/sglang/srt`

## **1. Runtime 请求数据流**

启动 SGLang 服务之后，前端 request 进入后端的大致路径为：

```python
http_server # /python/sglang/srt/entrypoint/http_server.py
  -> tokenizer/tokenize_manager # /python/sglang/srt/manager/tokenize_manager.py
  -> scheduler # /python/sglang/srt/manager/scheduler.py
  -> tp_worker # /python/sglang/srt/manager/tp_worker.py
  -> model_runner # /python/sglang/srt/model_executor/model_runner.py
  -> model.forward() # nn.Module
```

其中和模型并行通信关系最密切的是：

- `scheduler`：服务端调度层，负责接收 tokenized request、维护 `waiting_queue` / `running_batch`、管理 KV cache / radix cache、组 batch 并驱动 worker 前向。
- `tp_worker` / `model_runner`：模型执行层，负责构建 `ForwardBatch`，调用模型 `forward()`，再进入 sampling / logits processing
- `model.forward()`：模型结构层，例如 `Qwen2ForCausalLM -> Qwen2Model -> Qwen2DecoderLayer`，在 embedding、attention、MLP、logits processor 等位置触发通信

SGLang 的多 GPU 运行方式可以理解为 SPMD：每个 GPU 对应一个 `scheduler/model worker` 进程，每个进程运行同一份 Python 代码，但拥有不同的 `tp_rank` / `dp_rank` / `attn_tp_rank`。同一个通信组内的 rank 同时调用 collective，底层由 NCCL/Gloo/HCCL/自定义 all-reduce 等后端完成通信

## **2. 通信操作、通信组与 GroupCoordinator**

通信操作大多都封装在了 `python/srt/distributed/communication_op.py` 中

SGLang 使用 `GroupCoordinator` 封装通信组。不同并行维度会返回不同的 `GroupCoordinator`：

```python
# python/sglang/srt/distributed/communication_op.py
def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    return get_tp_group().all_reduce(input_)

def attention_tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    return get_attn_tp_group().all_reduce(input_)

def moe_tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    return get_moe_tp_group().all_reduce(input_)

def moe_expert_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    return get_moe_ep_group().all_reduce(input_)
```

常见通信组：

<table>
<tr>
<td>API<br/></td><td>通信语义<br/></td><td>典型用途<br/></td></tr>
<tr>
<td>`get_tp_group()`<br/></td><td>原始 Tensor Parallel 组<br/></td><td>普通 TP 下的 `RowParallelLinear`、MLP `down_proj`、vocab embedding<br/></td></tr>
<tr>
<td>`get_attn_tp_group()` / `get_attention_tp_group()`<br/></td><td>Attention TP 组<br/><br/></td><td>DP-Attention 下 attention / `o_proj` 内部通信<br/></td></tr>
<tr>
<td>`get_moe_tp_group()`<br/></td><td>MoE TP 组<br/></td><td>MoE 输出合并<br/></td></tr>
<tr>
<td>`get_moe_ep_group()`<br/></td><td>Expert Parallel 组<br/></td><td>MoE expert parallel 输出合并<br/></td></tr>
</table>

以 `tp_size=4, dp_size=2, enable_dp_attention=True` 为例：

```
原始 TP group:
  [0, 1, 2, 3]

attention TP groups:
  [0, 1]
  [2, 3]
```

每个 rank 进程里都有自己的 `GroupCoordinator` Python 对象。它们不是同一个对象，但拥有相同或相应的 `ranks` 列表，并指向同一组底层 distributed process group。

## **3. GroupCoordinator.all_reduce 的后端路由**

在通信时不直接调用 NCCL，而是调用 `GroupCoordinator.all_reduce()`。该函数根据设备、tensor 类型、是否在图捕获、是否使用对称内存、custom all-reduce 是否适配当前 tensor 等条件选择后端。

`GroupCoordinator.all-reduce` 实现：

```python
# python/sglang/srt/distributed/parallel_state.py
def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        """
        User-facing all-reduce function before we actually call the
        all-reduce operation.

        We need this because Dynamo does not support passing an arbitrary
        object (`self` in this case) to a custom op. We need to pass the
         group name as a string, and then look up the group coordinator from
         the group name, dispatch the all-reduce operation to the group
         coordinator.

        In addition, PyTorch custom ops do not support mutation or returning
        a new tensor in the same op. So we need to figure out if the op is
        in-place or out-of-place ahead of time.
        """
        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return input_

        if input_.is_cpu:
            if is_shm_available(input_.dtype, self.world_size, self.local_size):
                torch.ops.sgl_kernel.shm_allreduce(input_, REDUCE_OP_SUM)
            else:
                torch.distributed.all_reduce(input_, group=self.device_group)
            return input_

        if self.hpu_communicator is not None and not self.hpu_communicator.disabled:
            return self.hpu_communicator.all_reduce(input_)

        if self.xpu_communicator is not None and not self.xpu_communicator.disabled:
            return self.xpu_communicator.all_reduce(input_)

        if self.npu_communicator is not None and not self.npu_communicator.disabled:
            return self.npu_communicator.all_reduce(input_)

        if self.pynccl_comm is not None and self.is_symmetric_memory_enabled():
            self.debug_check_symmetric_mempool(self, {"input": input_}, "all_reduce")
            with self.pynccl_comm.change_state(enable=True):
                self.pynccl_comm.all_reduce(input_)
                return input_

        outplace_all_reduce_method = None
        if (
            self.ca_comm is not None
            and not self.ca_comm.disabled
            and self.ca_comm.should_custom_ar(input_)
        ):
            outplace_all_reduce_method = "ca"
        elif (
            self.qr_comm is not None
            and not self.qr_comm.disabled
            and self.qr_comm.should_quick_allreduce(input_)
        ):
            outplace_all_reduce_method = "qr"
        elif (
            self.pymscclpp_comm is not None
            and not self.pymscclpp_comm.disabled
            and self.pymscclpp_comm.should_mscclpp_allreduce(input_)
        ):
            outplace_all_reduce_method = "pymscclpp"
        elif (
            self.torch_symm_mem_comm is not None
            and not self.torch_symm_mem_comm.disabled
            and self.torch_symm_mem_comm.should_torch_symm_mem_allreduce(input_)
        ):
            outplace_all_reduce_method = "torch_symm_mem"
        elif is_in_piecewise_cuda_graph() and self.pynccl_comm is not None:
            # For piecewise cuda graph, we use pynccl outplace allreduce
            outplace_all_reduce_method = "pynccl"
        if outplace_all_reduce_method is not None:
            return outplace_all_reduce(
                input_,
                group_name=self.unique_name,
                outplace_all_reduce_method=outplace_all_reduce_method,
            )
        else:
            inplace_all_reduce(input_, group_name=self.unique_name)
            return input_
```

- `pynccl_comm` 存在不代表一定优先使用 PyNccl。只有 `pynccl_comm is not None and is_symmetric_memory_enabled()` 时才提前返回。
- `Custom AllReduce` 指 `ca_comm`，是 SGLang 按硬件/拓扑/tensor size 选择的自定义 all-reduce 实现，不是 PyTorch 原生 `torch.distributed.all_reduce`。
- `Quick AllReduce` 是 ROCm/AMD 上的快速 all-reduce 补充路径。
- `PyMscclpp` 是 MSCCL++ 路径，常和 graph 场景相关。
- `TorchSymmMem` 是 PyTorch symmetric memory all-reduce 路径。

## **4. Qwen2 中通信发生的位置**

我在本次调研中主要是通过 `qwen2.py` 中的模块以及相应的通信去进行的调研。当前 `Qwen2Attention` 使用普通 TP size，并没有显式接入 `use_dp_attention_reduce=True` 的 DP-Attention `o_proj` 路径。因此本节先按普通 TP 说明。

### **4.1 VocabParallelEmbedding：All-Reduce**

Qwen2 embedding 使用 `VocabParallelEmbedding`：

```python
# python/sglang/srt/models/qwen2.py
self.embed_tokens = VocabParallelEmbedding(
    config.vocab_size,
    config.hidden_size,
    quant_config=quant_config,
    use_attn_tp_group=is_dp_attention_enabled(),
    prefix=add_prefix("embed_tokens", prefix),
)
```

`VocabParallelEmbedding.forward()` 中，每个 TP rank 只持有一段 vocab shard。输入 token 如果不属于本 rank 的 vocab shard，会被 mask 掉。本地 embedding 后，需要把各 rank 的结果求和：

```python
# python/sglang/srt/layers/vocab_parallel_embedding.py
output_parallel = self.quant_method.embedding(self, masked_input.long())

if self.tp_size > 1:
    output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
    if not get_attn_tp_context().input_scattered:
        if self.use_attn_tp_group:
            output_parallel = attn_tp_all_reduce(output_parallel)
        else:
            output_parallel = tensor_model_parallel_all_reduce(output_parallel)
```

通信解释：

```
每个 rank 对自己 vocab shard 命中的 token 产生 embedding；
未命中的位置为 0；
all-reduce 求和后，每个 rank 得到完整 token embedding。
```

### **4.2 RowParallelLinear：All-Reduce**

`RowParallelLinear` 是多的 All-Reduce 触发点。

调用位置：

Qwen2 中：

```python
# python/sglang/srt/models/qwen2.py
class Qwen2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # MergedColumnParallelLinear支持All-Gather通信，一般不启用
        self.gate_up_proj = MergedColumnParallelLinear( 
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(
        self,
        x: torch.Tensor,
        forward_batch: ForwardBatch = None,
    ) -> torch.Tensor:
        if get_global_server_args().rl_on_policy_target is not None:
            x = x.bfloat16()

        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x, forward_batch=forward_batch)
        return x
```

`RowParallelLinear.forward()` 的通信逻辑：

```python
# python/sglang/srt/layers/linear.py
with symm_ctx:
    output_parallel = self.quant_method.apply(self, input_parallel, bias=bias_)

if self.reduce_results and self.tp_size > 1 and not skip_all_reduce:
    if self.use_dp_attention_reduce:
        output = get_attention_tp_group().all_reduce(output_parallel)
    else:
        quantize_communications = (
            (
                not forward_batch.forward_mode.is_decode_or_idle()
                and get_global_server_args().enable_quant_communications
            )
            if forward_batch is not None
            else False
        )
        if quantize_communications:
            output = tensor_model_parallel_quant_all_reduce(output_parallel)
        else:
            output = tensor_model_parallel_all_reduce(output_parallel)
else:
    output = output_parallel
```

通信语义：

```
RowParallelLinear 把输入维度 / weight 的 reduction 维度切给不同 TP rank；
每个 rank 计算一份 partial output；
All-Reduce 把 partial output 相加，得到完整 output。
```

普通 TP 下，`o_proj` 和 `down_proj` 通常都在 `tp_group` 上 all-reduce。DP-Attention （后面会解释什么是 DP-Attention，现在先按照普通的 TP 理解就好了）下，如果某个模型把 attention 的 `o_proj` 设置为 `use_dp_attention_reduce=True`，则 `o_proj` 只在 `attention_tp_group` 内 all-reduce，避免把不同 attention-DP 分片的 token 结果错误相加。

### **4.3 量化通信与普通通信**

`RowParallelLinear` 中的这段代码会在部分场景下启用量化通信：

```python
# python/sglang/srt/layers/linear.py
quantize_communications = (
    (
        not forward_batch.forward_mode.is_decode_or_idle()
        and get_global_server_args().enable_quant_communications
    )
    if forward_batch is not None
    else False
)
if quantize_communications:
    output = tensor_model_parallel_quant_all_reduce(output_parallel)
else:
    output = tensor_model_parallel_all_reduce(output_parallel)
```

区别：

- 普通路径：`tensor_model_parallel_all_reduce -> get_tp_group().all_reduce`
- 量化路径：`tensor_model_parallel_quant_all_reduce -> get_tp_group().quant_all_reduce`

`quant_all_reduce()` 目前主要是 NPU 特化；如果 NPU communicator 不可用，会 fallback 到普通 inplace all-reduce。启用条件排除了 decode/idle，说明它更偏向 prefill/extend 这类大 tensor 通信场景。

```python
# python/sglang/srt/distibuted/parallel_state.py
def quant_all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
    """
    User-facing quant-all-reduce function similar to all-reduce. 
    (NPU support only)
    """
    # Bypass the function if we are using only 1 GPU.
    if self.world_size == 1:
        return input_

    if self.npu_communicator is not None and not self.npu_communicator.disabled:
        return self.npu_communicator.quant_all_reduce(input_)
    else:
        inplace_all_reduce(input_, group_name=self.unique_name)
        return input_
```

### **4.4 LogitsProcessor：All-Gather**

`LogitsProcessor` 主要负责从 hidden states 计算 logits。LM head 通常按 vocab 维度(行维度)切分，因此每个 TP rank 先得到局部 vocab logits，然后在需要完整 logits 时做 all-gather。

计算 logits 后的 all-gather：

```python
# python/sglang/srt/layers/logits_processor.py
def __init__(self,……):
    self.use_attn_tp_group = get_global_server_args().enable_dp_lm_head
def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        logits_metadata: LogitsMetadata,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
    """Get logits from hidden_states.

    If sampled_logits_only is True, it means hidden_states only contain the
    last position (e.g., extend without input logprobs). The caller should
    guarantee the given hidden_states follow this constraint.
    """
    hidden_states, local_hidden_states = self._gather_dp_attn_hidden_states(
        hidden_states, logits_metadata
    ) # All-Gather hidden state

    logits = self._compute_lm_head(hidden_states, lm_head, embedding_bias)

    if self.logit_scale is not None:
        logits.mul_(self.logit_scale)

    if self.do_tensor_parallel_all_gather:
        if self.use_attn_tp_group:
            logits = self._gather_attn_tp_logits(logits) # 这种是每个dpgroup内有完整的lm-head的情况
        else:
            logits = tensor_model_parallel_all_gather(logits)

    logits = self._scatter_dp_attn_logits(
        logits, local_hidden_states, logits_metadata
    )

    logits = self._copy_logits_to_buffer(logits, logits_metadata)

    if self.final_logit_softcapping:
        if not _is_npu:
            fused_softcap(logits, self.final_logit_softcapping)
        else:
            logits = self.final_logit_softcapping * torch.tanh(
                logits / self.final_logit_softcapping
            )

    return logits
```

通信语义：

```
lm_head weight 按 vocab 切分；
每个 rank 计算局部 logits；
需要完整 vocab logits 时，在 TP 或 attention-TP 组上 all-gather
然后再把对应的 token 序列 scatter 到原来的dp rank上
```

## **5. DP-Attention 对通信路径的影响**

普通 TP attention 中，所有 TP rank 共同处理同一批 token，只是按 heads / hidden 维度切分。例如 `tp_size=4`：

```
rank0: heads 0..7
rank1: heads 8..15
rank2: heads 16..23
rank3: heads 24..31
```

DP-Attention 会把原始 TP group 再拆成多个 attention-DP 副本。例如 `tp_size=4, dp_size=2`：

```
attention_tp_size = tp_size // dp_size = 2

attention_tp_group 0 = [rank0, rank1]
attention_tp_group 1 = [rank2, rank3]
```

此时：

```
rank0, rank1 处理 token group A
rank2, rank3 处理 token group B
```

因此 attention 的 `o_proj` 只能在各自的 `attention_tp_group` 内 all-reduce，不能跨完整 `tp_group` all-reduce，否则会把不同请求/token group 的结果相加。

> [!TIP]
> 就是如果不开 DP Attention，那么比如有 8 张卡，来了 8 个 request 请求，那么不开 DP 的话就是这 8 张卡上部署一份 Attention 的权重，然后在 Attention 结束之后的 `o_proj` 中进行 8 卡的 All-Reduce，通信组过大在长序列的时候可能会造成性能损失，但是 FlashAttention 算的又很快，所以就做了一个计算和通信的 trade-off，SGLang 在这里做的就是将 Attention 分成多个副本，比如仍然是 8 卡，那么 `tp_size = 8, dp_size = 4, attn_dp_size = 2`,就是在这 8 张卡上部署 4 个 Attention 的权重副本，每两张卡作为一个 `dp_attn_group`,在来了 8 个 request 之后，有 4 个 dp 组，那么每个 DP 组就负责 8 / 4 = 2 个 request 的 Attention 计算，这样每张卡上的计算任务相比于之前不开 DP 就变多了，但是通信组变小了，在长序列以及 Prefill 的场景下会获得收益。

## **6. Attention 到 MLP 前的额外通信**

DP-Attention 下，attention 输出通常是 `TP_ATTN_FULL`(上面例子中的两个 request) 语义，而 dense MLP 默认需要 `FULL` （整个 batch 的 8 个 request）语义。

源码中的注释定义：

```python
# python/sglang/srt/layers/communicator.py
class ScatterMode(Enum):
    """
    Suppose we have TP=4, DP=2, enable-dp-attention, and the system handles seq a,b,c,d
    Model input/output: [ab, ab, cd, cd] for four ranks respectively
    SCATTERED: [a, b, c, d]
    TP_ATTN_FULL: [ab, ab, cd, cd], i.e. all ranks inside a TP attn group have full data of the group
    FULL: [abcd, abcd, abcd, abcd]
    """
```

`LayerCommunicator.prepare_mlp()` 会在 attention 后、MLP 前做布局转换。关键路径：

```python
# python/sglang/srt/layers/communicator.py
if context.attn_dp_size != 1:
    if use_layer_norm_before_gather and hidden_states.shape[0] != 0:
        hidden_states, residual = layernorm(hidden_states, residual)
    elif context.attn_tp_rank == 0:
        hidden_states += residual

    hidden_states, local_hidden_states = (
        get_global_dp_buffer(),
        hidden_states,
    )
    dp_gather_partial(hidden_states, local_hidden_states, forward_batch)

    if not use_layer_norm_before_gather:
        dp_scatter(residual, hidden_states, forward_batch)
        if hidden_states.shape[0] != 0:
            hidden_states = layernorm(hidden_states)
```

对于接入 `LayerCommunicator` 的 DP-Attention 模型，进入 MLP 前会有额外通信，把每个 attention-DP 分片中的 token 汇总成 dense MLP 所需的 `FULL` buffer。

## **7. DP-Attention gather 的 SUM_LEN 与 MAX_LEN**

`dp_gather_partial()` 有两类通信模式：`SUM_LEN` 和 `MAX_LEN`。选择逻辑：

```python
# python/sglang/srt/layers/dp_attention.py
if is_extend_in_batch and dp_size > 1:
    return DpPaddingMode.SUM_LEN

max_len = max(global_num_tokens)
sum_len = sum(global_num_tokens)
if sum_len * 2 >= max_len * dp_size:
    return cls.MAX_LEN
else:
    return cls.SUM_LEN
```

### **7.1 SUM_LEN**

`SUM_LEN` 是紧凑拼接。假设各 DP rank token 数为：

```
DP0: 2 tokens = ab
DP1: 5 tokens = cdefg
DP2: 1 token  = h
DP3: 3 tokens = ijk
```

则 global buffer 是：

```
[a b c d e f g h i j k]
长度 = 2 + 5 + 1 + 3 = 11
```

源码路径：

```python
# python/sglang/srt/layers/dp_attention.py
global_tokens.fill_(0)
memcpy_triton(global_tokens, local_tokens, 0, local_start_pos, local_num_tokens, False)
global_tokens[:] = tensor_model_parallel_all_reduce(global_tokens)
```

因为每个 rank 只在自己的 slice 写入非零数据，其他位置为 0，所以 all-reduce 后得到完整紧凑 buffer。

特点：

- 优点：通信量接近真实 token 数，padding 浪费少。
- 缺点：shape 不规则，对 CUDA graph、symmetric memory 和部分 collective 优化不如 MAX_LEN 友好。

### **7.2 MAX_LEN**

`MAX_LEN` 会把每个 DP 分片 pad 到相同长度。以上例中 `max_len=5`：

```
DP0: [a b _ _ _]
DP1: [c d e f g]
DP2: [h _ _ _ _]
DP3: [i j k _ _]

global buffer 长度 = max_len * dp_size = 5 * 4 = 20
```

源码路径：

```python
# python/sglang/srt/layers/dp_attention.py
scattered_local_tokens = local_tokens.tensor_split(get_attention_tp_size())[
    get_attention_tp_rank()
]
get_attention_tp_group().reduce_scatter_tensor(scattered_local_tokens, local_tokens)
get_tp_group().all_gather_into_tensor(global_tokens, scattered_local_tokens)
```

为什么先 `attention_tp_group.reduce_scatter_tensor()`，再 `tp_group.all_gather_into_tensor()`：

```
DP0(rank0, rank1) local block = [a, b, _, _]
DP1(rank2, rank3) local block = [c, d, e, _]

attention_tp_group.reduce_scatter_tensor() 之后
->
DP0 block [a,b,_,_] 被切成 2 份:
    rank0 保留 [a,b]
    rank1 保留 [_,_]

DP1 block [c,d,e,_] 被切成 2 份:
    rank2 保留 [c,d]
    rank3 保留 [e,_]

tp_group.all_gather_into_tensor()
->
rank0 贡献 [a,b]
rank1  [_,_]
rank2  [c,d]
rank3  [e,_]
allgather 结果:
rank0/1/2/3: [a,b, _,_, c,d, e,_]
```

特点：

- 优点：每个 rank 输入 shape 规则，适合 `all_gather_into_tensor`、CUDA graph、symmetric memory。
- 缺点：各 DP token 数差异大时 padding 浪费明显

prefill/extend 且 `dp_size > 1` 时源码直接选 `SUM_LEN`；decode 场景 token 数通常更均匀，MAX_LEN 的 padding 成本较小，规则 shape 的收益更明显。

## **8. 其他涉及 All-Reduce 的通信位置（AI 帮看）**

除 Qwen2 主干中的 embedding / row-parallel linear 外，还存在其他 All-Reduce 触发点：

### **8.1 LayerNorm 与 All-Reduce Fusion**

部分路径会尝试把 all-reduce 与 RMSNorm 融合：

```python
# python/sglang/srt/layers/layernorm.py
if world_size > 1:
    if _use_aiter:
        fused_result = tensor_model_parallel_fused_allreduce_rmsnorm(
            x, residual, weight, norm_module.variance_epsilon
        )
        if fused_result is not None:
            return fused_result
    else:
        fused_result = flashinfer_allreduce_residual_rmsnorm(...)
        if fused_result[0] is not None:
            return fused_result

    if _use_aiter and get_global_server_args().enable_aiter_allreduce_fusion:
        x = tensor_model_parallel_all_reduce(x)
        return norm_module.forward(x, residual, None)
```

该路径通常由 `LayerCommunicator` 或某些模型的 fused norm 调用触发。

### **8.2 MoE 输出合并**

MoE 模型中 expert parallel / tensor parallel 的输出合并可能触发：

```
moe_tensor_model_parallel_all_reduce # moe模型只开了TP的调用
moe_expert_parallel_all_reduce # 专家并行但是没有启用deepEP这种后端的时候的操作
tensor_model_parallel_all_reduce
```

这类路径一般出现在 MoE 的 post-expert combine 阶段。是否跳过 all-reduce 还会受到 `should_allreduce_fusion`、`use_reduce_scatter` 等参数影响。

### **8.3 Mamba / Hybrid 层中的局部统计量归约**

例如 `mixer2_rms_norm_gated.py` 中，当 RMSNorm 的 reduction 维度被 TP 切分后，需要对 local sum 做 all-reduce：

```python
# python/sglang/srt/layers/attention/mamba/mixer2_rms_norm_gated.py
if self.n_groups == 1:
    if self.tp_size > 1:
        local_sums = x.pow(2).sum(dim=-1, keepdim=True)
        global_sums = tensor_model_parallel_all_reduce(local_sums)
```

### **8.4 控制面或缓存状态同步**

并不是所有 `torch.distributed.all_reduce` 都走 `communication_op.py`。一些控制面、scheduler、cache、sampler、speculative worker 的状态同步会直接调用 PyTorch distributed。例如：

```
sampler token id 同步
radix / hierarchical cache 状态同步
scheduler / pipeline / disaggregation 控制信号
speculative worker 全局状态判断
```

这些通常不是模型主干 tensor-parallel 计算，而是服务端状态一致性或控制流同步。

## **9. 普通 DP 与 DP-Attention 的差异**

普通 DP 是复制整套 TP 模型；DP-Attention 是在一个 TP group 内让 attention 部分具有 DP 语义。

## 普通 DP

```
tp_size=4, dp_size=2
=> 2 套完整 TP 模型副本
=> 总 GPU / scheduler 进程数 = 4 * 2 = 8
```

布局示例：

```
DP0: tp0, tp1, tp2, tp3
DP1: tp0, tp1, tp2, tp3
```

## DP-Attention

```
tp_size=4, dp_size=2, enable_dp_attention=True
=> 通常仍是一个原始 TP group，共 4 个 rank
=> 在 attention 内拆出两个 attention-DP group
```

布局示例：

```
tp_group: [0, 1, 2, 3]
attn_tp_group0: [0, 1]
attn_tp_group1: [2, 3]
```

注意：DP 和 DP-Attention 不能同时打开

```python
class ModelRunner():
    def __init__():
        self.dp_size = server_args.dp_size if server_args.enable_dp_attention else 1
        
# data_parallel_controller中
if server_args.enable_dp_attention:
    self.launch_dp_attention_schedulers(server_args, port_args)
    # When local control broadcast is enabled, send control messages to
    # every DP group leader (attn_tp_rank=0) so each leader broadcasts
    # within its own attn_tp_group instead of the full tp_group.
    # Otherwise fall back to the original behaviour: send to only the
    # first leader, which then broadcasts over the full tp_group.
    local_ctrl = server_args.enable_dp_attention_local_control_broadcast
    self.control_message_step = 1 if local_ctrl else server_args.tp_size
else:
    self.launch_dp_schedulers(server_args, port_args)
    self.control_message_step = 1
```

## **10. 总结**

SGLang 中模型主路径的通信可以归纳为：

<table>
<tr>
<td>位置<br/></td><td>通信类型<br/></td><td>通信组<br/></td><td>目的<br/></td></tr>
<tr>
<td>`VocabParallelEmbedding.forward()`<br/></td><td>All-Reduce<br/></td><td>`tp_group` 或 `attn_tp_group`<br/></td><td>合并 vocab shard embedding<br/></td></tr>
<tr>
<td>`RowParallelLinear.forward()`<br/></td><td>All-Reduce<br/></td><td>`tp_group` 或 `attn_tp_group`<br/></td><td>合并 row-parallel partial output<br/></td></tr>
<tr>
<td>`LogitsProcessor._get_logits()`<br/></td><td>All-Gather<br/></td><td>`tp_group` 或 `attn_tp_group`<br/></td><td>合并 vocab-parallel logits<br/></td></tr>
<tr>
<td>`LayerCommunicator.prepare_mlp()`<br/></td><td>All-Reduce / Gather / Scatter<br/></td><td>`tp_group` / `attn_tp_group`<br/></td><td>DP-Attention 到 dense MLP 的布局转换<br/></td></tr>
<tr>
<td>DP-Attention gather `SUM_LEN`<br/></td><td>All-Reduce<br/></td><td>`tp_group`<br/></td><td>紧凑拼接 DP token buffer<br/></td></tr>
<tr>
<td>DP-Attention gather `MAX_LEN`<br/></td><td>Reduce-Scatter + All-Gather<br/></td><td>`attn_tp_group` + `tp_group`<br/></td><td>规则 shape 的 padded FULL buffer<br/></td></tr>
<tr>
<td>MoE post-expert combine<br/></td><td>All-Reduce<br/></td><td>`moe_tp_group` / `moe_ep_group`<br/></td><td>合并专家并行输出<br/></td></tr>
<tr>
<td>LayerNorm fusion<br/></td><td>fused All-Reduce + RMSNorm<br/></td><td>多种 group<br/></td><td>降低 all-reduce + norm 开销<br/></td></tr>
</table>

模型计算主路径大多通过：

```
communication_op.py wrapper
  -> get_*_group()
  -> GroupCoordinator collective API
  -> custom / PyNccl / MSCCLPP / TorchSymmMem / torch.distributed backend
```

服务端控制面、缓存、scheduler 等状态同步则可能直接调用 `torch.distributed.all_reduce`，不完全经过模型通信 wrapper。
