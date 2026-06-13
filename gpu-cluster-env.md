# 跨机 SGLang 环境描述

## 主机信息

### gpu04（头节点）
- **角色**: SGLang Rank 0
- **IP**: 172.16.41.80
- **GPU**: 2× NVIDIA RTX A6000 (PCI 0000:A8:00.0, 0000:B8:00.0)
- **管理网卡**: ens15f0 (172.16.41.80/24, 非 RDMA)
- **RDMA 网卡**: Mellanox ConnectX-6 Dx (双口)
  - **ens39f0np0** (mlx5_0): 192.168.2.13/24, MTU 9000, RoCEv2
  - **ens39f1np1** (mlx5_1): 192.168.1.13/24, MTU 9000, RoCEv2
- **NUMA**: GPU + NIC 均在 NUMA node 1
- **Hostname**: gpu04
- **OS**: Ubuntu 22.04 (Docker 容器)
- **Conda 环境**: sgl-dev (`/opt/conda/envs/sgl-dev/`)
- **Torch**: 2.9.1+cu128, NCCL 2.27.5
- **SSH**: 监听端口 10022; gpu04 → gpu03 免密

### gpu03（工作节点）
- **角色**: SGLang Rank 1
- **IP**: 172.16.41.73
- **GPU**: 2× NVIDIA RTX A6000 (PCI 0000:A8:00.0, 0000:B8:00.0)
- **管理网卡**: ens15f0 (172.16.41.73/24, 非 RDMA)
- **RDMA 网卡**: Mellanox ConnectX-6 Dx (双口)
  - **ens39f0np0** (mlx5_0): 192.168.2.12/24, MTU 9000, RoCEv2
  - **ens39f1np1** (mlx5_1): 192.168.1.12/24, MTU 9000, RoCEv2
- **NUMA**: GPU + NIC 均在 NUMA node 1
- **Hostname**: gpu03
- **OS**: Ubuntu 22.04 (Docker 容器)
- **Conda 环境**: sgl-dev (`/opt/conda/envs/sgl-dev/`)
- **Torch**: 2.9.1+cu128, NCCL 2.27.5
- **SSH**: 监听端口 10022; 未配到 gpu04 的 SSH 免密（不需要）

## RDMA 网卡详情

### 物理拓扑

```
gpu04 (172.16.41.80)              gpu03 (172.16.41.73)
┌─────────────────────────┐       ┌─────────────────────────┐
│ GPU0  GPU1              │       │ GPU0  GPU1              │
│ (A8)   (B8)             │       │ (A8)   (B8)             │
│                         │       │                         │
│ mlx5_0  mlx5_1          │       │ mlx5_0  mlx5_1          │
│ (D8:00) (D8:01)         │       │ (D8:00) (D8:01)         │
│  │        │              │       │  │        │              │
▼  ▼        ▼              ▼       ▼  ▼        ▼              ▼
 ens39f0np0  ens39f1np1         ens39f0np0  ens39f1np1
 192.168.2.13 192.168.1.13      192.168.2.12 192.168.1.12
    │            │                  │            │
    └────────────┼──────────────────┘            │
    192.168.2.0/24 subnet                        │
                 └───────────────────────────────┘
                  192.168.1.0/24 subnet
```

- 两个 Mellanox 口通过同一 PCI bridge (0000:d7:01.0) 连接到主机
- 所有 GPU 和 RDMA NIC 都在 NUMA node 1，避免了跨 NUMA 的数据传输开销
- 两个子网各有一条独立物理链路，可用于双通道 RDMA 或故障切换

### 工作模式

| 属性 | mlx5_0 (ens39f0np0) | mlx5_1 (ens39f1np1) |
|------|---------------------|---------------------|
| **类型** | Mellanox ConnectX-6 Dx | Mellanox ConnectX-6 Dx |
| **模式** | RoCE v2 (GID type 4) | RoCE v2 (GID type 4) |
| **链路状态** | ACTIVE / LINK_UP | ACTIVE / LINK_UP |
| **MTU** | 9000 (Jumbo Frame) | 9000 (Jumbo Frame) |
| **gpu04 IP** | 192.168.2.13/24 | 192.168.1.13/24 |
| **gpu03 IP** | 192.168.2.12/24 | 192.168.1.12/24 |
| **队列数** | TX 512, RX 63 | TX 512, RX 63 |
| **NUMA node** | 1 | 1 |

### 关键对照表

| 项目 | gpu04 (rank 0) | gpu03 (rank 1) |
|------|---------------|---------------|
| 管理 IP (ens15f0) | 172.16.41.80 | 172.16.41.73 |
| RDMA IP #1 (ens39f0np0) | 192.168.2.13 | 192.168.2.12 |
| RDMA IP #2 (ens39f1np1) | 192.168.1.13 | 192.168.1.12 |
| RDMA 设备 #1 | mlx5_0 | mlx5_0 |
| RDMA 设备 #2 | mlx5_1 | mlx5_1 |
| GPU 数量 | 2× RTX A6000 | 2× RTX A6000 |
| GPU PCI | A8:00.0, B8:00.0 | A8:00.0, B8:00.0 |

## SSH 连接

```bash
# gpu04 → gpu03 (免密)
ssh -p 10022 root@172.16.41.73

# gpu03 上激活 conda
source /opt/conda/etc/profile.d/conda.sh && conda activate sgl-dev
```

## 模型

- **模型**: Qwen/Qwen3-32B (62GB)
- **gpu04 路径**: `/root/.cache/huggingface/hub/models--Qwen--Qwen3-32B/`
- **gpu03 路径**: `/root/.cache/huggingface/hub/models--Qwen--Qwen3-32B/`
- **下载**: `source $(conda info --base)/etc/profile.d/conda.sh && conda activate sgl-dev && hf download Qwen/Qwen3-32B`

## RDMA 网卡绑定模式

参照 Megatron-LM Qwen 训练脚本的自动发现模式（详见 `Megatron-LM/examples/qwen/train_qwen_3b*.sh`）。

### 核心链路：HCA → sysfs → netdev → IP/NUMA

```
NCCL_IB_HCA=mlx5_1              # 1. 指定 HCA（单口绑定）
       ↓
/sys/class/infiniband/mlx5_1/device/net/    # 2. sysfs 查找 netdev
       ↓
ens39f1np1                      # 3. 得到接口名
       ↓
ip addr show ens39f1np1         # 4. 获取 RDMA IP
/sys/class/net/ens39f1np1/device/numa_node   # 5. NUMA node
/sys/class/net/ens39f1np1/device/local_cpulist  # 6. CPU 列表
       ↓
numactl --physcpubind=$CPUS --membind=$NUMA  # 7. NUMA 绑定
```

### 自动发现函数

```bash
extract_primary_hca() {
    local hca="${NCCL_IB_HCA:-}"
    hca="${hca%%,*}"       # "mlx5_1:1,mlx5_0:1" → "mlx5_1"
    hca="${hca%%:*}"       # 去掉端口后缀
    printf '%s' "$hca"
}

detect_iface_from_hca() {
    ls "/sys/class/infiniband/${1}/device/net" 2>/dev/null | head -n1
}

detect_ip_from_iface() {
    ip -4 -o addr show dev "${1}" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -n1
}

detect_numa_from_iface() {
    cat "/sys/class/net/${1}/device/numa_node" 2>/dev/null || echo "-1"
}

detect_cpulist_from_iface() {
    cat "/sys/class/net/${1}/device/local_cpulist" 2>/dev/null || echo ""
}
```

### 本集群上的映射结果

| 步骤 | 变量 | mlx5_1 口 | mlx5_0 口 |
|------|------|-----------|-----------|
| HCA | `NCCL_IB_HCA` | `mlx5_1` | `mlx5_0` |
| netdev | `DMLC_INTERFACE` | `ens39f1np1` | `ens39f0np0` |
| gpu04 IP | `DMLC_NODE_HOST` | `192.168.1.13` | `192.168.2.13` |
| gpu03 IP | `DMLC_NODE_HOST` | `192.168.1.12` | `192.168.2.12` |
| NUMA | NUMA node | `1` | `1` |
| 子网 | | 192.168.1.0/24 | 192.168.2.0/24 |

### 推荐绑定策略

Megatron 脚本默认只用 **单口** (`mlx5_1`)。双口可同时使用但需注意：
- 两个口都在同一 NUMA node、同一 PCI bridge，共享 PCIe 带宽
- 如果需要双通道（如 BytePS 控制面 + 数据面分离），可用 `mlx5_0` 走数据面、`mlx5_1` 走控制面

## NCCL + RDMA 跨机环境变量

### TCP 模式（当前已跑通）
```bash
export NCCL_SOCKET_IFNAME=ens15f0
export NCCL_IB_DISABLE=1
export GLOO_SOCKET_IFNAME=ens15f0
```

### RDMA/RoCE 模式（目标，参照 Megatron 训练脚本）
```bash
# NCCL
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_1              # 单口绑定（推荐 mlx5_1）
export NCCL_IB_GID_INDEX=0             # RoCEv2 GID index（0=默认 RoCEv2）
export NCCL_SOCKET_IFNAME=ens39f1np1   # Socket 也走 RDMA 口（非管理网卡）
export GLOO_SOCKET_IFNAME=ens39f1np1

# BytePS RDMA
export DMLC_ENABLE_RDMA=ibverbs        # BytePS 走 verbs RDMA
export DMLC_INTERFACE=ens39f1np1       # 从 HCA 自动发现
export DMLC_NODE_HOST=192.168.1.13     # 从 DMLC_INTERFACE IP 自动发现
export DMLC_USE_GDR=0                  # 暂不启用 GPUDirect RDMA
export BYTEPS_RDMA_RX_DEPTH=512
export BYTEPS_RDMA_START_DEPTH=32
```

## /etc/hosts 配置（必须）

```
# gpu04 上
172.16.41.80  gpu04
172.16.41.73  gpu03

# gpu03 上
172.16.41.80  gpu04
172.16.41.73  gpu03
```

## SGLang 跨机启动命令

### 默认 NCCL（已跑通，非 BytePS）

```bash
# gpu04 (Rank 0) — 先启动
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-32B --host 0.0.0.0 --port 30000 \
    --tp-size 4 --nnodes 2 --node-rank 0 \
    --dist-init-addr 172.16.41.80:50000 \
    --disable-piecewise-cuda-graph

# gpu03 (Rank 1) — 等 gpu04 打印 "Init torch distributed begin" 后
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-32B --host 0.0.0.0 --port 30000 \
    --tp-size 4 --nnodes 2 --node-rank 1 \
    --dist-init-addr 172.16.41.80:50000 \
    --disable-piecewise-cuda-graph
```

> 只有 Rank 0 暴露 HTTP API。跨机 TP 请求发到 `http://172.16.41.80:30000`。

## 已知问题

- gpu03 缺少部分命令行工具 (`ss`, `ip`, `lsof`)，不影响运行
- Docker 容器内 `/etc/hosts` 不可直接用 `sed -i`，需 `cat > /etc/hosts`
- gpu03 → gpu04 SSH 免密未配置（所有操作可从 gpu04 发起）
- RDMA 子网间 ping 不通（可能是容器网络策略限制 ICMP，需进一步排查 L2 连通性）
- 当前 `ibv_devinfo` / `ibstat` 命令在容器内不可用，RDMA 设备通过 sysfs 查看
- MTU 已配置为 9000 (Jumbo Frame)，但需确认交换机 / 对端支持
