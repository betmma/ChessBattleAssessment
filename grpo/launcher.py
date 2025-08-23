#!/usr/bin/env python3
import os
import subprocess
import sys
import torch
import torch.distributed as dist

# ----------------------------
# Environment overrides
# ----------------------------
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"
os.environ["NCCL_SOCKET_IFNAME"] = "lo"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_NET"] = "Socket"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["GLOO_SOCKET_IFNAME"] = "lo"

# ----------------------------
# Determine ranks for torchrun
# ----------------------------
world_size = int(os.environ.get("WORLD_SIZE", "1"))
rank = int(os.environ.get("RANK", "0"))
local_rank = int(os.environ.get("LOCAL_RANK", "0"))

# ----------------------------
# Initialize process group to prevent auto TCP hostname lookup
# ----------------------------
dist.init_process_group(
    backend="nccl",
    init_method=f"tcp://127.0.0.1:29500",
    world_size=world_size,
    rank=rank
)

# Assign GPU according to local rank
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)

# ----------------------------
# Launch the actual program
# ----------------------------
subprocess.run([sys.executable] + sys.argv[1:])
