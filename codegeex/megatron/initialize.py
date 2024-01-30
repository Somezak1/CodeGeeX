# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Megatron initialization."""

import random
import os
import time
import datetime

import numpy as np
import torch

from codegeex.megatron import get_adlr_autoresume
from codegeex.megatron import get_args
from codegeex.megatron import get_tensorboard_writer
from codegeex.megatron import mpu
from codegeex.megatron.global_vars import set_global_variables
from codegeex.megatron.mpu import (
    set_tensor_model_parallel_rank,
    set_tensor_model_parallel_world_size,
)

try:
    import wandb
except ImportError:
    wandb = None

import deepspeed


def initialize_megatron(
    extra_args_provider=None,
    args_defaults={},
    ignore_unknown_args=False,
    allow_no_cuda=False,
):
    """Set global variables, initialize distributed, and
    set autoresume and random seeds.
    `allow_no_cuda` should not be set unless using megatron for cpu only
    data processing. In general this arg should not be set unless you know
    what you are doing.
    Returns a function to finalize distributed env initialization
    (optionally, only when args.lazy_mpu_init == True)
    """

    # extra_args_provider: None
    # args_defaults: {"tokenizer_type": "GPT2BPETokenizer"}
    # ignore_unknown_args: False
    # allow_no_cuda: False

    if not allow_no_cuda:
        # Make sure cuda is available.
        assert torch.cuda.is_available(), "Megatron requires CUDA."

    # Parse args, build tokenizer, and set adlr-autoresume,
    # tensorboard-writer, and timers.
    set_global_variables(
        extra_args_provider=extra_args_provider,
        # extra_args_provider: None
        args_defaults=args_defaults,
        # args_defaults: {'tokenizer_type': 'GPT2BPETokenizer'}
        ignore_unknown_args=ignore_unknown_args,
        # ignore_unknown_args: False
    )

    # torch.distributed initialization
    def finish_mpu_init():
        args = get_args()
        # Pytorch distributed.
        # 分布式训练 通信初始化的核心函数
        _initialize_distributed()
        # 在_initialize_distributed()函数中, 使用torch.cuda.set_device(device)为各个进程分配了指定gpu
        # 在运行torch.cuda.set_device(device)前, 各个进程运行torch.cuda.current_device()的结果都为0
        # 在运行torch.cuda.set_device(device)后, 各个进程运行torch.cuda.current_device()的结果为自己指定gpu的编号

        # Random seeds for reproducibility.
        if args.rank == 0:
            print("> setting random seeds to {} ...".format(args.seed))
        # args.seed: 1234
        # 每个进程有两个随机数种子:

        # 1, 一个是常驻种子
        # 如果不使用流水线并行, 那么所有进程的常驻种子一样
        # 如果使用流水线并行, 那么同一流水线并行组组内各进程的常驻种子不同, 组间同对应位置的进程 常驻种子相同
        # 比如tp=2 pp=2 dp=2时
        # 流水线并行组1: [0, 4]
        # 流水线并行组2: [1, 5]
        # 流水线并行组3: [2, 6]
        # 流水线并行组4: [3, 7]
        # 那么0 1 2 3 进程的常驻种子相同, 4 5 6 7 进程的常驻种子相同, 0与4的常驻种子不同
        # 模型结构中部分参数的初始化使用的是常驻种子, 比如QueryEmbedding模块中的VocabParallelEmbedding, 所有的nn.Embedding
        # 部分dropout使用的是常驻种子, 比如Embedding模块中的Dropout, QueryEmbedding模块中的Dropout, ParallelTransformerLayer中ParallelSelfAttention后的Dropout

        # 2, 另一个是临时种子
        # 每个进程的临时种子= 常驻种子 + 一个该进程所属tp组的rank值 + 2718(可随意更改)
        # 比如对于本次运行/调试脚本 tp=4 pp=1 dp=2 的情况
        # 张量并行组1: [0, 1, 2, 3]
        # 张量并行组2: [4, 5, 6, 7]
        # 那么进程0的临时种子为args.seed + 0 + 2718
        # 那么进程4的临时种子为args.seed + 0 + 2718
        # 那么进程5的临时种子为args.seed + 1 + 2718
        # 那么进程7的临时种子为args.seed + 3 + 2718
        # 部分模型结构的参数初始化使用的是临时种子, 比如Embedding模块中的VocabParallelEmbedding, ParallelSelfAttention中的Q,K,V,O参数, ParallelMLP中的矩阵参数
        # 部分dropout使用的是临时种子, 比如ParaSelfAttention中的Dropout,

        _set_random_seed(args.seed)

    args = get_args()
    # args.lazy_mpu_init: None
    if args.lazy_mpu_init:
        args.use_cpu_initialization = True
        # delayed initialization of DDP-related stuff
        # We only set basic DDP globals
        set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
        # and return function for external DDP manager
        # to call when it has DDP initialized
        set_tensor_model_parallel_rank(args.rank)
        return finish_mpu_init
    else:
        # Megatron's MPU is the master. Complete initialization right away.
        finish_mpu_init()

        # Initialize memory buffers.
        _initialize_mem_buffs()

        # Autoresume.
        _init_autoresume()

        # No continuation function
        return None


def _compile_dependencies():

    args = get_args()

    # =========================
    # Compile dataset C++ code.
    # =========================
    # TODO: move this to ninja
    if torch.distributed.get_rank() == 0:
        start_time = time.time()
        print("> compiling dataset index builder ...")
        # from megatron.data.dataset_utils import compile_helper
        # compile_helper()
        print(
            ">>> done with dataset index builder. Compilation time: {:.3f} "
            "seconds".format(time.time() - start_time),
            flush=True,
        )

    # Custom kernel constraints check.
    seq_len = args.seq_length
    attn_batch_size = (
        args.num_attention_heads / args.tensor_model_parallel_size
    ) * args.micro_batch_size
    # Constraints on sequence length and attn_batch_size to enable warp based
    # optimization and upper triangular optimization (for causal mask)
    custom_kernel_constraint = (
        seq_len > 16
        and seq_len <= 2048
        and seq_len % 4 == 0
        and attn_batch_size % 4 == 0
    )
    # Print a warning.
    if not (
        (args.fp16 or args.bf16)
        and custom_kernel_constraint
        and args.masked_softmax_fusion
    ):
        if args.rank == 0:
            print(
                "WARNING: constraints for invoking optimized"
                " fused softmax kernel are not met. We default"
                " back to unfused kernel invocations.",
                flush=True,
            )

    # Always build on rank zero first.
    if torch.distributed.get_rank() == 0:
        start_time = time.time()
        print("> compiling and loading fused kernels ...", flush=True)
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()
    # Simple barrier to make sure all ranks have passed the
    # compilation phase successfully before moving on to the
    # rest of the program. We think this might ensure that
    # the lock is released.
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(
            ">>> done with compiling and loading fused kernels. "
            "Compilation time: {:.3f} seconds".format(time.time() - start_time),
            flush=True,
        )


def setup_deepspeed_random_and_activation_checkpointing(args):
    """Optional DeepSpeed Activation Checkpointing features.
    Gives access to partition activations, contiguous memory optimizations
    and cpu checkpointing.
    Activation checkpoint requires keep track of the random states
    and setting the random seed for each MP process. Megatron uses
    mpu.get_cuda_rng_tracker and mpu.model_parallel_cuda_manual_seed
    for keeping track of the random states and setting the random seeds.
    Since they are used in places outside of activation checkpointing,
    we overwrite them to maintain consistency.
    This must be called before all the calls to mpu.model_parallel_cuda_manual_seed
    """
    num_layers = args.num_layers // args.checkpoint_num_layers
    num_layers = (
        num_layers
        if args.num_layers % args.checkpoint_num_layers == 0
        else num_layers + 1
    )
    if args.split_transformers:
        num_layers *= 2

    deepspeed.checkpointing.configure(
        mpu,
        partition_activations=args.partition_activations,
        contiguous_checkpointing=args.contigious_checkpointing,
        num_checkpoints=num_layers,
        checkpoint_in_cpu=args.checkpoint_in_cpu,
        synchronize=args.synchronize_each_layer,
        profile=args.profile_backward,
    )

    mpu.checkpoint = deepspeed.checkpointing.checkpoint
    mpu.get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
    mpu.model_parallel_cuda_manual_seed = (
        deepspeed.checkpointing.model_parallel_cuda_manual_seed
    )


def _initialize_distributed():
    """Initialize torch.distributed and mpu."""
    """
    Initialize torch.distributed and mpu.
                |    Node1  |   Node2    |
    ____________| p1 |  p2  |  p3  |  p4 |
    local_rank  | 0  |   1  |  0   |   1 |
    rank        | 0  |   1  |  2   |   3 |

    node: 物理结点, 1台机器或者1个容器. 图中2个物理结点
    rank: 进程在全局上的序号. 图中4个进程
    local_rank: 进程在node上的序号.
    torch.cuda.device_count(): 当前进程所在的node上可使用的GPU的数量
    device: GPU在某个node上的编号

    该函数作用:
    1、设置分布式环境: 初始化进程, 分配GPU, 并设置进程大组（group）
    2、制定DP/TP/PP分组策略, 设置进程子组（subgroup）
    3、设置DeepSpeed ZeRO-R, 对activation进行优化
    """
    args = get_args()

    # device_count: 8, 当前进程所在的node上可使用的GPU的数量
    device_count = torch.cuda.device_count()

    # torch.distributed.is_initialized(): False
    if torch.distributed.is_initialized():
        # 如果已创建好分布式环境

        if args.rank == 0:
            print(
                "torch distributed is already initialized, "
                "skipping initialization ...",
                flush=True,
            )
        args.rank = torch.distributed.get_rank()
        # 取得当前进程的全局序号
        args.world_size = torch.distributed.get_world_size()
        # 取得全局进程的个数

    else:
        # 如果未创建好分布式环境

        if args.rank == 0:
            print("> initializing torch distributed ...", flush=True)
        # Manually set the device ids.
        # 1. 初始化进程, 分配GPU, 并设置进程大组（group）
        if device_count > 0:
            device = args.rank % device_count
            # 此次debug的device: 7
            # 1块GPU 1个进程, device为GPU在该机器上的编号. 例如图例中的进程9, 其所在机器上有8块卡. 因此进程9使用的gpu编号为9%8=1
            if args.local_rank is not None:
                assert (
                    args.local_rank == device
                ), "expected local-rank to be the same as rank % device-count."
            else:
                args.local_rank = device

            # args.force_device: None
            if args.force_device is not None:
                print(
                    f"  > forcefully set the device to {args.force_device}, originally {device}"
                )
                device = args.force_device
            torch.cuda.set_device(device)
            # 为当前进程分配GPU

        # Call the init process
        # 设置进程大组
        init_method = "tcp://"
        master_ip = os.getenv("MASTER_ADDR", "localhost")
        # 获取进程的ip
        # master_ip: 127.0.0.1
        master_port = os.getenv("MASTER_PORT", "6000")
        # 获取进程的端口号
        # master_port: 29501
        init_method += master_ip + ":" + master_port
        # init_method: 'tcp://127.0.0.1:29501'
        print(
            f"  > (rank={args.rank}) initializing process group: "
            f"world_size={args.world_size} "
            f"backend={args.distributed_backend} "
            f"init_method={init_method}",
            flush=True,
        )
        #   > (rank=7) initializing process group: world_size=8 backend=nccl init_method=tcp://127.0.0.1:29501
        timeout = datetime.timedelta(minutes=args.dist_timeout)
        # args.dist_timeout: 30
        torch.distributed.init_process_group(
            backend=args.distributed_backend,
            # distributed_backend: 'nccl'
            world_size=args.world_size,
            # world_size: 8
            rank=args.rank,
            init_method=init_method,
            # init_method: 'tcp://127.0.0.1:29501'
            timeout=timeout
            # timeout: 30
        )
        print(f"  > (rank={args.rank}) process group initialized")

    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    # 2、制定DP/TP/PP分组策略, 设置进程子组（subgroup）
    if device_count > 0:
        # mpu.model_parallel_is_initialized(): False
        if mpu.model_parallel_is_initialized():
            print("model parallel is already initialized")
        else:
            mpu.initialize_model_parallel(
                args.tensor_model_parallel_size,
                # args.tensor_model_parallel_size: 4
                args.pipeline_model_parallel_size,
                # args.pipeline_model_parallel_size: 1
                args.virtual_pipeline_model_parallel_size,
                # args.virtual_pipeline_model_parallel_size: None
            )

    # 设置DeepSpeed ZeRO-R, 对activation进行优化
    # args.deepspeed: True
    # args.deepspeed_activation_checkpointing: True
    if args.deepspeed and args.deepspeed_activation_checkpointing:
        setup_deepspeed_random_and_activation_checkpointing(args)


def _init_autoresume():
    """Set autoresume start time."""
    autoresume = get_adlr_autoresume()
    if autoresume:
        torch.distributed.barrier()
        autoresume.init()
        torch.distributed.barrier()


def _set_random_seed(seed_):
    """Set random seed for reproducability."""
    # seed_: 1234
    if seed_ is not None and seed_ > 0:
        # Ensure that different pipeline MP stages get different seeds.
        # 如果不使用流水线并行, 那么所有进程的随机数种子一样
        # 如果使用流水线并行, 那么同一流水线并行组组内各进程的随机数种子不同, 组间同对应位置的进程 随机数种子相同
        seed = seed_ + (100 * mpu.get_pipeline_model_parallel_rank())
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.device_count() > 0:
            mpu.model_parallel_cuda_manual_seed(seed)
        # 为CPU设置随机数种子: torch.manual_seed(seed)
        # 为特定GPU设置随机数种子: torch.cuda.manual_seed(seed)
    else:
        raise ValueError("Seed ({}) should be a positive integer.".format(seed))


def write_args_to_tensorboard():
    """Write arguments to tensorboard."""
    args = get_args()
    writer = get_tensorboard_writer()
    if writer:
        for arg in vars(args):
            writer.add_text(arg, str(getattr(args, arg)), global_step=args.iteration)


def initialize_wandb_experiment():
    """Initialize wandb experiment."""
    assert wandb is not None, "Fail to import wandb"

    args = get_args()
    config = args.__dict__

    wandb_id_path = os.path.join(args.save, "wandb_id.txt")
    if os.path.exists(wandb_id_path):
        wandb_id = open(wandb_id_path, "r").read().strip()
    else:
        wandb_id = wandb.util.generate_id()
        open(wandb_id_path, "w").write(wandb_id)

    wandb.init(id=wandb_id, project="megatron", config=config, resume="allow")


def _initialize_mem_buffs():
    """Initialize manually allocated static memory."""
    args = get_args()

    # Initialize memory for checkpointed activations.
    if args.distribute_checkpointed_activations:
        mpu.init_checkpointed_activations_memory_buffer()
