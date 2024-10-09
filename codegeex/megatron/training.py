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

"""Pretrain utilities."""
import os
from datetime import datetime
import math
import sys
import time
import json

# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()

import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from codegeex.megatron import get_args
from codegeex.megatron import get_timers
from codegeex.megatron import get_tensorboard_writer
from codegeex.megatron import get_current_global_batch_size
from codegeex.megatron import get_num_microbatches
from codegeex.megatron import is_last_rank
from codegeex.megatron import update_num_microbatches
from codegeex.megatron import mpu
from codegeex.megatron import print_rank_0
from codegeex.megatron import print_rank_last
from codegeex.megatron.checkpointing import load_checkpoint
from codegeex.megatron.checkpointing import save_checkpoint
from codegeex.megatron.model import Float16Module
from codegeex.megatron.optimizer import get_megatron_optimizer
from codegeex.megatron.initialize import initialize_megatron
from codegeex.megatron.initialize import write_args_to_tensorboard
from codegeex.megatron.initialize import initialize_wandb_experiment
from codegeex.megatron.learning_rates import AnnealingLR
from codegeex.megatron.model import DistributedDataParallel as LocalDDP
from codegeex.megatron.utils import check_adlr_autoresume_termination
from codegeex.megatron.utils import unwrap_model
from codegeex.megatron.data.data_samplers import build_pretraining_data_loader
from codegeex.megatron.utils import calc_params_l2_norm
from codegeex.megatron.schedules import forward_backward_no_pipelining
from codegeex.megatron.schedules import forward_backward_pipelining_without_interleaving
from codegeex.megatron.schedules import forward_backward_pipelining_with_interleaving
from codegeex.megatron.utils import report_memory, flops_calculator

import deepspeed
from deepspeed.runtime.utils import see_memory_usage

try:
    import wandb
except ImportError:
    wandb = None

from filelock import FileLock
import pathlib



def print_datetime(string):
    """Note that this call will sync across all ranks."""
    torch.distributed.barrier()
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print_rank_0("[" + string + "] datetime: {} ".format(time_str))


def pretrain(
    train_valid_test_dataset_provider,
    model_provider,
    forward_step_func,
    valid_forward_step_func=None,
    extra_args_provider=None,
    args_defaults={},
):
    """Main training program.

    This function will run the followings in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule using the model_provider.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the modle using the forward_step_func.

    Arguments:
        train_valid_test_dataset_provider: a function that takes the size of
            train/valid/test dataset and returns `train, valid, test` datasets.
        model_provider: a function that returns a vanilla version of the
            model. By vanilla we mean a simple model on cpu with no fp16 or ddp.
        forward_step_func: a function that takes a `data iterator` and `model`,
            and returns a `loss` scalar with a dictionary with key:values being
            the info we would like to monitor during training, for example
            `lm-loss: value`. We also require that this function add
            `batch generator` to the timers class.
        extra_args_provider: a function that takes a parser and adds arguments
            to it. It is used for programs to add their own arguments.
        args_defaults: a dictionary from argument-name to argument-value. It
            to set already parse arguments.
    """

    # valid_forward_step_func: None
    # extra_args_provider: None
    # args_defaults: {"tokenizer_type": "GPT2BPETokenizer"}

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(
        extra_args_provider=extra_args_provider, args_defaults=args_defaults
    )

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    # 在程序一开始运行时就使用 time.time() 对 _TRAIN_START_TIME 进行了初始化
    # 然后从各个子进程的 _TRAIN_START_TIME 变量中找一个最小值
    # 进而拿现在时间减去起始时间得到 megatron 初始化所用时长
    global _TRAIN_START_TIME
    start_time_tensor = torch.cuda.FloatTensor([_TRAIN_START_TIME])
    torch.distributed.all_reduce(start_time_tensor, op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()
    print_rank_0(
        "time to initialize megatron (seconds): {:.3f}".format(
            time.time() - _TRAIN_START_TIME
        )
    )
    print_datetime("after megatron is initialized")
    # 打印出如此样式的信息: "[after megatron is initialized] datetime: 2023-10-29 17:26:19"

    args = get_args()
    timers = get_timers()
    # 在函数 initialize_megatron 中对 _GLOBAL_ARGS、_GLOBAL_TOKENIZER、_GLOBAL_TIMERS 等全局变量完成了初始化
    # 后续只要不重新进行这些参数的初始化, 就能通过 get_args(), get_tokenizer(), get_timers() 等分别获取这些已完成初始化的变量

    # args.save: "/home/icksys/csw/CodeGeeX/scripts/pretrain-codegeex-13b-test"
    if args.local_rank == 0 and args.save is not None:
        print(f"Creating output dir ...")
        os.makedirs(args.save, exist_ok=True)

    # args.deepspeed: True
    if args.deepspeed:
        args.deepspeed_configuration = json.load(
            open(args.deepspeed_config, "r", encoding="utf-8")
        )
        # args.deepspeed_config: "/home/icksys/csw/CodeGeeX/scripts/ds_config.json"

    # Model, optimizer, and learning rate.
    timers("model-and-optimizer-setup").start()
    # 在 setup_model_and_optimizer 函数中完成了模型结构定义和权重加载
    model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider)
    timers("model-and-optimizer-setup").stop()
    print_datetime("after model, optimizer, and learning rate " "scheduler are built")

    # Data stuff.
    timers("train/valid/test-data-iterators-setup").start()
    # args.virtual_pipeline_model_parallel_size: None
    if args.virtual_pipeline_model_parallel_size is not None:
        all_data_iterators = [
            build_train_valid_test_data_iterators(train_valid_test_dataset_provider)
            for _ in range(len(model))
        ]
        train_data_iterator = [
            data_iterators[0] for data_iterators in all_data_iterators
        ]
        valid_data_iterator = [
            data_iterators[1] for data_iterators in all_data_iterators
        ]
        test_data_iterator = [
            data_iterators[2] for data_iterators in all_data_iterators
        ]
    else:
        (
            train_data_iterator,
            valid_data_iterator,
            test_data_iterator,
        ) = build_train_valid_test_data_iterators(train_valid_test_dataset_provider)
    timers("train/valid/test-data-iterators-setup").stop()
    print_datetime("after dataloaders are built")
    # 打印出如此样式的信息: "[after dataloaders are built] datetime: 2023-10-29 17:27:04"

    # Print setup timing.
    print_rank_0("done with setup ...")
    timers.log(["model-and-optimizer-setup", "train/valid/test-data-iterators-setup"])
    # 打印出如此样式的信息: "time (ms) | model-and-optimizer-setup: 44048.85 | train/valid/test-data-iterators-setup: 1007.76"
    print_rank_0("training ...")

    iteration = 0
    # args.do_train: True
    # args.train_iters: 25
    if args.do_train and args.train_iters > 0:
        iteration = train(
            forward_step_func,
            valid_forward_step_func,
            model,
            optimizer,
            lr_scheduler,
            train_data_iterator,
            valid_data_iterator,
        )
    print_datetime("after training is done")

    if args.do_valid:
        prefix = "the end of training for val data"
        if args.co_evaluation:
            for key, value in valid_data_iterator.items():
                evaluate_and_print_results(
                    prefix, valid_forward_step_func, value, model, iteration, False, tag=key
                )
        else:
            evaluate_and_print_results(
                prefix, valid_forward_step_func, valid_data_iterator, model, iteration, False
            )

    # args.save: "/home/icksys/csw/CodeGeeX/scripts/pretrain-codegeex-13b-test"
    if args.save and iteration != 0:
        save_checkpoint(iteration, model, optimizer, lr_scheduler)

    if args.do_test:
        # Run on test data.
        prefix = "the end of training for test data"
        if args.co_evaluation:
            for key, value in test_data_iterator.items():
                evaluate_and_print_results(
                    prefix, forward_step_func, value, model, 0, True, tag=key
                )
        else:
            evaluate_and_print_results(
                prefix, forward_step_func, test_data_iterator, model, 0, True
            )

    if args.wandb_logging and is_last_rank():
        wandb.finish()


def update_train_iters(args):

    # For iteration-based training, we don't need to do anything
    if args.train_iters:
        return

    # Constant batch size with sample-based training.
    if args.rampup_batch_size is None:
        args.train_iters = args.train_samples // args.global_batch_size

    else:
        # Sample based training with rampup batch size.
        iterations = 0
        consumed_samples = 0
        # Rampup phase.
        while consumed_samples <= int(args.rampup_batch_size[2]):
            update_num_microbatches(consumed_samples, consistency_check=False)
            consumed_samples += get_current_global_batch_size()
            iterations += 1
        # Reset
        update_num_microbatches(0, consistency_check=False)
        # Constant phase
        # Note that we throw away any partial last batch.
        iterations += (args.train_samples - consumed_samples) // args.global_batch_size
        args.train_iters = iterations

    print_rank_0("setting training iterations to {}".format(args.train_iters))


def get_model(model_provider_func):
    """Build the model."""
    args = get_args()

    # Build model.
    # mpu.get_pipeline_model_parallel_world_size(): 1, 获取当前进程所属 Pipeline Model Parallel Group 的成员数量
    if (
        mpu.get_pipeline_model_parallel_world_size() > 1
        and args.virtual_pipeline_model_parallel_size is not None
    ):
        # 当分布式框架采用 virtual pipeline (是 NVDIA 后续提出的对 Megatron 的优化方法, 可先忽略不看)
        model = []
        for i in range(args.virtual_pipeline_model_parallel_size):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            # Set pre_process and post_process only after virtual rank is set.
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            this_model = model_provider_func(
                pre_process=pre_process, post_process=post_process
            )
            model.append(this_model)
    else:
        # This way
        # 判断当前进程是否是 PP 组的第一个进程
        # mpu.is_pipeline_first_stage(): True
        pre_process = mpu.is_pipeline_first_stage()
        # 判断当前进程是否是 PP 组的最后一个进程
        # mpu.is_pipeline_last_stage(): True
        post_process = mpu.is_pipeline_last_stage()
        # 如果 PP 组 size 为 1, 即每个进程既是第一个进程也是最后一个进程, 则 pre_process 和 post_process 都是 True
        # 构建 CodeGeeX 模型
        # 如果使用的是 CPU 版权重初始化方法, 那么每个进程模型子块 (model) 的所有权重都在内存中
        # 如果使用的是 GPU 版权重初始化方法, 那么每个进程模型子块 (model) 的绝大部分权重在 GPU 上, 少量权重在内存上, 因此后面将模型整体挪到 GPU 上时才会有显存占用的少量增加
        model = model_provider_func(pre_process=pre_process, post_process=post_process)

    if not isinstance(model, list):
        model = [model]
    # model: [ CodeGeeXModel(...) ]

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            mpu.set_defaults_if_not_set_tensor_model_parallel_attributes(param)
            # 为模型中所有未含如下属性的参数设置指定值
            # "tensor_model_parallel": False,
            # "partition_dim": -1,
            # "partition_stride": 1,

    # Print number of parameters.
    # mpu.get_data_parallel_rank(): Return local rank for the data parallel group.
    if mpu.get_data_parallel_rank() == 0:
        print(
            " > number of parameters on (tensor, pipeline) "
            "model parallel rank ({}, {}): {}".format(
                mpu.get_tensor_model_parallel_rank(),
                mpu.get_pipeline_model_parallel_rank(),
                sum(
                    [
                        sum(
                            [
                                p.ds_numel if hasattr(p, "ds_id") else p.nelement()
                                for p in model_module.parameters()
                            ]
                        )
                        for model_module in model
                    ]
                ),
            ),
            flush=True,
        )

    # 如果采用 Megatron-DeepSpeed 的方式, 则直接返回模型, 后面的搬运, 数据并行等工作将由 deepspeed 来完成
    # ref: https://www.deepspeed.ai/tutorials/megatron/
    # args.deepspeed: True
    if args.deepspeed:
        return model

    # GPU allocation.
    # 将当前进程所维护的模型子块, 从 CPU 搬运到 GPU 上（GPU 即为在初始化时为当前进程分配的那块 GPU）
    # 这步过后显存占用略增一点点, 因为少量模型权重从内存移到了显存中

    # [2024-03-12 21:51:08,535] [INFO] [utils.py:828:see_memory_usage] Before moving to GPU
    # [2024-03-12 21:51:08,536] [INFO] [utils.py:829:see_memory_usage] MA 5.99 GB         Max_MA 5.99 GB         CA 6.23 GB         Max_CA 6 GB
    # [2024-03-12 21:51:08,536] [INFO] [utils.py:837:see_memory_usage] CPU Virtual Memory:  used = 93.11 GB, percent = 5.0%
    see_memory_usage(f"Before moving to GPU", force=True)
    print(f" > moving model to GPU ...", flush=True)
    for model_module in model:
        model_module.cuda(torch.cuda.current_device())
    print(f" > moving to GPU done", flush=True)
    see_memory_usage(f"After moving to GPU", force=True)
    # [2024-03-12 21:51:08,654] [INFO] [utils.py:828:see_memory_usage] After moving to GPU
    # [2024-03-12 21:51:08,655] [INFO] [utils.py:829:see_memory_usage] MA 6.01 GB         Max_MA 6.01 GB         CA 6.25 GB         Max_CA 6 GB
    # [2024-03-12 21:51:08,655] [INFO] [utils.py:837:see_memory_usage] CPU Virtual Memory:  used = 93.09 GB, percent = 5.0%

    # Float16Module(...) 封装前
    # Embedding 中 word_embeddings: torch.float16
    # Embedding 中 position_embeddings: torch.float16
    # QueryEmbedding 中 top_query_embeddings: torch.float16
    # ParallelTransformerLayer 中 input_layernorm: torch.float32
    # ParallelTransformerLayer 中 post_attention_layernorm: torch.float32
    # ParallelSelfAttention 中 query: torch.float16
    # ParallelSelfAttention 中 dense: torch.float16
    # ParallelMLP 中 dense_h_to_4h: torch.float16
    # ParallelTransformer 中 final_layernorm: torch.float32

    # Fp16 conversion.
    # fp16 转换（pytorch 默认模型参数精度为 fp32, 依需决定计算过程中是否要转成 fp16, 节省显存）
    # args.fp16: True

    # [2024-03-12 21:51:08,654] [INFO] [utils.py:828:see_memory_usage] Before Float16Module Warpped
    # [2024-03-12 21:51:08,655] [INFO] [utils.py:829:see_memory_usage] MA 6.01 GB         Max_MA 6.01 GB         CA 6.25 GB         Max_CA 6 GB
    # [2024-03-12 21:51:08,655] [INFO] [utils.py:837:see_memory_usage] CPU Virtual Memory:  used = 93.09 GB, percent = 5.0%
    see_memory_usage(f"Before Float16Module Warpped", force=True)
    if args.fp16 or args.bf16:
        print(f" > converting model to fp16 ...", flush=True)
        model = [Float16Module(model_module, args) for model_module in model]
        # model: [Float16Module(CodeGeeXModel(...))]
        # Float16Module(...) 封装后对 CodeGeeXModel(...) 的前向计算没影响
        print(f" > converting to fp16 done", flush=True)
    see_memory_usage(f"After Float16Module Warpped", force=True)
    # [2024-03-12 21:51:08,771] [INFO] [utils.py:828:see_memory_usage] After Float16Module Warpped
    # [2024-03-12 21:51:08,772] [INFO] [utils.py:829:see_memory_usage] MA 6.01 GB         Max_MA 6.01 GB         CA 6.25 GB         Max_CA 6 GB
    # [2024-03-12 21:51:08,772] [INFO] [utils.py:837:see_memory_usage] CPU Virtual Memory:  used = 93.02 GB, percent = 5.0%

    # Float16Module(...) 封装后 LayerNorm 相关层从 fp32 变成了 fp16
    # Embedding 中 word_embeddings: torch.float16
    # Embedding 中 position_embeddings: torch.float16
    # QueryEmbedding 中 top_query_embeddings: torch.float16
    # ParallelTransformerLayer 中 input_layernorm: torch.float16
    # ParallelTransformerLayer 中 post_attention_layernorm: torch.float16
    # ParallelSelfAttention 中 query: torch.float16
    # ParallelSelfAttention 中 dense: torch.float16
    # ParallelMLP 中 dense_h_to_4h: torch.float16
    # ParallelTransformer 中 final_layernorm: torch.float16

    # 采用 pytorch 定义的 DistributedDataParallel 管理数据并行
    # args.DDP_impl: 'local'
    if args.DDP_impl == "torch":
        i = torch.cuda.current_device()
        model = [
            torchDDP(
                model_module,
                device_ids=[i],
                output_device=i,
                process_group=mpu.get_data_parallel_group(),
            )
            for model_module in model
        ]
        # model: [ torchDDP( Float16Module( CodeGeeXModel(...) ) ) ]
        return model

    # 采用自定义的 DistributedDataParallel 管理数据并行
    if args.DDP_impl == "local":
        # 自定义的数据并行类在 megatron/model/distributed.py 下
        print(f" > creating DDP model ...", flush=True)
        see_memory_usage(f"Before LocalDDP Wrapped", force=True)
        # [2024-03-12 21:51:08,771] [INFO] [utils.py:828:see_memory_usage] Before LocalDDP Wrapped
        # [2024-03-12 21:51:08,772] [INFO] [utils.py:829:see_memory_usage] MA 6.01 GB         Max_MA 6.01 GB         CA 6.25 GB         Max_CA 6 GB
        # [2024-03-12 21:51:08,772] [INFO] [utils.py:837:see_memory_usage] CPU Virtual Memory:  used = 93.02 GB, percent = 5.0%
        model = [
            LocalDDP(
                model_module,
                args.accumulate_allreduce_grads_in_fp32,
                # args.accumulate_allreduce_grads_in_fp32: True
                args.use_contiguous_buffers_in_ddp,
                # args.use_contiguous_buffers_in_ddp: True
            )
            for model_module in model
        ]
        # model: [ LocalDDP ( Float16Module( CodeGeeXModel(...) ) ) ]
        print(f" > creating DDP model done", flush=True)
        see_memory_usage(f"After LocalDDP Wrapped", force=True)
        # [2024-03-17 13:48:05,678] [INFO] [utils.py:828:see_memory_usage] After LocalDDP Wrapped
        # [2024-03-17 13:48:05,678] [INFO] [utils.py:829:see_memory_usage] MA 18.04 GB         Max_MA 18.04 GB         CA 18.27 GB         Max_CA 18 GB
        # [2024-03-17 13:48:05,679] [INFO] [utils.py:837:see_memory_usage] CPU Virtual Memory:  used = 93.03 GB, percent = 5.0%

        # LocalDDP 中初始化了一个两倍于模型子块显存大小的缓存空间, 用于存放每个 global_batch_size 训练时的梯度累积信息
        # fp16 的模型子块占用显存 6.01G, 而该梯度缓存空间占用 12.02G, 正好就是 LocalDDP 初始化后显存增加的这部分
        return model

    raise NotImplementedError(
        "Unknown DDP implementation specified: {}. " "Exiting.".format(args.DDP_impl)
    )


def get_learning_rate_scheduler(optimizer):
    """Build the learning rate scheduler."""
    args = get_args()

    # Iteration-based training.
    if args.train_iters:
        if args.lr_decay_iters is None:
            args.lr_decay_iters = args.train_iters
        decay_steps = args.lr_decay_iters * args.global_batch_size
        if args.lr_warmup_fraction is not None:
            warmup_steps = args.lr_warmup_fraction * decay_steps
        else:
            warmup_steps = args.lr_warmup_iters * args.global_batch_size
    # Sample-based training.
    elif args.train_samples:
        # We need to set training iters for later use. Technically
        # we need to adjust the training samples too (due to last
        # batch being incomplete) but we leave it as is for now.
        update_train_iters(args)
        if args.lr_decay_samples is None:
            args.lr_decay_samples = args.train_samples
        decay_steps = args.lr_decay_samples
        if args.lr_warmup_fraction is not None:
            warmup_steps = args.lr_warmup_fraction * decay_steps
        else:
            warmup_steps = args.lr_warmup_samples
    else:
        raise Exception("either train-iters or train-samples should be provided.")

    lr_scheduler = AnnealingLR(
        optimizer,
        max_lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        decay_style=args.lr_decay_style,
        use_checkpoint_lr_scheduler=args.use_checkpoint_lr_scheduler,
        override_lr_scheduler=args.override_lr_scheduler,
    )

    return lr_scheduler


def setup_model_and_optimizer(model_provider_func):
    """Setup model and optimizer."""
    args = get_args()

    model = get_model(model_provider_func)
    # 如果训练脚本不使用 deepspeed, 那么 model: [ LocalDDP ( Float16Module( CodeGeeXModel(...) ) ) ], 即
    # [DistributedDataParallel(
    #   (module): Float16Module(
    #     (module): CodeGeeXModel(
    #       (language_model): TransformerLanguageModel(
    #         (embedding): Embedding(
    #           (word_embeddings): VocabParallelEmbedding()
    #           (position_embeddings): Embedding(2048, 5120)
    #           (embedding_dropout): Dropout(p=0.1, inplace=False)
    #         )
    #         (topQueryEmbedding): QueryEmbedding(
    #           (top_query_embeddings): VocabParallelEmbedding()
    #           (embedding_dropout): Dropout(p=0.1, inplace=False)
    #         )
    #         (transformer): ParallelTransformer(
    #           (layers): ModuleList(
    #             (0-38): 39 x ParallelTransformerLayer(
    #               (input_layernorm): LayerNorm((5120,), eps=1e-05, elementwise_affine=True)
    #               (attention): ParallelSelfAttention(
    #                 (query): ColumnParallelLinear()
    #                 (key): ColumnParallelLinear()
    #                 (value): ColumnParallelLinear()
    #                 (softmax): Softmax(dim=-1)
    #                 (attention_dropout): Dropout(p=0.1, inplace=False)
    #                 (dense): RowParallelLinear()
    #               )
    #               (post_attention_layernorm): LayerNorm((5120,), eps=1e-05, elementwise_affine=True)
    #               (mlp): ParallelMLP(
    #                 (dense_h_to_4h): ColumnParallelLinear()
    #                 (dense_4h_to_h): RowParallelLinear()
    #               )
    #             )
    #           )
    #           (topQueryLayer): ParallelTopQueryLayer(
    #             (input_layernorm): LayerNorm((5120,), eps=1e-05, elementwise_affine=True)
    #             (attention): ParallelTopQuerySelfAttention(
    #               (query): ColumnParallelLinear()
    #               (key): ColumnParallelLinear()
    #               (value): ColumnParallelLinear()
    #               (softmax): Softmax(dim=-1)
    #               (attention_dropout): Dropout(p=0.1, inplace=False)
    #               (dense): RowParallelLinear()
    #             )
    #             (post_attention_layernorm): LayerNorm((5120,), eps=1e-05, elementwise_affine=True)
    #             (mlp): ParallelMLP(
    #               (dense_h_to_4h): ColumnParallelLinear()
    #               (dense_4h_to_h): RowParallelLinear()
    #             )
    #           )
    #           (final_layernorm): LayerNorm((5120,), eps=1e-05, elementwise_affine=True)
    #         )
    #       )
    #     )
    #   )
    # )]

    # 但因为当前训练脚本使用了 deepspeed, 所以 model: [ CodeGeeXModel(...) ], 即
    # [CodeGeeXModel(
    #   (language_model): TransformerLanguageModel(
    #     (embedding): Embedding(
    #       (word_embeddings): VocabParallelEmbedding()
    #       (position_embeddings): Embedding(2048, 5120)
    #       (embedding_dropout): Dropout(p=0.1, inplace=False)
    #     )
    #     (topQueryEmbedding): QueryEmbedding(
    #       (top_query_embeddings): VocabParallelEmbedding()
    #       (embedding_dropout): Dropout(p=0.1, inplace=False)
    #     )
    #     (transformer): ParallelTransformer(
    #       (layers): ModuleList(
    #         (0-38): 39 x ParallelTransformerLayer(
    #           (input_layernorm): LayerNorm((5120,), eps=1e-05, elementwise_affine=True)
    #           (attention): ParallelSelfAttention(
    #             (query): ColumnParallelLinear()
    #             (key): ColumnParallelLinear()
    #             (value): ColumnParallelLinear()
    #             (softmax): Softmax(dim=-1)
    #             (attention_dropout): Dropout(p=0.1, inplace=False)
    #             (dense): RowParallelLinear()
    #           )
    #           (post_attention_layernorm): LayerNorm((5120,), eps=1e-05, elementwise_affine=True)
    #           (mlp): ParallelMLP(
    #             (dense_h_to_4h): ColumnParallelLinear()
    #             (dense_4h_to_h): RowParallelLinear()
    #           )
    #         )
    #       )
    #       (topQueryLayer): ParallelTopQueryLayer(
    #         (input_layernorm): LayerNorm((5120,), eps=1e-05, elementwise_affine=True)
    #         (attention): ParallelTopQuerySelfAttention(
    #           (query): ColumnParallelLinear()
    #           (key): ColumnParallelLinear()
    #           (value): ColumnParallelLinear()
    #           (softmax): Softmax(dim=-1)
    #           (attention_dropout): Dropout(p=0.1, inplace=False)
    #           (dense): RowParallelLinear()
    #         )
    #         (post_attention_layernorm): LayerNorm((5120,), eps=1e-05, elementwise_affine=True)
    #         (mlp): ParallelMLP(
    #           (dense_h_to_4h): ColumnParallelLinear()
    #           (dense_4h_to_h): RowParallelLinear()
    #         )
    #       )
    #       (final_layernorm): LayerNorm((5120,), eps=1e-05, elementwise_affine=True)
    #     )
    #   )
    # )]

    unwrapped_model = unwrap_model(model, (torchDDP, LocalDDP, Float16Module))
    # 如果 model 是被 torchDDP, LocalDDP, Float16Module 等对象封装着, 那么 unwrap_model 就会帮 model 脱去包装;
    # 如果 model 没被 torchDDP, LocalDDP, Float16Module 等对象封装着, 那么就无事发生
    # 如果训练脚本不使用 deepspeed, 那么 unwrapped_model: [ CodeGeeXModel(...) ]
    # 但因为当前训练脚本使用了 deepspeed, 所以 unwrapped_model: [ CodeGeeXModel(...) ]

    optimizer = get_megatron_optimizer(unwrapped_model)
    see_memory_usage(f"After Float16OptimizerWithFloat16Params Wrapped", force=True)
    # [2024-03-17 14:49:05,184] [INFO] [utils.py:828:see_memory_usage] After Float16OptimizerWithFloat16Params Wrapped
    # [2024-03-17 14:49:05,185] [INFO] [utils.py:829:see_memory_usage] MA 30.14 GB         Max_MA 30.18 GB         CA 30.45 GB         Max_CA 30 GB
    # [2024-03-17 14:49:05,185] [INFO] [utils.py:837:see_memory_usage] CPU Virtual Memory:  used = 75.59 GB, percent = 4.0%
    # Float16OptimizerWithFloat16Params 不再指向 fp16 的模型子块, 而是根据 fp16 的模型子块复制得到了一份 fp32 的高精度模型权重
    # 这个 fp32 的权重占用 12.02G 左右的空间, 正好约为显存增量值

    # 如果训练脚本不使用 deepspeed, 那么 optimizer: Float16OptimizerWithFloat16Params( FusedAdam(...) )
    # 但因为当前训练脚本使用了 deepspeed, 所以 optimizer: FusedAdam(...)

    lr_scheduler = get_learning_rate_scheduler(optimizer)

    if args.deepspeed:
        print_rank_0("DeepSpeed is enabled.")
        # mpu.get_pipeline_model_parallel_world_size(): 1
        pp = mpu.get_pipeline_model_parallel_world_size()
        print_rank_0(pp)

        # args.no_pipeline_parallel: True
        # Initialize the DeepSpeed Engine
        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model[0],
            optimizer=optimizer,
            args=args,
            lr_scheduler=lr_scheduler,
            mpu=mpu if args.no_pipeline_parallel else None,
        )
        # model: DeepSpeedEngine(CodeGeeXModel(...)), 即
        # DeepSpeedEngine(
        #   (module): CodeGeeXModel(
        #     (language_model): TransformerLanguageModel(
        #       (embedding): Embedding(
        #         (word_embeddings): VocabParallelEmbedding()
        #         (position_embeddings): Embedding(2048, 5120)
        #         (embedding_dropout): Dropout(p=0.1, inplace=False)
        #       )
        #       (topQueryEmbedding): QueryEmbedding(
        #         (top_query_embeddings): VocabParallelEmbedding()
        #         (embedding_dropout): Dropout(p=0.1, inplace=False)
        #       )
        #       (transformer): ParallelTransformer(
        #         (layers): ModuleList(
        #           (0-38): 39 x ParallelTransformerLayer(
        #             (input_layernorm): LayerNorm((5120,), eps=1e-05, elementwise_affine=True)
        #             (attention): ParallelSelfAttention(
        #               (query): ColumnParallelLinear()
        #               (key): ColumnParallelLinear()
        #               (value): ColumnParallelLinear()
        #               (softmax): Softmax(dim=-1)
        #               (attention_dropout): Dropout(p=0.1, inplace=False)
        #               (dense): RowParallelLinear()
        #             )
        #             (post_attention_layernorm): LayerNorm((5120,), eps=1e-05, elementwise_affine=True)
        #             (mlp): ParallelMLP(
        #               (dense_h_to_4h): ColumnParallelLinear()
        #               (dense_4h_to_h): RowParallelLinear()
        #             )
        #           )
        #         )
        #         (topQueryLayer): ParallelTopQueryLayer(
        #           (input_layernorm): LayerNorm((5120,), eps=1e-05, elementwise_affine=True)
        #           (attention): ParallelTopQuerySelfAttention(
        #             (query): ColumnParallelLinear()
        #             (key): ColumnParallelLinear()
        #             (value): ColumnParallelLinear()
        #             (softmax): Softmax(dim=-1)
        #             (attention_dropout): Dropout(p=0.1, inplace=False)
        #             (dense): RowParallelLinear()
        #           )
        #           (post_attention_layernorm): LayerNorm((5120,), eps=1e-05, elementwise_affine=True)
        #           (mlp): ParallelMLP(
        #             (dense_h_to_4h): ColumnParallelLinear()
        #             (dense_4h_to_h): RowParallelLinear()
        #           )
        #         )
        #         (final_layernorm): LayerNorm((5120,), eps=1e-05, elementwise_affine=True)
        #       )
        #     )
        #   )
        # )
        print_rank_0("FinishInitialization.")
        # isinstance(model, deepspeed.PipelineEngine): False
        if isinstance(model, deepspeed.PipelineEngine):
            # hack to get batch_fn from pretrain_gpt.py
            print_rank_0("InstancePipelineEngine.")
            model.set_batch_fn(model.module._megatron_batch_fn)

            assert (
                model.grid.get_pipe_parallel_rank()
                == mpu.get_pipeline_model_parallel_rank()
            )
            assert (
                model.grid.get_slice_parallel_rank()
                == mpu.get_tensor_model_parallel_rank()
            )
            assert model.grid.get_data_parallel_rank() == mpu.get_data_parallel_rank()
        model = [model]
        print_rank_0("Finishparallel.")

    # args.load: None
    # args.load 是加载模型权重、优化器状态和 Scheduler 状态
    # args.load_state 则是只加载模型权重
    if args.load is not None:
        timers = get_timers()
        # Extra barrier is added to make sure all ranks report the
        # max time.
        torch.distributed.barrier()
        timers("load-checkpoint").start()
        # args.low_memory_load: None
        if args.low_memory_load:
            load_start = time.perf_counter()
            with FileLock(os.path.join(pathlib.Path.home(), "checkpoint_lock"), timeout=-1):
                this_rank_load_start = time.perf_counter()
                print(f"Rank {args.rank} is loading checkpoint ...")
                args.iteration = load_checkpoint(model, optimizer, lr_scheduler)
                this_rank_load_time = time.perf_counter() - this_rank_load_start
                load_time = time.perf_counter() - load_start
                print(f"Rank {args.rank} loaded checkpoint, this rank time: {this_rank_load_time}, total time: {load_time}")
        else:
            args.iteration = load_checkpoint(model, optimizer, lr_scheduler)
        print(f"Rank {args.rank} loaded checkpoint and waiting for other ranks")
        torch.distributed.barrier()
        timers("load-checkpoint").stop()
        timers.log(["load-checkpoint"])
    else:
        args.iteration = 0

    # We only support local DDP with multiple micro-batches.
    # len(model): 1
    # mpu.get_pipeline_model_parallel_world_size(): 1
    if len(model) > 1 or mpu.get_pipeline_model_parallel_world_size() > 1:
        assert args.DDP_impl == "local"

    # get model without FP16 and/or TorchDDP wrappers
    if (
        args.iteration == 0
        and len(unwrapped_model) == 1
        # unwrapped_model: [ CodeGeeXModel(...) ]
        and hasattr(unwrapped_model[0], "init_state_dict_from_bert")
        # hasattr(unwrapped_model[0], "init_state_dict_from_bert"): False
    ):
        print_rank_0("Initializing ICT from pretrained BERT model")
        unwrapped_model[0].init_state_dict_from_bert()
        if args.fp16:
            optimizer.reload_model_params()

    return model, optimizer, lr_scheduler


def train_step(forward_step_func, data_iterator, model, optimizer, lr_scheduler):
    """Single training step."""
    # train_step: 在一个 global_batch_size 的数据上进行迭代训练
    args = get_args()
    timers = get_timers()

    # 如果训练脚本不使用 deepspeed, 那么
    #       model: [ LocalDDP ( Float16Module( CodeGeeXModel(...) ) ) ]
    #       optimizer: Float16OptimizerWithFloat16Params( FusedAdam(...) )
    # 但因为当前训练脚本使用了 deepspeed, 所以
    #       model: [ DeepSpeedEngine( CodeGeeXModel(...) ) ]
    #       optimizer: FusedAdam(...)

    # args.deepspeed: True
    # args.ds_pipeline_enabled: False
    if args.deepspeed and args.ds_pipeline_enabled:
        skipped_iter = 0
        num_zeros_in_grad = 0
        assert isinstance(model[0], deepspeed.PipelineEngine)
        loss = model[0].train_batch(data_iter=data_iterator)
        grad_norm = model[0].get_global_grad_norm()
        return {"lm loss": loss}, skipped_iter, grad_norm, num_zeros_in_grad

    # Set grad to zero.
    if not args.deepspeed:
        # args.DDP_impl: "local"
        # args.use_contiguous_buffers_in_ddp: True
        if args.DDP_impl == "local" and args.use_contiguous_buffers_in_ddp:
            # 将缓存区的梯度归零, 但不释放缓存
            report_memory("(iterations {} before 【model_param.main_grad = 0】)".format(args.iteration), True)
            for partition in model:
                # model_param.grad 不用清零吗 (在训练时 model_param.main_grad += model_param.grad; model_param.grad = None  所以 model_param.grad 其实是清零了的)
                # 即 model_param.main_grad = 0
                partition.zero_grad_buffer()
            report_memory("(iterations {} after  【model_param.main_grad = 0】)".format(args.iteration), True)
        else:
            optimizer.zero_grad()

    # mpu.get_pipeline_model_parallel_world_size(): 1, 获取当前进程所属 Pipeline Model Parallel Group 的成员数量
    if mpu.get_pipeline_model_parallel_world_size() > 1:
        if args.virtual_pipeline_model_parallel_size is not None:
            # print_rank_0("===> fb_func = w/ interleaving")
            forward_backward_func = forward_backward_pipelining_with_interleaving
            assert get_num_microbatches() % args.pipeline_model_parallel_size == 0, (
                "number of microbatches is not divisible by pipeline-parallel "
                "size when using interleaved schedule"
            )
        else:
            # print_rank_0("===> fb_func = w/o interleaving")
            forward_backward_func = forward_backward_pipelining_without_interleaving
    else:
        # print_rank_0("===> fb_func = no_pp")
        forward_backward_func = forward_backward_no_pipelining
    # print_rank_0("===> running fb_func")
    losses_reduced = forward_backward_func(
        forward_step_func, data_iterator, model, optimizer, timers, forward_only=False
    )
    # forward_backward_func() 进行 get_num_microbatches() 次前向计算和反向传播积累梯度, 但不进行梯度更新
    # losses_reduced 是一个列表, 里面有 get_num_microbatches() 个元素, 每个元素是类似 {'lm loss': tensor(3.9247)} 的字典
    # 字典中的值表示当前所有数据并行模型在该 micro_batch_size 数据上的平均损失值

    # All-reduce if needed.
    # args.DDP_impl: "local"
    if not args.deepspeed and args.DDP_impl == "local":
        timers("backward-params-all-reduce").start()
        report_memory("(iterations {} before 【all reduce gradients】)".format(args.iteration), True)
        for model_module in model:
            model_module.allreduce_gradients()
            # 将数据并行组内部各个进程的累积梯度 model_param.main_grad 做个平均
        report_memory("(iterations {} after  【all reduce gradients】)".format(args.iteration), True)
        timers("backward-params-all-reduce").stop()

    # All-reduce word_embeddings' grad across first and last stages to ensure
    # that word_embeddings parameters stay in sync.
    # This should only run for models that support pipelined model parallelism
    # (BERT and GPT-2).
    if not args.deepspeed:
        timers("backward-embedding-all-reduce").start()
        if (
            mpu.is_pipeline_first_stage(ignore_virtual=True)
            or mpu.is_pipeline_last_stage(ignore_virtual=True)
        ) and mpu.get_pipeline_model_parallel_world_size() > 1:
            if mpu.is_pipeline_first_stage(ignore_virtual=True):
                unwrapped_model = model[0]
            elif mpu.is_pipeline_last_stage(ignore_virtual=True):
                unwrapped_model = model[-1]
            unwrapped_model = unwrap_model(
                unwrapped_model, (torchDDP, LocalDDP, Float16Module)
            )

            if unwrapped_model.share_word_embeddings:
                word_embeddings_weight = unwrapped_model.word_embeddings_weight()
                if args.DDP_impl == "local":
                    grad = word_embeddings_weight.main_grad
                else:
                    grad = word_embeddings_weight.grad
                torch.distributed.all_reduce(grad, group=mpu.get_embedding_group())
        timers("backward-embedding-all-reduce").stop()

    # Update parameters.
    timers("optimizer").start()
    # print_rank_0("===> start of update params")
    if args.deepspeed:
        increment = (
            get_num_microbatches() * args.micro_batch_size * args.data_parallel_size
        )
        # model[0] is a DeepSpeedEngine object
        # .step(): Execute the weight update step after forward and backward propagation on effective_train_batch
        # 这步才会更新参数, 而在上面的 forward_backward_func() 中不会更新参数
        report_memory("(iterations {} before 【update model】)".format(args.iteration), True)
        model[0].step(lr_kwargs={"increment": increment})
        update_successful = model[0].was_step_applied()
        report_memory("(iterations {} after  【update model】)".format(args.iteration), True)
    else:
        # optimizer: Float16OptimizerWithFloat16Params( FusedAdam(...) )
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
        # optimizer.step() 分为如下几步:
        #   main_param.grad = model_param.main_grad.float()
        #   梯度裁剪和缩放
        #   使用 main_param.grad 更新 main_param
        #   model_param = main_param

    # print_rank_0("===> end of update params")
    timers("optimizer").stop()
    # 下面是从训练时每次迭代的日志中截取的一段关于迭代用时的输出字段 (训练脚本未使用 deepspeed)
    # time (ms) | forward-compute: 152.33 | backward-compute: 157.91 | backward-params-all-reduce: 101.10 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 0.72 | optimizer-unscale-and-check-inf: 18.83 | optimizer-clip-main-grad: 28.35 | optimizer-copy-main-to-model-params: 19.37 | optimizer: 129.43 | batch-generator: 0.64
    # forward-compute 和 backward-compute 分别是在一个 global_batch_size 数据上前向传播 和 反向传播 (不包含梯度更新) 的总耗时
    # optimizer: 129.43 表示完成 optimizer.step() 的总耗时, 由 backward-params-all-reduce 到 optimizer-copy-main-to-model-params 以及 参数更新 阶段构成
    # 注意, 从日志中截取的这段输出字段并不包含 optimizer 进行参数更新的耗时
    # batch-generator: 0.64 是从 dataloader 中取下个批次 global_batch_size 数据 (包含取数据, 分发数据) 的耗时

    # Update learning rate.
    if args.deepspeed:
        skipped_iter = 0
        grad_norm = None
        num_zeros_in_grad = None

        loss_reduced = {}
        for key in losses_reduced[0]:
            losses_reduced_for_key = [x[key] for x in losses_reduced]
            loss_reduced[key] = sum(losses_reduced_for_key) / len(
                losses_reduced_for_key
            )
        # loss_reduced: {'lm loss': 该 global_batch_size 的平均损失}
        return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
    else:
        if update_successful:
            increment = (
                get_num_microbatches() * args.micro_batch_size * args.data_parallel_size
            )
            lr_scheduler.step(increment=increment)
            skipped_iter = 0
        else:
            skipped_iter = 1

        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            # Average loss across microbatches.
            loss_reduced = {}
            # losses_reduced 是一个列表, 里面有 get_num_microbatches() 个元素, 每个元素是类似 {'lm loss': tensor(3.9247)} 的字典
            # 字典中的值表示当前所有数据并行模型在该 micro_batch_size 数据上的平均损失值
            for key in losses_reduced[0]:
                losses_reduced_for_key = [x[key] for x in losses_reduced]
                # losses_reduced_for_key: 将 losses_reduced 中 get_num_microbatches() 个 loss 值取出来放到列表中
                loss_reduced[key] = sum(losses_reduced_for_key) / len(
                    losses_reduced_for_key
                )
                # loss_reduced['lm loss'] = get_num_microbatches() 个 loss 的平均值, 相当于当前所有数据并行模型在该 global_batch_size 数据上的平均损失值
            return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
    return {}, skipped_iter, grad_norm, num_zeros_in_grad


def training_log(
    loss_dict,
    total_loss_dict,
    learning_rate,
    iteration,
    loss_scale,
    report_memory_flag,
    skipped_iter,
    grad_norm,
    params_norm,
    num_zeros_in_grad,
    model=None,
):
    """Log training information such as losses, timing, ...."""
    args = get_args()
    timers = get_timers()
    writer = get_tensorboard_writer()

    # Advanced, skipped, and Nan iterations.
    advanced_iters_key = "advanced iterations"
    skipped_iters_key = "skipped iterations"
    nan_iters_key = "nan iterations"
    # Advanced iterations.
    # skipped_iter: 0
    if not skipped_iter:
        total_loss_dict[advanced_iters_key] = (
            total_loss_dict.get(advanced_iters_key, 0) + 1
        )
    else:
        if advanced_iters_key not in total_loss_dict:
            total_loss_dict[advanced_iters_key] = 0
    # Skipped iterations.
    total_loss_dict[skipped_iters_key] = (
        total_loss_dict.get(skipped_iters_key, 0) + skipped_iter
    )
    # 用 total_loss_dict[advanced_iters_key] 记录当前 iteration 中已成功运行的 iterations 数量 (要么 0 要么 1)
    # 用 total_loss_dict[skipped_iters_key]  记录目前 iteration 中已跳过运行的 iterations 数量 (要么 0 要么 1)

    # Update losses and set nan iterations
    got_nan = False
    for key in loss_dict:
        if not skipped_iter:
            total_loss_dict[key] = (
                total_loss_dict.get(key, torch.cuda.FloatTensor([0.0])) + loss_dict[key]
            )
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float("inf") or value == -float("inf") or value != value
            got_nan = got_nan or is_nan
    total_loss_dict[nan_iters_key] = total_loss_dict.get(nan_iters_key, 0) + int(
        got_nan
    )

    # Logging.
    timers_to_log = []

    def add_to_logging(name):
        if name in timers.timers:
            timers_to_log.append(name)

    add_to_logging("forward-compute")
    add_to_logging("forward-recv")
    add_to_logging("forward-send")
    add_to_logging("forward-backward-send-forward-backward-recv")
    add_to_logging("backward-compute")
    add_to_logging("backward-recv")
    add_to_logging("backward-send")
    add_to_logging("backward-send-forward-recv")
    add_to_logging("backward-send-backward-recv")
    add_to_logging("backward-params-all-reduce")
    add_to_logging("backward-embedding-all-reduce")
    add_to_logging("optimizer-copy-to-main-grad")
    add_to_logging("optimizer-unscale-and-check-inf")
    add_to_logging("optimizer-clip-main-grad")
    add_to_logging("optimizer-copy-main-to-model-params")
    add_to_logging("optimizer")
    add_to_logging("batch-generator")

    # Calculate batch size.
    batch_size = (
        args.micro_batch_size * args.data_parallel_size * get_num_microbatches()
    )

    total_iterations = (
        total_loss_dict[advanced_iters_key] + total_loss_dict[skipped_iters_key]
    )

    # wandb logging.
    # args.wandb_logging: False
    if (
        args.wandb_logging
        and (iteration % args.wandb_log_interval == 0)
        and is_last_rank()
    ):
        wandb.log(
            {
                "train/tokens": args.consumed_train_tokens,
                "train/lr": learning_rate,
            },
            step=iteration,
        )

        for k, v in loss_dict.items():
            wandb.log({f"train/{k}": v}, step=iteration)

        for k in timers_to_log:
            value = timers.timers[k].elapsed(reset=False)
            wandb.log({f"timer/{k}": value}, step=iteration)

    # Tensorboard values.
    # args.tensorboard_log_interval: 1
    # is_last_rank(): 当前进程是否为全局序号末尾的那个进程
    if writer and (iteration % args.tensorboard_log_interval == 0) and is_last_rank():
        writer.add_scalar(
            "steps-vs-samples/y=steps,x=samples", iteration, args.consumed_train_samples
        )
        writer.add_scalar(
            "steps-vs-samples/y=samples,x=steps", args.consumed_train_samples, iteration
        )
        writer.add_scalar(
            "steps-vs-tokens/y=steps,x=tokens", iteration, args.consumed_train_tokens
        )
        writer.add_scalar(
            "steps-vs-tokens/y=tokens,x=steps", args.consumed_train_tokens, iteration
        )
        if args.log_learning_rate_to_tensorboard:
            writer.add_scalar("learning-rate/learning-rate", learning_rate, iteration)
            writer.add_scalar(
                "learning-rate/learning-rate vs samples",
                learning_rate,
                args.consumed_train_samples,
            )
            writer.add_scalar(
                "learning-rate/learning-rate vs tokens",
                learning_rate,
                args.consumed_train_tokens,
            )
        if args.log_batch_size_to_tensorboard:
            writer.add_scalar("batch-size/batch-size", batch_size, iteration)
            writer.add_scalar(
                "batch-size/batch-size vs samples",
                batch_size,
                args.consumed_train_samples,
            )
        for key in loss_dict:
            writer.add_scalar(f"lm-loss-training/{key}", loss_dict[key], iteration)
            # writer.add_scalar(
            #     f"lm-loss-training/{key}" + " vs samples",
            #     loss_dict[key],
            #     args.consumed_train_samples,
            # )
            # writer.add_scalar(
            #     f"lm-loss-training/{key}" + " vs tokens",
            #     loss_dict[key],
            #     args.consumed_train_tokens,
            # )
        if args.log_loss_scale_to_tensorboard:
            writer.add_scalar("loss-scale/loss-scale", loss_scale, iteration)
            writer.add_scalar(
                "loss-scale/loss-scale vs samples",
                loss_scale,
                args.consumed_train_samples,
            )
            writer.add_scalar(
                "loss-scale/loss-scale vs tokens",
                loss_scale,
                args.consumed_train_tokens,
            )
        if grad_norm is not None:
            writer.add_scalar("grad-norm/grad-norm", grad_norm, iteration)
            writer.add_scalar(
                "grad-norm/grad-norm vs samples", grad_norm, args.consumed_train_samples
            )
            writer.add_scalar(
                "grad-norm/grad-norm vs tokens", grad_norm, args.consumed_train_tokens
            )
        if num_zeros_in_grad is not None:
            writer.add_scalar("num-zeros/num-zeros", num_zeros_in_grad, iteration)
            writer.add_scalar(
                "num-zeros/num-zeros vs samples",
                num_zeros_in_grad,
                args.consumed_train_samples,
            )
            writer.add_scalar(
                "num-zeros/num-zeros vs tokens",
                num_zeros_in_grad,
                args.consumed_train_tokens,
            )
        if params_norm is not None:
            writer.add_scalar("params-norm/params-norm", params_norm, iteration)
            writer.add_scalar(
                "params-norm/params-norm vs samples",
                params_norm,
                args.consumed_train_samples,
            )
            writer.add_scalar(
                "params-norm/params-norm vs tokens",
                params_norm,
                args.consumed_train_tokens,
            )
        # args.log_timers_to_tensorboard: False
        if args.log_timers_to_tensorboard:
            timers.write(timers_to_log, writer, iteration, normalizer=total_iterations)

    # args.log_interval: 1
    if iteration % args.log_interval == 0:
        elapsed_time = timers("interval-time").elapsed()
        # timers 有以下两种用法:
        # ①
        # timers("xxx").start()  # A
        # timers("xxx").stop()   # B
        # timers("xxx").start()  # C
        # timers("xxx").stop()   # D
        # elapsed_time = timers("xxx").elapsed()   # elapsed_time = (B-A) + (D-C)
        # timers("xxx").start()  # E
        # timers("xxx").stop()   # F
        # timers("xxx").start()  # G
        # timers("xxx").stop()   # H
        # elapsed_time = timers("xxx").elapsed()   # elapsed_time = (F-E) + (H-G)
        #
        # ②
        # timers("xxx").start()  # A
        # elapsed_time = timers("xxx").elapsed()   # B, elapsed_time = B-A
        # elapsed_time = timers("xxx").elapsed()   # C, elapsed_time = C-B

        elapsed_time_per_iteration = elapsed_time / total_iterations
        # elapsed_time_per_iteration: 当前 iteration 迭代训练一次 global_batch_size 数据所需的时间

        # log iteration time to wandb
        if args.wandb_logging and is_last_rank():
            wandb.log(
                {
                    "train/iteration-time": elapsed_time_per_iteration,
                },
                step=iteration,
            )

        # only the last rank process has a non-None _GLOBAL_TENSORBOARD_WRITER
        if writer and is_last_rank():
            if args.log_timers_to_tensorboard:
                writer.add_scalar(
                    "iteration-time/iteration-time",
                    elapsed_time_per_iteration,
                    iteration,
                )
                writer.add_scalar(
                    "iteration-time/iteration-time vs samples",
                    elapsed_time_per_iteration,
                    args.consumed_train_samples,
                )
                writer.add_scalar(
                    "iteration-time/iteration-time vs tokens",
                    elapsed_time_per_iteration,
                    args.consumed_train_tokens,
                )
        log_string = "==> iteration {:8d}/{:8d} |".format(iteration, args.train_iters)
        log_string += " consumed samples: {:12d} |".format(args.consumed_train_samples)
        log_string += " consumed tokens: {:12d} |".format(args.consumed_train_tokens)
        log_string += " elapsed time per iteration (ms): {:.1f} |".format(
            elapsed_time_per_iteration * 1000.0
        )
        log_string += " learning rate: {:.3E} |".format(learning_rate)
        log_string += " global batch size: {:5d} |".format(batch_size)
        for key in total_loss_dict:
            if key not in [advanced_iters_key, skipped_iters_key, nan_iters_key]:
                avg = total_loss_dict[key].item() / float(
                    max(1, total_loss_dict[advanced_iters_key])
                )
                if avg > 0.0:
                    log_string += " {}: {:.6E} |".format(key, avg)
                total_loss_dict[key] = torch.cuda.FloatTensor([0.0])
        log_string += " loss scale: {:.1f} |".format(loss_scale)
        if grad_norm is not None:
            log_string += " grad norm: {:.3f} |".format(grad_norm)
        if num_zeros_in_grad is not None:
            log_string += " num zeros: {:.1f} |".format(num_zeros_in_grad)
        if params_norm is not None:
            log_string += " params norm: {:.3f} |".format(params_norm)
        log_string += " number of skipped iterations: {:3d} |".format(
            total_loss_dict[skipped_iters_key]
        )
        log_string += " number of nan iterations: {:3d} |".format(
            total_loss_dict[nan_iters_key]
        )

        # 重置
        total_loss_dict[advanced_iters_key] = 0
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[nan_iters_key] = 0
        print_rank_last(log_string)
        # report_memory_flag: True
        if report_memory_flag and learning_rate > 0.0:
            # Report memory after optimizer state has been initialized.
            report_memory("(after {} iterations)".format(iteration))
            report_memory_flag = False
            # 只记录一次显存占用
        # args.log_interval: 1
        timers.log(timers_to_log, normalizer=args.log_interval)
        flops_calculator(model, args, elapsed_time)

    return report_memory_flag


def save_checkpoint_and_time(iteration, model, optimizer, lr_scheduler):
    timers = get_timers()
    # Extra barrier is added to make sure
    # all ranks report the max time.
    torch.distributed.barrier()
    timers("save-checkpoint").start()
    save_checkpoint(iteration, model, optimizer, lr_scheduler)
    torch.distributed.barrier()
    timers("save-checkpoint").stop()
    timers.log(["save-checkpoint"])


def train(
    forward_step_func,
    valid_forward_step_func,
    model,
    optimizer,
    lr_scheduler,
    train_data_iterator,
    valid_data_iterator,
):
    # 如果训练脚本不使用 deepspeed, 那么 model: [ LocalDDP ( Float16Module( CodeGeeXModel(...) ) ) ], 即
    # [DistributedDataParallel(
    #   (module): Float16Module(
    #     (module): CodeGeeXModel(
    #       (language_model): TransformerLanguageModel(
    #         (embedding): Embedding(
    #           (word_embeddings): VocabParallelEmbedding()
    #           (position_embeddings): Embedding(2048, 5120)
    #           (embedding_dropout): Dropout(p=0.1, inplace=False)
    #         )
    #         (topQueryEmbedding): QueryEmbedding(
    #           (top_query_embeddings): VocabParallelEmbedding()
    #           (embedding_dropout): Dropout(p=0.1, inplace=False)
    #         )
    #         (transformer): ParallelTransformer(
    #           (layers): ModuleList(
    #             (0-38): 39 x ParallelTransformerLayer(
    #               (input_layernorm): LayerNorm((5120,), eps=1e-05, elementwise_affine=True)
    #               (attention): ParallelSelfAttention(
    #                 (query): ColumnParallelLinear()
    #                 (key): ColumnParallelLinear()
    #                 (value): ColumnParallelLinear()
    #                 (softmax): Softmax(dim=-1)
    #                 (attention_dropout): Dropout(p=0.1, inplace=False)
    #                 (dense): RowParallelLinear()
    #               )
    #               (post_attention_layernorm): LayerNorm((5120,), eps=1e-05, elementwise_affine=True)
    #               (mlp): ParallelMLP(
    #                 (dense_h_to_4h): ColumnParallelLinear()
    #                 (dense_4h_to_h): RowParallelLinear()
    #               )
    #             )
    #           )
    #           (topQueryLayer): ParallelTopQueryLayer(
    #             (input_layernorm): LayerNorm((5120,), eps=1e-05, elementwise_affine=True)
    #             (attention): ParallelTopQuerySelfAttention(
    #               (query): ColumnParallelLinear()
    #               (key): ColumnParallelLinear()
    #               (value): ColumnParallelLinear()
    #               (softmax): Softmax(dim=-1)
    #               (attention_dropout): Dropout(p=0.1, inplace=False)
    #               (dense): RowParallelLinear()
    #             )
    #             (post_attention_layernorm): LayerNorm((5120,), eps=1e-05, elementwise_affine=True)
    #             (mlp): ParallelMLP(
    #               (dense_h_to_4h): ColumnParallelLinear()
    #               (dense_4h_to_h): RowParallelLinear()
    #             )
    #           )
    #           (final_layernorm): LayerNorm((5120,), eps=1e-05, elementwise_affine=True)
    #         )
    #       )
    #     )
    #   )
    # )]

    # 但因为当前训练脚本使用了 deepspeed, 所以 model: [ DeepSpeedEngine( CodeGeeXModel(...) ) ], 即
    # [DeepSpeedEngine(
    #   (module): CodeGeeXModel(
    #     (language_model): TransformerLanguageModel(
    #       (embedding): Embedding(
    #         (word_embeddings): VocabParallelEmbedding()
    #         (position_embeddings): Embedding(2048, 5120)
    #         (embedding_dropout): Dropout(p=0.1, inplace=False)
    #       )
    #       (topQueryEmbedding): QueryEmbedding(
    #         (top_query_embeddings): VocabParallelEmbedding()
    #         (embedding_dropout): Dropout(p=0.1, inplace=False)
    #       )
    #       (transformer): ParallelTransformer(
    #         (layers): ModuleList(
    #           (0-38): 39 x ParallelTransformerLayer(
    #             (input_layernorm): LayerNorm((5120,), eps=1e-05, elementwise_affine=True)
    #             (attention): ParallelSelfAttention(
    #               (query): ColumnParallelLinear()
    #               (key): ColumnParallelLinear()
    #               (value): ColumnParallelLinear()
    #               (softmax): Softmax(dim=-1)
    #               (attention_dropout): Dropout(p=0.1, inplace=False)
    #               (dense): RowParallelLinear()
    #             )
    #             (post_attention_layernorm): LayerNorm((5120,), eps=1e-05, elementwise_affine=True)
    #             (mlp): ParallelMLP(
    #               (dense_h_to_4h): ColumnParallelLinear()
    #               (dense_4h_to_h): RowParallelLinear()
    #             )
    #           )
    #         )
    #         (topQueryLayer): ParallelTopQueryLayer(
    #           (input_layernorm): LayerNorm((5120,), eps=1e-05, elementwise_affine=True)
    #           (attention): ParallelTopQuerySelfAttention(
    #             (query): ColumnParallelLinear()
    #             (key): ColumnParallelLinear()
    #             (value): ColumnParallelLinear()
    #             (softmax): Softmax(dim=-1)
    #             (attention_dropout): Dropout(p=0.1, inplace=False)
    #             (dense): RowParallelLinear()
    #           )
    #           (post_attention_layernorm): LayerNorm((5120,), eps=1e-05, elementwise_affine=True)
    #           (mlp): ParallelMLP(
    #             (dense_h_to_4h): ColumnParallelLinear()
    #             (dense_4h_to_h): RowParallelLinear()
    #           )
    #         )
    #         (final_layernorm): LayerNorm((5120,), eps=1e-05, elementwise_affine=True)
    #       )
    #     )
    #   )
    # )]

    # 如果训练脚本不使用 deepspeed, 那么 optimizer: Float16OptimizerWithFloat16Params( FusedAdam(...) )
    # 但因为当前训练脚本使用了 deepspeed, 所以 optimizer: FusedAdam(...)

    """Train the model function."""
    args = get_args()
    timers = get_timers()

    # Write args to tensorboard
    write_args_to_tensorboard()

    # args.wandb_logging: False
    if args.wandb_logging:
        torch.distributed.barrier()
        print_datetime("before the initialization of wandb")
        timers("wandb-init").start()
        if is_last_rank():
            initialize_wandb_experiment()
        torch.distributed.barrier()
        timers("wandb-init").stop()
        timers.log(["wandb-init"])

    # Turn on training mode which enables dropout.
    # len(model): 1
    for model_module in model:
        model_module.train()

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    iteration = args.iteration
    # args.iteration: 0
    # args.train_iters: 25

    timers("interval-time").start()
    print_datetime("before the start of training step")
    report_memory_flag = True

    # report_memory_flag 会记录训练时的显存占用信息, 未使用 deepspeed 时相关输出的形式如下:
    # [Rank 1] (after 1 iterations) memory (MB) | allocated: 55809.58203125 | max allocated: 55809.58251953125 | reserved: 56272.0 | max reserved: 56272.0
    # [Rank 3] (after 1 iterations) memory (MB) | allocated: 55809.58203125 | max allocated: 55809.58251953125 | reserved: 56272.0 | max reserved: 56272.0
    # [Rank 2] (after 1 iterations) memory (MB) | allocated: 55809.58203125 | max allocated: 55809.58251953125 | reserved: 56272.0 | max reserved: 56272.0
    # [Rank 0] (after 1 iterations) memory (MB) | allocated: 55810.08203125 | max allocated: 55810.08251953125 | reserved: 56166.0 | max reserved: 56166.0

    # report_memory_flag 会记录训练时的显存占用信息, 使用 deepspeed 时相关输出的形式如下:
    # [2024-03-31 11:14:23,226] [INFO] [stage_1_and_2.py:1651:step] [deepspeed] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 4096, reducing to 4096
    # [2024-03-31 11:14:23,228] [INFO] [logging.py:69:log_dist] [Rank 0] rank=0 time (ms) | forward_microstep: 1425.28 | backward_microstep: 211.13 | backward_inner_microstep: 209.05 | backward_allreduce_microstep: 1.99 | step_microstep: 649.10
    # [2024-03-31 11:14:23,228] [INFO] [logging.py:69:log_dist] [Rank 0] rank=0 time (ms) | forward: 1425.20 | backward: 211.12 | backward_inner: 209.05 | backward_allreduce: 1.99 | step: 649.10

    # args.train_tokens: None
    while iteration < args.train_iters and (
        args.train_tokens is None or args.consumed_train_tokens < args.train_tokens
    ):
        # print_rank_0(f'=> iteration {iteration}')
        # 首次进入时 args.consumed_train_samples: 0
        # update_num_microbatches(args.consumed_train_samples) 因为代码没完成的缘故无事发生
        update_num_microbatches(args.consumed_train_samples)
        if args.deepspeed:
            # inform deepspeed of any batch size changes
            global_batch_size = (
                mpu.get_data_parallel_world_size()
                * args.micro_batch_size
                * get_num_microbatches()
            )
            model[0].set_train_batch_size(global_batch_size)
            # set_train_batch_size 是 DeepSpeedEngine 的方法

        # print_rank_0(f"==> running train step for iteration {iteration}")
        # train_step: 训练一个 global_batch_size 的数据
        loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = train_step(
            forward_step_func, train_data_iterator, model, optimizer, lr_scheduler
        )
        # loss_dict: {'lm loss': 在当前 global_batch_size 数据上计算得到的 dp 组之间 (各个模型) 的平均损失}
        # skipped_iter: 0
        # grad_norm: None
        # num_zeros_in_grad: None
        # 这几个返回值都会展示在训练的日志里

        iteration += 1
        args.iteration = iteration
        new_samples = (
            mpu.get_data_parallel_world_size()
            * args.micro_batch_size
            * get_num_microbatches()
        )
        # new_samples: 等价于 global_batch_size

        # 在训练开始前 args.consumed_train_samples 和 args.consumed_train_tokens 都为 0
        # 每迭代一次后 args.consumed_train_samples 和 args.consumed_train_tokens 就都增加一些
        args.consumed_train_samples += new_samples
        args.consumed_train_tokens += new_samples * args.seq_length

        # Logging.
        if args.deepspeed:
            loss_scale = model[0].optimizer.cur_scale
        else:
            loss_scale = optimizer.get_loss_scale().item()
            # loss_scale: 12.0
        params_norm = None

        # args.log_params_norm: False
        if args.log_params_norm:
            params_norm = calc_params_l2_norm(model)

        report_memory_flag = training_log(
            loss_dict,
            total_loss_dict,
            optimizer.param_groups[0]["lr"],
            iteration,
            loss_scale,
            report_memory_flag,
            skipped_iter,
            grad_norm,
            params_norm,
            num_zeros_in_grad,
            model,
        )
        # training_log(...) 方法记录训练信息, 未使用 deepspeed 时相关输出的形式如下:
        # ==> iteration        8/      25 | consumed samples:           32 | consumed tokens:        16384 | elapsed time per iteration (ms): 548.5 | learning rate: 1.067E-06 | global batch size:     4 | lm loss: 1.495719E+00 | loss scale: 12.0 | grad norm: 157.681 | number of skipped iterations:   0 | number of nan iterations:   0 |
        # time (ms) | forward-compute: 152.33 | backward-compute: 157.91 | backward-params-all-reduce: 101.10 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 0.72 | optimizer-unscale-and-check-inf: 18.83 | optimizer-clip-main-grad: 28.35 | optimizer-copy-main-to-model-params: 19.37 | optimizer: 129.43 | batch-generator: 0.64

        # training_log(...) 方法记录训练信息, 使用 deepspeed 时相关输出的形式如下:
        # ==> iteration        8/      25 | consumed samples:           32 | consumed tokens:        16384 | elapsed time per iteration (ms): 535.8 | learning rate: 2.667E-07 | global batch size:     4 | lm loss: 3.425794E+00 | loss scale: 128.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
        # time (ms) | forward-compute: 177.59 | backward-compute: 197.18 | optimizer: 160.12 | batch-generator: 0.88

        # consumed samples: 当前已迭代的样本数, 一般为 iteration * global_batch_size
        # consumed tokens: 当前已迭代的 tokens 数量, 一般为 consumed samples * s
        # elapsed time per iteration (ms): 本次 iteration 耗时
        # global batch size: 一般为 dp_size * micro_batch_size * 梯度累积次数
        # lm loss: 所有数据并行模型在当前 global_batch_size 数据上的平均损失
        # loss scale: 损失缩放系数
        # forward-compute: 在一个 global_batch_size 数据上前向传播的总耗时, 是数个 micro_batch_size 数据前向传播时间的总和
        # backward-compute: 在一个 global_batch_size 数据上反向传播 (不包含梯度更新) 的总耗时, 是数个 micro_batch_size 数据反向传播时间的总和
        # optimizer: 表示完成 optimizer.step() 的总耗时 (由 backward-params-all-reduce 到 optimizer-copy-main-to-model-params 以及 参数更新 阶段构成), 注意, 从日志中截取的这 2 个输出字段并不包含 optimizer 进行参数更新的耗时
        # batch-generator: 从 dataloader 中取下个批次 global_batch_size 数据 (包含取数据, 分发数据) 的耗时, 是取数个 micro_batch_size 数据时间的总和

        # Autoresume
        # args.adlr_autoresume: False
        if args.adlr_autoresume and (iteration % args.adlr_autoresume_interval == 0):
            check_adlr_autoresume_termination(iteration, model, optimizer, lr_scheduler)

        # Evaluation
        # args.do_valid: 0
        if args.eval_interval and iteration % args.eval_interval == 0 and args.do_valid:
            prefix = "iteration {}".format(iteration)
            if args.co_evaluation:
                for key, value in valid_data_iterator.items():
                    evaluate_and_print_results(
                        prefix, valid_forward_step_func, value, model, iteration, False, tag=key
                    )
            else:
                if args.gold:
                    evaluate_and_print_results_gold(
                        prefix, forward_step_func, valid_data_iterator, model, iteration, False
                    )
                evaluate_and_print_results(
                    prefix, valid_forward_step_func, valid_data_iterator, model, iteration, False
                )

        # Checkpointing
        saved_checkpoint = False
        # args.save: '/home/icksys/csw/CodeGeeX/scripts/pretrain-codegeex-13b-test'
        # args.save_interval: 100
        if args.save and args.save_interval and (iteration % args.save_interval == 0):  # debugging
            save_checkpoint_and_time(iteration, model, optimizer, lr_scheduler)
            # save_checkpoint_and_time(...) 其实运行的也是 save_checkpoint(iteration, model, optimizer, lr_scheduler)
            saved_checkpoint = True

        # Exiting based on duration
        # args.exit_duration_in_mins: None
        if args.exit_duration_in_mins:
            train_time = (time.time() - _TRAIN_START_TIME) / 60.0
            done_cuda = torch.cuda.IntTensor([train_time > args.exit_duration_in_mins])
            torch.distributed.all_reduce(done_cuda, op=torch.distributed.ReduceOp.MAX)
            done = done_cuda.item()
            if done:
                if not saved_checkpoint:
                    save_checkpoint_and_time(iteration, model, optimizer, lr_scheduler)
                print_datetime("exiting program after {} minutes".format(train_time))
                sys.exit()

        # Exiting based on iterations
        # args.exit_interval: None
        if args.exit_interval and iteration % args.exit_interval == 0:
            if not saved_checkpoint:
                save_checkpoint_and_time(iteration, model, optimizer, lr_scheduler)
            torch.distributed.barrier()
            print_datetime("exiting program at iteration {}".format(iteration))
            sys.exit()

    return iteration


def evaluate(forward_step_func, data_iterator, model, verbose=False):
    """Evaluation."""
    args = get_args()

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

    total_loss_dict = {}

    with torch.no_grad():
        iteration = 0
        while iteration < args.eval_iters:
            iteration += 1
            if verbose and iteration % args.log_interval == 0:
                print_rank_0("Evaluating iter {}/{}".format(iteration, args.eval_iters))

            if mpu.get_pipeline_model_parallel_world_size() > 1:
                if args.virtual_pipeline_model_parallel_size is not None:
                    forward_backward_func = (
                        forward_backward_pipelining_with_interleaving
                    )
                else:
                    forward_backward_func = (
                        forward_backward_pipelining_without_interleaving
                    )
            else:
                forward_backward_func = forward_backward_no_pipelining

            if args.deepspeed and not args.no_pipeline_parallel:
                # DeepSpeed uses eval_batch() and already aggregates losses.
                assert isinstance(model, list) and len(model) == 1
                loss = model[0].eval_batch(data_iterator)
                loss_dicts = [{"lm loss": loss}] * get_num_microbatches()
            else:
                loss_dicts = forward_backward_func(
                    forward_step_func,
                    data_iterator,
                    model,
                    optimizer=None,
                    timers=None,
                    forward_only=True,
                )

            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                # Reduce across processes.
                for loss_dict in loss_dicts:
                    for key in loss_dict:
                        total_loss_dict[key] = (
                            total_loss_dict.get(key, torch.cuda.FloatTensor([0.0]))
                            + loss_dict[key]
                        )

            args.consumed_valid_samples += (
                mpu.get_data_parallel_world_size()
                * args.micro_batch_size
                * get_num_microbatches()
            )
    # Move model back to the train mode.
    for model_module in model:
        model_module.train()

    for key in total_loss_dict:
        total_loss_dict[key] /= args.eval_iters * get_num_microbatches()

    return total_loss_dict


def evaluate_and_print_results(
    prefix, forward_step_func, data_iterator, model, iteration, verbose=False, tag=None
):
    """Helper function to evaluate and dump results on screen."""
    args = get_args()
    writer = get_tensorboard_writer()

    total_loss_dict = evaluate(forward_step_func, data_iterator, model, verbose)
    if tag is None:
        string = " validation loss at {} | ".format(prefix)
    else:
        string = " validation loss for {} at {} | ".format(tag, prefix)
    for key in total_loss_dict:
        string += "{} value: {:.6E} | ".format(key, total_loss_dict[key].item())
        ppl = math.exp(min(20, total_loss_dict[key].item()))
        string += "{} PPL: {:.6E} | ".format(key, ppl)

        if tag is not None:
            display_key = tag + "-" + key
        else:
            display_key = key

        if args.wandb_logging and is_last_rank():
            wandb.log(
                {
                    f"eval/{display_key}": total_loss_dict[key].item(),
                },
                step=iteration,
            )

        if writer and is_last_rank():
            writer.add_scalar(
                f"lm-loss-validation/{display_key} validation",
                total_loss_dict[key].item(),
                iteration,
            )
            # writer.add_scalar(
            #     f"lm-loss-validation/{display_key} validation vs samples",
            #     total_loss_dict[key].item(),
            #     args.consumed_train_samples,
            # )
            # writer.add_scalar(
            #     f"lm-loss-validation/{display_key} validation vs tokens",
            #     total_loss_dict[key].item(),
            #     args.consumed_train_tokens,
            # )
            if args.log_validation_ppl_to_tensorboard:
                writer.add_scalar(
                    f"lm-loss-validation/{display_key} validation ppl", ppl, iteration
                )
                writer.add_scalar(
                    f"lm-loss-validation/{display_key} validation ppl vs samples",
                    ppl,
                    args.consumed_train_samples,
                )
                writer.add_scalar(
                    f"lm-loss-validation/{display_key} validation ppl vs tokens",
                    ppl,
                    args.consumed_train_tokens,
                )

    length = len(string) + 1
    print_rank_last("-" * length)
    print_rank_last(string)
    print_rank_last("-" * length)


def evaluate_and_print_results_gold(
    prefix, forward_step_func, data_iterator, model, iteration, verbose=False, tag=None
):
    """Helper function to evaluate and dump results on screen."""
    args = get_args()
    writer = get_tensorboard_writer()

    total_loss_dict = evaluate(forward_step_func, data_iterator, model, verbose)
    if tag is None:
        string = " validation loss (gold) at {} | ".format(prefix)
    else:
        string = " validation loss (gold) for {} at {} | ".format(tag, prefix)
    for key in total_loss_dict:
        string += "{} value: {:.6E} | ".format(key, total_loss_dict[key].item())
        ppl = math.exp(min(20, total_loss_dict[key].item()))
        string += "{} PPL: {:.6E} | ".format(key, ppl)

        if tag is not None:
            display_key = tag + "-" + key
        else:
            display_key = key

        if args.wandb_logging and is_last_rank():
            wandb.log(
                {
                    f"eval/{display_key}": total_loss_dict[key].item(),
                },
                step=iteration,
            )

        if writer and is_last_rank():
            writer.add_scalar(
                f"lm-loss-validation-gold/{display_key} validation",
                total_loss_dict[key].item(),
                iteration,
            )
            if args.log_validation_ppl_to_tensorboard:
                writer.add_scalar(
                    f"lm-loss-validation/{display_key} validation ppl", ppl, iteration
                )
                writer.add_scalar(
                    f"lm-loss-validation/{display_key} validation ppl vs samples",
                    ppl,
                    args.consumed_train_samples,
                )
                writer.add_scalar(
                    f"lm-loss-validation/{display_key} validation ppl vs tokens",
                    ppl,
                    args.consumed_train_tokens,
                )

    length = len(string) + 1
    print_rank_last("-" * length)
    print_rank_last(string)
    print_rank_last("-" * length)
    

def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x


def build_train_valid_test_data_iterators(build_train_valid_test_datasets_provider):
    args = get_args()

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    print_rank_0("> building train, validation, and test datasets ...")

    # Backward compatibility, assume fixed batch size.
    # args.iteration: 0
    # args.consumed_train_samples: 0
    # args.consumed_valid_samples: 0

    # 如果不指定 args.load 参数加载模型权重、优化器状态和 Scheduler 状态, 那么 args.iteration 默认为 0
    # 如果指定   args.load 参数, 那么 args.iteration 则为 checkpoint 中已训练的 iterations 数
    if args.iteration > 0 and args.consumed_train_samples == 0:
        assert (
            args.train_samples is None
        ), "only backward compatibility support for iteration-based training"
        args.consumed_train_samples = args.iteration * args.global_batch_size
    if args.iteration > 0 and args.consumed_valid_samples == 0:
        assert (
            args.train_samples is None
        ), "only backward compatibility support for iteration-based training"
        args.consumed_valid_samples = (
            (args.iteration // args.eval_interval)
            * args.eval_iters
            * args.global_batch_size
        )

    # Data loader only on rank 0 of each model parallel group.
    # 只在张量并行组组内 local rank=0 的进程内构建 dataloader, 其余 gpu 对应的进程返回 dataloader=None
    # 比如对于本次运行/调试脚本 tp=4 pp=1 dp=2 的情况
    # 张量并行组 1: [0, 1, 2, 3]
    # 张量并行组 2: [4, 5, 6, 7]
    # 那么本函数只在 gpu0 和 gpu4 所在的进程内构建 datasets, 该 datasets 是 PromptDataset 对象, 迭代时每次返回一个包含 input_ids、attention_mask、labels 键值对的字典
    # 然后会使用 sampler 和 torch.utils.data.DataLoader 包装下该 datasets 得到 dataloader, 使之成为一个 torch.utils.data.DataLoader 对象, 注意 gpu0 和 gpu4 中的 dataloader 完全一样
    # 遍历该 dataloader 对象时, 张量并行组组内 local rank=0 的每个进程会视自己所在数据并行组的 local rank, 返回指定的 micro_batch_size 个数据
    # 比如对于本次运行/调试脚本 tp=4 pp=1 dp=2 的情况, micro_batch_size = 2
    # 遍历 0 号进程（GPU）的 dataloader 时, 每次取 4（dp_size * micro_batch_size） 个样本序号, 但只返回前 2 个样本序号
    # 遍历 4 号进程（GPU）的 dataloader 时, 每次也是取同 0 号进程完全一样的 4（dp_size * micro_batch_size） 个样本序号, 但只返回后 2 个样本序号
    if mpu.get_tensor_model_parallel_rank() == 0:

        # Number of train/valid/test samples.
        # args.train_samples: None
        if args.train_samples:
            train_samples = args.train_samples
        else:
            train_samples = args.train_iters * args.global_batch_size
        # args.train_iters: 25
        # args.eval_interval: 10
        # args.eval_iters: 10
        eval_iters = (args.train_iters // args.eval_interval + 1) * args.eval_iters
        test_iters = args.eval_iters
        # eval_iters: 30
        # test_iters: 10
        # args.global_batch_size: 4
        train_val_test_num_samples = [
            train_samples,
            eval_iters * args.global_batch_size,
            test_iters * args.global_batch_size,
        ]
        # train_val_test_num_samples: [100, 120, 40]
        print_rank_0(" > datasets target sizes (minimum size):")
        print_rank_0("    train:      {}".format(train_val_test_num_samples[0]))
        print_rank_0("    validation: {}".format(train_val_test_num_samples[1]))
        print_rank_0("    test:       {}".format(train_val_test_num_samples[2]))

        # Build the datasets.
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets_provider(
            train_val_test_num_samples
        )
        # 返回的 train_dataset, valid_dataset, test_dataset 一般都是 PromptDataset 类
        # PromptDataset 类继承自 torch.utils.data.Dataset, 遍历时每次返回一个如下形式的字典, 表示一个样本
        # {
        #     "input_ids": np.array(input_ids, dtype=np.int64),
        #     "attention_mask": np.array(attention_mask, dtype=np.int64),
        #     "labels": np.array(labels, dtype=np.int64),
        # }
        # 当然, 如果 splits_string = [100, 0, 0], 则没有样本用于构建 valid 和 test 数据集, 因此 valid_dataset 和 test_dataset 都为 None

        # Build dataloders.
        # args.consumed_train_samples: 0
        # args.consumed_valid_samples: 0
        # build_pretraining_data_loader 使用 sampler 和 torch.utils.data.DataLoader 包装下 dataset
        train_dataloader = build_pretraining_data_loader(
            train_ds, args.consumed_train_samples
        )
        # args.co_evaluation: False
        if args.co_evaluation:
            valid_dataloader = {}
            for key, value in valid_ds.items():
                valid_dataloader[key] = build_pretraining_data_loader(
                    value, args.consumed_valid_samples
                )
        else:
            valid_dataloader = build_pretraining_data_loader(
                valid_ds, args.consumed_valid_samples
            )
        if args.co_evaluation:
            if test_ds is not None:
                test_dataloader = {}
                for key, value in test_ds.items():
                    test_dataloader[key] = build_pretraining_data_loader(value, 0)
            else:
                test_dataloader = None
        else:
            test_dataloader = build_pretraining_data_loader(test_ds, 0)

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader is not None and args.train_iters > 0
        do_valid = valid_dataloader is not None and args.eval_iters > 0
        do_test = test_dataloader is not None and args.eval_iters > 0
        # Need to broadcast num_tokens and num_type_tokens.
        flags = torch.cuda.LongTensor([int(do_train), int(do_valid), int(do_test)])
    else:
        flags = torch.cuda.LongTensor([0, 0, 0])

    # Broadcast num tokens.
    # 比如对于本次运行/调试脚本 tp=4 pp=1 dp=2 的情况
    # 张量并行组 1: [0, 1, 2, 3]
    # 张量并行组 2: [4, 5, 6, 7]
    # 那么如下函数是将 flags 张量从 GPU0 向 GPU1、GPU2、GPU3 广播, 以及从 GPU4 向 GPU5、GPU6、GPU7 广播
    torch.distributed.broadcast(
        flags,
        mpu.get_tensor_model_parallel_src_rank(),
        group=mpu.get_tensor_model_parallel_group(),
    )
    args.do_train = flags[0].item()
    args.do_valid = flags[1].item()
    args.do_test = flags[2].item()
    # args.do_train: 1
    # args.do_valid: 0
    # args.do_test: 0

    # Build iterators.
    # args.dataloader_type: 'single'
    dl_type = args.dataloader_type
    assert dl_type in ["single", "cyclic"]

    if train_dataloader is not None:
        train_data_iterator = (
            iter(train_dataloader)
            if dl_type == "single"
            else iter(cyclic_iter(train_dataloader))
        )
    else:
        train_data_iterator = None

    # args.co_evaluation: False
    if valid_dataloader is not None:
        if args.co_evaluation:
            valid_data_iterator = {}
            for key, value in valid_dataloader.items():
                valid_data_iterator[key] = (
                    iter(value)
                    if dl_type == "single"
                    else iter(cyclic_iter(value))
                )
        else:
            valid_data_iterator = (
                iter(valid_dataloader)
                if dl_type == "single"
                else iter(cyclic_iter(valid_dataloader))
            )
    else:
        valid_data_iterator = None

    if test_dataloader is not None:
        if args.co_evaluation:
            test_data_iterator = {}
            for key, value in test_dataloader.items():
                test_data_iterator[key] = (
                    iter(value)
                    if dl_type == "single"
                    else iter(cyclic_iter(value))
                )
        else:
            test_data_iterator = (
                iter(test_dataloader)
                if dl_type == "single"
                else iter(cyclic_iter(test_dataloader))
            )
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator
