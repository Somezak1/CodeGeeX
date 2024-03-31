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

import torch

from contextlib import contextmanager
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from codegeex.megatron import get_args
from codegeex.megatron import get_num_microbatches
from codegeex.megatron import get_timers
from codegeex.megatron import mpu
from codegeex.megatron import p2p_communication
from codegeex.megatron.utils import unwrap_model
from codegeex.megatron.model import DistributedDataParallel as LocalDDP
from codegeex.megatron.model import Float16Module


def get_forward_backward_func():
    args = get_args()
    if mpu.get_pipeline_model_parallel_world_size() > 1:
        if args.virtual_pipeline_model_parallel_size is not None:
            forward_backward_func = forward_backward_pipelining_with_interleaving
        else:
            forward_backward_func = forward_backward_pipelining_without_interleaving
    else:
        forward_backward_func = forward_backward_no_pipelining
    return forward_backward_func


def forward_step(forward_step_func, data_iterator, model, input_tensor, losses_reduced):
    """Forward step for passed-in model.

    If first stage, input tensor is obtained from data_iterator, otherwise
    passed-in input_tensor is used.

    Returns output tensor."""

    # input_tensor: None
    # 如果训练脚本不使用deepspeed, 那么 model: LocalDDP ( Float16Module( CodeGeeXModel(...) ) )
    # 但因为当前训练脚本使用了deepspeed, 所以 model: DeepSpeedEngine( CodeGeeXModel(...) )

    timers = get_timers()

    args = get_args()

    timers("forward-compute").start()
    unwrapped_model = unwrap_model(model, (torchDDP, LocalDDP, Float16Module))
    # unwrap_model 帮 model 脱去包装
    # 如果训练脚本不使用deepspeed, 那么 unwrapped_model: CodeGeeXModel(...)
    # 但因为当前训练脚本使用了deepspeed, 所以 unwrapped_model: DeepSpeedEngine( CodeGeeXModel(...) )
    if not args.deepspeed:
        unwrapped_model.set_input_tensor(input_tensor)
    else:
        unwrapped_model.module.set_input_tensor(input_tensor)
    # 流水线并行时才会用到这个input_tensor

    output_tensor, loss_func = forward_step_func(data_iterator, model)
    # output_tensor.shape: [b, s], dtype: torch.float32, 代表该进程所属模型并行组的模型在当前 micro_batch_size 数据上的损失矩阵
    # output_tensor 是一个已经在同一张量并行组内各进程间经过 All Reduce, 未坍缩的损失矩阵, 同一张量并行组内所有进程的 output_tensor 一样

    # mpu.is_pipeline_last_stage(): True
    if mpu.is_pipeline_last_stage():
        output_tensor = loss_func(output_tensor)
        loss, loss_reduced = output_tensor
        # loss: 形如tensor(3.9353), 代表该进程所属模型并行组的模型在当前 micro_batch_size 数据上的损失值, 此时同一张量并行组内各进程的 loss 一致
        # loss_reduced: 形如{"lm loss": : tensor(3.9247)}, 表示当前所有数据并行模型在该 micro_batch_size 数据上的平均损失值

        # get_num_microbatches(): 梯度累积次数
        output_tensor = loss / get_num_microbatches()
        losses_reduced.append(loss_reduced)
    timers("forward-compute").stop()

    return output_tensor


def backward_step(
    optimizer, input_tensor, output_tensor, output_tensor_grad, model=None
):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage)."""

    # 如果训练脚本不使用deepspeed, 那么 optimizer: Float16OptimizerWithFloat16Params( FusedAdam(...) )
    # 但因为当前训练脚本使用了deepspeed, 所以 optimizer: FusedAdam(...)
    # input_tensor: None
    # output_tensor: 当前进程所在模型在当前 micro_batch_size 数据上的平均训练损失 / 梯度累积次数
    # output_tensor_grad: None

    args = get_args()

    if args.deepspeed:
        assert model is not None

    timers = get_timers()
    timers("backward-compute").start()

    # Retain the grad on the input_tensor.
    # 流水线并行时用到
    if input_tensor is not None:
        input_tensor.retain_grad()

    if args.deepspeed:
        # Execute backward pass on the loss
        model.backward(output_tensor)
    else:
        # Backward pass.
        if output_tensor_grad is None:
            output_tensor = optimizer.scale_loss(output_tensor)
            # optimizer.scale_loss(output_tensor): 等价于 torch.cuda.FloatTensor([12.0]) * output_tensor
        torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad)
        # 梯度保存在当前进程模型子块 model_param.grad 中

    # Collect the grad of the input_tensor.
    input_tensor_grad = None
    if input_tensor is not None:
        input_tensor_grad = input_tensor.grad

    timers("backward-compute").stop()

    return input_tensor_grad


@contextmanager
# 上下文管理器是指在一段代码执行之前执行一段代码, 用于一些预处理工作; 执行之后再执行一段代码, 用于一些清理工作
def dummy_handler():
    try:
        yield
    finally:
        pass


def forward_backward_no_pipelining(
    forward_step_func, data_iterator, model, optimizer, timers, forward_only
):
    """Run forward and backward passes with no pipeline parallelism
    (no inter-stage communication).
    # 进行get_num_microbatches()次前向计算和反向传播积累梯度, 但不进行梯度更新

    # 如果训练脚本不使用deepspeed, 那么
    #       model: [ LocalDDP ( Float16Module( CodeGeeXModel(...) ) ) ]
    #       optimizer: Float16OptimizerWithFloat16Params( FusedAdam(...) )
    # 但因为当前训练脚本使用了deepspeed, 所以
    #       model: [ DeepSpeedEngine( CodeGeeXModel(...) ) ]
    #       optimizer: FusedAdam(...)
    # forward_only: False

    Returns dictionary with losses."""
    assert len(model) == 1
    model = model[0]

    args = get_args()

    # 无事发生的上下文管理器
    context_handler = dummy_handler

    # isinstance(model, torchDDP): False
    if isinstance(model, torchDDP):
        context_handler = model.no_sync

    # args.deepspeed: True
    if args.deepspeed:
        # 看set_gradient_accumulation_boundary的函数注释
        # model此时是一个DeepSpeedEngine(Module)类
        model.set_gradient_accumulation_boundary(False)

    losses_reduced = []
    input_tensor, output_tensor_grad = None, None
    with context_handler():
        # get_num_microbatches(): 梯度累积次数
        for i in range(get_num_microbatches() - 1):
            # print_rank_0("====> start of microstep {i}")
            # print_rank_0("====> forward")
            output_tensor = forward_step(
                forward_step_func, data_iterator, model, input_tensor, losses_reduced
            )
            # loss: 形如tensor(3.9353), 代表该进程所属模型并行组的模型在当前 micro_batch_size 数据上的损失值, 此时同一张量并行组内各进程的 loss 一致
            # output_tensor = loss / get_num_microbatches()
            # print_rank_0("====> backward")
            if not forward_only:
                backward_step(
                    optimizer, input_tensor, output_tensor, output_tensor_grad, model
                )
                # backward_step()这步相当于只做了model.backward(output_tensor)
                # 将 output_tensor 反向传播计算的梯度累积到 model_param.main_grad 中
            # print_rank_0("====> end of microstep {i}")

    if args.deepspeed:
        model.set_gradient_accumulation_boundary(True)

    # Run computation for last microbatch out of context handler (want to synchronize gradients).
    # print_rank_0("====> start of the last microstep")
    # print_rank_0("====> forward")
    output_tensor = forward_step(
        forward_step_func, data_iterator, model, input_tensor, losses_reduced
    )
    # print_rank_0("====> backward")
    if not forward_only:
        backward_step(optimizer, input_tensor, output_tensor, output_tensor_grad, model)
    # print_rank_0("====> end of the last microstep")

    return losses_reduced


def forward_backward_pipelining_with_interleaving(
    forward_step_func, data_iterator, model, optimizer, timers, forward_only
):
    """Run interleaved 1F1B schedule (model split into model chunks), with
    communication between pipeline stages as needed.

    Returns dictionary with losses if the last stage, empty dict otherwise."""
    input_tensors = [[] for _ in range(len(model))]
    output_tensors = [[] for _ in range(len(model))]
    losses_reduced = []
    if not forward_only:
        output_tensor_grads = [[] for _ in range(len(model))]

    pipeline_parallel_size = mpu.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = mpu.get_pipeline_model_parallel_rank()

    # Compute number of warmup and remaining microbatches.
    num_model_chunks = len(model)
    num_microbatches = get_num_microbatches() * num_model_chunks
    all_warmup_microbatches = False
    if forward_only:
        num_warmup_microbatches = num_microbatches
    else:
        # Run all forward passes and then all backward passes if number of
        # microbatches is just the number of pipeline stages.
        # Otherwise, perform (num_model_chunks-1)*pipeline_parallel_size on
        # all workers, followed by more microbatches after depending on
        # stage ID (more forward passes for earlier stages, later stages can
        # immediately start with 1F1B).
        if get_num_microbatches() == pipeline_parallel_size:
            num_warmup_microbatches = num_microbatches
            all_warmup_microbatches = True
        else:
            num_warmup_microbatches = (
                pipeline_parallel_size - pipeline_parallel_rank - 1
            ) * 2
            num_warmup_microbatches += (num_model_chunks - 1) * pipeline_parallel_size
            num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches

    def get_model_chunk_id(microbatch_id, forward):
        """Helper method to get the model chunk ID given the iteration number."""
        microbatch_id_in_group = microbatch_id % (
            pipeline_parallel_size * num_model_chunks
        )
        model_chunk_id = microbatch_id_in_group // pipeline_parallel_size
        if not forward:
            model_chunk_id = num_model_chunks - model_chunk_id - 1
        return model_chunk_id

    def forward_step_helper(microbatch_id):
        """Helper method to run forward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        forward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=True)
        mpu.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        if mpu.is_pipeline_first_stage():
            if len(input_tensors[model_chunk_id]) == len(
                output_tensors[model_chunk_id]
            ):
                input_tensors[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id][-1]
        output_tensor = forward_step(
            forward_step_func,
            data_iterator[model_chunk_id],
            model[model_chunk_id],
            input_tensor,
            losses_reduced,
        )
        output_tensors[model_chunk_id].append(output_tensor)

        return output_tensor

    def backward_step_helper(microbatch_id):
        """Helper method to run backward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        backward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=False)
        mpu.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        if mpu.is_pipeline_last_stage():
            if len(output_tensor_grads[model_chunk_id]) == 0:
                output_tensor_grads[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id].pop(0)
        output_tensor = output_tensors[model_chunk_id].pop(0)
        output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)
        input_tensor_grad = backward_step(
            optimizer, input_tensor, output_tensor, output_tensor_grad
        )

        return input_tensor_grad

    # Run warmup forward passes.
    mpu.set_virtual_pipeline_model_parallel_rank(0)
    input_tensors[0].append(p2p_communication.recv_forward(timers))
    for k in range(num_warmup_microbatches):
        output_tensor = forward_step_helper(k)

        # Determine if tensor should be received from previous stage.
        next_forward_model_chunk_id = get_model_chunk_id(k + 1, forward=True)
        recv_prev = True
        if mpu.is_pipeline_first_stage(ignore_virtual=True):
            if next_forward_model_chunk_id == 0:
                recv_prev = False
        if k == (num_microbatches - 1):
            recv_prev = False

        # Don't send tensor downstream if on last stage.
        if mpu.is_pipeline_last_stage():
            output_tensor = None

        # Send and receive tensors as appropriate (send tensors computed
        # in this iteration; receive tensors for next iteration).
        if (
            k == (num_warmup_microbatches - 1)
            and not forward_only
            and not all_warmup_microbatches
        ):
            input_tensor_grad = None
            recv_next = True
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                recv_next = False
            (
                input_tensor,
                output_tensor_grad,
            ) = p2p_communication.send_forward_backward_recv_forward_backward(
                output_tensor,
                input_tensor_grad,
                recv_prev=recv_prev,
                recv_next=recv_next,
                timers=timers,
            )
            output_tensor_grads[num_model_chunks - 1].append(output_tensor_grad)
        else:
            input_tensor = p2p_communication.send_forward_recv_forward(
                output_tensor, recv_prev, timers
            )
        input_tensors[next_forward_model_chunk_id].append(input_tensor)

    # Run 1F1B in steady state.
    for k in range(num_microbatches_remaining):
        # Forward pass.
        forward_k = k + num_warmup_microbatches
        output_tensor = forward_step_helper(forward_k)

        # Backward pass.
        backward_k = k
        input_tensor_grad = backward_step_helper(backward_k)

        # Send output_tensor and input_tensor_grad, receive input_tensor
        # and output_tensor_grad.

        # Determine if current stage has anything to send in either direction,
        # otherwise set tensor to None.
        forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
        mpu.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)
        if mpu.is_pipeline_last_stage():
            output_tensor = None

        backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
        mpu.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)
        if mpu.is_pipeline_first_stage():
            input_tensor_grad = None

        # Determine if peers are sending, and where in data structure to put
        # received tensors.
        recv_prev = True
        if mpu.is_pipeline_first_stage(ignore_virtual=True):
            # First stage is ahead of last stage by (pipeline_parallel_size - 1).
            next_forward_model_chunk_id = get_model_chunk_id(
                forward_k - (pipeline_parallel_size - 1), forward=True
            )
            if next_forward_model_chunk_id == (num_model_chunks - 1):
                recv_prev = False
            next_forward_model_chunk_id += 1
        else:
            next_forward_model_chunk_id = get_model_chunk_id(
                forward_k + 1, forward=True
            )

        recv_next = True
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            # Last stage is ahead of first stage by (pipeline_parallel_size - 1).
            next_backward_model_chunk_id = get_model_chunk_id(
                backward_k - (pipeline_parallel_size - 1), forward=False
            )
            if next_backward_model_chunk_id == 0:
                recv_next = False
            next_backward_model_chunk_id -= 1
        else:
            next_backward_model_chunk_id = get_model_chunk_id(
                backward_k + 1, forward=False
            )

        # If last iteration, don't receive; we already received one extra
        # before the start of the for loop.
        if k == (num_microbatches_remaining - 1):
            recv_prev = False

        # Communicate tensors.
        (
            input_tensor,
            output_tensor_grad,
        ) = p2p_communication.send_forward_backward_recv_forward_backward(
            output_tensor,
            input_tensor_grad,
            recv_prev=recv_prev,
            recv_next=recv_next,
            timers=timers,
        )

        # Put input_tensor and output_tensor_grad in data structures in the
        # right location.
        if recv_prev:
            input_tensors[next_forward_model_chunk_id].append(input_tensor)
        if recv_next:
            output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)

    # Run cooldown backward passes (flush out pipeline).
    if not forward_only:
        if all_warmup_microbatches:
            output_tensor_grads[num_model_chunks - 1].append(
                p2p_communication.recv_backward(timers)
            )
        for k in range(num_microbatches_remaining, num_microbatches):
            input_tensor_grad = backward_step_helper(k)
            next_backward_model_chunk_id = get_model_chunk_id(k + 1, forward=False)
            recv_next = True
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                if next_backward_model_chunk_id == (num_model_chunks - 1):
                    recv_next = False
            if k == (num_microbatches - 1):
                recv_next = False
            output_tensor_grads[next_backward_model_chunk_id].append(
                p2p_communication.send_backward_recv_backward(
                    input_tensor_grad, recv_next, timers
                )
            )

    return losses_reduced


def forward_backward_pipelining_without_interleaving(
    forward_step_func, data_iterator, model, optimizer, timers, forward_only
):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages.

    Returns dictionary with losses if the last stage, empty dict otherwise."""
    timers = get_timers()

    assert len(model) == 1
    model = model[0]

    # Compute number of warmup microbatches.
    num_microbatches = get_num_microbatches()
    num_warmup_microbatches = (
        mpu.get_pipeline_model_parallel_world_size()
        - mpu.get_pipeline_model_parallel_rank()
        - 1
    )
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches

    input_tensors = []
    output_tensors = []
    losses_reduced = []

    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        input_tensor = p2p_communication.recv_forward(timers)
        output_tensor = forward_step(
            forward_step_func, data_iterator, model, input_tensor, losses_reduced
        )
        p2p_communication.send_forward(output_tensor, timers)

        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        input_tensor = p2p_communication.recv_forward(timers)

    # Run 1F1B in steady state.
    for i in range(num_microbatches_remaining):
        last_iteration = i == (num_microbatches_remaining - 1)

        output_tensor = forward_step(
            forward_step_func, data_iterator, model, input_tensor, losses_reduced
        )
        if forward_only:
            p2p_communication.send_forward(output_tensor, timers)
        else:
            output_tensor_grad = p2p_communication.send_forward_recv_backward(
                output_tensor, timers
            )

        # Add input_tensor and output_tensor to end of list, then pop from the
        # start of the list for backward pass.
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

        if forward_only:
            if not last_iteration:
                input_tensor = p2p_communication.recv_forward(timers)
        else:
            input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)

            input_tensor_grad = backward_step(
                optimizer, input_tensor, output_tensor, output_tensor_grad, model
            )

            if last_iteration:
                input_tensor = None
                p2p_communication.send_backward(input_tensor_grad, timers)
            else:
                input_tensor = p2p_communication.send_backward_recv_forward(
                    input_tensor_grad, timers
                )

    # Run cooldown backward passes.
    if not forward_only:
        for i in range(num_warmup_microbatches):
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            output_tensor_grad = p2p_communication.recv_backward(timers)

            input_tensor_grad = backward_step(
                optimizer, input_tensor, output_tensor, output_tensor_grad, model
            )

            p2p_communication.send_backward(input_tensor_grad, timers)

    return losses_reduced
