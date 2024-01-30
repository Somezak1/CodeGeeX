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

from .initialize import get_tensor_model_parallel_group
from .initialize import get_tensor_model_parallel_rank
from .initialize import get_tensor_model_parallel_world_size
from .utils import VocabUtility


class _VocabParallelCrossEntropy(torch.autograd.Function):
    """
    分布式计算Loss
    """
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target):
        # vocab_parallel_logits.shape: [b, s, vocab_size/p], dtype: torch.float32
        # target.shape: [b, s], dtype: torch.int64

        # Maximum value along vocab dimension across all GPUs.
        # 1. logit - global max(logit)操作, 主要目的是防溢出
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        # logits_max.shape: [b, s]
        torch.distributed.all_reduce(
            logits_max,
            op=torch.distributed.ReduceOp.MAX,
            # 找全局最大值
            group=get_tensor_model_parallel_group(),
        )
        # Subtract the maximum value.
        # 原始GPU上维护的logits减去每行最大值（防止溢出）
        vocab_parallel_logits.sub_(logits_max.unsqueeze(dim=-1))
        # vocab_parallel_logits.shape: [b, s, vocab_size/p]

        # Get the partition's vocab indecies
        # 2、根据当前进程id, 取出当前进程所维护词表序号等信息
        # 函数, 能够获取当前进程所维护词表的start_index和end_index
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        # 这块GPU上logits最后一维的大小, 等于所维护的词表的大小
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        # 取得当前进程所在TP组中的序号
        rank = get_tensor_model_parallel_rank()
        # 取得当前进程所在TP组的总进程数
        world_size = get_tensor_model_parallel_world_size()
        # 取得当前进程所维护的词表的start_index和end_index
        vocab_start_index, vocab_end_index = get_vocab_range(
            partition_vocab_size, rank, world_size
        )

        # 3. 基于真值, 取出每个token在真值位置上的logit（即和真值的相似度）
        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        # target.shape: [b, s]
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        # logits_2d.shape: [b*s, vocab_size/p]
        masked_target_1d = masked_target.view(-1)
        # masked_target_1d.shape: [b*s]
        arange_1d = torch.arange(
            start=0, end=logits_2d.size()[0], device=logits_2d.device
        )
        # arange_1d.shape: [b*s]
        # logits_2d.shape: [arange_1d, masked_target_1d]
        # tensor的切片操作。arange_1d: 取出所有的行。masked_target_1d: 取出logit
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        # predicted_logits_1d.shape: [b*s]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        # predicted_logits.shape: [b, s]
        predicted_logits[target_mask] = 0.0
        # All reduce is needed to get the chunks from other GPUs.
        # allreduce之后得到的logit矩阵为[b, s], 每一个位置表示对应真值位置的预测logit
        torch.distributed.all_reduce(
            predicted_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=get_tensor_model_parallel_group(),
        )

        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = vocab_parallel_logits
        # exp_logits.shape: [b, s, vocab_size/p]
        torch.exp(vocab_parallel_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        # sum_exp_logits.shape: [b, s]
        torch.distributed.all_reduce(
            sum_exp_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=get_tensor_model_parallel_group(),
        )

        # Loss = log(sum(exp(logits))) - predicted-logit.
        # 4. 计算Loss = log(sum(exp(logits))) - predicted-logit.
        loss = torch.log(sum_exp_logits) - predicted_logits
        # loss = - (torch.log(exp^(predicted_logits)) - torch.log(sum_exp_logits))
        #      = - (torch.log(exp^(predicted_logits) / sum_exp_logits))
        #      = - (torch.log(yi_hat))
        # loss.shape: [b, s], dtype: torch.float32

        # Store softmax, target-mask and masked-target for backward pass.
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

        return loss

    @staticmethod
    def backward(ctx, grad_output):

        # Retreive tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensors

        # All the inputs have softmax as their gradient.
        grad_input = softmax
        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)
        grad_2d[arange_1d, masked_target_1d] -= 1.0 - target_mask.view(-1).float()

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input, None


def vocab_parallel_cross_entropy(vocab_parallel_logits, target):
    """Helper function for the cross entropy."""
    return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target)
